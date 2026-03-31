"""
Pipeline action implementations for OmicsNavigator CLI demo.
Each action corresponds to a step in the spatial omics analysis pipeline.
"""

import asyncio
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .progress import PipelineProgress


class PipelineActions:
    """Individual action implementations with mock data."""

    def __init__(self, console: Console, mockdata_dir: Path):
        self.console = console
        self.mockdata_dir = mockdata_dir

    def _load_mock_data(self, filename: str) -> str:
        """Load mock data from file."""
        file_path = self.mockdata_dir / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                return f.read()
        return ""

    async def action_1_data_analyst(self) -> None:
        """
        Action 1: DataAnalyst
        Print dataset analysis (sparsity, cell types, spatially variable genes).
        Each section title is displayed first, followed by a delay, then the content.
        """
        import asyncio

        content = self._load_mock_data("H1_DataAnalyst.txt")

        # Split content into sections based on [X/3] markers
        lines = content.split("\n")
        sections = []
        current_section = []

        for line in lines:
            if line.strip().startswith("[") and "/3]" in line:
                # This is a section header
                if current_section:
                    sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        # Don't forget the last section
        if current_section:
            sections.append("\n".join(current_section))

        # Print the separator line first
        if sections and "===" in sections[0]:
            # Find and print the separator from the first section
            first_section_lines = sections[0].split("\n")
            for line in first_section_lines:
                if line.startswith("==="):
                    self.console.print(line)
                    break

        # Process each section
        for i, section in enumerate(sections):
            lines = section.split("\n")
            header_line = None
            content_lines = []

            for line in lines:
                if line.strip().startswith("[") and "/3]" in line:
                    header_line = line
                elif line:
                    content_lines.append(line)

            # Print header
            if header_line:
                self.console.print(header_line)

                # Wait 2 seconds to simulate processing
                await asyncio.sleep(2)

            # Print content
            for line in content_lines:
                self.console.print(line)

        # Print final separator if exists in original content
        if content.strip().endswith("=" * 60):
            self.console.print()
            self.console.print("=" * 60)

    async def action_2_literature_reviewer(self) -> None:
        """
        Action 2: LiteratureReviewer
        Print literature summaries (3 papers).
        """
        import asyncio

        # Print log message
        self.console.print("[dim]Querying literature database for domain-specific biological priors...[/dim]")
        await asyncio.sleep(2)
        self.console.print()

        content = self._load_mock_data("H1_LiteratureReviewer.txt")

        # Parse the literature summaries by finding paper boundaries
        lines = content.split("\n")
        papers = []
        current_paper = {}

        for line in lines:
            line = line.rstrip()
            # Check if this is a field header
            if line.startswith("Title:"):
                # Save previous paper if exists
                if current_paper:
                    papers.append(current_paper)
                current_paper = {"title": line.replace("Title: ", "")}
            elif line.startswith("Source:"):
                current_paper["source"] = line.replace("Source: ", "")
            elif line.startswith("Relevance (Why Selected):"):
                current_paper["relevance"] = []
            elif line.startswith("UI Tag:"):
                current_paper["ui_tag"] = line.replace("UI Tag: ", "")
            elif not line:  # Empty line
                continue
            elif "relevance" in current_paper and isinstance(current_paper.get("relevance"), list):
                # This is relevance content
                current_paper["relevance"].append(line)

        # Don't forget the last paper
        if current_paper:
            papers.append(current_paper)

        # Display each paper with delay
        for i, paper in enumerate(papers):
            # Add delay between papers (except first)
            if i > 0:
                await asyncio.sleep(2)

            title = paper.get("title", "")
            source = paper.get("source", "")
            relevance_lines = paper.get("relevance", [])
            ui_tag = paper.get("ui_tag", "")

            # Join relevance lines
            relevance = " ".join(relevance_lines) if relevance_lines else ""

            # Format as panel
            paper_content = f"[bold cyan]Title:[/bold cyan] {title}\n\n"
            paper_content += f"[bold]Source:[/bold] {source}\n\n"
            paper_content += f"[bold yellow]Relevance (Why Selected):[/bold yellow]\n{relevance}\n\n"
            paper_content += f"[dim]UI Tag: {ui_tag}[/dim]"

            panel = Panel(paper_content, border_style="blue", padding=(0, 1))
            self.console.print(panel)
            self.console.print()

    async def action_3_planner(self) -> None:
        """
        Action 3: Planner (PI)
        Print background knowledge and analysis plan with enhanced visual formatting.
        """
        import asyncio

        # Print log message
        self.console.print("[dim]Synthesizing executable blueprint from domain knowledge and constraints...[/dim]")
        await asyncio.sleep(2)
        self.console.print()

        content = self._load_mock_data("H1_Planner.txt")

        # Parse and format the content with better visual hierarchy
        lines = content.split("\n")
        current_section = None
        content_lines = []

        for line in lines:
            line = line.rstrip()

            # Main section headers (=== Title ===)
            if line.startswith("===") and line.endswith("==="):
                # Print previous content if exists
                if content_lines:
                    self._print_planner_content(content_lines)
                    content_lines = []

                # Extract title from === Title ===
                title = line.replace("=== ", "").replace(" ===", "")
                self.console.print()
                self.console.print(f"[bold cyan]═══ {title} ═══[/bold cyan]")

            # Subsection headers ([1], [2], etc.) - but NOT [x] or [SYSTEM]
            elif (line.strip().startswith("[") and "] " in line and
                  not line.strip().startswith("[x]") and
                  not line.strip().startswith("[SYSTEM]") and
                  not line.strip().startswith("[CRITICAL")):
                # Print previous content if any
                if content_lines:
                    self._print_planner_content(content_lines)
                    content_lines = []

                current_section = line.strip()
                # Print subsection header
                self.console.print()
                self.console.print(f"[bold yellow]{line}[/bold yellow]")

            # Separator lines (------)
            elif line.strip().startswith("-----"):
                if content_lines:
                    self._print_planner_content(content_lines)
                    content_lines = []
                self.console.print(f"[dim]{line}[/dim]")

            # Status indicators line with dashes
            elif line.strip().startswith("--------------------------------------------------"):
                continue

            # Empty lines
            elif not line.strip():
                continue

            # Regular content
            else:
                content_lines.append(line)

        # Print remaining content
        if content_lines:
            self._print_planner_content(content_lines)

        self.console.print()

    def _print_planner_content(self, lines: list) -> None:
        """Print planner content with appropriate formatting."""
        from rich.text import Text

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Status indicators ([x])
            if line.strip().startswith("[x]"):
                # Use Text object to properly escape [x] tag
                text = Text()
                text.append("[x] ", style="green")
                text.append(line[4:], style="green")
                self.console.print(text)

            # Bullet points (*)
            elif line.strip().startswith("* "):
                self.console.print(f"[dim cyan]{line}[/dim cyan]")

            # Indented items (with -)
            elif line.strip().startswith("  -"):
                self.console.print(f"[dim white]{line}[/dim white]")

            # Critical override notes
            elif "[CRITICAL OVERRIDE]" in line:
                # Use Text object to properly escape the tag
                text = Text()
                idx = line.index("[CRITICAL OVERRIDE]")
                text.append(line[:idx], style="bold red")
                text.append("[CRITICAL OVERRIDE]", style="bold red")
                text.append(line[idx+19:], style="bold red")
                self.console.print(text)

            # Dimension headers
            elif line.strip().startswith("Dimension "):
                self.console.print(f"[bold magenta]{line}[/bold magenta]")

            # Plan status and target
            elif line.strip().startswith("Plan STATUS:") or line.strip().startswith("TARGET HYPOTHESIS"):
                self.console.print(f"[bold]{line}[/bold]")

            # Section dividers (---)
            elif line.strip().startswith("---") and not line.strip().startswith("-----"):
                self.console.print(f"[dim]{line}[/dim]")

            # System messages
            elif line.strip().startswith("[SYSTEM]"):
                # Use Text object to properly escape the tag
                text = Text()
                text.append("[SYSTEM] ", style="bold green")
                text.append(line[8:], style="bold green")
                self.console.print(text)

            # Guardrails applied line
            elif "Guardrails Applied:" in line:
                self.console.print(f"[bold]{line}[/bold]")

            # Regular content
            else:
                self.console.print(line)

    def _print_planner_section(self, title: str, lines: list) -> None:
        """Print a planner section (placeholder for compatibility)."""
        pass

    async def action_4_anchor_cluster_expand(self) -> None:
        """
        Action 4: Anchor-Cluster-Expand sampling
        10-second progress bar for ROI sampling.
        """
        description = "[Phase 2] Sampling ROIs using Anchor-Cluster-Expand algorithm..."
        await PipelineProgress.show_progress(self.console, description, 10.0)
        PipelineProgress.format_completion(
            self.console,
            "Sampling complete: 10,000 ROIs generated and stored in data/s255/s255_pivot_ROIs_registry.pkl"
        )

    async def action_5_high_throughput_interpretation(self) -> None:
        """
        Action 5: High-throughput interpretation
        10-second progress bar for interpretation + FAISS storage.
        """
        description = "[Phase 2] Generating OmicsProfile and VisualProfile..."
        await PipelineProgress.show_progress(self.console, description, 10.0)
        PipelineProgress.format_completion(
            self.console,
            "Interpretation complete: 10,000 ROIs processed"
        )
        PipelineProgress.format_completion(
            self.console,
            "All interpretation reports stored in FAISS database"
        )

    async def action_6_semantic_retriever(self) -> None:
        """
        Action 6: SemanticRetriever
        3-second progress + FAISS search results.
        """
        self.console.print("[Phase 3] Searching FAISS database with keyword: 'Proximal Tubules'...")

        # Show search progress
        await PipelineProgress.show_progress(self.console, "Searching...", 3.0)

        # Display results
        result_table = Table(title="Search Results", border_style="green")
        result_table.add_column("Metric", style="bold")
        result_table.add_column("Value", style="cyan")

        result_table.add_row("Keyword", "Proximal Tubules")
        result_table.add_row("Matched ROIs", "160")
        result_table.add_row("Database", "FAISS vector index")

        self.console.print(result_table)
        self.console.print()

    async def action_7_hypothesis_validator(self) -> None:
        """
        Action 7: HypothesisValidator
        Print conclusion: VERIFIED with findings.
        """
        import asyncio

        # Print log message
        self.console.print("[dim]Generating analysis report and validating hypothesis...[/dim]")
        await asyncio.sleep(2)
        self.console.print()

        content = self._load_mock_data("H1_HypothesisValidator.txt")

        # Parse and format the conclusion
        lines = [line.rstrip() for line in content.strip().split("\n") if line.rstrip()]

        # Print conclusion header
        conclusion_line = lines[0]
        self.console.print(f"[bold green]{conclusion_line}[/bold green]")
        self.console.print()

        # Print main finding
        if len(lines) > 1:
            finding = lines[1]
            self.console.print(f"[bold]{finding}[/bold]")
            self.console.print()

        # Known section headers in this file
        section_headers = ["Key Evidence", "Follow-up Recommendation"]

        # Parse sections by looking for section headers and bullet points
        current_section = None
        section_content = []

        for line in lines[2:]:
            # Check if this is a section header
            if line in section_headers:
                # Print previous section if exists
                if current_section and section_content:
                    self.console.print(f"[bold yellow]{current_section}[/bold yellow]")
                    for item in section_content:
                        self.console.print(item)
                    self.console.print()

                current_section = line
                section_content = []
            else:
                # Add content to current section
                section_content.append(f"  {line}")

        # Print last section
        if current_section and section_content:
            self.console.print(f"[bold yellow]{current_section}[/bold yellow]")
            for item in section_content:
                self.console.print(item)

        self.console.print()

    async def execute_action(self, action_num: int) -> None:
        """
        Execute a specific action by number.

        Args:
            action_num: Action number (1-7)
        """
        action_methods = {
            1: self.action_1_data_analyst,
            2: self.action_2_literature_reviewer,
            3: self.action_3_planner,
            4: self.action_4_anchor_cluster_expand,
            5: self.action_5_high_throughput_interpretation,
            6: self.action_6_semantic_retriever,
            7: self.action_7_hypothesis_validator,
        }

        action_names = {
            1: "DataAnalyst",
            2: "LiteratureReviewer",
            3: "Planner (PI)",
            4: "Anchor-Cluster-Expand Sampling",
            5: "High-throughput Interpretation",
            6: "SemanticRetriever",
            7: "HypothesisValidator",
        }

        if action_num in action_methods:
            method = action_methods[action_num]

            # Check if method is async
            if asyncio.iscoroutinefunction(method):
                await method()
            else:
                method()
        else:
            self.console.print(f"[bold red]Error:[/bold red] Invalid action number: {action_num}")
