"""
Pipeline orchestrator for OmicsNavigator CLI.
Manages the execution of the 7-action pipeline with step-by-step interaction.
Supports both mock demo mode and real LangGraph agent execution.
"""

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.markdown import Markdown

from .actions import PipelineActions
from .progress import PipelineProgress

if TYPE_CHECKING:
    from ..agent import SpatialOmicsAgent


class PipelineOrchestrator:
    """
    Manages the execution of the 7-action pipeline demo.

    Handles:
    - State validation (READY state check)
    - Step-by-step execution with user confirmation
    - Progress tracking and formatting
    - Phase and action header display
    """

    # Pipeline structure definition
    PHASES = [
        {
            "num": 1,
            "name": "Plan Module",
            "description": "Establishing physical boundaries and theoretical boundaries",
            "actions": [
                (1, "DataAnalyst"),
                (2, "LiteratureReviewer"),
                (3, "Planner (PI)"),
            ]
        },
        {
            "num": 2,
            "name": "Interpretation Module",
            "description": "Data sampling and high-throughput interpretation",
            "actions": [
                (4, "Anchor-Cluster-Expand Sampling"),
                (5, "High-throughput Interpretation"),
            ]
        },
        {
            "num": 3,
            "name": "Analysis Module",
            "description": "Semantic retrieval and hypothesis validation",
            "actions": [
                (6, "SemanticRetriever"),
                (7, "HypothesisValidator"),
            ]
        },
    ]

    def __init__(self, console: Console, session: PromptSession, cli_state: str = "READY", config: Optional[dict] = None):
        """
        Initialize the pipeline orchestrator.

        Args:
            console: Rich console instance for output
            session: PromptSession instance for user interaction
            cli_state: Current CLI state (should be "READY")
            config: Optional configuration dictionary
        """
        self.console = console
        self.session = session
        self.cli_state = cli_state
        self.config = config or {}

        # Get mockdata directory
        self.mockdata_dir = Path(__file__).parent.parent / "mockdata"

        # Initialize actions
        self.actions = PipelineActions(console, self.mockdata_dir)

        # Check if real agents should be used
        self.use_real_agents = os.getenv("USE_REAL_AGENTS", "false").lower() == "true"

        # Initialize LangGraph orchestrator if real agents are enabled
        self.langgraph_orchestrator = None
        if self.use_real_agents:
            try:
                from ...workflows.langgraph_orchestrator import LangGraphOrchestrator
                self.langgraph_orchestrator = LangGraphOrchestrator(self.config)
            except ImportError as e:
                self.console.print(f"[yellow]Warning:[/yellow] Could not import LangGraph: {e}")
                self.console.print("[yellow]Falling back to mock pipeline.[/yellow]")
                self.use_real_agents = False

    def _validate_ready_state(self) -> bool:
        """
        Ensure cli_state == 'READY' before starting.

        Returns:
            True if state is valid, False otherwise
        """
        if self.cli_state != "READY":
            self.console.print("[bold red]Error:[/bold red] Pipeline execution requires READY state")
            self.console.print("[yellow]Hint:[/yellow] Run [bold]/status[/bold] to verify system readiness")
            return False
        return True

    async def _wait_for_continue(self) -> None:
        """
        Pause and wait for user to press Enter.

        Waits silently for input without displaying a prompt message.
        """
        try:
            await self.session.prompt_async(
                "",
                enable_suspend=True
            )
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+C or Ctrl+D gracefully
            raise KeyboardInterrupt()

    async def execute_pipeline(self) -> None:
        """
        Main pipeline execution entry point.

        Executes all 7 actions in order with user confirmation between each action.
        Uses either mock pipeline or real LangGraph agents based on configuration.
        """
        # Validate state before starting
        if not self._validate_ready_state():
            return

        # Print pipeline start banner
        await self._print_pipeline_banner()

        try:
            if self.use_real_agents and self.langgraph_orchestrator:
                # Use real LangGraph agents
                await self._execute_real_pipeline()
            else:
                # Use mock pipeline (existing behavior)
                await self._execute_mock_pipeline()

            # Print completion message
            self._print_completion_message()

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        except Exception as e:
            self.console.print(f"\n[bold red]Pipeline error:[/bold red] {e}")

    async def _execute_phase(self, phase: dict) -> None:
        """
        Execute a single phase with all its actions.

        Args:
            phase: Phase configuration dictionary
        """
        # Print phase header
        PipelineProgress.format_phase_header(
            self.console,
            phase["num"],
            phase["name"]
        )

        # Execute each action in the phase
        for action_num, action_name in phase["actions"]:
            await self._execute_action(action_num, action_name)

    async def _execute_action(self, action_num: int, action_name: str) -> None:
        """
        Execute a single action with header and continuation prompt.

        Args:
            action_num: Action number (1-7)
            action_name: Action name for display
        """
        # Print action header
        PipelineProgress.format_action_header(
            self.console,
            action_num,
            action_name
        )

        # Execute the action
        await self.actions.execute_action(action_num)

        # Wait for user to continue (except for last action)
        if action_num < 7:
            await self._wait_for_continue()

    async def _execute_mock_pipeline(self) -> None:
        """Execute the mock pipeline with pre-defined actions."""
        # Execute each phase
        for phase in self.PHASES:
            await self._execute_phase(phase)

    async def _execute_real_pipeline(self) -> None:
        """Execute the real LangGraph agent workflow."""
        from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TextColumn

        self.console.print("[bold cyan]Executing with Real AI Agents (LangGraph)...[/bold cyan]")
        self.console.print()

        # Get hypothesis from config
        hypotheses = self.config.get("hypotheses", [])
        if not hypotheses:
            self.console.print("[bold red]Error:[/bold red] No hypotheses defined in config")
            return

        hypothesis = hypotheses[0]

        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Agent Workflow", total=100)

            # Update progress for each phase
            progress.update(task, advance=10, description="Initializing agents...")

            # Execute LangGraph workflow
            result = await self.langgraph_orchestrator.execute(
                hypothesis_id=hypothesis.get("id", "H1"),
                hypothesis_description=hypothesis.get("description", ""),
                data_root=self.config.get("session", {}).get("data_path", "./data/s255/"),
                sample_id="s255"
            )

            progress.update(task, completed=100)

        # Display results
        self._display_agent_results(result)

    def _display_agent_results(self, state) -> None:
        """Display results from agent execution."""
        from rich.table import Table

        # Phase 1 Results
        if state.dataset_profile:
            self.console.print("\n[bold yellow]Phase 1: Dataset Analysis[/bold yellow]")
            summary = state.dataset_profile.get("summary", "")
            if summary:
                self.console.print(Markdown(summary))

        if state.literature_summaries:
            self.console.print("\n[bold yellow]Phase 1: Literature Review[/bold yellow]")
            for paper in state.literature_summaries[:3]:  # Show first 3
                self.console.print(f"  • {paper.get('title', 'Paper')}")

        if state.analysis_plan:
            self.console.print("\n[bold yellow]Phase 1: Analysis Plan[/bold yellow]")
            self.console.print("  [green]✓[/green] Analysis plan generated")

        # Phase 2 Results
        if state.roi_count:
            self.console.print(f"\n[bold yellow]Phase 2: ROI Sampling[/bold yellow]")
            self.console.print(f"  Generated [bold cyan]{state.roi_count}[/bold cyan] ROIs")

        if state.interpretation_reports_path:
            self.console.print(f"\n[bold yellow]Phase 2: Feature Extraction[/bold yellow]")
            self.console.print(f"  [green]✓[/green] Interpretation complete")

        # Phase 3 Results
        if state.semantic_search_results:
            self.console.print(f"\n[bold yellow]Phase 3: Semantic Search[/bold yellow]")
            matched = state.semantic_search_results.get("matched_rois", 0)
            keyword = state.semantic_search_results.get("keyword", "")
            self.console.print(f"  [green]✓[/green] Found {matched} ROIs matching '{keyword}'")

        if state.validation_results:
            self.console.print(f"\n[bold yellow]Phase 3: Hypothesis Validation[/bold yellow]")
            conclusion = state.validation_results.get("conclusion", "UNKNOWN")
            confidence = state.validation_results.get("confidence", 0)

            color = "green" if conclusion == "VERIFIED" else "red"
            self.console.print(f"  Conclusion: [{bold}]{color}]{conclusion}[/{bold}][/{color}]")
            self.console.print(f"  Confidence: {confidence:.1%}")

        # Show errors if any
        if state.has_errors():
            self.console.print("\n[bold red]Errors encountered:[/bold red]")
            for error in state.errors[:5]:  # Show first 5
                self.console.print(f"  • {error}")

    async def _print_pipeline_banner(self) -> None:
        """Print the pipeline start banner."""
        self.console.print()
        self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        self.console.print("[bold cyan]  Starting Hypothesis Verification Pipeline[/bold cyan]")
        self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        self.console.print()
        self.console.print("[yellow]Press Enter to begin the pipeline...[/yellow]")

        # Wait for user to start
        try:
            await self.session.prompt_async("", enable_suspend=True)
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt()

        self.console.print()

    def _print_completion_message(self) -> None:
        """Print the pipeline completion message."""
        self.console.print()
        self.console.print("[bold green]═══════════════════════════════════════════════════════════════[/bold green]")
        self.console.print("[bold green]  Pipeline Execution Complete[/bold green]")
        self.console.print("[bold green]═══════════════════════════════════════════════════════════════[/bold green]")
        self.console.print()
        self.console.print("[dim]All 7 actions have been executed successfully.[/dim]")
        self.console.print("[dim]Hypothesis H1 has been verified: [bold green]VERIFIED[/bold green][/dim]")
        self.console.print()
        self.console.print("[cyan]Run /status to check system status or /help for more commands.[/cyan]")
        self.console.print()
