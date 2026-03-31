import asyncio
import yaml
from pathlib import Path
from typing import Dict, List, Callable, Awaitable, Optional, Any, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.text import Text
# Use relative import assuming execution context is properly set up,
# or absolute import if running as a package (e.g., from src.cli.agent import ...)
from .agent import SpatialOmicsAgent, AgentResponse
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document


class SlashCommandCompleter(Completer):
    """
    Advanced custom completer that provides a dropdown menu with
    command descriptions (meta-information) and dynamic filtering.
    """

    def __init__(self, commands: Dict[str, str]):
        self.commands_dict: Dict[str, str] = commands

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        # Capture exactly what the user has typed before the cursor
        text: str = document.text_before_cursor

        # Only trigger the completion menu if the input starts with a slash
        if not text.startswith('/'):
            return

        # If there's a space, the user has finished typing the command, stop completing
        if " " in text:
            return

        # Extract the current word being typed (e.g., "/" or "/a")
        current_word: str = text.split(" ")[0]

        # Filter and yield matching commands dynamically
        for cmd, description in self.commands_dict.items():
            if cmd.startswith(current_word):
                yield Completion(
                    text=cmd,
                    # Replace the partially typed text with the full command
                    start_position=-len(current_word),
                    # The actual text shown in the left column
                    display=cmd,
                    # The description shown in the right column (like Claude Code)
                    display_meta=description
                )


class AgentCLI:
    """
    Command-Line Interface manager for the Spatial Omics tool.
    Handles the asynchronous REPL and slash command routing.
    """

    def __init__(self, work_dir: Path):
        self.work_dir: Path = work_dir
        self.console: Console = Console()
        self.agent: SpatialOmicsAgent = SpatialOmicsAgent()

        # Load configuration
        self.config: Optional[Dict[str, Any]] = self._load_config()
        self.cli_state: str = "INIT"  # States: INIT, READY, RUNNING

        # Define native slash commands and their descriptions
        self.commands: Dict[str, str] = {
            "/help": "Show available commands and usage.",
            "/model": "View current model configuration from config.yaml.",
            "/data": "View current dataset configuration from config.yaml.",
            "/hypothesis": "View current hypothesis configuration from config.yaml.",
            "/status": "Display diagnostic status of AnalysisContext (READY state check).",
            "/start": "Execute the full 7-action spatial omics analysis pipeline demo.",
            "/clear": "Clear the terminal screen.",
            "/exit": "Safely shutdown the CLI session."
        }

        # Initialize the interactive auto-completer
        self.completer: SlashCommandCompleter = SlashCommandCompleter(self.commands)

        self.session: PromptSession = PromptSession(completer=self.completer)

    def _load_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from config.yaml."""
        config_path = Path(__file__).parent / "config.yaml"

        if not config_path.exists():
            self.console.print(f"[bold red]Config file not found:[/bold red] {config_path}")
            return None

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.console.print(f"[bold red]Error loading config:[/bold red] {e}")
            return None

    async def run_loop(self) -> None:
        """
        The core asynchronous REPL that maintains the terminal session.
        """
        self._print_banner()

        while True:
            try:
                # Asynchronously wait for user input
                raw_input: str = await self.session.prompt_async("OmicsNavigator> ")
                text: str = raw_input.strip()

                if not text:
                    continue

                # Command Routing Logic
                if text == "/exit":
                    self.console.print("[dim]Shutting down OmicsNavigator Agent interface...[/dim]")
                    break
                elif text == "/help":
                    self._show_help()
                elif text == "/clear":
                    self.console.clear()
                    self._print_banner()
                elif text == "/model":
                    self._cmd_model()
                elif text == "/data":
                    self._cmd_data()
                elif text == "/hypothesis":
                    self._cmd_hypothesis()
                elif text == "/status":
                    self._cmd_status()
                elif text == "/start":
                    await self._cmd_start()
                elif text.startswith("/"):
                    self.console.print(
                        f"[bold red]Unknown command:[/bold red] {text}. Type [bold cyan]/help[/bold cyan] for options.")
                else:
                    # Pass standard inputs to the mock LLM agent
                    response: AgentResponse = self.agent.process_query(text)
                    self._render_response(response)

            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully without exiting the application
                continue
            except EOFError:
                # Handle Ctrl+D to exit
                self.console.print("\n[dim]Session terminated by user (EOF).[/dim]")
                break

    def _render_response(self, response: AgentResponse) -> None:
        """
        Translates the strict dataclass schema into rich terminal formatting.
        """
        self.console.print(Markdown(response.content))
        if response.tool_used:
            self.console.print(f"[dim italic]System log: Triggered backend tool -> {response.tool_used}[/dim italic]\n")
        else:
            self.console.print()  # Add a newline for readability

    def _show_help(self) -> None:
        """
        Renders a formatted table or list of available slash commands.
        """
        help_text: str = "\n".join([f"**{cmd}** : {desc}" for cmd, desc in self.commands.items()])
        panel = Panel(Markdown(help_text), title="Available Commands", border_style="cyan")
        self.console.print(panel)
        self.console.print()

    # ==========================================
    # Config-based Command Handlers
    # ==========================================

    def _cmd_model(self) -> None:
        """
        Display current model configuration from config.yaml.
        """
        if self.config is None:
            self.console.print("[bold red]Error:[/bold red] Configuration not loaded.")
            self.console.print("[yellow]Hint:[/yellow] Ensure config.yaml exists in src/cli/")
            return

        system_config = self.config.get("system", {})

        if not system_config or "default_model" not in system_config:
            self.console.print("[bold yellow]Warning:[/bold yellow] Model configuration not found in config.yaml")
            self.console.print("[dim]Required: system.default_model[/dim]")
            return

        # Create a table for model info
        table = Table(title="Model Configuration", border_style="cyan")
        table.add_column("Parameter", style="bold cyan")
        table.add_column("Value", style="green")

        table.add_row("Default Model", system_config.get("default_model", "Not set"))
        table.add_row("API Key Env Var", system_config.get("api_key_env_var", "Not set"))
        table.add_row("Log Level", system_config.get("log_level", "Not set"))
        table.add_row("Max Workers", str(system_config.get("max_workers", "Not set")))

        self.console.print(table)
        self.console.print()

    def _cmd_data(self) -> None:
        """
        Display current dataset configuration from config.yaml.
        """
        if self.config is None:
            self.console.print("[bold red]Error:[/bold red] Configuration not loaded.")
            return

        session_config = self.config.get("session", {})
        metadata_config = self.config.get("metadata", {})

        if not session_config:
            self.console.print("[bold yellow]Warning:[/bold yellow] Dataset configuration not found in config.yaml")
            self.console.print("[dim]Required: session section with data_path[/dim]")
            return

        # Create a table for dataset info
        table = Table(title="Dataset Configuration", border_style="blue")
        table.add_column("Parameter", style="bold blue")
        table.add_column("Value", style="green")

        data_path = session_config.get("data_path", "Not set")
        output_dir = session_config.get("output_dir", "Not set")

        # Check if paths exist
        data_path_obj = Path(data_path)
        path_status = "[green]✓ Exists[/green]" if data_path_obj.exists() else "[red]✗ Not found[/red]"

        table.add_row("Data Path", data_path)
        table.add_row("Path Status", path_status)
        table.add_row("Output Directory", output_dir)

        self.console.print(table)

        # Show metadata if available
        if metadata_config:
            self.console.print("\n[bold]Metadata Context:[/bold]")
            meta_table = Table(show_header=False, box=None, padding=(0, 2))
            meta_table.add_column("Key", style="cyan")
            meta_table.add_column("Value", style="white")

            for key, value in metadata_config.items():
                meta_table.add_row(key.capitalize(), str(value))

            self.console.print(meta_table)

        self.console.print()

    def _cmd_hypothesis(self) -> None:
        """
        Display current hypothesis configuration from config.yaml.
        """
        if self.config is None:
            self.console.print("[bold red]Error:[/bold red] Configuration not loaded.")
            return

        hypotheses = self.config.get("hypotheses", [])

        if not hypotheses:
            self.console.print("[bold yellow]Warning:[/bold yellow] No hypotheses defined in config.yaml")
            self.console.print("[dim]Required: hypotheses section with at least one hypothesis[/dim]")
            self.console.print("\n[yellow]Example configuration:[/yellow]")
            example_yaml = """
hypotheses:
  - id: "H1"
    description: "Your hypothesis description here..."
"""
            self.console.print(Markdown(f"```yaml{example_yaml}```"))
            return

        # Create a table for hypotheses
        table = Table(title=f"Defined Hypotheses ({len(hypotheses)})", border_style="magenta")
        table.add_column("ID", style="bold magenta", width=8)
        table.add_column("Description", style="white", width=60)

        for hyp in hypotheses:
            hyp_id = hyp.get("id", "Unknown")
            description = hyp.get("description", "No description")
            table.add_row(hyp_id, description)

        self.console.print(table)
        self.console.print()

    def _cmd_status(self) -> None:
        """
        Diagnostic command to verify the AnalysisContext.
        Displays dataset, metadata keys, and hypothesis status.
        """
        if self.config is None:
            self.console.print("[bold red]Error:[/bold red] Configuration not loaded.")
            return

        # Check all components
        checks = {
            "Model": False,
            "Dataset": False,
            "Metadata": False,
            "Hypothesis": False
        }

        # Check Model
        system_config = self.config.get("system", {})
        if system_config and "default_model" in system_config:
            checks["Model"] = True

        # Check Dataset
        session_config = self.config.get("session", {})
        data_path = session_config.get("data_path", "") if session_config else ""
        if data_path and Path(data_path).exists():
            checks["Dataset"] = True

        # Check Metadata
        metadata_config = self.config.get("metadata", {})
        if metadata_config and all(k in metadata_config for k in ["technology", "tissue"]):
            checks["Metadata"] = True

        # Check Hypothesis
        hypotheses = self.config.get("hypotheses", [])
        if hypotheses:
            checks["Hypothesis"] = True

        # Build status table
        table = Table(title="AnalysisContext Status", border_style="white")
        table.add_column("Component", style="bold white")
        table.add_column("Status", width=12)
        table.add_column("Details", style="dim")

        for component, status in checks.items():
            status_text = "[green]✓ OK[/green]" if status else "[red]✗ Missing[/red]"
            details = ""

            if component == "Model" and status:
                details = f"Using: {system_config.get('default_model')}"
            elif component == "Model" and not status:
                details = "Set system.default_model in config.yaml"

            elif component == "Dataset" and status:
                details = f"Path: {data_path}"
            elif component == "Dataset" and not status:
                details = "Set session.data_path in config.yaml"

            elif component == "Metadata" and status:
                details = f"{metadata_config.get('technology')} / {metadata_config.get('tissue')}"
            elif component == "Metadata" and not status:
                details = "Set metadata.technology and metadata.tissue"

            elif component == "Hypothesis" and status:
                details = f"{len(hypotheses)} hypothesis(es) defined"
            elif component == "Hypothesis" and not status:
                details = "Add hypotheses to config.yaml"

            table.add_row(component, status_text, details)

        self.console.print(table)

        # Overall status
        all_ok = all(checks.values())
        if all_ok:
            self.cli_state = "READY"
            self.console.print("\n[bold green]✓ System Status: READY[/bold green]")
            self.console.print("[dim]All components validated. You can proceed with analysis.[/dim]")
        else:
            missing = [k for k, v in checks.items() if not v]
            self.cli_state = "INIT"
            self.console.print(f"\n[bold yellow]⚠ System Status: INIT[/bold yellow]")
            self.console.print(f"[dim]Missing components: {', '.join(missing)}[/dim]")
            self.console.print("[yellow]Please configure the missing items in config.yaml[/yellow]")

        self.console.print()

    async def _cmd_start(self) -> None:
        """
        Execute the spatial omics analysis pipeline demo.
        Only works when cli_state == 'READY'.
        """
        from .pipeline.orchestrator import PipelineOrchestrator

        # State validation
        if self.cli_state != "READY":
            self.console.print("[bold red]Error:[/bold red] Pipeline execution requires READY state")
            self.console.print("[yellow]Hint:[/yellow] Run [bold]/status[/bold] to verify system readiness")
            return

        # Create and execute pipeline (pass config for LangGraph support)
        orchestrator = PipelineOrchestrator(
            self.console,
            self.session,
            self.cli_state,
            self.config  # Pass config to support LangGraph agents
        )
        await orchestrator.execute_pipeline()

    def _print_banner(self) -> None:
        """
        Renders a Claude-Code style welcome banner with a pixel art logo,
        split-pane layout, and system information.
        """
        # 1. Define the Pixel Art Logo using Unicode blocks and Rich color tags
        # Replace this string with your own generated ASCII/ANSI art
        bot_color: str = "#AEC6CF"  # Light pastel blue
        eye_color: str = "#000000"  # Black

        # Using a multiline f-string to draw the bot
        logo_art: str = (
            f"[{bot_color}]▄▄▄▄▄▄▄ [/]\n"
            f"[{bot_color}]██[{eye_color}]█[{bot_color}]█[{eye_color}]█[{bot_color}]██[/]\n"
            f"[{bot_color}]█▀▀▀▀▀█[/]\n"
            f"[{bot_color}]█     █  [/]"
        )

        # 2. Construct the Left Column Content
        model_name: str = self.agent.current_model
        workspace_path: str = str(self.work_dir.resolve())

        left_content: str = (
            "\n[bold]Welcome back![/bold]\n\n"
            f"{logo_art}\n\n"
            f"[dim]{model_name} · Local Environment\n"
            f"{workspace_path}[/dim]"
        )

        # 3. Construct the Right Column Content
        right_color: str = "#D58B8B"  # Muted red/pink from the screenshot
        right_content: str = (
            f"\n[bold {right_color}]Tips for getting started[/bold {right_color}]\n"
            "Run [bold]/help[/bold] to see available commands or [bold]/model[/bold] to switch backends.\n\n"
            f"[bold {right_color}]Recent activity[/bold {right_color}]\n"
            "[dim]No recent activity found.[/dim]\n"
        )

        # 4. Create a borderless Grid Table for layout
        grid: Table = Table.grid(expand=True, padding=(0, 2))
        # Left column for logo (centered), Right column for text (left-aligned)
        grid.add_column(justify="center", ratio=4)
        grid.add_column(justify="left", ratio=6)

        # Add the content to the grid, adding a vertical separator line conceptually
        # Rich's grid doesn't draw internal lines by default, which matches the UI goal
        grid.add_row(Align.center(left_content), right_content)

        # 5. Wrap everything in a styled Panel
        banner: Panel = Panel(
            grid,
            title="[bold {right_color}]OmicsNavigator v0.1.0[/]",
            title_align="left",
            border_style=right_color,
            padding=(1, 2)
        )

        self.console.print(banner)
        self.console.print()  # Add padding before the prompt starts


async def app_entry() -> None:
    """
    Standard entry point for the asyncio event loop.
    """
    # Utilizing Pathlib for robust directory resolution
    base_dir: Path = Path.cwd() / "data"
    base_dir.mkdir(parents=True, exist_ok=True)

    cli = AgentCLI(work_dir=base_dir)
    await cli.run_loop()


if __name__ == "__main__":
    asyncio.run(app_entry())