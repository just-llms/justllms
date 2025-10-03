import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import questionary
from questionary import Style
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from justllms import JustLLM
from justllms.sxs.core import ParallelExecutor
from justllms.sxs.models import ResponseStatus

console = Console()

custom_style = Style(
    [
        ("qmark", "fg:#00d7ff bold"),
        ("question", "bold"),
        ("answer", "fg:#00ff00 bold"),
        ("pointer", "fg:#00d7ff bold"),
        ("highlighted", "fg:#00d7ff bold"),
        ("selected", "fg:#00ff00"),
        ("separator", "fg:#808080"),
        ("instruction", "fg:#808080 italic"),
        ("text", ""),
    ]
)


def get_all_providers() -> List[str]:
    """Get list of all supported providers."""
    return ["openai", "anthropic", "google", "xai", "deepseek", "azure_openai", "ollama"]


def select_providers_checkbox() -> List[str]:
    """Select providers using checkbox interface."""
    console.print("\n[bold cyan]Select providers to compare:[/bold cyan]\n")

    providers = get_all_providers()

    selected = questionary.checkbox(
        "Select providers (use space to select, enter to confirm):",
        choices=providers,
        style=custom_style,
        instruction="(Use arrow keys to navigate, space to select, enter to confirm)",
    ).ask()

    if not selected:
        console.print("[red]No providers selected. Please select at least one.[/red]")
        return select_providers_checkbox()

    return list(selected)


def collect_api_keys(providers: List[str]) -> Dict[str, str]:
    """Collect API keys for selected providers."""
    api_keys = {}

    console.print("\n[bold cyan]API Key Configuration:[/bold cyan]\n")

    for provider in providers:
        if provider == "ollama":
            default_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
            base_url = Prompt.ask(
                f"Enter base URL for [bold]{provider}[/bold]",
                default=default_base,
            )
            if base_url:
                os.environ["OLLAMA_API_BASE"] = base_url
                os.environ["OLLAMA_ENABLED"] = "1"
                console.print(f"✓ {provider}: [green]Using base URL {base_url}[/green]")
            else:
                console.print(
                    f"[yellow]⚠ {provider} will use default base URL {default_base}[/yellow]"
                )
            continue

        env_var = f"{provider.upper()}_API_KEY"
        existing_key = os.environ.get(env_var)

        if existing_key:
            console.print(f"✓ {provider}: [green]Using existing key from {env_var}[/green]")
            api_keys[provider] = existing_key
        else:
            key = Prompt.ask(f"Enter API key for [bold]{provider}[/bold]", password=True)
            if key:
                api_keys[provider] = key
                os.environ[env_var] = key
            else:
                console.print(f"[yellow]⚠ Skipping {provider} (no API key provided)[/yellow]")

    return api_keys


def get_all_available_models(client: Optional[JustLLM], provider: str) -> List[str]:
    """Get all available models for a provider using dynamic model discovery."""
    try:
        if client and hasattr(client, "list_models"):
            models_info = client.list_models(provider=provider)
            if isinstance(models_info, dict) and provider in models_info:
                provider_models = models_info[provider]
                if isinstance(provider_models, dict):
                    return list(provider_models.keys())
                elif isinstance(provider_models, list):
                    return provider_models

        from justllms.providers import get_provider_class

        provider_class = get_provider_class(provider)
        if provider_class:
            from justllms.core.models import ProviderConfig

            provider_kwargs: Dict[str, Any] = {"name": provider}
            if getattr(provider_class, "requires_api_key", True):
                provider_kwargs["api_key"] = "temp"
            else:
                env_base = os.environ.get(f"{provider.upper()}_API_BASE") or os.environ.get(
                    f"{provider.upper()}_HOST"
                )
                if env_base:
                    provider_kwargs["base_url"] = env_base

            temp_config = ProviderConfig(**provider_kwargs)
            temp_provider = provider_class(temp_config)
            models = temp_provider.get_available_models()
            return list(models.keys())

    except Exception as e:
        console.print(f"[dim]Could not fetch models for {provider}: {e}[/dim]")

    fallback_models = {
        "openai": ["gpt-4o", "gpt-4o-mini"],
        "anthropic": ["claude-3-5-sonnet-20241022"],
        "google": ["gemini-1.5-flash"],
        "deepseek": ["deepseek-chat"],
        "xai": ["grok-3"],
        "grok": ["grok-3"],
        "azure_openai": ["gpt-4o"],
        "ollama": ["llama3.1:8b"],
    }

    return fallback_models.get(provider, [])


def select_models_checkbox(
    client: Optional[JustLLM], providers: List[str]
) -> List[Tuple[str, str]]:
    """Select models using checkbox interface for each provider."""
    selected_models = []

    console.print("\n[bold cyan]Select models for comparison:[/bold cyan]\n")

    for provider in providers:
        console.print(f"\n[bold yellow]{provider.upper()} Models:[/bold yellow]")

        # Get all available models for this provider
        models = get_all_available_models(client, provider)

        if not models:
            # If no models found, ask for custom input
            console.print(f"[yellow]No models found for {provider}[/yellow]")
            custom = Prompt.ask(f"Enter custom model names for {provider} (comma-separated)")
            for model in custom.split(","):
                model = model.strip()
                if model:
                    selected_models.append((provider, model))
        else:
            # Use checkbox selection for models
            selected = questionary.checkbox(
                f"Select {provider} models:",
                choices=models,
                style=custom_style,
                instruction="(Space to select, Enter to confirm)",
            ).ask()

            if selected:
                for model in selected:
                    selected_models.append((provider, model))
            else:
                console.print(f"[yellow]No models selected for {provider}[/yellow]")

    if not selected_models:
        console.print("[red]No models selected! Please select at least one model.[/red]")
        return select_models_checkbox(client, providers)

    return selected_models


def get_prompt() -> str:
    """Get prompt from user."""
    console.print("\n[bold cyan]Enter your prompt:[/bold cyan]")
    console.print(
        "[dim]Tip: For multiple lines, use \\n (e.g., 'Compare:\\nPython\\nJavaScript')[/dim]\n"
    )

    # Simple single-line text input
    prompt = questionary.text("Prompt:", style=custom_style).ask()

    # Convert \n to actual newlines if present
    if prompt and "\\n" in prompt:
        prompt = prompt.replace("\\n", "\n")

    return prompt if prompt else ""


def display_live_results(
    models: List[Tuple[str, str]],
    prompt: str,
    executor: Any,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Display results in real-time using Rich Live."""
    console.print("\n" + "=" * 80)
    console.print(f"\n[bold cyan]Prompt:[/bold cyan]\n{prompt}\n")
    console.print("=" * 80 + "\n")

    # Store responses as they complete
    responses = {f"{p}/{m}": "" for p, m in models}
    completed = {f"{p}/{m}": False for p, m in models}

    # Animation frames for loading
    loading_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    frame_index = 0

    def create_display() -> Any:
        """Create the current display layout."""
        nonlocal frame_index

        panels = []
        for model_id in responses:
            if completed[model_id]:
                content = responses[model_id] or ""
                border_style = "green"
            else:
                frame = loading_frames[frame_index % len(loading_frames)]
                content = f"[dim]Generating... {frame}[/dim]"
                border_style = "yellow"

            panel = Panel(
                Text(content),
                title=f"[bold]{model_id}[/bold]",
                expand=True,
                border_style=border_style,
            )
            panels.append(panel)

        frame_index += 1
        return Group(*panels)

    def on_model_complete(model_id: str, result: Any) -> None:
        """Callback for model completion."""
        completed[model_id] = True
        if result.status == ResponseStatus.COMPLETED:
            responses[model_id] = result.content
        else:
            responses[model_id] = f"[red]Error: {result.error}[/red]"

    with Live(create_display(), refresh_per_second=10, console=console) as live:
        import threading
        import time

        animation_running = True

        def update_animation() -> None:
            while animation_running and not all(completed.values()):
                live.update(create_display())
                time.sleep(0.1)

        animation_thread = threading.Thread(target=update_animation)
        animation_thread.start()

        # Execute comparison
        results = executor.execute_comparison(
            prompt=prompt,
            models=models,
            temperature=temperature,
            max_tokens=max_tokens,
            on_model_complete=on_model_complete,
        )

        # Stop animation
        animation_running = False
        animation_thread.join()

        # Final update
        live.update(create_display())

    return dict(results)


def run_interactive_sxs() -> None:
    """Run the interactive SxS comparison flow with checkbox selection."""
    console.print("[bold cyan]🚀 JustLLMs Side-by-Side Comparison[/bold cyan]\n")

    # Step 1: Select providers using checkboxes
    selected_providers = select_providers_checkbox()
    if not selected_providers:
        console.print("[red]No providers selected. Exiting.[/red]")
        sys.exit(1)
    console.print(f"\n[green]✓ Selected providers:[/green] {', '.join(selected_providers)}")

    # Step 2: Collect API keys
    collect_api_keys(selected_providers)

    # Initialize client with API keys
    try:
        client = JustLLM()
    except Exception as e:
        console.print(f"[yellow]Warning: JustLLM client initialization issue: {e}[/yellow]")
        console.print("[dim]Will continue with manual configuration[/dim]\n")
        client = None

    # Step 3: Select models using checkboxes
    selected_models = select_models_checkbox(client, selected_providers)
    for provider, model in selected_models:
        console.print(f"  - {provider}/{model}")

    # Step 4: Get prompt
    prompt = get_prompt()
    if not prompt:
        console.print("[red]No prompt provided. Exiting.[/red]")
        sys.exit(1)

    # Reinitialize client to ensure API keys are loaded
    try:
        client = JustLLM()
    except Exception as e:
        console.print(f"[red]Error initializing client: {e}[/red]")
        sys.exit(1)

    executor = ParallelExecutor(client)

    # Use live display
    results = display_live_results(
        models=selected_models, prompt=prompt, executor=executor, temperature=0.7, max_tokens=None
    )

    # Show final metrics
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]Metrics Summary:[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", min_width=20)
    table.add_column("Status", justify="center")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost ($)", justify="right")

    for model_id, response in results.items():
        status_style = "green" if response.status == ResponseStatus.COMPLETED else "red"
        status_text = "✓ Success" if response.status == ResponseStatus.COMPLETED else "✗ Error"

        table.add_row(
            model_id,
            f"[{status_style}]{status_text}[/{status_style}]",
            f"{response.latency:.2f}" if response.latency is not None else "-",
            str(response.tokens) if response.tokens is not None else "-",
            f"{response.cost:.4f}" if response.cost is not None else "-",
        )

    console.print(table)

    # Ask if user wants to continue
    console.print("\n")
    if Confirm.ask("Would you like to run another comparison?"):
        run_interactive_sxs()


if __name__ == "__main__":
    try:
        run_interactive_sxs()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
