import sys

import click


@click.group()
@click.version_option()
def main() -> None:
    pass


@main.command()
def sxs() -> None:
    """Launch side-by-side model comparison."""
    from justllms.sxs.cli import run_interactive_sxs

    try:
        run_interactive_sxs()
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        sys.exit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.group()
def tools() -> None:
    """Tool discovery and management commands."""
    pass


@tools.command()
@click.option("--provider", "-p", help="Filter tools by provider (e.g., 'google', 'all')")
@click.option("--native", is_flag=True, help="Show only native provider tools")
def list(provider: str, native: bool) -> None:
    """List available tools.

    Examples:
        justllms tools list
        justllms tools list --provider google
        justllms tools list --native
    """
    from justllms.tools.registry import GlobalToolRegistry

    try:
        # Show native tools if requested
        if native or provider:
            if provider == "google" or (native and not provider):
                click.echo("Native Tools (Google):")
                click.echo("-" * 60)

                from justllms.tools.native.google_tools import GOOGLE_NATIVE_TOOLS

                for _tool_name, tool_class in GOOGLE_NATIVE_TOOLS.items():
                    tool_instance = tool_class()
                    click.echo(f"\n  {tool_instance.name}")
                    click.echo(f"    Description: {tool_instance.description}")
                    click.echo(f"    Namespace: {tool_instance.namespace}")
                    click.echo("    Provider: google")

            if not native and provider not in ["google", "all"]:
                click.echo(f"No native tools available for provider: {provider}", err=True)

        # Show registered user tools
        if not native:
            registry = GlobalToolRegistry()
            registered_tools = registry.list_tools()

            if registered_tools:
                click.echo("\n\nRegistered User Tools:")
                click.echo("-" * 60)
                for tool_name in registered_tools:
                    tool = registry.get_tool(tool_name)
                    if tool:
                        click.echo(f"\n  {tool.name}")
                        click.echo(f"    Description: {tool.description}")
                        if tool.namespace:
                            click.echo(f"    Namespace: {tool.namespace}")
            else:
                if not provider and not native:
                    click.echo("\nNo user tools registered.")
                    click.echo("Use @tool decorator or Client.register_tools() to add tools.")

    except Exception as e:
        click.echo(f"Error listing tools: {e}", err=True)
        sys.exit(1)


@tools.command()
@click.argument("tool_name")
@click.option("--provider", "-p", help="Provider for native tools (e.g., 'google')")
def describe(tool_name: str, provider: str) -> None:
    """Show detailed information about a specific tool.

    Examples:
        justllms tools describe my_tool
        justllms tools describe google_search --provider google
    """
    from justllms.tools.registry import GlobalToolRegistry

    try:
        # Check native tools first if provider specified
        if provider == "google":
            from justllms.tools.native.google_tools import get_google_native_tool

            try:
                native_tool = get_google_native_tool(tool_name)
                click.echo(f"Tool: {native_tool.name}")
                click.echo("Type: Native Tool")
                click.echo("Provider: google")
                click.echo(f"Namespace: {native_tool.namespace}")
                click.echo(f"Description: {native_tool.description}")
                click.echo("\nConfiguration:")
                # Config is stored in metadata
                config_keys = [k for k in native_tool.metadata if k != "provider"]
                if config_keys:
                    for key in config_keys:
                        click.echo(f"  {key}: {native_tool.metadata[key]}")
                else:
                    click.echo("  No configuration")
                return
            except ValueError:
                click.echo(f"Native tool '{tool_name}' not found for provider 'google'", err=True)
                sys.exit(1)

        # Check user tools
        registry = GlobalToolRegistry()
        user_tool = registry.get_tool(tool_name)

        if not user_tool:
            click.echo(f"Tool '{tool_name}' not found", err=True)
            click.echo("\nUse 'justllms tools list' to see available tools.")
            sys.exit(1)

        # Display tool details
        click.echo(f"Tool: {user_tool.name}")
        click.echo("Type: User Tool")
        if user_tool.namespace:
            click.echo(f"Namespace: {user_tool.namespace}")
        click.echo(f"Description: {user_tool.description}")

        # Show parameters
        if user_tool.parameters:
            click.echo("\nParameters:")
            for param_name, param_info in user_tool.parameters.items():
                required = " (required)" if param_info.required else ""
                default = (
                    f" [default: {param_info.default}]" if param_info.default is not None else ""
                )
                desc = user_tool.parameter_descriptions.get(param_name, "")
                click.echo(f"  {param_name}: {param_info.type}{required}{default}")
                if desc:
                    click.echo(f"    {desc}")

        # Show return type
        if user_tool.return_type:
            click.echo(f"\nReturns: {user_tool.return_type}")

        # Show metadata
        if user_tool.metadata:
            click.echo("\nMetadata:")
            for key, value in user_tool.metadata.items():
                click.echo(f"  {key}: {value}")

    except Exception as e:
        click.echo(f"Error describing tool: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
