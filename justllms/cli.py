import click
import sys


@click.group()
@click.version_option()
def main():
    pass


@main.command()
def sxs():
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


if __name__ == "__main__":
    main()