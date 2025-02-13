#!/usr/bin/env python3
"""Command line utility for refreshing the model registry from remote source."""

import sys
from typing import Optional

import click

from ..errors import ModelNotSupportedError, ModelVersionError
from ..model_registry import ModelRegistry


def refresh_registry(
    verbose: bool = False,
    force: bool = False,
    url: str | None = None,
    validate: bool = False,
) -> int:
    """Refresh the model registry from remote source.

    Args:
        verbose: Whether to print verbose output
        force: Skip confirmation prompt
        url: Custom config URL
        validate: Validate without updating

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        registry = ModelRegistry.get_instance()

        if validate:
            registry._load_capabilities()  # Force revalidation
            print("✅ Config validation successful")
            return 0

        if not force:
            if not click.confirm(
                "Update model configurations from remote?", default=True
            ):
                return 0

        if registry.refresh_from_remote(url):
            print(
                "✅ Successfully refreshed model registry from remote source"
            )
            if verbose:
                print("\nAvailable models:")
                for model, caps in registry.models.items():
                    if not model.endswith("-latest"):
                        print(f"- {model}")
                        print(f"  Context window: {caps.context_window}")
                        print(f"  Max output tokens: {caps.max_output_tokens}")
                        print(
                            f"  Supports streaming: {caps.supports_streaming}"
                        )
                        print()
            return 0
        else:
            print("❌ Failed to refresh model registry")
            return 1

    except ModelNotSupportedError as e:
        print(f"❌ Config error: {e}")
        return 1
    except ModelVersionError as e:
        print(f"❌ Config error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error refreshing model registry: {e}")
        return 1


@click.command()
@click.option("-v", "--verbose", is_flag=True, help="Show verbose output")
@click.option("-f", "--force", is_flag=True, help="Skip confirmation prompt")
@click.option("--url", help="Custom config URL")
@click.option("--validate", is_flag=True, help="Validate without updating")
def main(
    verbose: bool = False,
    force: bool = False,
    url: Optional[str] = None,
    validate: bool = False,
) -> int:
    """Update model registry from remote source.

    This command updates the local model registry configuration from a remote
    source. By default, it fetches the configuration from the official repository.

    The command will:
    1. Download the latest model configuration
    2. Validate the configuration format and values
    3. Update the local configuration file

    Examples:
        # Basic update with confirmation
        $ openai-structured-refresh

        # Update with verbose output
        $ openai-structured-refresh -v

        # Update from custom URL without confirmation
        $ openai-structured-refresh -f --url https://example.com/models.yml

        # Validate current configuration without updating
        $ openai-structured-refresh --validate
    """
    return refresh_registry(
        verbose=verbose, force=force, url=url, validate=validate
    )


if __name__ == "__main__":
    sys.exit(main())
