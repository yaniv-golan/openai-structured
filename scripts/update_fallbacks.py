#!/usr/bin/env python3
"""Script to update fallback models in the model registry.

This script reads the models.yml configuration and updates the fallback models
in model_registry.py. It is designed to be run as part of CI/CD to ensure
the fallback models stay in sync with the configuration.
"""

import re
from pathlib import Path

import yaml


def generate_fallbacks(config_path: Path, registry_path: Path) -> None:
    """Generate fallback models from configuration.

    Args:
        config_path: Path to models.yml configuration
        registry_path: Path to model_registry.py
    """
    # Read current configuration
    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Convert to Python dict format
    fallbacks = {
        "version": data["version"],
        "dated_models": {},
        "aliases": data["aliases"],
    }

    # Process dated models
    for name, config in data["dated_models"].items():
        model_config = {
            "context_window": config["context_window"],
            "max_output_tokens": config["max_output_tokens"],
            "supports_structured": config.get("supports_structured", True),
            "supports_streaming": config.get("supports_streaming", True),
            "supported_parameters": config["supported_parameters"],
            "description": config.get("description", ""),
        }
        if "min_version" in config:
            # Keep min_version as a dictionary instead of a ModelVersion instance
            model_config["min_version"] = config["min_version"]
        fallbacks["dated_models"][name] = model_config

    # Generate Python code with proper indentation
    lines = ["    _fallback_models = {"]

    # Add version
    lines.append(f'        "version": "{fallbacks["version"]}",')

    # Add dated models
    lines.append('        "dated_models": {')
    for name, config in fallbacks["dated_models"].items():
        lines.append(f"            {name!r}: {{")
        for key, value in config.items():
            if key == "min_version":
                # Format min_version as a dictionary
                v = value
                lines.append('                "min_version": {')
                lines.append(f'                    "year": {v["year"]},')
                lines.append(f'                    "month": {v["month"]},')
                lines.append(f'                    "day": {v["day"]},')
                lines.append("                },")
            elif key == "supported_parameters":
                lines.append('                "supported_parameters": [')
                for param in value:
                    lines.append(f"                    {param},")
                lines.append("                ],")
            else:
                lines.append(f"                {key!r}: {value!r},")
        lines.append("            },")
    lines.append("        },")

    # Add aliases
    lines.append('        "aliases": {')
    for alias, target in fallbacks["aliases"].items():
        lines.append(f"            {alias!r}: {target!r},")
    lines.append("        },")

    lines.append("    }")
    fallback_code = "\n".join(lines)

    # Read current registry file
    with open(registry_path) as f:
        content = f.read()

    # Replace fallback section
    pattern = re.compile(
        r"# AUTO-GENERATED FALLBACK START\n.*?# AUTO-GENERATED FALLBACK END",
        re.DOTALL,
    )
    updated = pattern.sub(
        f"# AUTO-GENERATED FALLBACK START\n{fallback_code}\n# AUTO-GENERATED FALLBACK END",
        content,
    )

    # Write updated file
    with open(registry_path, "w") as f:
        f.write(updated)


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    config_path = root / "src" / "openai_structured" / "config" / "models.yml"
    registry_path = root / "src" / "openai_structured" / "model_registry.py"

    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)

    if not registry_path.exists():
        print(f"Error: Registry file not found at {registry_path}")
        exit(1)

    try:
        generate_fallbacks(config_path, registry_path)
        print("✅ Successfully updated fallback models")
    except Exception as e:
        print(f"❌ Failed to update fallbacks: {e}")
        exit(1)
