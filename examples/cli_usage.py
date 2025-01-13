"""Example demonstrating CLI usage with openai-structured."""

import json
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, Field


class RecipeStep(BaseModel):
    """A single step in a recipe."""

    step_number: int = Field(
        ..., description="The order of this step in the recipe"
    )
    instruction: str = Field(..., description="The instruction for this step")
    time_minutes: int = Field(
        ..., description="Estimated time in minutes for this step"
    )


class Recipe(BaseModel):
    """A cooking recipe with structured information."""

    name: str = Field(..., description="The name of the recipe")
    cuisine: Literal[
        "Italian", "Mexican", "Japanese", "Indian", "American"
    ] = Field(..., description="The type of cuisine")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="The difficulty level of the recipe"
    )
    prep_time: int = Field(
        ..., description="Total preparation time in minutes"
    )
    ingredients: List[str] = Field(
        ..., min_items=2, description="List of ingredients needed"
    )
    steps: List[RecipeStep] = Field(
        ..., min_items=1, description="Steps to prepare the recipe"
    )
    serves: int = Field(
        ..., gt=0, description="Number of servings this recipe makes"
    )


def main():
    """Create example files for CLI usage."""
    # Create schema file
    schema = Recipe.model_json_schema()
    with open("recipe_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    print("Created recipe_schema.json with Recipe schema")

    # Create system prompt
    system_prompt = "You are a professional chef who creates detailed, easy-to-follow recipes."
    Path("system_prompt.txt").write_text(system_prompt)
    print("Created system_prompt.txt with chef persona")

    # Create user prompt template
    user_prompt = "Create a recipe for a classic Italian pasta dish that serves 4 people."
    Path("user_prompt.txt").write_text(user_prompt)
    print("Created user_prompt.txt with recipe request")

    print("\nTo use the CLI tool:")
    print("1. Make a call to generate a recipe:")
    print("   ostruct --system-prompt system_prompt.txt \\")
    print("                      --template user_prompt.txt \\")
    print("                      --schema-file recipe_schema.json \\")
    print("                      --output recipe_output.json")
    print("\n2. View the generated recipe:")
    print("   cat recipe_output.json")


if __name__ == "__main__":
    main()
