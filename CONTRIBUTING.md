# Contributing to openai-structured

We welcome contributions to `openai-structured`! Here's how you can contribute:

## Ways to Contribute

* **Report Bugs:** If you find a bug, please open an issue on GitHub with a clear description of the problem and steps to reproduce it.
* **Suggest Enhancements:** If you have an idea for a new feature or an improvement, please open an issue to discuss it.
* **Submit Pull Requests:** If you've fixed a bug or implemented a new feature, you can submit a pull request.

## Development Setup

1. **Fork the repository** on GitHub.
2. **Clone your fork** to your local machine:

   ```bash
   git clone https://github.com/your-username/openai-structured.git
   cd openai-structured
   ```

3. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows
   ```

4. **Install dependencies:**

   ```bash
   poetry install
   ```

## Type Checking with Pydantic and mypy

This project uses Pydantic with mypy for robust type checking. Here's how to ensure proper setup:

1. **Install Pydantic with mypy support:**

   In `pyproject.toml`, ensure Pydantic is installed with mypy extras:

   ```toml
   [tool.poetry.dependencies]
   pydantic = { version = "^2.6.3", extras = ["mypy"] }
   ```

2. **Configure mypy for Pydantic:**

   Add these settings to `pyproject.toml`:

   ```toml
   [tool.mypy]
   plugins = ["pydantic.mypy"]
   follow_imports = "normal"
   strict = true
   show_error_codes = true
   warn_unused_configs = true

   # Path configuration
   mypy_path = ["src", "tests"]
   files = ["src/openai_structured", "tests"]

   [tool.pydantic-mypy]
   init_forbid_extra = true
   init_typed = true
   warn_required_dynamic_aliases = true
   warn_untyped_fields = true
   ```

3. **Run type checking:**

   ```bash
   poetry run mypy
   ```

   Note: Always run mypy through Poetry to ensure it uses the correct virtual environment.

4. **Common issues and solutions:**
   * If mypy can't find Pydantic plugin: Ensure you've installed Pydantic with mypy extras
   * If getting "Cannot find implementation or library stub" errors: Check mypy_path setting
   * For example files that don't need strict typing: Add them to exclude patterns or use `# type: ignore`

## Making Changes

1. **Create a new branch** for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure your code follows the project's coding style (run `poetry run flake8`).
3. **Add tests** for your changes to ensure they work correctly.
4. **Run the tests:**

   ```bash
   poetry run pytest
   ```

5. **Commit your changes:**

   ```bash
   git add .
   git commit -m "Add your commit message here"
   ```

6. **Push your branch** to your fork on GitHub:

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a pull request** on the main repository.

## Pull Request Guidelines

* Keep pull requests focused on a single issue or feature.
* Provide a clear and concise description of your changes in the pull request.
* Link any related issues in the pull request description.
* Ensure all tests pass before submitting your pull request.
* Follow the project's coding style guidelines.
* Be responsive to feedback on your pull request.

By contributing to `openai-structured`, you help make it a better tool for everyone. Thank you!
