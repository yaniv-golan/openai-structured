"""Tests for CLI integration."""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from openai_structured.cli.cli import ExitCode, _main

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Mock tiktoken
class MockEncoding:
    def encode(self, text: str) -> List[int]:
        return [0] * len(text.split())  # Simple mock: one token per word

    def decode(self, tokens: List[int]) -> str:
        return " " * len(tokens)


class MockTiktoken:
    @staticmethod
    def get_encoding(encoding_name: str) -> MockEncoding:
        return MockEncoding()

    @staticmethod
    def encoding_for_model(model_name: str) -> MockEncoding:
        return MockEncoding()


@pytest_asyncio.fixture  # type: ignore[misc]
async def mock_openai_client() -> AsyncOpenAI:
    """Create mock OpenAI client."""
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()

    async def mock_stream(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Mock streaming response."""
        # First chunk with the content
        yield ChatCompletionChunk(
            id="test",
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content='{"result": "test response", "status": "success"}'
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1234567890,
            model="gpt-4",
            object="chat.completion.chunk",
        )

        # Final chunk indicating completion
        yield ChatCompletionChunk(
            id="test",
            choices=[
                Choice(
                    delta=ChoiceDelta(content=""),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=1234567890,
            model="gpt-4",
            object="chat.completion.chunk",
        )

    mock_client.chat.completions.create = AsyncMock(side_effect=mock_stream)
    return mock_client


@pytest.fixture(autouse=True)  # type: ignore[misc]
def mock_tiktoken() -> Generator[None, None, None]:
    """Mock tiktoken for testing."""
    with patch("openai_structured.cli.cli.tiktoken", MockTiktoken()):
        yield


@pytest.fixture  # type: ignore[misc]
def test_files(tmp_path: Path) -> Dict[str, str]:
    # Create test files and directories
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    input_file = test_dir / "input.txt"
    input_file.write_text("test input")

    template_file = test_dir / "template.txt"
    template_file.write_text(
        """---
system_prompt: You are a helpful assistant
---
test template {{ input }}"""
    )

    schema_file = test_dir / "schema.json"
    schema_file.write_text(
        '{"type": "object", "properties": {"result": {"type": "string"}}}'
    )

    # Create subdirectories for testing recursive functionality
    subdir = test_dir / "subdir"
    subdir.mkdir()
    (subdir / "subfile.txt").write_text("test subfile")

    return {
        "test_dir": str(test_dir),
        "input_file": str(input_file),
        "template_file": str(template_file),
        "schema_file": str(schema_file),
    }


@pytest.fixture(autouse=True)  # type: ignore[misc]
def cleanup_test_dirs(
    test_files: Dict[str, str]
) -> Generator[None, None, None]:
    """Clean up test directories after each test."""
    yield
    # Clean up external directories after tests
    test_dir = Path(test_files["test_dir"])
    for dir_name in ["outside_data", "external_data1", "external_data2"]:
        ext_dir = test_dir.parent / dir_name
        if ext_dir.exists():
            shutil.rmtree(ext_dir)


@pytest.mark.asyncio  # type: ignore[misc]
async def test_basic_cli_execution(
    mock_openai_client: AsyncOpenAI,
    test_files: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(os.path.dirname(test_files["test_dir"]))
    with (
        patch(
            "sys.argv",
            [
                "cli.py",
                "--task",
                test_files["template_file"],
                "--file",
                f'input={test_files["input_file"]}',
                "--schema",
                test_files["schema_file"],
            ],
        ),
        patch(
            "openai_structured.cli.cli.AsyncOpenAI",
            return_value=mock_openai_client,
        ),
    ):
        result = await _main()
        assert result == ExitCode.SUCCESS


@pytest.mark.asyncio  # type: ignore[misc]
async def test_cli_with_directory_input(
    mock_openai_client: AsyncOpenAI,
    test_files: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = Path(test_files["test_dir"])
    monkeypatch.chdir(test_dir.parent)
    dir_name = test_dir.name
    print("\nDEBUG INFO for test_cli_with_directory_input:")
    print(f"test_dir = {test_dir}")
    print(f"dir_name = {dir_name}")
    print(f"current working directory = {os.getcwd()}")
    print(f"test_dir exists? {os.path.exists(dir_name)}")
    print(f"test_dir is directory? {os.path.isdir(dir_name)}")
    print(f"directory contents: {os.listdir('.')}")
    with (
        patch(
            "sys.argv",
            [
                "cli.py",
                "--task",
                test_files["template_file"],
                "--dir",
                f"input={dir_name}",
                "--schema",
                test_files["schema_file"],
            ],
        ),
        patch(
            "openai_structured.cli.cli.AsyncOpenAI",
            return_value=mock_openai_client,
        ),
    ):
        result = await _main()
        assert result == ExitCode.SUCCESS


@pytest.mark.asyncio  # type: ignore[misc]
async def test_cli_with_multiple_files(
    mock_openai_client: AsyncOpenAI,
    test_files: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(os.path.dirname(test_files["test_dir"]))
    with (
        patch(
            "sys.argv",
            [
                "cli.py",
                "--task",
                test_files["template_file"],
                "--file",
                f'input={test_files["input_file"]}',
                "--schema",
                test_files["schema_file"],
            ],
        ),
        patch(
            "openai_structured.cli.cli.AsyncOpenAI",
            return_value=mock_openai_client,
        ),
    ):
        result = await _main()
        assert result == ExitCode.SUCCESS


@pytest.mark.asyncio  # type: ignore[misc]
async def test_cli_dry_run(
    mock_openai_client: AsyncOpenAI,
    test_files: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(os.path.dirname(test_files["test_dir"]))
    with (
        patch(
            "sys.argv",
            [
                "cli.py",
                "--task",
                test_files["template_file"],
                "--file",
                f'input={test_files["input_file"]}',
                "--schema",
                test_files["schema_file"],
                "--dry-run",
            ],
        ),
        patch(
            "openai_structured.cli.cli.AsyncOpenAI",
            return_value=mock_openai_client,
        ),
    ):
        result = await _main()
        assert result == ExitCode.SUCCESS


@pytest.mark.asyncio  # type: ignore[misc]
async def test_cli_with_output_file(
    mock_openai_client: AsyncOpenAI,
    test_files: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(os.path.dirname(test_files["test_dir"]))
    output_file = os.path.join(test_files["test_dir"], "output.json")

    with (
        patch(
            "sys.argv",
            [
                "cli.py",
                "--task",
                test_files["template_file"],
                "--file",
                f'input={test_files["input_file"]}',
                "--schema",
                test_files["schema_file"],
                "--output-file",
                output_file,
            ],
        ),
        patch(
            "openai_structured.cli.cli.AsyncOpenAI",
            return_value=mock_openai_client,
        ),
    ):
        result = await _main()
        assert result == ExitCode.SUCCESS
        assert os.path.exists(output_file)

        # Verify file contents
        with open(output_file, "r") as f:
            content = f.read().strip()
            expected = json.dumps(
                {"result": "test response", "status": "success"}, indent=2
            )
            assert content == expected


@pytest.mark.asyncio  # type: ignore[misc]
async def test_cli_with_recursive_directory(
    mock_openai_client: AsyncOpenAI,
    test_files: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = Path(test_files["test_dir"])
    monkeypatch.chdir(test_dir.parent)
    dir_name = test_dir.name
    print("\nDEBUG INFO for test_cli_with_recursive_directory:")
    print(f"test_dir = {test_dir}")
    print(f"dir_name = {dir_name}")
    print(f"current working directory = {os.getcwd()}")
    print(f"test_dir exists? {os.path.exists(dir_name)}")
    print(f"test_dir is directory? {os.path.isdir(dir_name)}")
    print(f"directory contents: {os.listdir('.')}")
    with (
        patch(
            "sys.argv",
            [
                "cli.py",
                "--task",
                test_files["template_file"],
                "--dir",
                f"input={dir_name}",
                "--recursive",
                "--schema",
                test_files["schema_file"],
            ],
        ),
        patch(
            "openai_structured.cli.cli.AsyncOpenAI",
            return_value=mock_openai_client,
        ),
    ):
        result = await _main()
        assert result == ExitCode.SUCCESS


@pytest.mark.asyncio  # type: ignore[misc]
async def test_cli_with_allowed_dir(
    mock_openai_client: AsyncOpenAI,
    test_files: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CLI with --allowed-dir argument."""
    test_dir = Path(test_files["test_dir"])
    outside_dir = test_dir.parent / "outside_data"
    outside_dir.mkdir(exist_ok=True)
    outside_file = outside_dir / "external.txt"
    outside_file.write_text("external data")

    monkeypatch.chdir(test_dir)

    with (
        patch(
            "sys.argv",
            [
                "cli.py",
                "--task",
                test_files["template_file"],
                "--file",
                f"external={outside_file}",
                "--schema",
                test_files["schema_file"],
                "--allowed-dir",
                str(outside_dir.resolve()),
            ],
        ),
        patch(
            "openai_structured.cli.cli.AsyncOpenAI",
            return_value=mock_openai_client,
        ),
    ):
        result = await _main()
        assert result == ExitCode.SUCCESS


@pytest.mark.asyncio  # type: ignore[misc]
async def test_cli_with_allowed_dir_file(
    mock_openai_client: AsyncOpenAI,
    test_files: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CLI with --allowed-dir @file argument."""
    # Create test directories outside main test directory
    test_dir = Path(test_files["test_dir"])
    outside_dir1 = test_dir.parent / "external_data1"
    outside_dir2 = test_dir.parent / "external_data2"
    outside_dir1.mkdir()
    outside_dir2.mkdir()

    # Create test files in external directories
    file1 = outside_dir1 / "data1.txt"
    file2 = outside_dir2 / "data2.txt"
    file1.write_text("external data 1")
    file2.write_text("external data 2")

    # Create allowed dirs file
    allowed_dirs_file = test_dir / "allowed_dirs.txt"
    allowed_dirs_file.write_text(f"{outside_dir1}\n{outside_dir2}")

    monkeypatch.chdir(test_dir)

    with (
        patch(
            "sys.argv",
            [
                "cli.py",
                "--task",
                test_files["template_file"],
                "--file",
                f"data1={file1}",
                "--file",
                f"data2={file2}",
                "--schema",
                test_files["schema_file"],
                "--allowed-dir",
                f"@{allowed_dirs_file}",
            ],
        ),
        patch(
            "openai_structured.cli.cli.AsyncOpenAI",
            return_value=mock_openai_client,
        ),
    ):
        result = await _main()
        assert result == ExitCode.SUCCESS


@pytest.mark.asyncio  # type: ignore[misc]
async def test_cli_with_multiple_allowed_dirs(
    mock_openai_client: AsyncOpenAI,
    test_files: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CLI with multiple --allowed-dir arguments."""
    test_dir = Path(test_files["test_dir"])
    outside_dir1 = test_dir.parent / "external_data1"
    outside_dir2 = test_dir.parent / "external_data2"
    outside_dir1.mkdir()
    outside_dir2.mkdir()

    file1 = outside_dir1 / "data1.txt"
    file2 = outside_dir2 / "data2.txt"
    file1.write_text("external data 1")
    file2.write_text("external data 2")

    monkeypatch.chdir(test_dir)

    with (
        patch(
            "sys.argv",
            [
                "cli.py",
                "--task",
                test_files["template_file"],
                "--file",
                f"data1={file1}",
                "--file",
                f"data2={file2}",
                "--schema",
                test_files["schema_file"],
                "--allowed-dir",
                str(outside_dir1.resolve()),
                "--allowed-dir",
                str(outside_dir2.resolve()),
            ],
        ),
        patch(
            "openai_structured.cli.cli.AsyncOpenAI",
            return_value=mock_openai_client,
        ),
    ):
        result = await _main()
        assert result == ExitCode.SUCCESS


@pytest.mark.asyncio  # type: ignore[misc]
async def test_cli_with_disallowed_dir(
    mock_openai_client: AsyncOpenAI,
    test_files: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CLI with file in disallowed directory."""
    test_dir = Path(test_files["test_dir"])
    outside_dir = test_dir.parent / "outside_data"
    outside_dir.mkdir()
    outside_file = outside_dir / "external.txt"
    outside_file.write_text("external data")

    monkeypatch.chdir(test_dir)

    with (
        patch(
            "sys.argv",
            [
                "cli.py",
                "--task",
                test_files["template_file"],
                "--file",
                f"external={outside_file}",
                "--schema",
                test_files["schema_file"],
            ],
        ),
        patch(
            "openai_structured.cli.cli.AsyncOpenAI",
            return_value=mock_openai_client,
        ),
    ):
        result = await _main()
        assert result == ExitCode.SECURITY_ERROR
