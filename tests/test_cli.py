"""Tests for the CLI module."""

import argparse
import json
from io import StringIO
from typing import Any, AsyncIterator, Dict, Union
from unittest.mock import MagicMock, patch

import pytest
from pyfakefs.fake_filesystem import FakeFilesystem

from openai_structured.cli.cli import ExitCode, _main, create_template_context
from openai_structured.cli.file_list import FileInfoList
from openai_structured.cli.security import SecurityManager


# Core CLI Tests
class TestCLICore:
    """Test core CLI functionality."""

    @pytest.mark.asyncio
    async def test_basic_execution(self, fs: FakeFilesystem) -> None:
        """Test basic CLI execution with minimal arguments."""
        # Create test files
        fs.create_file("schema.json", contents='{"type": "string"}')
        fs.create_file("task.txt", contents="Process this: {{ input }}")
        fs.create_file("input.txt", contents="test content")

        # Create mock structured stream
        async def mock_structured_stream(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[Dict[str, str]]:
            yield {"result": "test response"}

        with (
            patch(
                "sys.argv",
                [
                    "ostruct",
                    "--task",
                    "@task.txt",
                    "--schema",
                    "schema.json",
                    "--file",
                    "input=input.txt",
                    "--api-key",
                    "test-key",
                ],
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch(
                "openai_structured.cli.cli.async_openai_structured_stream",
                mock_structured_stream,
            ),
            patch("tiktoken.get_encoding") as mock_get_encoding,
        ):
            # Mock tiktoken
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_get_encoding.return_value = mock_encoding

            result = await _main()
            assert result == ExitCode.SUCCESS

    @pytest.mark.asyncio
    async def test_help_text(self) -> None:
        """Test help text display."""
        with (
            patch("sys.argv", ["ostruct", "--help"]),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            with pytest.raises(SystemExit) as exc_info:
                await _main()
            assert exc_info.value.code == 0
            output = mock_stdout.getvalue().lower()
            assert "usage:" in output
            assert "--help" in output

    @pytest.mark.asyncio
    async def test_version_info(self) -> None:
        """Test version information display."""
        with (
            patch("sys.argv", ["ostruct", "--version"]),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            with pytest.raises(SystemExit) as exc_info:
                await _main()
            assert exc_info.value.code == 0
            output = mock_stdout.getvalue()
            # Check that output matches format: "ostruct X.Y.Z" or "ostruct unknown"
            assert output.startswith("ostruct ")
            assert (
                len(output.strip().split()) == 2
            )  # program name + version number

    @pytest.mark.asyncio
    async def test_missing_required_args(self) -> None:
        """Test error handling for missing required arguments."""
        with (
            patch("sys.argv", ["ostruct"]),
            patch("sys.stderr", new_callable=StringIO) as mock_stderr,
        ):
            with pytest.raises(SystemExit) as exc_info:
                await _main()
            assert exc_info.value.code == 2
            output = mock_stderr.getvalue()
            assert "required" in output.lower()


# Variable Handling Tests
class TestCLIVariables:
    """Test variable handling in CLI."""

    @pytest.mark.asyncio
    async def test_basic_variable(self, fs: FakeFilesystem) -> None:
        """Test basic variable assignment."""
        fs.create_file("schema.json", contents='{"type": "string"}')
        fs.create_file("task.txt", contents="Value is: {{ test_var }}")

        # Create mock structured stream
        async def mock_structured_stream(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[Dict[str, str]]:
            yield {"result": "test value processed"}

        with (
            patch(
                "sys.argv",
                [
                    "ostruct",
                    "--task",
                    "@task.txt",
                    "--schema",
                    "schema.json",
                    "--var",
                    "test_var=test_value",
                    "--api-key",
                    "test-key",
                ],
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch(
                "openai_structured.cli.cli.async_openai_structured_stream",
                mock_structured_stream,
            ),
            patch("tiktoken.get_encoding") as mock_get_encoding,
        ):
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_get_encoding.return_value = mock_encoding

            result = await _main()
            assert result == ExitCode.SUCCESS

    @pytest.mark.asyncio
    async def test_json_variable(self, fs: FakeFilesystem) -> None:
        """Test JSON variable assignment."""
        fs.create_file("schema.json", contents='{"type": "string"}')
        fs.create_file("task.txt", contents="Config: {{ config.key }}")

        # Create mock structured stream
        async def mock_structured_stream(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[Dict[str, str]]:
            yield {"result": "config value processed"}

        with (
            patch(
                "sys.argv",
                [
                    "ostruct",
                    "--task",
                    "@task.txt",
                    "--schema",
                    "schema.json",
                    "--json-var",
                    'config={"key": "value"}',
                    "--api-key",
                    "test-key",
                ],
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch(
                "openai_structured.cli.cli.async_openai_structured_stream",
                mock_structured_stream,
            ),
            patch("tiktoken.get_encoding") as mock_get_encoding,
        ):
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_get_encoding.return_value = mock_encoding

            result = await _main()
            assert result == ExitCode.SUCCESS

    @pytest.mark.asyncio
    async def test_invalid_variable_name(self, fs: FakeFilesystem) -> None:
        """Test error handling for invalid variable names."""
        fs.create_file("schema.json", contents='{"type": "string"}')
        fs.create_file("task.txt", contents="Test")

        with (
            patch(
                "sys.argv",
                [
                    "ostruct",
                    "--task",
                    "@task.txt",
                    "--schema",
                    "schema.json",
                    "--var",
                    "123invalid=value",  # Invalid: starts with a number
                    "--api-key",
                    "test-key",
                ],
            ),
            patch("logging.getLogger") as mock_logger,
            patch("tiktoken.get_encoding") as mock_get_encoding,
        ):
            # Setup logger mock to capture error messages
            mock_error_logger = MagicMock()
            mock_logger.return_value = mock_error_logger

            # Setup tiktoken mock
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_get_encoding.return_value = mock_encoding

            result = await _main()
            assert result == ExitCode.DATA_ERROR

            # Check that the error was logged
            mock_error_logger.error.assert_called()
            error_msg = mock_error_logger.error.call_args[0][0].lower()
            assert "invalid variable name" in error_msg
            assert "123invalid" in error_msg

    @pytest.mark.asyncio
    async def test_invalid_json_variable(self, fs: FakeFilesystem) -> None:
        """Test error handling for invalid JSON variables."""
        fs.create_file("schema.json", contents='{"type": "string"}')
        fs.create_file("task.txt", contents="Test")

        with (
            patch(
                "sys.argv",
                [
                    "ostruct",
                    "--task",
                    "@task.txt",
                    "--schema",
                    "schema.json",
                    "--json-var",
                    "config={invalid_json}",
                    "--api-key",
                    "test-key",
                ],
            ),
            patch("logging.getLogger") as mock_logger,
            patch("tiktoken.get_encoding") as mock_get_encoding,
        ):
            # Setup logger mock to capture error messages
            mock_error_logger = MagicMock()
            mock_logger.return_value = mock_error_logger

            # Setup tiktoken mock
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_get_encoding.return_value = mock_encoding

            result = await _main()
            assert result == ExitCode.DATA_ERROR

            # Check that the error was logged
            mock_error_logger.error.assert_called()
            error_msg = mock_error_logger.error.call_args[0][0].lower()
            assert "invalid json" in error_msg
            assert "config" in error_msg
            assert (
                "{invalid_json}" in error_msg
            )  # Check for the actual invalid JSON value


# I/O Tests
class TestCLIIO:
    """Test I/O handling in CLI."""

    @pytest.mark.asyncio
    async def test_file_input(self, fs: FakeFilesystem) -> None:
        """Test file input handling."""
        fs.create_file("schema.json", contents='{"type": "string"}')
        fs.create_file("task.txt", contents="Content: {{ input[0].content }}")
        fs.create_file("input.txt", contents="test content")

        # Create mock structured stream
        async def mock_structured_stream(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[Dict[str, str]]:
            yield {"result": "file content processed"}

        with (
            patch(
                "sys.argv",
                [
                    "ostruct",
                    "--task",
                    "@task.txt",
                    "--schema",
                    "schema.json",
                    "--file",
                    "input=input.txt",
                    "--api-key",
                    "test-key",
                ],
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch(
                "openai_structured.cli.cli.async_openai_structured_stream",
                mock_structured_stream,
            ),
            patch("tiktoken.get_encoding") as mock_get_encoding,
        ):
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_get_encoding.return_value = mock_encoding

            result = await _main()
            assert result == ExitCode.SUCCESS

    @pytest.mark.asyncio
    async def test_stdin_input(self, fs: FakeFilesystem) -> None:
        """Test stdin input handling."""
        fs.create_file("schema.json", contents='{"type": "string"}')
        fs.create_file("task.txt", contents="Input: {{ stdin }}")

        # Create mock structured stream
        async def mock_structured_stream(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[Dict[str, str]]:
            yield {"result": "stdin content processed"}

        with (
            patch(
                "sys.argv",
                [
                    "ostruct",
                    "--task",
                    "@task.txt",
                    "--schema",
                    "schema.json",
                    "--api-key",
                    "test-key",
                ],
            ),
            patch("sys.stdin.isatty", return_value=False),
            patch("sys.stdin.read", return_value="test input"),
            patch(
                "openai_structured.cli.cli.async_openai_structured_stream",
                mock_structured_stream,
            ),
            patch("tiktoken.get_encoding") as mock_get_encoding,
        ):
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_get_encoding.return_value = mock_encoding

            result = await _main()
            assert result == ExitCode.SUCCESS

    @pytest.mark.asyncio
    async def test_directory_input(self, fs: FakeFilesystem) -> None:
        """Test directory input handling."""
        # Create test directory structure
        fs.create_file("schema.json", contents='{"type": "string"}')
        fs.create_file("task.txt", contents="Files: {{ files | length }}")
        fs.create_dir("test_dir")
        fs.create_file("test_dir/file1.txt", contents="content 1")
        fs.create_file("test_dir/file2.txt", contents="content 2")

        # Create mock structured stream
        async def mock_structured_stream(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[Dict[str, str]]:
            yield {"result": "directory content processed"}

        with (
            patch(
                "sys.argv",
                [
                    "ostruct",
                    "--task",
                    "@task.txt",
                    "--schema",
                    "schema.json",
                    "--dir",
                    "files=test_dir",
                    "--api-key",
                    "test-key",
                ],
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch(
                "openai_structured.cli.cli.async_openai_structured_stream",
                mock_structured_stream,
            ),
            patch("tiktoken.get_encoding") as mock_get_encoding,
        ):
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_get_encoding.return_value = mock_encoding

            result = await _main()
            assert result == ExitCode.SUCCESS


# Integration Tests
class TestCLIIntegration:
    """Test integration between different CLI features."""

    @pytest.mark.asyncio
    async def test_template_variable_integration(
        self, fs: FakeFilesystem
    ) -> None:
        """Test integration between templates and variables."""
        fs.create_file("schema.json", contents='{"type": "string"}')
        fs.create_file(
            "task.txt",
            contents="""---
system_prompt: Test prompt with {{ var1 }}
---
Template with {{ var2 }} and {{ input[0].content }}""",
        )
        fs.create_file("input.txt", contents="test content")

        # Create mock structured stream
        async def mock_structured_stream(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[Dict[str, str]]:
            yield {"result": "template and variables processed"}

        with (
            patch(
                "sys.argv",
                [
                    "ostruct",
                    "--task",
                    "@task.txt",
                    "--schema",
                    "schema.json",
                    "--file",
                    "input=input.txt",
                    "--var",
                    "var1=value1",
                    "--var",
                    "var2=value2",
                    "--api-key",
                    "test-key",
                ],
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch(
                "openai_structured.cli.cli.async_openai_structured_stream",
                mock_structured_stream,
            ),
            patch("tiktoken.get_encoding") as mock_get_encoding,
        ):
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_get_encoding.return_value = mock_encoding

            result = await _main()
            assert result == ExitCode.SUCCESS

    @pytest.mark.asyncio
    async def test_schema_validation_integration(
        self, fs: FakeFilesystem
    ) -> None:
        """Test integration between schema validation and API response."""
        schema_content = {
            "type": "object",
            "properties": {
                "analysis": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["analysis", "score"],
        }
        fs.create_file("schema.json", contents=json.dumps(schema_content))
        fs.create_file("task.txt", contents="Analyze: {{ input }}")
        fs.create_file("input.txt", contents="test content")

        # Create mock structured stream
        async def mock_structured_stream(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[Dict[str, Union[str, float]]]:
            yield {"analysis": "Test analysis", "score": 0.95}

        with (
            patch(
                "sys.argv",
                [
                    "ostruct",
                    "--task",
                    "@task.txt",
                    "--schema",
                    "schema.json",
                    "--file",
                    "input=input.txt",
                    "--api-key",
                    "test-key",
                ],
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch(
                "openai_structured.cli.cli.async_openai_structured_stream",
                mock_structured_stream,
            ),
            patch("tiktoken.get_encoding") as mock_get_encoding,
        ):
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_get_encoding.return_value = mock_encoding

            result = await _main()
            assert result == ExitCode.SUCCESS

    def test_template_context_single_file(self, fs: FakeFilesystem) -> None:
        """Test template context creation with single file."""
        fs.create_file("test.txt", contents="hello")
        args = argparse.Namespace(
            file=["doc=test.txt"],
            files=None,
            dir=None,
            recursive=False,
            ext=None,
            var=None,
            json_var=None,
        )

        context = create_template_context(args, SecurityManager())
        assert "doc" in context
        assert isinstance(context["doc"], FileInfoList)
        assert context["doc"].content == "hello"  # Test direct access
        assert (
            context["doc"][0].content == "hello"
        )  # Test backward compatibility
        assert context["doc"].path == "test.txt"

    def test_template_context_multiple_files(self, fs: FakeFilesystem) -> None:
        """Test template context creation with multiple files."""
        fs.create_file("test1.txt", contents="hello")
        fs.create_file("test2.txt", contents="world")
        args = argparse.Namespace(
            file=None,
            files=["docs=*.txt"],
            dir=None,
            recursive=False,
            ext=None,
            var=None,
            json_var=None,
        )

        context = create_template_context(args, SecurityManager())
        assert "docs" in context
        assert isinstance(context["docs"], FileInfoList)
        assert context["docs"].content == ["hello", "world"]
        assert context["docs"].path == ["test1.txt", "test2.txt"]
        assert context["docs"][0].content == "hello"
        assert context["docs"][1].content == "world"

    def test_template_context_mixed_sources(self, fs: FakeFilesystem) -> None:
        """Test template context creation with mixed file sources."""
        # Create test files
        fs.create_file("single.txt", contents="single")
        fs.create_file("test1.py", contents="test1")
        fs.create_file("test2.py", contents="test2")
        fs.create_file("dir/test3.txt", contents="test3")

        args = argparse.Namespace(
            file=["doc=single.txt"],
            files=["py_files=*.py"],
            dir=["txt_files=dir"],
            recursive=True,
            ext=".txt",
            var=["name=value"],
            json_var=['data={"key":"value"}'],
        )

        context = create_template_context(args, SecurityManager())

        # Check single file
        assert isinstance(context["doc"], FileInfoList)
        assert context["doc"].content == "single"

        # Check pattern-matched files
        assert isinstance(context["py_files"], FileInfoList)
        assert context["py_files"].content == ["test1", "test2"]

        # Check directory files
        assert isinstance(context["txt_files"], FileInfoList)
        assert context["txt_files"].content == ["test3"]

        # Check other variables
        assert context["name"] == "value"
        assert context["data"] == {"key": "value"}
