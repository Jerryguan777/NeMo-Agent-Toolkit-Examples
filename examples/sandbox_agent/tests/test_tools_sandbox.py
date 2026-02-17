# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for sandbox tools (shell, python, browser, file_editor)."""

from unittest.mock import AsyncMock

import pytest

from nat_sandbox_agent.sandbox.base import CommandResult
from nat_sandbox_agent.sandbox.base import IPythonResult
from nat_sandbox_agent.tools.sandbox import SandboxToolExecutor
from nat_sandbox_agent.tools.sandbox.browser import browser_action
from nat_sandbox_agent.tools.sandbox.browser import create_browser_tool
from nat_sandbox_agent.tools.sandbox.execution import create_python_tool
from nat_sandbox_agent.tools.sandbox.execution import create_shell_tool
from nat_sandbox_agent.tools.sandbox.execution import execute_python
from nat_sandbox_agent.tools.sandbox.execution import execute_shell
from nat_sandbox_agent.tools.sandbox.file_ops import create_file_editor_tool
from nat_sandbox_agent.tools.sandbox.file_ops import file_editor


class TestSandboxToolExecutor:
    """Tests for SandboxToolExecutor."""

    def test_init_with_defaults(self, mock_sandbox):
        """Test executor initialization with default values."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        assert executor.sandbox is mock_sandbox
        assert executor.max_output_chars == 16000
        assert executor.default_timeout == 120

    def test_init_with_custom_values(self, mock_sandbox):
        """Test executor initialization with custom values."""
        executor = SandboxToolExecutor(
            sandbox=mock_sandbox,
            max_output_chars=5000,
            default_timeout=60,
        )

        assert executor.max_output_chars == 5000
        assert executor.default_timeout == 60

    def test_truncate_short_text(self, mock_sandbox):
        """Test that short text is not truncated."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox, max_output_chars=100)

        result = executor.truncate("Short text")

        assert result == "Short text"

    def test_truncate_long_text(self, mock_sandbox):
        """Test that long text is truncated with head+tail preserved."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox, max_output_chars=200)

        long_text = "A" * 1000
        result = executor.truncate(long_text)

        assert len(result) < 1000
        assert "truncated" in result

    @pytest.mark.asyncio
    async def test_list_generated_files_success(self, mock_sandbox):
        """Test listing generated files using shell command."""
        mock_sandbox.run_command = AsyncMock(
            return_value=CommandResult(exit_code=0, stdout="output.txt\ndata.json\n", stderr="")
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        files = await executor.list_generated_files()

        assert files == ["/workspace/output/output.txt", "/workspace/output/data.json"]

    @pytest.mark.asyncio
    async def test_list_generated_files_handles_exception(self, mock_sandbox):
        """Test that list_generated_files handles exceptions gracefully."""
        mock_sandbox.run_command = AsyncMock(side_effect=Exception("Directory not found"))
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        files = await executor.list_generated_files()

        assert files == []

    @pytest.mark.asyncio
    async def test_list_generated_files_empty_on_error(self, mock_sandbox):
        """Test that list_generated_files returns empty list on command failure."""
        mock_sandbox.run_command = AsyncMock(
            return_value=CommandResult(exit_code=1, stdout="", stderr="ls: cannot access")
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        files = await executor.list_generated_files()

        assert files == []


class TestShellTool:
    """Tests for shell command execution tool."""

    @pytest.mark.asyncio
    async def test_execute_shell_success(self, mock_sandbox):
        """Test successful shell command execution."""
        mock_sandbox.run_command = AsyncMock(
            return_value=CommandResult(exit_code=0, stdout="file1.txt\nfile2.py", stderr="")
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await execute_shell(executor, "ls -la")

        assert result["status"] == "success"
        assert "file1.txt" in result["stdout"]
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_execute_shell_failure(self, mock_sandbox):
        """Test shell command failure."""
        mock_sandbox.run_command = AsyncMock(
            return_value=CommandResult(exit_code=1, stdout="", stderr="Command not found")
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await execute_shell(executor, "invalid_command")

        assert result["status"] == "error"
        assert result["exit_code"] == 1
        assert "Command not found" in result["stderr"]

    @pytest.mark.asyncio
    async def test_execute_shell_truncates_output(self, mock_sandbox):
        """Test that long output is truncated."""
        long_output = "X" * 20000
        mock_sandbox.run_command = AsyncMock(
            return_value=CommandResult(exit_code=0, stdout=long_output, stderr="")
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox, max_output_chars=200)

        result = await execute_shell(executor, "cat bigfile.txt")

        assert len(result["stdout"]) < 20000
        assert "truncated" in result["stdout"]

    def test_create_shell_tool_returns_structured_tool(self, mock_sandbox):
        """Test that create_shell_tool creates a StructuredTool."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox)
        tool = create_shell_tool(executor)

        assert tool.name == "shell"
        assert "persistent" in tool.description.lower()


class TestPythonTool:
    """Tests for Python code execution tool (IPython)."""

    @pytest.mark.asyncio
    async def test_execute_python_success(self, mock_sandbox):
        """Test successful Python code execution via IPython."""
        mock_sandbox.execute_ipython = AsyncMock(
            return_value=IPythonResult(exit_code=0, text_output="42", error_output="")
        )
        mock_sandbox.run_command = AsyncMock(
            return_value=CommandResult(exit_code=0, stdout="result.txt\n", stderr="")
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await execute_python(executor, "print(6 * 7)")

        assert result["status"] == "success"
        assert "42" in result["stdout"]
        assert "generated_files" in result
        assert "image_paths" in result

    @pytest.mark.asyncio
    async def test_execute_python_error(self, mock_sandbox):
        """Test Python execution with error."""
        mock_sandbox.execute_ipython = AsyncMock(
            return_value=IPythonResult(
                exit_code=1,
                text_output="",
                error_output="NameError: name 'undefined_var' is not defined",
            )
        )
        mock_sandbox.run_command = AsyncMock(
            return_value=CommandResult(exit_code=0, stdout="", stderr="")
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await execute_python(executor, "print(undefined_var)")

        assert result["status"] == "error"
        assert "NameError" in result["stderr"]

    @pytest.mark.asyncio
    async def test_execute_python_with_images_saves_to_file(self, mock_sandbox):
        """Test Python execution saves base64 images as files and returns paths."""
        mock_sandbox.execute_ipython = AsyncMock(
            return_value=IPythonResult(
                exit_code=0,
                text_output="",
                error_output="",
                image_urls=["data:image/png;base64,iVBORw0KGgo="],
            )
        )
        mock_sandbox.run_command = AsyncMock(
            return_value=CommandResult(exit_code=0, stdout="", stderr="")
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await execute_python(executor, "import matplotlib; plt.plot([1,2,3])")

        assert result["status"] == "success"
        assert len(result["image_paths"]) == 1
        assert result["image_paths"][0].startswith("/workspace/output/plot_")
        assert result["image_paths"][0].endswith(".png")
        # Verify run_command was called to write the image
        assert mock_sandbox.run_command.call_count >= 2  # image write + list_generated_files

    @pytest.mark.asyncio
    async def test_execute_python_non_base64_url_passthrough(self, mock_sandbox):
        """Test that non-base64 image URLs pass through unchanged."""
        mock_sandbox.execute_ipython = AsyncMock(
            return_value=IPythonResult(
                exit_code=0,
                text_output="",
                error_output="",
                image_urls=["https://example.com/image.png"],
            )
        )
        mock_sandbox.run_command = AsyncMock(
            return_value=CommandResult(exit_code=0, stdout="", stderr="")
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await execute_python(executor, "display(Image(url='...'))")

        assert result["image_paths"] == ["https://example.com/image.png"]

    def test_create_python_tool_returns_structured_tool(self, mock_sandbox):
        """Test that create_python_tool creates a StructuredTool."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox)
        tool = create_python_tool(executor)

        assert tool.name == "python"
        assert "persistent" in tool.description.lower()
        assert "ipython" in tool.description.lower()


class TestBrowserTool:
    """Tests for interactive browser tool."""

    @pytest.mark.asyncio
    async def test_browser_action_goto(self, mock_sandbox):
        """Test browser goto action."""
        mock_sandbox.browser_action = AsyncMock(
            return_value={
                "status": "success",
                "url": "https://example.com",
                "title": "Example Domain",
            }
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await browser_action(executor, "goto", url="https://example.com")

        assert result["status"] == "success"
        assert result["title"] == "Example Domain"
        mock_sandbox.browser_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_browser_action_get_text(self, mock_sandbox):
        """Test browser get_text action."""
        mock_sandbox.browser_action = AsyncMock(
            return_value={
                "status": "success",
                "url": "https://example.com",
                "title": "Example",
                "content": "Page content here",
            }
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await browser_action(executor, "get_text")

        assert result["status"] == "success"
        assert result["content"] == "Page content here"

    @pytest.mark.asyncio
    async def test_browser_action_click(self, mock_sandbox):
        """Test browser click action."""
        mock_sandbox.browser_action = AsyncMock(
            return_value={
                "status": "success",
                "url": "https://example.com/page2",
                "title": "Page 2",
            }
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await browser_action(executor, "click", selector="a.link")

        assert result["status"] == "success"
        call_args = mock_sandbox.browser_action.call_args[0][0]
        assert call_args["type"] == "click"
        assert call_args["selector"] == "a.link"

    @pytest.mark.asyncio
    async def test_browser_action_screenshot_strips_base64(self, mock_sandbox):
        """Test that raw base64 screenshot data is stripped from results."""
        mock_sandbox.browser_action = AsyncMock(
            return_value={
                "status": "success",
                "url": "https://example.com",
                "title": "Example",
                "screenshot": "iVBORw0KGgoAAAANS..." * 1000,
            }
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await browser_action(executor, "screenshot")

        assert "screenshot" not in result

    @pytest.mark.asyncio
    async def test_browser_action_screenshot_path_with_guidance(self, mock_sandbox):
        """Test that screenshot_path is preserved with actionable guidance."""
        mock_sandbox.browser_action = AsyncMock(
            return_value={
                "status": "success",
                "url": "https://example.com",
                "title": "Example",
                "screenshot_path": "/workspace/output/screenshot_123.png",
            }
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await browser_action(executor, "screenshot")

        assert result["screenshot_path"] == "/workspace/output/screenshot_123.png"
        assert "screenshot_info" in result
        assert "PIL" in result["screenshot_info"]

    @pytest.mark.asyncio
    async def test_browser_action_error(self, mock_sandbox):
        """Test browser action error handling."""
        mock_sandbox.browser_action = AsyncMock(
            return_value={"status": "error", "error": "Element not found"}
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await browser_action(executor, "click", selector="nonexistent")

        assert result["status"] == "error"

    def test_create_browser_tool_returns_structured_tool(self, mock_sandbox):
        """Test that create_browser_tool creates a StructuredTool."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox)
        tool = create_browser_tool(executor)

        assert tool.name == "browser"
        assert "interactive" in tool.description.lower() or "browser" in tool.description.lower()


class TestFileEditorTool:
    """Tests for unified file editor tool."""

    @pytest.mark.asyncio
    async def test_file_editor_view(self, mock_sandbox):
        """Test file editor view command."""
        mock_sandbox.file_editor = AsyncMock(
            return_value={"status": "success", "content": "1\tline one\n2\tline two\n", "total_lines": 2}
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await file_editor(executor, "view", "/workspace/test.txt")

        assert result["status"] == "success"
        assert "content" in result
        mock_sandbox.file_editor.assert_called_once_with(
            command="view", path="/workspace/test.txt"
        )

    @pytest.mark.asyncio
    async def test_file_editor_create(self, mock_sandbox):
        """Test file editor create command."""
        mock_sandbox.file_editor = AsyncMock(
            return_value={"status": "success", "path": "/workspace/new.txt", "size": 11}
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await file_editor(
            executor, "create", "/workspace/new.txt", file_text="hello world"
        )

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_file_editor_str_replace(self, mock_sandbox):
        """Test file editor str_replace command."""
        mock_sandbox.file_editor = AsyncMock(
            return_value={"status": "success", "path": "/workspace/test.txt", "replacements": 1}
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await file_editor(
            executor, "str_replace", "/workspace/test.txt",
            old_str="hello", new_str="goodbye"
        )

        assert result["status"] == "success"
        mock_sandbox.file_editor.assert_called_once_with(
            command="str_replace",
            path="/workspace/test.txt",
            old_str="hello",
            new_str="goodbye",
        )

    @pytest.mark.asyncio
    async def test_file_editor_insert(self, mock_sandbox):
        """Test file editor insert command."""
        mock_sandbox.file_editor = AsyncMock(
            return_value={"status": "success", "path": "/workspace/test.txt", "total_lines": 3}
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await file_editor(
            executor, "insert", "/workspace/test.txt",
            insert_line=1, new_str="new line"
        )

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_file_editor_view_truncates_content(self, mock_sandbox):
        """Test that file editor truncates long view content."""
        long_content = "X" * 20000
        mock_sandbox.file_editor = AsyncMock(
            return_value={"status": "success", "content": long_content}
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox, max_output_chars=200)

        result = await file_editor(executor, "view", "/workspace/bigfile.txt")

        assert result["status"] == "success"
        assert len(result["content"]) < 20000
        assert "truncated" in result["content"]

    @pytest.mark.asyncio
    async def test_file_editor_error(self, mock_sandbox):
        """Test file editor error handling."""
        mock_sandbox.file_editor = AsyncMock(
            return_value={"status": "error", "error": "File not found: /workspace/nope.txt"}
        )
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await file_editor(executor, "view", "/workspace/nope.txt")

        assert result["status"] == "error"

    def test_create_file_editor_tool_returns_structured_tool(self, mock_sandbox):
        """Test that create_file_editor_tool creates a StructuredTool."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox)
        tool = create_file_editor_tool(executor)

        assert tool.name == "file_editor"
        assert "view" in tool.description.lower()
        assert "str_replace" in tool.description.lower()
