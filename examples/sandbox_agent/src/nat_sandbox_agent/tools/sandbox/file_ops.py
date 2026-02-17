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
"""Unified file editor tool - view, create, write, str_replace, insert.

All file operations go through the shell server's /file_editor endpoint.
Content is transmitted via JSON, so special characters are handled safely.
"""

import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import Field

from nat_sandbox_agent.tools.sandbox.executor import SandboxToolExecutor

logger = logging.getLogger(__name__)


class FileEditorInput(BaseModel):
    """Input schema for the unified file editor."""

    command: str = Field(
        description=(
            "File operation to perform. One of: "
            "view (read file with line numbers), "
            "create (create new file), "
            "write (overwrite file), "
            "str_replace (replace exact string - must be unique in file), "
            "insert (insert text after a line number)"
        ),
    )
    path: str = Field(
        description="Absolute file path in the sandbox.",
    )
    view_range: list[int] | None = Field(
        default=None,
        description="[start_line, end_line] for 'view' command (1-indexed, inclusive).",
    )
    file_text: str | None = Field(
        default=None,
        description="Full file content for 'create' or 'write' commands.",
    )
    old_str: str | None = Field(
        default=None,
        description="Exact string to find for 'str_replace' (must be unique in file).",
    )
    new_str: str | None = Field(
        default=None,
        description="Replacement string for 'str_replace' or text for 'insert'.",
    )
    insert_line: int | None = Field(
        default=None,
        description="Line number to insert after for 'insert' (0 = beginning).",
    )


async def file_editor(
    executor: SandboxToolExecutor,
    command: str,
    path: str,
    view_range: list[int] | None = None,
    file_text: str | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | None = None,
) -> dict[str, Any]:
    """Execute a file operation via the shell server's /file_editor endpoint.

    Content is transmitted via JSON - special characters (quotes,
    backslashes, newlines) are handled automatically.
    """
    logger.info(f"File editor: {command} {path}")

    kwargs = {}
    if view_range is not None:
        kwargs["view_range"] = view_range
    if file_text is not None:
        kwargs["file_text"] = file_text
    if old_str is not None:
        kwargs["old_str"] = old_str
    if new_str is not None:
        kwargs["new_str"] = new_str
    if insert_line is not None:
        kwargs["insert_line"] = insert_line

    result = await executor.sandbox.file_editor(
        command=command,
        path=path,
        **kwargs,
    )

    # Truncate view content
    if result.get("content"):
        result["content"] = executor.truncate(result["content"])

    return result


def create_file_editor_tool(executor: SandboxToolExecutor) -> StructuredTool:
    """Create the unified file editor tool."""
    return StructuredTool.from_function(
        coroutine=lambda command, path, view_range=None, file_text=None, old_str=None, new_str=None, insert_line=None: (
            file_editor(executor, command, path, view_range, file_text, old_str, new_str, insert_line)
        ),
        name="file_editor",
        description=(
            "Unified file operations tool. Commands: "
            "view (read file with line numbers, optional view_range=[start,end]), "
            "create (create new file with file_text), "
            "write (overwrite existing file with file_text), "
            "str_replace (replace old_str with new_str - old_str must be unique), "
            "insert (insert new_str after insert_line, 0=beginning)."
        ),
        args_schema=FileEditorInput,
    )
