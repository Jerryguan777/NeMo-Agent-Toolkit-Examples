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
"""Factory for creating all agent tools.

All tools are sandbox-side. Web search is done via tavily-python inside
the IPython kernel (TAVILY_API_KEY env var passed to container).
"""

from langchain_core.tools import StructuredTool

from nat_sandbox_agent.sandbox.base import BaseSandbox
from nat_sandbox_agent.tools.common import DEFAULT_MAX_OUTPUT_CHARS
from nat_sandbox_agent.tools.sandbox import create_sandbox_tools


def create_all_tools(
    sandbox: BaseSandbox,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    include_tools: list[str] | None = None,
) -> list[StructuredTool]:
    """Create all tools (sandbox-only).

    Tools: shell, python, browser, file_editor.
    Web search is done via tavily-python in IPython.

    Args:
        sandbox: Sandbox instance for tools.
        max_output_chars: Maximum characters for tool output truncation.
        include_tools: Optional list of tool names to include.

    Returns:
        List of all tools.
    """
    return create_sandbox_tools(
        sandbox=sandbox,
        max_output_chars=max_output_chars,
        include_tools=include_tools,
    )


def get_tool_descriptions() -> str:
    """Get formatted descriptions of all available tools."""
    tools_info = [
        ("python", "Execute Python in persistent IPython (variables persist)"),
        ("shell", "Execute bash in persistent shell (cd/export persist)"),
        ("browser", "Interactive web browser (multi-step browsing)"),
        ("file_editor", "File operations (view/create/write/str_replace/insert)"),
    ]

    lines = ["Available tools:"]
    for name, desc in tools_info:
        lines.append(f"  - {name}: {desc}")

    return "\n".join(lines)
