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
"""Shell and Python execution tools for sandbox.

- python: Executes in persistent IPython kernel (variables/imports persist)
- shell: Executes in persistent bash (cd/export/alias persist)
"""

import base64 as b64_module
import logging
import time
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import Field

from nat_sandbox_agent.tools.sandbox.executor import SandboxToolExecutor

logger = logging.getLogger(__name__)


class ShellInput(BaseModel):
    """Input schema for shell command execution."""

    command: str = Field(description="The shell command to execute in the sandbox.")


class PythonInput(BaseModel):
    """Input schema for Python code execution."""

    code: str = Field(description="Python code to execute in the persistent IPython kernel.")


async def execute_shell(
    executor: SandboxToolExecutor,
    command: str,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Execute a shell command in the persistent shell.

    cd, export, alias persist across calls.
    """
    logger.info(f"Executing shell command: {command[:80]}...")

    result = await executor.sandbox.run_command(
        command=command,
        timeout=timeout or executor.default_timeout,
    )

    return {
        "status": "success" if result.success else "error",
        "stdout": executor.truncate(result.stdout),
        "stderr": executor.truncate(result.stderr),
        "exit_code": result.exit_code,
    }


async def execute_python(
    executor: SandboxToolExecutor,
    code: str,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Execute Python code in the persistent IPython kernel.

    Variables, imports, and state persist across calls.
    """
    logger.info(f"Executing Python code ({len(code)} chars)")

    result = await executor.sandbox.execute_ipython(
        code=code,
        timeout=timeout or executor.default_timeout,
    )

    # Save base64 images to container files, return paths only
    image_paths = []
    for i, url in enumerate(result.image_urls):
        if url.startswith("data:image/"):
            try:
                b64_data = url.split(",", 1)[1]
                img_bytes = b64_module.b64decode(b64_data)
                img_path = f"/workspace/output/plot_{int(time.time() * 1000)}_{i}.png"
                # Write via chunked shell commands to avoid command-line length limits
                chunk_size = 500_000  # ~500K chars per chunk, safe for shell
                if len(b64_data) <= chunk_size:
                    await executor.sandbox.run_command(
                        f"echo '{b64_data}' | base64 -d > {img_path}",
                        timeout=10,
                    )
                else:
                    # Chunked write for large images
                    await executor.sandbox.run_command(f"> {img_path}", timeout=5)
                    for offset in range(0, len(b64_data), chunk_size):
                        chunk = b64_data[offset:offset + chunk_size]
                        await executor.sandbox.run_command(
                            f"echo '{chunk}' | base64 -d >> {img_path}",
                            timeout=10,
                        )
                image_paths.append(img_path)
            except Exception:
                logger.exception("Failed to save image to container, skipping")
                image_paths.append("[image save failed]")
        else:
            image_paths.append(url)  # Non-base64 URLs pass through

    # List any generated files in output directory
    generated_files = await executor.list_generated_files()

    return {
        "status": "success" if result.success else "error",
        "stdout": executor.truncate(result.text_output),
        "stderr": executor.truncate(result.error_output),
        "image_paths": image_paths,
        "generated_files": generated_files,
    }


def create_shell_tool(executor: SandboxToolExecutor) -> StructuredTool:
    """Create the shell command tool."""
    return StructuredTool.from_function(
        coroutine=lambda command: execute_shell(executor, command),
        name="shell",
        description=(
            "Execute bash commands in a PERSISTENT shell. "
            "cd, export, alias persist across calls. "
            "Root access: use apt-get install -y for system packages. "
            "Use for: file management, downloads (curl/wget), git, system commands. "
            "Do NOT use for data processing - use python instead."
        ),
        args_schema=ShellInput,
    )


def create_python_tool(executor: SandboxToolExecutor) -> StructuredTool:
    """Create the Python execution tool."""
    return StructuredTool.from_function(
        coroutine=lambda code: execute_python(executor, code),
        name="python",
        description=(
            "Execute Python code in a PERSISTENT IPython environment. "
            "Variables, imports, and state persist across calls. "
            "Use %pip install to install packages. "
            "Pre-installed: pandas, numpy, matplotlib, pillow, requests, httpx, "
            "beautifulsoup4, openpyxl, pyyaml, tavily-python. "
            "Save outputs to /workspace/output/."
        ),
        args_schema=PythonInput,
    )
