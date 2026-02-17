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
"""Base sandbox interface with HTTP service communication.

Architecture:
    Container runs 3 HTTP services. Host communicates via httpx/websockets.
    - Port 8888: Jupyter Kernel Gateway (IPython, persistent variables)
    - Port 8889: Browser Server (Playwright, persistent session)
    - Port 8890: Shell Server (Bash, persistent state + file editor)

    Subclasses implement URL construction + lifecycle management.
    All communication logic lives in this base class.
"""

import json
import logging
import re
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from types import TracebackType

import httpx
import websockets
from websockets import State

# ============ Workspace Constants ============
# These constants define the standard workspace directory structure.
# All sandbox implementations should use these paths.

WORKSPACE_ROOT = "/workspace"
WORKSPACE_INPUT = f"{WORKSPACE_ROOT}/input"
WORKSPACE_OUTPUT = f"{WORKSPACE_ROOT}/output"
WORKSPACE_TEMP = f"{WORKSPACE_ROOT}/temp"
WORKSPACE_DOWNLOADS = f"{WORKSPACE_ROOT}/downloads"

# Command to initialize workspace directories
WORKSPACE_INIT_COMMAND = f"mkdir -p {WORKSPACE_INPUT} {WORKSPACE_OUTPUT} {WORKSPACE_TEMP} {WORKSPACE_DOWNLOADS}"

logger = logging.getLogger(__name__)

# Regex to strip ANSI escape sequences from output
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


@dataclass
class CommandResult:
    """Result of a command execution in the sandbox."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """Check if the command executed successfully."""
        return self.exit_code == 0

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "success": self.success,
        }


@dataclass
class IPythonResult:
    """Result of IPython code execution."""

    exit_code: int
    text_output: str
    error_output: str
    image_urls: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.exit_code == 0


class BaseSandbox(ABC):
    """Abstract base class for sandbox implementations.

    Subclasses must implement:
    - Lifecycle: start, cleanup
    - URL construction: _get_service_url
    - Service lifecycle: start_ipython_kernel, stop_ipython_kernel,
      start_browser_server, stop_browser_server,
      start_persistent_shell, stop_persistent_shell
    - Bootstrap: _exec_run (for pre-service commands)

    Communication methods (execute_ipython, browser_action, run_command,
    file_editor) are implemented here using _get_service_url().
    """

    def __init__(self) -> None:
        import asyncio

        self._kernel_id: str | None = None
        self._kernel_ws: websockets.WebSocketClientProtocol | None = None
        self._kernel_lock = asyncio.Lock()
        self._browser_started: bool = False

    # ============ Lifecycle Management ============

    @abstractmethod
    async def start(self) -> None:
        """Start the sandbox environment (container/workspace)."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Destroy the sandbox completely."""

    # ============ URL Construction (core abstraction) ============

    @abstractmethod
    def _get_service_url(self, port: int) -> str:
        """Get the HTTP URL for a service running on the given port.

        Args:
            port: The container-side port (8888, 8889, or 8890).

        Returns:
            Full HTTP URL accessible from the host.
        """

    # ============ Bootstrap (pre-service) ============

    @abstractmethod
    async def _exec_run(self, command: str, timeout: float = 60) -> CommandResult:
        """Execute a command directly in the container (bypassing services).

        Used only for bootstrapping services before they're available.
        """

    # ============ Service Lifecycle ============

    @abstractmethod
    async def start_ipython_kernel(self) -> None:
        """Start Jupyter Kernel Gateway and create a kernel."""

    @abstractmethod
    async def stop_ipython_kernel(self) -> None:
        """Stop the IPython kernel and Kernel Gateway."""

    @abstractmethod
    async def start_browser_server(self) -> None:
        """Start the browser server (Playwright + FastAPI)."""

    @abstractmethod
    async def stop_browser_server(self) -> None:
        """Stop the browser server."""

    @abstractmethod
    async def start_persistent_shell(self) -> None:
        """Start the persistent shell server."""

    @abstractmethod
    async def stop_persistent_shell(self) -> None:
        """Stop the persistent shell server."""

    # ============ IPython Communication ============

    async def _connect_kernel_ws(self) -> None:
        """Connect WebSocket to the kernel channels endpoint."""
        if self._kernel_ws is not None:
            try:
                await self._kernel_ws.close()
            except Exception:
                pass

        url = self._get_service_url(8888)
        ws_url = url.replace("http://", "ws://").replace("https://", "wss://")
        self._kernel_ws = await websockets.connect(
            f"{ws_url}/api/kernels/{self._kernel_id}/channels",
            max_size=100 * 1024 * 1024,  # 100MB max message size
        )

    async def execute_ipython(self, code: str, timeout: float = 120) -> IPythonResult:
        """Execute Python code in the persistent IPython kernel.

        Variables, imports, and state persist across calls.
        Uses a lock to serialize concurrent WebSocket access.

        Args:
            code: Python code to execute.
            timeout: Maximum execution time in seconds.

        Returns:
            IPythonResult with output and optional images.
        """
        import asyncio
        import uuid as _uuid

        if not self._kernel_id:
            raise RuntimeError("IPython kernel not started")

        async with self._kernel_lock:
            # Reconnect WebSocket if needed
            ws_closed = (
                self._kernel_ws is None
                or getattr(self._kernel_ws, "state", None) == State.CLOSED
                or getattr(self._kernel_ws, "closed", False)
            )
            if ws_closed:
                try:
                    await self._connect_kernel_ws()
                except Exception:
                    # Kernel may have crashed - try to recreate
                    logger.warning("WebSocket connection failed, recreating kernel")
                    await self._recreate_kernel()

            msg_id = _uuid.uuid4().hex

            # Jupyter wire protocol execute_request
            execute_msg = {
                "header": {
                    "msg_id": msg_id,
                    "msg_type": "execute_request",
                    "username": "sandbox",
                    "session": msg_id,
                    "version": "5.3",
                },
                "parent_header": {},
                "metadata": {},
                "content": {
                    "code": code,
                    "silent": False,
                    "store_history": True,
                    "user_expressions": {},
                    "allow_stdin": False,
                    "stop_on_error": True,
                },
                "buffers": [],
                "channel": "shell",
            }

            try:
                await self._kernel_ws.send(json.dumps(execute_msg))
            except Exception:
                # WebSocket broken, reconnect and retry once
                await self._connect_kernel_ws()
                await self._kernel_ws.send(json.dumps(execute_msg))

            # Collect output
            text_parts: list[str] = []
            error_parts: list[str] = []
            image_urls: list[str] = []
            exit_code = 0

            try:
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            self._kernel_ws.recv(), timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        # Interrupt the kernel
                        await self._interrupt_kernel()
                        return IPythonResult(
                            exit_code=-1,
                            text_output="\n".join(text_parts),
                            error_output=f"Execution timed out after {timeout}s",
                            image_urls=image_urls,
                        )

                    msg = json.loads(raw)
                    msg_type = msg.get("msg_type", "")
                    parent_msg_id = msg.get("parent_header", {}).get("msg_id", "")

                    # Only process messages for our request
                    if parent_msg_id != msg_id:
                        continue

                    content = msg.get("content", {})

                    if msg_type == "stream":
                        text = content.get("text", "")
                        text_parts.append(_ANSI_ESCAPE_RE.sub("", text))

                    elif msg_type == "execute_result":
                        data = content.get("data", {})
                        if "text/plain" in data:
                            text_parts.append(data["text/plain"])
                        if "image/png" in data:
                            image_urls.append(
                                f"data:image/png;base64,{data['image/png']}"
                            )

                    elif msg_type == "display_data":
                        data = content.get("data", {})
                        if "text/plain" in data:
                            text_parts.append(data["text/plain"])
                        if "image/png" in data:
                            image_urls.append(
                                f"data:image/png;base64,{data['image/png']}"
                            )

                    elif msg_type == "error":
                        exit_code = 1
                        traceback = content.get("traceback", [])
                        # Clean ANSI codes from traceback
                        clean_tb = [
                            _ANSI_ESCAPE_RE.sub("", line) for line in traceback
                        ]
                        error_parts.extend(clean_tb)

                    elif msg_type in ("execute_reply",):
                        status = content.get("status", "")
                        if status == "error":
                            exit_code = 1
                        break  # Done

            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket closed during execution")
                self._kernel_ws = None
                if not text_parts and not error_parts:
                    return IPythonResult(
                        exit_code=-1,
                        text_output="",
                        error_output="Kernel connection lost during execution",
                    )

        return IPythonResult(
            exit_code=exit_code,
            text_output="\n".join(text_parts),
            error_output="\n".join(error_parts),
            image_urls=image_urls,
        )

    async def _interrupt_kernel(self) -> None:
        """Send interrupt request to the kernel."""
        if not self._kernel_id:
            return
        try:
            url = self._get_service_url(8888)
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{url}/api/kernels/{self._kernel_id}/interrupt",
                    timeout=10,
                )
        except Exception as e:
            logger.warning(f"Failed to interrupt kernel: {e}")

    async def _recreate_kernel(self) -> None:
        """Recreate kernel after crash (delete old + create new + reconnect WS)."""
        url = self._get_service_url(8888)
        async with httpx.AsyncClient() as client:
            # Try to delete old kernel
            if self._kernel_id:
                try:
                    await client.delete(f"{url}/api/kernels/{self._kernel_id}", timeout=5)
                except Exception:
                    pass

            # Create new kernel
            resp = await client.post(f"{url}/api/kernels", timeout=30)
            resp.raise_for_status()
            self._kernel_id = resp.json()["id"]

        # Reconnect WebSocket
        await self._connect_kernel_ws()
        logger.info(f"Recreated kernel: {self._kernel_id}")

    # ============ Browser Communication ============

    async def _ensure_browser_started(self) -> None:
        """Lazily start browser server on first use."""
        if not self._browser_started:
            await self.start_browser_server()
            self._browser_started = True

    async def browser_action(self, action: dict) -> dict:
        """Execute a browser action (goto, click, fill, get_text, etc.).

        Browser is lazily initialized on first call.

        Args:
            action: Dict with 'type' and action-specific fields.

        Returns:
            Dict with status, url, title, and content.
        """
        await self._ensure_browser_started()

        url = self._get_service_url(8889)
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{url}/action",
                    json=action,
                    timeout=60,
                )
                return resp.json()
            except httpx.TimeoutException:
                return {"status": "error", "error": "Browser action timed out after 60s"}
            except Exception as e:
                return {"status": "error", "error": str(e)}

    # ============ Shell Communication ============

    async def run_command(self, command: str, timeout: float = 120) -> CommandResult:
        """Execute a shell command in the persistent shell.

        cd, export, alias persist across calls.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds.

        Returns:
            CommandResult with exit_code, stdout, stderr.
        """
        url = self._get_service_url(8890)
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{url}/execute",
                    json={"command": command, "timeout": timeout},
                    timeout=timeout + 10,
                )
                data = resp.json()
                return CommandResult(
                    exit_code=data.get("exit_code", -1),
                    stdout=data.get("stdout", ""),
                    stderr=data.get("stderr", ""),
                )
            except httpx.TimeoutException:
                return CommandResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {timeout}s",
                )
            except Exception as e:
                return CommandResult(
                    exit_code=-1,
                    stdout="",
                    stderr=str(e),
                )

    async def file_editor(self, command: str, path: str, **kwargs) -> dict:
        """Unified file operation via shell server's /file_editor endpoint.

        Content is transmitted via JSON - safe for special characters.

        Args:
            command: One of 'view', 'create', 'write', 'str_replace', 'insert'.
            path: Absolute file path.
            **kwargs: Additional args (file_text, old_str, new_str, etc.).

        Returns:
            Dict with status and operation result.
        """
        url = self._get_service_url(8890)
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{url}/file_editor",
                    json={"command": command, "path": path, **kwargs},
                    timeout=30,
                )
                return resp.json()
            except Exception as e:
                return {"status": "error", "error": str(e)}

    # ============ Context Manager Support ============

    async def __aenter__(self) -> "BaseSandbox":
        """Async context manager entry - starts the sandbox."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit - cleans up the sandbox."""
        await self.cleanup()
