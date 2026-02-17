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
"""Daytona cloud-based sandbox implementation with HTTP service architecture.

Workspace runs 3 HTTP services:
- Port 8888: Jupyter Kernel Gateway (IPython)
- Port 8889: Browser Server (Playwright)
- Port 8890: Shell Server (persistent bash + file editor)

Host communicates via Daytona preview links.
"""

import asyncio
import logging
import time

import httpx

from nat_sandbox_agent.sandbox.base import WORKSPACE_INIT_COMMAND
from nat_sandbox_agent.sandbox.base import BaseSandbox
from nat_sandbox_agent.sandbox.base import CommandResult
from nat_sandbox_agent.sandbox.servers.browser_server import BROWSER_SERVER_SOURCE
from nat_sandbox_agent.sandbox.servers.shell_server import SHELL_SERVER_SOURCE

logger = logging.getLogger(__name__)


class DaytonaSandbox(BaseSandbox):
    """Daytona cloud-based sandbox with HTTP service architecture.

    Uses Daytona preview links for service URL construction.
    """

    DEFAULT_IMAGE = "daytonaio/workspace:latest"

    def __init__(
        self,
        api_key: str,
        server_url: str = "https://api.daytona.io",
        target: str = "us",
        image: str = DEFAULT_IMAGE,
        cpu: int = 4,
        memory: int = 4,
        disk: int = 10,
        auto_stop_interval: int = 30,
    ):
        super().__init__()
        self._api_key = api_key
        self._server_url = server_url
        self._target = target
        self._image = image
        self._cpu = cpu
        self._memory = memory
        self._disk = disk
        self._auto_stop_interval = auto_stop_interval

        self._client = None
        self._sandbox = None

    def _get_client(self):
        """Get or create Daytona client."""
        if self._client is None:
            try:
                from daytona_sdk import Daytona
                from daytona_sdk import DaytonaConfig

                config = DaytonaConfig(api_key=self._api_key)
                self._client = Daytona(config)
            except ImportError:
                raise ImportError(
                    "daytona-sdk package is required for DaytonaSandbox. "
                    "Install it with: pip install daytona-sdk"
                )
        return self._client

    # ============ Lifecycle ============

    async def start(self) -> None:
        """Start the Daytona sandbox."""
        logger.info("Starting Daytona sandbox")

        try:
            client = self._get_client()

            from daytona_sdk import CreateSandboxFromImageParams
            from daytona_sdk import Resources

            params = CreateSandboxFromImageParams(
                image=self._image,
                resources=Resources(
                    cpu=self._cpu,
                    memory=self._memory,
                    disk=self._disk,
                ),
                auto_stop_interval=self._auto_stop_interval,
            )

            self._sandbox = await asyncio.get_running_loop().run_in_executor(
                None, lambda: client.create(params)
            )

            # Initialize workspace directories
            await self._exec_run(WORKSPACE_INIT_COMMAND)

            logger.info(f"Daytona sandbox started: {self._sandbox.id}")

        except Exception as e:
            logger.error(f"Failed to start Daytona sandbox: {e}")
            raise

    async def cleanup(self) -> None:
        """Delete the Daytona sandbox."""
        if self._sandbox:
            logger.info(f"Cleaning up Daytona sandbox: {self._sandbox.id}")
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, self._sandbox.delete
                )
                self._sandbox = None
            except Exception as e:
                logger.error(f"Failed to cleanup Daytona sandbox: {e}")
                raise
            finally:
                self._client = None

    # ============ URL Construction ============

    def _get_service_url(self, port: int) -> str:
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")
        return self._sandbox.get_preview_link(port).url

    # ============ Bootstrap ============

    async def _exec_run(self, command: str, timeout: float = 60) -> CommandResult:
        """Execute command via Daytona process.exec (bootstrap only)."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")

        try:
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self._sandbox.process.exec(
                        command,
                        cwd="/workspace",
                        timeout=int(timeout),
                    ),
                ),
                timeout=timeout + 5,
            )

            return CommandResult(
                exit_code=result.exit_code,
                stdout=result.result if result.result else "",
                stderr="",
            )
        except TimeoutError:
            return CommandResult(exit_code=-1, stdout="", stderr=f"Timed out after {timeout}s")
        except Exception as e:
            return CommandResult(exit_code=-1, stdout="", stderr=str(e))

    # ============ Service Lifecycle: IPython ============

    async def start_ipython_kernel(self) -> None:
        """Start Jupyter Kernel Gateway via Daytona process.exec."""
        logger.info("Starting Jupyter Kernel Gateway (Daytona)...")

        # Install kernel gateway if not present
        await self._exec_run(
            "pip install -q ipykernel jupyter_kernel_gateway 2>/dev/null || true"
        )

        # Start kernel gateway
        await self._exec_run(
            "nohup jupyter kernelgateway "
            "--KernelGatewayApp.ip=0.0.0.0 "
            "--KernelGatewayApp.port=8888 "
            "--KernelGatewayApp.allow_origin='*' "
            "--JupyterApp.answer_yes=true "
            "> /tmp/kernel_gateway.log 2>&1 &"
        )

        # Wait for kernel gateway to be ready
        url = self._get_service_url(8888)
        await self._poll_health(url, "/api", timeout=60, label="Kernel Gateway")

        # Create kernel
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{url}/api/kernels", timeout=15)
            resp.raise_for_status()
            self._kernel_id = resp.json()["id"]

        # Connect WebSocket
        await self._connect_kernel_ws()
        logger.info(f"IPython kernel ready: {self._kernel_id}")

    async def stop_ipython_kernel(self) -> None:
        """Stop IPython kernel."""
        if self._kernel_ws:
            try:
                await self._kernel_ws.close()
            except Exception:
                pass
            self._kernel_ws = None

        if self._kernel_id and self._sandbox:
            try:
                url = self._get_service_url(8888)
                async with httpx.AsyncClient() as client:
                    await client.delete(f"{url}/api/kernels/{self._kernel_id}", timeout=5)
            except Exception:
                pass
            self._kernel_id = None

        if self._sandbox:
            try:
                await self._exec_run("pkill -f 'jupyter.kernelgateway' || true")
            except Exception:
                pass

    # ============ Service Lifecycle: Browser ============

    async def start_browser_server(self) -> None:
        """Start browser server in Daytona workspace."""
        logger.info("Starting Browser Server (Daytona)...")

        # Install dependencies if needed
        await self._exec_run(
            "pip install -q fastapi uvicorn playwright 2>/dev/null && "
            "playwright install chromium --with-deps 2>/dev/null || true"
        )

        # Write and start server
        await self._write_server_script("/opt/browser_server.py", BROWSER_SERVER_SOURCE)
        await self._exec_run(
            "nohup python3 /opt/browser_server.py > /tmp/browser_server.log 2>&1 &"
        )

        url = self._get_service_url(8889)
        await self._poll_health(url, "/health", timeout=60, label="Browser Server")
        logger.info("Browser Server ready")

    async def stop_browser_server(self) -> None:
        if self._sandbox and self._browser_started:
            try:
                await self._exec_run("pkill -f 'browser_server.py' || true")
            except Exception:
                pass

    # ============ Service Lifecycle: Shell ============

    async def start_persistent_shell(self) -> None:
        """Start shell server in Daytona workspace."""
        logger.info("Starting Shell Server (Daytona)...")

        # Install dependencies if needed
        await self._exec_run(
            "pip install -q fastapi uvicorn 2>/dev/null || true"
        )

        # Write and start server
        await self._write_server_script("/opt/shell_server.py", SHELL_SERVER_SOURCE)
        await self._exec_run(
            "nohup python3 /opt/shell_server.py > /tmp/shell_server.log 2>&1 &"
        )

        url = self._get_service_url(8890)
        await self._poll_health(url, "/health", timeout=30, label="Shell Server")
        logger.info("Shell Server ready")

    async def stop_persistent_shell(self) -> None:
        if self._sandbox:
            try:
                await self._exec_run("pkill -f 'shell_server.py' || true")
            except Exception:
                pass

    # ============ Helpers ============

    async def _write_server_script(self, path: str, source: str) -> None:
        """Write a Python script into the Daytona workspace."""
        import base64

        encoded = base64.b64encode(source.encode()).decode()
        result = await self._exec_run(
            f"echo '{encoded}' | base64 -d > {path}"
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to write {path}: {result.stderr}")

    async def _poll_health(
        self, base_url: str, path: str, timeout: float = 60, label: str = "service"
    ) -> None:
        """Poll a service health endpoint until ready."""
        deadline = time.monotonic() + timeout
        last_error = None
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                try:
                    resp = await client.get(f"{base_url}{path}", timeout=5)
                    if resp.status_code == 200:
                        return
                except Exception as e:
                    last_error = e
                await asyncio.sleep(1.0)

        raise TimeoutError(
            f"{label} not ready after {timeout}s at {base_url}{path}: {last_error}"
        )
