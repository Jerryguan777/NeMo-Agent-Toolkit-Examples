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
"""Docker-based sandbox implementation with HTTP service architecture.

Container runs 3 HTTP services:
- Port 8888: Jupyter Kernel Gateway (IPython)
- Port 8889: Browser Server (Playwright)
- Port 8890: Shell Server (persistent bash + file editor)

Host communicates via httpx/websockets through Docker port mapping.
"""

import asyncio
import logging
import shlex
import uuid

import docker
import httpx
from docker.errors import ContainerError
from docker.errors import ImageNotFound
from docker.errors import NotFound

from nat_sandbox_agent.sandbox.base import WORKSPACE_INIT_COMMAND
from nat_sandbox_agent.sandbox.base import WORKSPACE_ROOT
from nat_sandbox_agent.sandbox.base import BaseSandbox
from nat_sandbox_agent.sandbox.base import CommandResult
from nat_sandbox_agent.sandbox.servers.browser_server import BROWSER_SERVER_SOURCE
from nat_sandbox_agent.sandbox.servers.shell_server import SHELL_SERVER_SOURCE

logger = logging.getLogger(__name__)

# Ports used by in-container services
SERVICE_PORTS = [8888, 8889, 8890]


class DockerSandbox(BaseSandbox):
    """Docker-based sandbox with HTTP service architecture.

    Each container exposes 3 services via Docker port mapping.
    _get_service_url() maps container ports to host-accessible URLs.
    """

    DEFAULT_IMAGE = "nat-sandbox:latest"
    DEFAULT_WORK_DIR = WORKSPACE_ROOT

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        memory_limit: str = "4g",
        cpu_limit: float = 4.0,
        network_enabled: bool = True,
        work_dir: str = DEFAULT_WORK_DIR,
        container_name: str | None = None,
        auto_remove: bool = False,
        environment: dict[str, str] | None = None,
        volumes: dict[str, dict[str, str]] | None = None,
    ):
        super().__init__()
        self._image = image
        self._memory_limit = memory_limit
        self._cpu_limit = cpu_limit
        self._network_enabled = network_enabled
        self._work_dir = work_dir
        self._container_name = container_name or f"sandbox_{uuid.uuid4().hex[:8]}"
        self._auto_remove = auto_remove
        self._environment = environment or {}
        self._volumes = volumes or {}

        self._client: docker.DockerClient | None = None
        self._container = None
        self._host_ports: dict[int, int] = {}

    # ============ Lifecycle ============

    async def start(self) -> None:
        """Start the Docker container with port mapping for services."""
        logger.info(f"Starting Docker sandbox: {self._container_name}")

        try:
            self._client = docker.from_env()

            # Ensure image is available
            try:
                self._client.images.get(self._image)
            except ImageNotFound:
                logger.info(f"Pulling image: {self._image}")
                await asyncio.get_running_loop().run_in_executor(
                    None, self._client.images.pull, self._image
                )

            # Container configuration with port mapping
            container_config = {
                "image": self._image,
                "name": self._container_name,
                "detach": True,
                "tty": True,
                "stdin_open": True,
                "working_dir": self._work_dir,
                "mem_limit": self._memory_limit,
                "nano_cpus": int(self._cpu_limit * 1e9),
                "network_mode": "bridge" if self._network_enabled else "none",
                "command": "/bin/bash",
                "auto_remove": self._auto_remove,
                "environment": self._environment,
                # Docker auto-assigns host ports (None = random available port)
                "ports": {f"{p}/tcp": None for p in SERVICE_PORTS},
            }

            if self._volumes:
                container_config["volumes"] = self._volumes
                logger.info(f"Mounting volumes: {list(self._volumes.keys())}")

            # Create and start container
            self._container = await asyncio.get_running_loop().run_in_executor(
                None, lambda: self._client.containers.create(**container_config)
            )
            await asyncio.get_running_loop().run_in_executor(None, self._container.start)

            # Parse assigned host ports
            self._container.reload()
            ports_info = self._container.ports
            for port in SERVICE_PORTS:
                key = f"{port}/tcp"
                if key in ports_info and ports_info[key]:
                    self._host_ports[port] = int(ports_info[key][0]["HostPort"])
                    logger.debug(f"Port {port} -> host:{self._host_ports[port]}")

            # Initialize workspace directories
            await self._exec_run(WORKSPACE_INIT_COMMAND)

            logger.info(
                f"Docker sandbox started: {self._container_name}, "
                f"ports={self._host_ports}"
            )

        except Exception as e:
            logger.error(f"Failed to start Docker sandbox: {e}")
            if self._client:
                self._client.close()
                self._client = None
            raise

    async def cleanup(self) -> None:
        """Remove the Docker container completely."""
        if self._container:
            logger.info(f"Cleaning up Docker sandbox: {self._container_name}")
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: self._container.remove(force=True)
                )
                self._container = None
            except NotFound:
                self._container = None
            except Exception as e:
                logger.error(f"Failed to cleanup Docker sandbox: {e}")
                raise
            finally:
                if self._client:
                    self._client.close()
                    self._client = None

    # ============ URL Construction ============

    def _get_service_url(self, port: int) -> str:
        if port not in self._host_ports:
            raise RuntimeError(
                f"Port {port} not mapped. Available: {self._host_ports}"
            )
        return f"http://localhost:{self._host_ports[port]}"

    # ============ Bootstrap ============

    async def _exec_run(self, command: str, timeout: float = 60) -> CommandResult:
        """Execute command directly via container.exec_run (bootstrap only)."""
        if not self._container:
            raise RuntimeError("Sandbox not started")

        timeout_int = int(timeout)
        wrapped = f"timeout {timeout_int} /bin/bash -c {shlex.quote(command)}"

        try:
            exec_result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self._container.exec_run(
                        cmd=wrapped,
                        workdir=self._work_dir,
                        demux=True,
                    ),
                ),
                timeout=timeout + 5,
            )

            exit_code = exec_result.exit_code
            stdout_bytes, stderr_bytes = exec_result.output
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

            if exit_code == 124:
                return CommandResult(exit_code=-1, stdout=stdout, stderr=f"Timed out after {timeout_int}s")

            return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

        except TimeoutError:
            return CommandResult(exit_code=-1, stdout="", stderr=f"Timed out after {timeout}s")
        except ContainerError as e:
            return CommandResult(exit_code=e.exit_status, stdout="", stderr=str(e))
        except Exception as e:
            return CommandResult(exit_code=-1, stdout="", stderr=str(e))

    # ============ Service Lifecycle: IPython ============

    async def start_ipython_kernel(self) -> None:
        """Start Jupyter Kernel Gateway and create a kernel."""
        logger.info("Starting Jupyter Kernel Gateway...")

        # Start kernel gateway in background
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
        await self._poll_health(url, "/api", timeout=30, label="Kernel Gateway")

        # Create a kernel
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{url}/api/kernels", timeout=15)
            resp.raise_for_status()
            self._kernel_id = resp.json()["id"]

        # Connect WebSocket
        await self._connect_kernel_ws()
        logger.info(f"IPython kernel ready: {self._kernel_id}")

    async def stop_ipython_kernel(self) -> None:
        """Stop IPython kernel and Kernel Gateway."""
        if self._kernel_ws:
            try:
                await self._kernel_ws.close()
            except Exception:
                pass
            self._kernel_ws = None

        if self._kernel_id and self._container:
            try:
                url = self._get_service_url(8888)
                async with httpx.AsyncClient() as client:
                    await client.delete(
                        f"{url}/api/kernels/{self._kernel_id}",
                        timeout=5,
                    )
            except Exception:
                pass
            self._kernel_id = None

        # Kill the kernel gateway process
        if self._container:
            try:
                await self._exec_run("pkill -f 'jupyter.kernelgateway' || true")
            except Exception:
                pass

    # ============ Service Lifecycle: Browser ============

    async def start_browser_server(self) -> None:
        """Write browser server script and start it."""
        logger.info("Starting Browser Server...")

        # Write the server script into the container
        await self._write_server_script("/opt/browser_server.py", BROWSER_SERVER_SOURCE)

        # Start in background
        await self._exec_run(
            "nohup python3 /opt/browser_server.py > /tmp/browser_server.log 2>&1 &"
        )

        # Wait for health
        url = self._get_service_url(8889)
        await self._poll_health(url, "/health", timeout=30, label="Browser Server")
        logger.info("Browser Server ready")

    async def stop_browser_server(self) -> None:
        """Stop the browser server."""
        if self._container and self._browser_started:
            try:
                await self._exec_run("pkill -f 'browser_server.py' || true")
            except Exception:
                pass

    # ============ Service Lifecycle: Shell ============

    async def start_persistent_shell(self) -> None:
        """Write shell server script and start it."""
        logger.info("Starting Shell Server...")

        # Write the server script into the container
        await self._write_server_script("/opt/shell_server.py", SHELL_SERVER_SOURCE)

        # Start in background
        await self._exec_run(
            "nohup python3 /opt/shell_server.py > /tmp/shell_server.log 2>&1 &"
        )

        # Wait for health
        url = self._get_service_url(8890)
        await self._poll_health(url, "/health", timeout=15, label="Shell Server")
        logger.info("Shell Server ready")

    async def stop_persistent_shell(self) -> None:
        """Stop the shell server."""
        if self._container:
            try:
                await self._exec_run("pkill -f 'shell_server.py' || true")
            except Exception:
                pass

    # ============ Helpers ============

    async def _write_server_script(self, path: str, source: str) -> None:
        """Write a Python script into the container using exec_run + heredoc."""
        # Use base64 to avoid any quoting issues
        import base64

        encoded = base64.b64encode(source.encode()).decode()
        result = await self._exec_run(
            f"echo '{encoded}' | base64 -d > {path}"
        )
        if not result.success:
            raise RuntimeError(f"Failed to write {path}: {result.stderr}")

    async def _poll_health(
        self, base_url: str, path: str, timeout: float = 30, label: str = "service"
    ) -> None:
        """Poll a service health endpoint until ready."""
        import time

        deadline = time.monotonic() + timeout
        last_error = None
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                try:
                    resp = await client.get(f"{base_url}{path}", timeout=3)
                    if resp.status_code == 200:
                        return
                except Exception as e:
                    last_error = e
                await asyncio.sleep(0.5)

        raise TimeoutError(
            f"{label} not ready after {timeout}s at {base_url}{path}: {last_error}"
        )
