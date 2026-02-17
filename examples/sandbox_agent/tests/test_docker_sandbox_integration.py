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
"""Integration tests for Docker sandbox.

These tests require Docker to be running and will create real containers.
Mark with @pytest.mark.integration to allow skipping in CI environments.

NOTE: These tests use _exec_run() for direct command execution since
they test the container infrastructure, not the HTTP services.
"""

import pytest
import pytest_asyncio

from nat_sandbox_agent.sandbox.docker_sandbox import DockerSandbox


def docker_available():
    """Check if Docker is available."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


# Check Docker availability at module load time
_docker_available = docker_available()

# Skip all tests in this module if Docker is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


# Use a fixture-based skip instead of pytestmark for conditional skipping
@pytest.fixture(autouse=True)
def skip_without_docker():
    """Skip tests if Docker is not available."""
    if not _docker_available:
        pytest.skip("Docker not available")


class TestDockerSandboxLifecycle:
    """Integration tests for Docker sandbox lifecycle."""

    @pytest_asyncio.fixture
    async def sandbox(self):
        """Create a Docker sandbox for testing."""
        sandbox = DockerSandbox(
            image="python:3.12-slim",
            memory_limit="256m",
            cpu_limit=0.5,
            network_enabled=True,
        )
        yield sandbox
        # Cleanup after test
        try:
            await sandbox.cleanup()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_start_and_cleanup(self, sandbox):
        """Test sandbox start and cleanup lifecycle."""
        await sandbox.start()

        # Container should exist
        assert sandbox._container is not None
        # Port mapping should be populated
        assert len(sandbox._host_ports) == 3

        await sandbox.cleanup()

        assert sandbox._container is None


class TestDockerSandboxBootstrap:
    """Integration tests for Docker sandbox bootstrap commands."""

    @pytest_asyncio.fixture
    async def running_sandbox(self):
        """Create and start a Docker sandbox."""
        sandbox = DockerSandbox(
            image="python:3.12-slim",
            memory_limit="256m",
            cpu_limit=0.5,
            network_enabled=True,
        )
        await sandbox.start()
        yield sandbox
        try:
            await sandbox.cleanup()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_exec_run_success(self, running_sandbox):
        """Test running a simple command via _exec_run."""
        result = await running_sandbox._exec_run("echo Hello World")

        assert result.success is True
        assert result.exit_code == 0
        assert "Hello World" in result.stdout

    @pytest.mark.asyncio
    async def test_exec_run_failure(self, running_sandbox):
        """Test running a command that fails via _exec_run."""
        result = await running_sandbox._exec_run("exit 1")

        assert result.success is False
        assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_exec_run_python(self, running_sandbox):
        """Test running Python code via _exec_run."""
        result = await running_sandbox._exec_run('python3 -c "print(2 ** 10)"')

        assert result.success is True
        assert "1024" in result.stdout

    @pytest.mark.asyncio
    async def test_workspace_directories_exist(self, running_sandbox):
        """Test that workspace directories are created."""
        result = await running_sandbox._exec_run(
            "ls -d /workspace/input /workspace/output /workspace/temp /workspace/downloads"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_port_mapping(self, running_sandbox):
        """Test that all service ports are mapped."""
        assert 8888 in running_sandbox._host_ports
        assert 8889 in running_sandbox._host_ports
        assert 8890 in running_sandbox._host_ports

        # All ports should be valid port numbers
        for port in running_sandbox._host_ports.values():
            assert 1024 <= port <= 65535


class TestDockerSandboxNetwork:
    """Integration tests for Docker sandbox network operations."""

    @pytest_asyncio.fixture
    async def running_sandbox(self):
        """Create and start a Docker sandbox with network enabled."""
        sandbox = DockerSandbox(
            image="python:3.12-slim",
            memory_limit="256m",
            cpu_limit=0.5,
            network_enabled=True,
        )
        await sandbox.start()
        yield sandbox
        try:
            await sandbox.cleanup()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_network_access(self, running_sandbox):
        """Test that sandbox has network access."""
        result = await running_sandbox._exec_run("getent hosts google.com")
        assert result.exit_code == 0

    @pytest_asyncio.fixture
    async def no_network_sandbox(self):
        """Create and start a Docker sandbox without network."""
        sandbox = DockerSandbox(
            image="python:3.12-slim",
            memory_limit="256m",
            cpu_limit=0.5,
            network_enabled=False,
        )
        await sandbox.start()
        yield sandbox
        try:
            await sandbox.cleanup()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_no_network_access(self, no_network_sandbox):
        """Test that sandbox without network cannot access internet."""
        result = await no_network_sandbox._exec_run(
            "python3 -c \"import socket; socket.gethostbyname('google.com')\"",
            timeout=10,
        )
        assert result.exit_code != 0


class TestDockerSandboxResourceLimits:
    """Integration tests for Docker sandbox resource limits."""

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Test that command timeout works."""
        sandbox = DockerSandbox(
            image="python:3.12-slim",
            memory_limit="256m",
            cpu_limit=0.5,
        )

        try:
            await sandbox.start()
            result = await sandbox._exec_run("sleep 10", timeout=2)
            assert result.success is False

        finally:
            await sandbox.cleanup()
