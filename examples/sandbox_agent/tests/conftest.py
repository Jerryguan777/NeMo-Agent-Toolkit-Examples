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
"""Pytest configuration and fixtures for sandbox agent tests."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from nat_sandbox_agent.sandbox.base import BaseSandbox
from nat_sandbox_agent.sandbox.base import CommandResult
from nat_sandbox_agent.sandbox.base import IPythonResult


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """Create a mock sandbox instance for testing.

    Mocks all BaseSandbox methods including:
    - Lifecycle: start, cleanup
    - Services: start/stop for ipython, browser, shell
    - Communication: execute_ipython, browser_action, run_command, file_editor
    """
    sandbox = MagicMock(spec=BaseSandbox)

    # Mock lifecycle
    sandbox.start = AsyncMock(return_value=None)
    sandbox.cleanup = AsyncMock(return_value=None)

    # Mock service lifecycle
    sandbox.start_ipython_kernel = AsyncMock(return_value=None)
    sandbox.stop_ipython_kernel = AsyncMock(return_value=None)
    sandbox.start_browser_server = AsyncMock(return_value=None)
    sandbox.stop_browser_server = AsyncMock(return_value=None)
    sandbox.start_persistent_shell = AsyncMock(return_value=None)
    sandbox.stop_persistent_shell = AsyncMock(return_value=None)

    # Mock communication
    sandbox.run_command = AsyncMock(
        return_value=CommandResult(exit_code=0, stdout="output", stderr="")
    )
    sandbox.execute_ipython = AsyncMock(
        return_value=IPythonResult(exit_code=0, text_output="output", error_output="")
    )
    sandbox.browser_action = AsyncMock(
        return_value={"status": "success", "url": "https://example.com", "title": "Example"}
    )
    sandbox.file_editor = AsyncMock(
        return_value={"status": "success"}
    )

    # Mock internal methods
    sandbox._exec_run = AsyncMock(
        return_value=CommandResult(exit_code=0, stdout="", stderr="")
    )
    sandbox._get_service_url = MagicMock(return_value="http://localhost:8888")

    return sandbox


@pytest.fixture
def docker_sandbox_config() -> dict:
    """Configuration for Docker sandbox testing."""
    return {
        "type": "docker",
        "image": "nat-sandbox:latest",
        "memory_limit": "4g",
        "cpu_limit": 4.0,
        "network_enabled": True,
        "work_dir": "/workspace",
    }


@pytest.fixture
def daytona_sandbox_config() -> dict:
    """Configuration for Daytona sandbox testing."""
    return {
        "type": "daytona",
        "api_key": "test-api-key",
        "server_url": "https://api.daytona.io",
        "target": "us",
        "image": "daytonaio/workspace:latest",
        "cpu": 4,
        "memory": 4,
        "disk": 10,
    }
