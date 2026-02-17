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
"""Tests for Daytona sandbox implementation using mocks."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat_sandbox_agent.sandbox.daytona_sandbox import DaytonaSandbox


class TestDaytonaSandboxInit:
    """Tests for DaytonaSandbox initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        sandbox = DaytonaSandbox(api_key="test-key")

        assert sandbox._api_key == "test-key"
        assert sandbox._server_url == "https://api.daytona.io"
        assert sandbox._target == "us"
        assert sandbox._image == "daytonaio/workspace:latest"
        assert sandbox._cpu == 4
        assert sandbox._memory == 4
        assert sandbox._disk == 10
        assert sandbox._auto_stop_interval == 30

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        sandbox = DaytonaSandbox(
            api_key="custom-key",
            server_url="https://custom.daytona.io",
            target="eu",
            image="custom/image:latest",
            cpu=8,
            memory=16,
            disk=20,
            auto_stop_interval=60,
        )

        assert sandbox._api_key == "custom-key"
        assert sandbox._server_url == "https://custom.daytona.io"
        assert sandbox._target == "eu"
        assert sandbox._image == "custom/image:latest"
        assert sandbox._cpu == 8
        assert sandbox._memory == 16
        assert sandbox._disk == 20
        assert sandbox._auto_stop_interval == 60

    def test_client_is_lazy_loaded(self):
        """Test that client is None until first use."""
        sandbox = DaytonaSandbox(api_key="test-key")

        assert sandbox._client is None
        assert sandbox._sandbox is None


class TestDaytonaSandboxClient:
    """Tests for Daytona client initialization."""

    def test_get_client_import_error(self):
        """Test error when daytona_sdk is not installed."""
        sandbox = DaytonaSandbox(api_key="test-key")

        with patch.dict("sys.modules", {"daytona_sdk": None}):
            with pytest.raises(ImportError, match="daytona-sdk"):
                sandbox._get_client()

    def test_get_client_creates_client(self):
        """Test that _get_client creates a Daytona client."""
        sandbox = DaytonaSandbox(api_key="test-key")

        mock_daytona = MagicMock()
        mock_config = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "daytona_sdk": MagicMock(
                    Daytona=mock_daytona,
                    DaytonaConfig=mock_config,
                ),
            },
        ):
            sandbox._get_client()

            mock_config.assert_called_once_with(api_key="test-key")
            assert sandbox._client is not None


class TestDaytonaSandboxLifecycle:
    """Tests for Daytona sandbox lifecycle using mocks."""

    @pytest.fixture
    def mock_daytona_sandbox(self):
        """Create a DaytonaSandbox with mocked SDK."""
        sandbox = DaytonaSandbox(api_key="test-key")

        # Mock the sandbox instance
        mock_instance = MagicMock()
        mock_instance.id = "mock-sandbox-id"

        # Mock client
        mock_client = MagicMock()
        mock_client.create.return_value = mock_instance

        sandbox._client = mock_client
        sandbox._sandbox = mock_instance

        return sandbox

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_daytona_sandbox):
        """Test sandbox cleanup."""
        mock_delete = MagicMock()
        mock_daytona_sandbox._sandbox.delete = mock_delete

        await mock_daytona_sandbox.cleanup()

        mock_delete.assert_called_once()
        assert mock_daytona_sandbox._sandbox is None


class TestDaytonaSandboxBootstrap:
    """Tests for bootstrap command execution in Daytona sandbox."""

    @pytest.fixture
    def running_sandbox(self):
        """Create a running Daytona sandbox with mocked SDK."""
        sandbox = DaytonaSandbox(api_key="test-key")

        mock_instance = MagicMock()
        mock_instance.id = "test-id"

        # Mock process.exec
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.result = "output"
        mock_instance.process.exec.return_value = mock_result

        sandbox._sandbox = mock_instance
        return sandbox

    @pytest.mark.asyncio
    async def test_exec_run_success(self, running_sandbox):
        """Test successful bootstrap command execution."""
        result = await running_sandbox._exec_run("echo hello")

        assert result.exit_code == 0
        assert result.stdout == "output"
        running_sandbox._sandbox.process.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_exec_run_not_started(self):
        """Test bootstrap command fails if sandbox not started."""
        sandbox = DaytonaSandbox(api_key="test-key")

        with pytest.raises(RuntimeError, match="not started"):
            await sandbox._exec_run("echo hello")

    @pytest.mark.asyncio
    async def test_exec_run_failure(self, running_sandbox):
        """Test bootstrap command execution failure."""
        running_sandbox._sandbox.process.exec.side_effect = Exception("Command failed")

        result = await running_sandbox._exec_run("invalid_command")

        assert result.exit_code == -1
        assert "Command failed" in result.stderr


class TestDaytonaSandboxURLConstruction:
    """Tests for service URL construction."""

    def test_get_service_url_not_started(self):
        """Test URL construction fails if sandbox not started."""
        sandbox = DaytonaSandbox(api_key="test-key")

        with pytest.raises(RuntimeError, match="not started"):
            sandbox._get_service_url(8888)

    def test_get_service_url(self):
        """Test URL construction via preview link."""
        sandbox = DaytonaSandbox(api_key="test-key")

        mock_instance = MagicMock()
        mock_preview = MagicMock()
        mock_preview.url = "https://sandbox-8888.preview.daytona.io"
        mock_instance.get_preview_link.return_value = mock_preview
        sandbox._sandbox = mock_instance

        url = sandbox._get_service_url(8888)

        assert url == "https://sandbox-8888.preview.daytona.io"
        mock_instance.get_preview_link.assert_called_once_with(8888)
