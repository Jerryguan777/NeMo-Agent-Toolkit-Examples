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
"""
Sandbox Agent - A general-purpose agent with persistent execution environments.

Architecture:
    Container runs 3 HTTP services communicating via httpx/websockets:
    - Port 8888: Jupyter Kernel Gateway (persistent IPython)
    - Port 8889: Browser Server (persistent Playwright session)
    - Port 8890: Shell Server (persistent bash + file editor)

Features:
    - Persistent Python execution (variables/imports preserved)
    - Persistent shell (cd/export/alias preserved)
    - Interactive web browsing (multi-step click/fill/scroll)
    - Unified file editor (view/create/write/str_replace/insert)
    - Root access for apt-get install

Usage:
    # CLI mode
    nat run configs/config.yaml

    # Evaluation mode
    nat eval configs/config_gaia.yaml
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    __version__ = version("nat_sandbox_agent")
except PackageNotFoundError:
    __version__ = "unknown"

# Import registration to enable NAT component discovery
from . import register

__all__ = ["register", "__version__"]
