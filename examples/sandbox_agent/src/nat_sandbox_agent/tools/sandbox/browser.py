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
"""Interactive browser tool - action-based web interaction.

Supports multi-step browsing: goto -> click -> fill -> get_text -> scroll etc.
Browser session persists across calls (lazy-initialized on first use).
"""

import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import Field

from nat_sandbox_agent.tools.sandbox.executor import SandboxToolExecutor

logger = logging.getLogger(__name__)


class BrowserActionInput(BaseModel):
    """Input schema for browser actions."""

    action: str = Field(
        description=(
            "Browser action to perform. One of: "
            "goto, click, fill, get_text, scroll, screenshot, "
            "select, press, wait, back, forward"
        ),
    )
    url: str | None = Field(
        default=None,
        description="URL for 'goto' action.",
    )
    selector: str | None = Field(
        default=None,
        description="CSS selector for click/fill/get_text/select/wait actions.",
    )
    value: str | None = Field(
        default=None,
        description="Value for 'fill' or 'select' actions.",
    )
    pixels: int | None = Field(
        default=None,
        description="Pixels to scroll for 'scroll' action (default: 500).",
    )
    key: str | None = Field(
        default=None,
        description="Key for 'press' action (e.g., 'Enter', 'Tab').",
    )


async def browser_action(
    executor: SandboxToolExecutor,
    action: str,
    url: str | None = None,
    selector: str | None = None,
    value: str | None = None,
    pixels: int | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    """Execute a browser action.

    Browser is lazily initialized on first call. Session persists.
    """
    logger.info(f"Browser action: {action}")

    payload = {"type": action}
    if url is not None:
        payload["url"] = url
    if selector is not None:
        payload["selector"] = selector
    if value is not None:
        payload["value"] = value
    if pixels is not None:
        payload["pixels"] = pixels
    if key is not None:
        payload["key"] = key

    result = await executor.sandbox.browser_action(payload)

    # Strip any raw base64 screenshot data (defense in depth)
    if "screenshot" in result:
        del result["screenshot"]

    # Provide actionable guidance for screenshot results
    if result.get("screenshot_path"):
        result["screenshot_info"] = (
            f"Screenshot saved to {result['screenshot_path']}. "
            "Use python with PIL/pytesseract to analyze it."
        )

    # Truncate content if present
    if "content" in result and result["content"]:
        result["content"] = executor.truncate(result["content"])

    return result


def create_browser_tool(executor: SandboxToolExecutor) -> StructuredTool:
    """Create the interactive browser tool."""
    return StructuredTool.from_function(
        coroutine=lambda action, url=None, selector=None, value=None, pixels=None, key=None: (
            browser_action(executor, action, url, selector, value, pixels, key)
        ),
        name="browser",
        description=(
            "Interactive web browser with persistent session. "
            "Actions: goto (navigate to URL), click (CSS selector), "
            "fill (type into input), get_text (extract page text), "
            "scroll (scroll page), screenshot (capture page), "
            "select (dropdown), press (keyboard key), "
            "wait (wait for element), back, forward. "
            "Session persists across calls - use multi-step browsing."
        ),
        args_schema=BrowserActionInput,
    )
