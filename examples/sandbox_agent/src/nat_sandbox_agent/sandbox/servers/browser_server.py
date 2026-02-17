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
"""Browser Server - runs INSIDE the container on port 8889.

This file is read as a string on the host side and written into the
container at startup. It is NOT imported on the host.

Provides a persistent Playwright Chromium session with action-based API.
"""

# This is the source code that gets written into the container.
# It is stored here for version control and readability.
# The actual deployment happens in docker_sandbox.py / daytona_sandbox.py
# via _exec_run("cat > /opt/browser_server.py << 'PYEOF' ... PYEOF")

BROWSER_SERVER_SOURCE = r'''
import asyncio
import os
import time as _time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from playwright.async_api import async_playwright, Browser, Page


class BrowserAction(BaseModel):
    type: str  # goto, click, fill, get_text, scroll, screenshot, select, press, wait, back, forward
    url: Optional[str] = None
    selector: Optional[str] = None
    value: Optional[str] = None
    pixels: Optional[int] = None
    key: Optional[str] = None


browser: Browser = None
page: Page = None
playwright_ctx = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global browser, page, playwright_ctx
    playwright_ctx = await async_playwright().start()
    browser = await playwright_ctx.chromium.launch(headless=True)
    ctx = await browser.new_context(viewport={"width": 1280, "height": 720})
    page = await ctx.new_page()
    os.makedirs("/workspace/output", exist_ok=True)
    yield
    await browser.close()
    await playwright_ctx.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/action")
async def execute_action(action: BrowserAction):
    global page
    text = ""
    screenshot_path = None
    try:
        match action.type:
            case "goto":
                await page.goto(action.url, wait_until="domcontentloaded", timeout=30000)
            case "click":
                await page.click(action.selector, timeout=10000)
            case "fill":
                await page.fill(action.selector, action.value or "", timeout=10000)
            case "get_text":
                sel = action.selector or "body"
                text = await page.text_content(sel) or ""
                text = text[:50000]  # limit response size
            case "scroll":
                px = action.pixels or 500
                await page.evaluate(f"window.scrollBy(0, {px})")
            case "screenshot":
                raw = await page.screenshot(type="png")
                path = f"/workspace/output/screenshot_{int(_time.time() * 1000)}.png"
                with open(path, "wb") as f:
                    f.write(raw)
                screenshot_path = path
            case "select":
                await page.select_option(action.selector, action.value, timeout=10000)
            case "press":
                await page.keyboard.press(action.key or "Enter")
            case "wait":
                await page.wait_for_selector(action.selector, timeout=10000)
            case "back":
                await page.go_back(wait_until="domcontentloaded", timeout=15000)
            case "forward":
                await page.go_forward(wait_until="domcontentloaded", timeout=15000)
            case _:
                return {"status": "error", "error": f"Unknown action: {action.type}"}

        result = {
            "status": "success",
            "url": page.url,
            "title": await page.title(),
        }
        if text:
            result["content"] = text
        if screenshot_path:
            result["screenshot_path"] = screenshot_path
        return result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "url": page.url if page else "",
        }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8889)
'''
