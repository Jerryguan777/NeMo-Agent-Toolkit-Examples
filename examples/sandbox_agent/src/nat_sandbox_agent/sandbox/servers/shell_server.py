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
"""Shell Server - runs INSIDE the container on port 8890.

This file is read as a string on the host side and written into the
container at startup. It is NOT imported on the host.

Provides:
- /execute: Persistent bash shell (cd, export, alias preserved)
- /file_editor: Unified file operations (view, create, write, str_replace, insert)
"""

SHELL_SERVER_SOURCE = r'''
import asyncio
import os
import subprocess
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel


class ShellCommand(BaseModel):
    command: str
    timeout: float = 120


class FileEditorRequest(BaseModel):
    command: str  # view, create, write, str_replace, insert
    path: str
    file_text: Optional[str] = None
    old_str: Optional[str] = None
    new_str: Optional[str] = None
    insert_line: Optional[int] = None
    view_range: Optional[list[int]] = None


bash_process: subprocess.Popen = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bash_process
    bash_process = subprocess.Popen(
        ["/bin/bash", "--norc", "--noprofile"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
        env={**os.environ, "PS1": ""},
    )
    yield
    if bash_process and bash_process.poll() is None:
        bash_process.terminate()
        try:
            bash_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            bash_process.kill()


app = FastAPI(lifespan=lifespan)


@app.post("/execute")
async def execute(cmd: ShellCommand):
    global bash_process

    # Restart bash if it died
    if bash_process is None or bash_process.poll() is not None:
        bash_process = subprocess.Popen(
            ["/bin/bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            env={**os.environ, "PS1": ""},
        )

    delimiter = f"__END_{uuid.uuid4().hex[:8]}__"

    # Write command + delimiter that captures exit code and cwd
    full_cmd = f"{cmd.command}\n_EC=$?\necho {delimiter} $_EC $(pwd)\necho {delimiter} >&2\n"

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_in_bash, full_cmd, delimiter, cmd.timeout),
            timeout=cmd.timeout + 5,
        )
        return result
    except asyncio.TimeoutError:
        # Kill the hung bash and start a new one
        if bash_process and bash_process.poll() is None:
            bash_process.kill()
            bash_process.wait()
        bash_process = subprocess.Popen(
            ["/bin/bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            env={**os.environ, "PS1": ""},
        )
        return {"exit_code": -1, "stdout": "", "stderr": f"Command timed out after {cmd.timeout}s"}


def _run_in_bash(full_cmd: str, delimiter: str, timeout: float) -> dict:
    """Run command in persistent bash, parse output until delimiter."""
    try:
        bash_process.stdin.write(full_cmd)
        bash_process.stdin.flush()
    except (BrokenPipeError, OSError):
        return {"exit_code": -1, "stdout": "", "stderr": "Bash process died"}

    stdout_lines = []
    stderr_lines = []

    import select
    import time

    deadline = time.monotonic() + timeout
    stdout_fd = bash_process.stdout.fileno()
    stderr_fd = bash_process.stderr.fileno()
    stdout_done = False
    stderr_done = False

    # Set non-blocking
    import fcntl
    flags_out = fcntl.fcntl(stdout_fd, fcntl.F_GETFL)
    fcntl.fcntl(stdout_fd, fcntl.F_SETFL, flags_out | os.O_NONBLOCK)
    flags_err = fcntl.fcntl(stderr_fd, fcntl.F_GETFL)
    fcntl.fcntl(stderr_fd, fcntl.F_SETFL, flags_err | os.O_NONBLOCK)

    exit_code = 0
    cwd = ""

    try:
        while not (stdout_done and stderr_done):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"exit_code": -1, "stdout": "\n".join(stdout_lines), "stderr": "Timeout"}

            readable, _, _ = select.select(
                [fd for fd, done in [(stdout_fd, stdout_done), (stderr_fd, stderr_done)] if not done],
                [], [], min(remaining, 0.1),
            )

            if stdout_fd in readable:
                try:
                    data = os.read(stdout_fd, 65536).decode("utf-8", errors="replace")
                    if data:
                        for line in data.split("\n"):
                            if delimiter in line:
                                # Parse: __END_xxxx__ <exit_code> <cwd>
                                parts = line.split(delimiter, 1)[1].strip().split(None, 1)
                                if parts:
                                    try:
                                        exit_code = int(parts[0])
                                    except ValueError:
                                        pass
                                    if len(parts) > 1:
                                        cwd = parts[1]
                                stdout_done = True
                                break
                            else:
                                stdout_lines.append(line)
                except (BlockingIOError, OSError):
                    pass

            if stderr_fd in readable:
                try:
                    data = os.read(stderr_fd, 65536).decode("utf-8", errors="replace")
                    if data:
                        for line in data.split("\n"):
                            if delimiter in line:
                                stderr_done = True
                                break
                            else:
                                stderr_lines.append(line)
                except (BlockingIOError, OSError):
                    pass

            # If stdout is done but stderr hasn't seen delimiter yet, wait briefly
            if stdout_done and not stderr_done:
                remaining_brief = min(deadline - time.monotonic(), 0.5)
                if remaining_brief <= 0:
                    stderr_done = True
                    break
                readable2, _, _ = select.select([stderr_fd], [], [], remaining_brief)
                if stderr_fd in readable2:
                    try:
                        data = os.read(stderr_fd, 65536).decode("utf-8", errors="replace")
                        if data:
                            for line in data.split("\n"):
                                if delimiter in line:
                                    stderr_done = True
                                    break
                                else:
                                    stderr_lines.append(line)
                    except (BlockingIOError, OSError):
                        pass
                stderr_done = True
    finally:
        # Restore blocking mode
        fcntl.fcntl(stdout_fd, fcntl.F_SETFL, flags_out)
        fcntl.fcntl(stderr_fd, fcntl.F_SETFL, flags_err)

    # Clean up trailing empty lines
    while stdout_lines and not stdout_lines[-1].strip():
        stdout_lines.pop()

    while stderr_lines and not stderr_lines[-1].strip():
        stderr_lines.pop()

    return {
        "exit_code": exit_code,
        "stdout": "\n".join(stdout_lines),
        "stderr": "\n".join(stderr_lines),
        "cwd": cwd,
    }


@app.post("/file_editor")
async def file_editor(req: FileEditorRequest):
    """Unified file operations - content via JSON, no shell escaping needed."""
    try:
        match req.command:
            case "view":
                return _file_view(req.path, req.view_range)
            case "create":
                return _file_create(req.path, req.file_text or "")
            case "write":
                return _file_write(req.path, req.file_text or "")
            case "str_replace":
                return _file_str_replace(req.path, req.old_str, req.new_str)
            case "insert":
                return _file_insert(req.path, req.insert_line, req.new_str or "")
            case _:
                return {"status": "error", "error": f"Unknown command: {req.command}"}
    except FileNotFoundError:
        return {"status": "error", "error": f"File not found: {req.path}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _file_view(path: str, view_range: list[int] | None) -> dict:
    with open(path, "r") as f:
        lines = f.readlines()

    if view_range and len(view_range) == 2:
        start, end = view_range
        # 1-indexed, inclusive
        start = max(1, start) - 1
        end = min(len(lines), end)
        selected = lines[start:end]
        numbered = "".join(f"{i + start + 1}\t{line}" for i, line in enumerate(selected))
    else:
        numbered = "".join(f"{i + 1}\t{line}" for i, line in enumerate(lines))

    return {"status": "success", "content": numbered, "total_lines": len(lines)}


def _file_create(path: str, content: str) -> dict:
    if os.path.exists(path):
        return {"status": "error", "error": f"File already exists: {path}. Use 'write' to overwrite."}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return {"status": "success", "path": path, "size": len(content)}


def _file_write(path: str, content: str) -> dict:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return {"status": "success", "path": path, "size": len(content)}


def _file_str_replace(path: str, old_str: str | None, new_str: str | None) -> dict:
    if old_str is None:
        return {"status": "error", "error": "old_str is required for str_replace"}
    if new_str is None:
        new_str = ""

    with open(path, "r") as f:
        content = f.read()

    count = content.count(old_str)
    if count == 0:
        return {"status": "error", "error": f"old_str not found in {path}"}
    if count > 1:
        return {"status": "error", "error": f"old_str found {count} times in {path}. Must be unique."}

    new_content = content.replace(old_str, new_str, 1)
    with open(path, "w") as f:
        f.write(new_content)

    return {"status": "success", "path": path, "replacements": 1}


def _file_insert(path: str, insert_line: int | None, new_str: str) -> dict:
    if insert_line is None:
        return {"status": "error", "error": "insert_line is required for insert"}

    with open(path, "r") as f:
        lines = f.readlines()

    if insert_line < 0 or insert_line > len(lines):
        return {"status": "error", "error": f"insert_line {insert_line} out of range (0-{len(lines)})"}

    # Insert after the specified line (0 = insert at beginning)
    new_lines = new_str if new_str.endswith("\n") else new_str + "\n"
    lines.insert(insert_line, new_lines)

    with open(path, "w") as f:
        f.writelines(lines)

    return {"status": "success", "path": path, "total_lines": len(lines)}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8890)
'''
