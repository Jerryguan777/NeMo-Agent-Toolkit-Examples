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
"""Conversation history compression to prevent context overflow."""

from langchain_core.messages import ToolMessage


def compress_messages(
    messages: list,
    max_chars: int = 800_000,
    keep_recent: int = 6,
    truncate_chars: int = 1000,
) -> list:
    """Progressively compress message history to stay within context limits.

    Phase 1: Truncate old ToolMessage content (preserve head + tail).
    Phase 2: If still over limit, drop oldest messages (keep first HumanMessage + recent).

    Args:
        messages: List of LangChain message objects.
        max_chars: Maximum total characters before compression triggers.
        keep_recent: Number of recent messages to always preserve intact.
        truncate_chars: Max chars for old ToolMessage content after truncation.

    Returns:
        Compressed message list.
    """
    total_chars = sum(len(str(m.content)) for m in messages)
    if total_chars <= max_chars:
        return messages

    # Phase 1: Truncate old ToolMessage outputs
    compressed = []
    cutoff = len(messages) - keep_recent
    for i, msg in enumerate(messages):
        if i >= cutoff:
            compressed.append(msg)  # Keep recent messages intact
        elif isinstance(msg, ToolMessage):
            content = str(msg.content)
            if len(content) > truncate_chars:
                half = truncate_chars // 2
                head = content[:half]
                tail = content[-half:]
                summary = f"{head}\n...[compressed {len(content)} chars]...\n{tail}"
                compressed.append(
                    ToolMessage(content=summary, tool_call_id=msg.tool_call_id)
                )
            else:
                compressed.append(msg)
        else:
            compressed.append(msg)

    total_chars = sum(len(str(m.content)) for m in compressed)
    if total_chars <= max_chars:
        return compressed

    # Phase 2: Drop oldest messages (keep first HumanMessage + recent)
    while total_chars > max_chars and len(compressed) > keep_recent + 1:
        removed = compressed.pop(1)  # Index 0 is HumanMessage, start from 1
        total_chars -= len(str(removed.content))

    return compressed
