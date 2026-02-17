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
"""Common utilities and constants for tools."""

# Default limit: 16,000 chars ~ 4,000 tokens
# This helps prevent context overflow when multiple tools are called
DEFAULT_MAX_OUTPUT_CHARS = 16000


def truncate_output(text: str, max_chars: int = DEFAULT_MAX_OUTPUT_CHARS) -> str:
    """Truncate output preserving head and tail.

    Keeps the beginning and end of the text, inserting a truncation
    marker in the middle. Error messages are often at the end, so
    tail-only truncation would lose them.

    Args:
        text: Text to truncate.
        max_chars: Maximum characters to keep.

    Returns:
        Truncated text with indicator if truncation occurred.
    """
    if len(text) <= max_chars:
        return text

    # Keep 60% head, 40% tail (reserve ~80 chars for the marker)
    marker_budget = 80
    head_chars = int(max_chars * 0.6)
    tail_chars = max_chars - head_chars - marker_budget

    if tail_chars <= 0:
        # max_chars too small for head+tail+marker — head-only truncation
        truncated = len(text) - head_chars
        marker = f"\n\n... [truncated {truncated} chars] ...\n\n"
        return text[:head_chars] + marker

    truncated = len(text) - head_chars - tail_chars
    marker = f"\n\n... [truncated {truncated} chars] ...\n\n"
    return text[:head_chars] + marker + text[-tail_chars:]
