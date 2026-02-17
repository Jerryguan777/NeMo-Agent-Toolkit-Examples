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
"""Tests for conversation history compression."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nat_sandbox_agent.utils.context_management import compress_messages


class TestCompressMessages:
    """Tests for compress_messages function."""

    def test_no_compression_when_under_limit(self):
        """Messages under the limit should be returned unchanged."""
        messages = [
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
        ]
        result = compress_messages(messages, max_chars=1000)
        assert result == messages

    def test_truncates_old_tool_messages(self):
        """Old ToolMessage content should be truncated when over limit."""
        big_content = "X" * 5000
        messages = [
            HumanMessage(content="task"),
            ToolMessage(content=big_content, tool_call_id="tc1"),
            ToolMessage(content=big_content, tool_call_id="tc2"),
            AIMessage(content="thinking"),
            AIMessage(content="step"),
            HumanMessage(content="cont"),
            AIMessage(content="done"),
            ToolMessage(content="small", tool_call_id="tc3"),
            AIMessage(content="final"),
        ]
        # Set max_chars so it triggers compression, keep_recent=4
        result = compress_messages(messages, max_chars=2000, keep_recent=4, truncate_chars=200)

        # Recent 4 messages should be unchanged
        assert result[-4:] == messages[-4:]

        # Old ToolMessages should be truncated
        for msg in result[:-4]:
            if isinstance(msg, ToolMessage) and msg.tool_call_id in ("tc1", "tc2"):
                assert "compressed" in str(msg.content)
                assert len(str(msg.content)) < 5000

    def test_drops_old_messages_when_still_over_limit(self):
        """When truncation isn't enough, old messages should be dropped."""
        huge_content = "Y" * 10000
        messages = [
            HumanMessage(content="original task"),
            ToolMessage(content=huge_content, tool_call_id="tc1"),
            AIMessage(content=huge_content),
            ToolMessage(content=huge_content, tool_call_id="tc2"),
            AIMessage(content="recent1"),
            ToolMessage(content="recent tool", tool_call_id="tc3"),
            AIMessage(content="recent2"),
        ]
        result = compress_messages(messages, max_chars=500, keep_recent=3, truncate_chars=100)

        # First message (HumanMessage) should always be preserved
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "original task"

        # Recent 3 messages should be intact
        assert result[-3:] == messages[-3:]

    def test_keeps_first_human_message(self):
        """The first HumanMessage should never be dropped."""
        messages = [
            HumanMessage(content="important task description"),
            AIMessage(content="A" * 10000),
            ToolMessage(content="B" * 10000, tool_call_id="tc1"),
            AIMessage(content="recent"),
        ]
        result = compress_messages(messages, max_chars=100, keep_recent=1, truncate_chars=50)

        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "important task description"
        assert result[-1] == messages[-1]

    def test_keep_recent_preserves_exact_count(self):
        """keep_recent parameter should preserve exactly N recent messages."""
        messages = [
            HumanMessage(content="task"),
            ToolMessage(content="X" * 5000, tool_call_id="tc1"),
            AIMessage(content="old"),
            AIMessage(content="r1"),
            AIMessage(content="r2"),
            AIMessage(content="r3"),
        ]
        result = compress_messages(messages, max_chars=100, keep_recent=3, truncate_chars=50)

        # Last 3 should be untouched
        assert result[-3:] == messages[-3:]

    def test_tool_call_id_preserved_after_truncation(self):
        """ToolMessage tool_call_id must be preserved after truncation."""
        messages = [
            HumanMessage(content="task"),
            ToolMessage(content="Z" * 5000, tool_call_id="unique_id_123"),
            AIMessage(content="recent1"),
            AIMessage(content="recent2"),
        ]
        result = compress_messages(messages, max_chars=500, keep_recent=2, truncate_chars=100)

        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "unique_id_123"

    def test_empty_messages(self):
        """Empty message list should return empty."""
        result = compress_messages([], max_chars=1000)
        assert result == []

    def test_single_message(self):
        """Single message should be returned unchanged."""
        messages = [HumanMessage(content="hello")]
        result = compress_messages(messages, max_chars=1000)
        assert result == messages
