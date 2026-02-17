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
"""Sandbox Agent workflow registration for NAT framework.

Architecture:
    Container runs 3 HTTP services (IPython, Browser, Shell).
    IPython + Shell start in parallel at init.
    Browser starts lazily on first use.
"""

import asyncio
import logging
import uuid
from typing import Annotated
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import Field
from typing_extensions import TypedDict

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat_sandbox_agent.prompts import get_system_prompt
from nat_sandbox_agent.sandbox.factory import create_sandbox_from_dict
from nat_sandbox_agent.tools.factory import create_all_tools
from nat_sandbox_agent.utils.answer_cleaning import clean_answer
from nat_sandbox_agent.utils.context_management import compress_messages

logger = logging.getLogger(__name__)

# ============ Configuration Classes ============


class SandboxAgentWorkflowConfig(FunctionBaseConfig, name="sandbox_agent"):
    """Configuration for the Sandbox Agent workflow."""

    llm_name: LLMRef = Field(
        ...,
        description="Reference to the LLM to use for the agent.",
    )
    max_iterations: int = Field(
        default=20,
        description="Maximum number of agent iterations before stopping.",
    )
    max_observation_tokens: int = Field(
        default=10000,
        description="Maximum tokens for tool observation results.",
    )
    sandbox_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "docker",
            "image": "nat-sandbox:latest",
            "memory_limit": "4g",
            "cpu_limit": 4.0,
            "network_enabled": True,
        },
        description="Sandbox configuration dictionary.",
    )
    enabled_tools: list[str] | None = Field(
        default=None,
        description="List of tool names to enable. If None, all tools are enabled.",
    )
    max_context_chars: int = Field(
        default=800_000,
        description="Maximum characters in conversation history before compression. "
        "Default ~200K tokens, safe for models with 256K+ context windows.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt. If None, uses the default prompt.",
    )
    additional_instructions: str | None = Field(
        default=None,
        description="Additional instructions to append to the system prompt.",
    )


# ============ Agent State ============


class AgentState(TypedDict):
    """State for the sandbox agent graph."""

    messages: Annotated[list, add_messages]
    iteration_count: int
    sandbox_id: str


# ============ Workflow Registration ============


@register_function(
    config_type=SandboxAgentWorkflowConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def sandbox_agent_workflow(config: SandboxAgentWorkflowConfig, builder: Builder):
    """Sandbox Agent workflow - Execute tasks in an isolated sandbox environment.

    This workflow creates a ReAct-style agent with persistent execution
    environments: IPython kernel (variables persist), bash shell (cd/export
    persist), and interactive browser (session persists).

    Args:
        config: Workflow configuration.
        builder: NAT builder for dependency injection.

    Yields:
        The agent response function.
    """
    logger.info("Initializing Sandbox Agent workflow")

    session_id = str(uuid.uuid4())[:8]

    # Create sandbox instance
    sandbox = create_sandbox_from_dict(config.sandbox_config)
    logger.info(f"Created sandbox with config: {config.sandbox_config.get('type', 'docker')}")

    try:
        # Start the sandbox container/workspace
        await sandbox.start()
        logger.info(f"Sandbox started: {session_id}")

        # Start IPython + Shell in parallel (browser is lazy)
        await asyncio.gather(
            sandbox.start_ipython_kernel(),
            sandbox.start_persistent_shell(),
        )
        logger.info("IPython kernel and Shell server ready")

        # Create tools bound to this sandbox
        max_output_chars = config.max_observation_tokens * 4
        tools = create_all_tools(
            sandbox=sandbox,
            include_tools=config.enabled_tools,
            max_output_chars=max_output_chars,
        )
        logger.info(f"Created {len(tools)} tools: {[t.name for t in tools]}")

        # Get LLM from builder
        llm = await builder.get_llm(
            config.llm_name,
            wrapper_type=LLMFrameworkEnum.LANGCHAIN,
        )
        llm_with_tools = llm.bind_tools(tools)

        # Build system prompt
        system_prompt = config.system_prompt or get_system_prompt(
            additional_instructions=config.additional_instructions,
            available_tools=[t.name for t in tools] if config.enabled_tools else None,
        )

        # ============ Define Agent Nodes ============
        async def agent_node(state: AgentState) -> dict:
            """Agent reasoning node."""
            messages = state["messages"]
            iteration = state.get("iteration_count", 0)

            if iteration >= config.max_iterations:
                logger.warning(f"Reached max iterations ({config.max_iterations})")
                return {
                    "messages": [
                        AIMessage(
                            content="I've reached the maximum number of steps. "
                            "Here's a summary of what was accomplished so far."
                        )
                    ],
                    "iteration_count": iteration,
                }

            # Compress message history to prevent context overflow
            compressed = compress_messages(
                list(messages),
                max_chars=config.max_context_chars,
            )

            full_messages = [SystemMessage(content=system_prompt)] + compressed
            response = await llm_with_tools.ainvoke(full_messages)

            return {
                "messages": [response],
                "iteration_count": iteration + 1,
            }

        def should_continue(state: AgentState) -> str:
            """Decide whether to continue to tools or end."""
            messages = state["messages"]
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END

        # ============ Build Graph ============
        tool_node = ToolNode(tools)
        graph = StateGraph(AgentState)

        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", END: END},
        )
        graph.add_edge("tools", "agent")

        agent_graph = graph.compile()
        recursion_limit = config.max_iterations * 2 + 10

        logger.info("Agent graph compiled successfully")

        # ============ Response Function ============
        async def _response_fn(input_message: str) -> str:
            """Process user input and return agent response."""
            logger.info(f"Processing input: {input_message[:100]}...")

            initial_state = {
                "messages": [HumanMessage(content=input_message)],
                "iteration_count": 0,
                "sandbox_id": session_id,
            }

            try:
                result = await agent_graph.ainvoke(
                    initial_state,
                    config={"recursion_limit": recursion_limit},
                )

                final_message = result["messages"][-1]
                if hasattr(final_message, "content"):
                    raw_response = final_message.content
                else:
                    raw_response = str(final_message)

                cleaned_response = clean_answer(raw_response)
                logger.debug(f"Raw response: {raw_response[:100]}...")
                logger.debug(f"Cleaned response: {cleaned_response}")

                return cleaned_response

            except Exception as e:
                logger.error(f"Error during agent execution: {e}")
                return f"An error occurred: {str(e)}"

        yield _response_fn

    except Exception as e:
        logger.error(f"Error in sandbox agent workflow: {e}")
        raise

    finally:
        # Cleanup all services + sandbox
        logger.info(f"Cleaning up sandbox: {session_id}")
        try:
            await sandbox.stop_ipython_kernel()
        except Exception as e:
            logger.error(f"Error stopping IPython: {e}")
        try:
            await sandbox.stop_browser_server()
        except Exception as e:
            logger.error(f"Error stopping browser: {e}")
        try:
            await sandbox.stop_persistent_shell()
        except Exception as e:
            logger.error(f"Error stopping shell: {e}")
        try:
            await sandbox.cleanup()
        except Exception as e:
            logger.error(f"Error during sandbox cleanup: {e}")
