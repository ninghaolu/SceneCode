"""Exhausted planner tools remain callable without invoking expensive agents."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents import Agent, function_tool
from agents.tool_context import ToolContext

from scenecode.agent_utils.base_stateful_agent import BaseStatefulAgent


def make_tools(rounds):
    @function_tool
    async def reset_scene_to_checkpoint() -> str:
        """Reset the design checkpoint."""
        return "reset"

    owner = SimpleNamespace(
        cfg=SimpleNamespace(max_critique_rounds=rounds),
        _is_placement_agent=False,
        _request_initial_design_impl=AsyncMock(return_value="initial"),
        _request_critique_impl=AsyncMock(return_value="critique"),
        _request_design_change_impl=AsyncMock(return_value="changed"),
        _create_reset_checkpoint_tool=lambda: reset_scene_to_checkpoint,
    )
    return owner, BaseStatefulAgent._create_planner_tools(owner)


@pytest.mark.parametrize("name,args,implementation", [
    ("request_critique", "{}", "_request_critique_impl"),
    ("request_design_change", '{"instruction":"fix support"}',
     "_request_design_change_impl"),
])
def test_exhausted_tool_stays_available_and_does_not_run_agent(name, args, implementation):
    async def check():
        owner, tools = make_tools(2)
        agent = Agent(name="planner", tools=tools)
        context = ToolContext(context=None, tool_name=name, tool_call_id="test", tool_arguments=args)
        tool = next(t for t in tools if t.name == name)
        for _ in range(2):
            await tool.on_invoke_tool(context, args)
        # This SDK filtering removed the tool and caused job 19106372 to abort.
        available = {t.name: t for t in await agent.get_all_tools(context)}
        assert name in available
        for _ in range(2):
            result = await available[name].on_invoke_tool(context, args)
            assert "budget exhausted" in result
        assert getattr(owner, implementation).await_count == 2
        other = "_request_design_change_impl" if name == "request_critique" else "_request_critique_impl"
        getattr(owner, other).assert_not_awaited()
    asyncio.run(check())


def test_zero_rounds_still_omits_critique_tools():
    _, tools = make_tools(0)
    assert [t.name for t in tools] == ["request_initial_design"]


def test_new_workflow_has_fresh_budget():
    async def check():
        owner, tools = make_tools(1)
        context = ToolContext(context=None, tool_name="request_critique", tool_call_id="test", tool_arguments="{}")
        tool = next(t for t in tools if t.name == "request_critique")
        await tool.on_invoke_tool(context, "{}")
        assert "budget exhausted" in await tool.on_invoke_tool(context, "{}")
        fresh = BaseStatefulAgent._create_planner_tools(owner)
        fresh_tool = next(t for t in fresh if t.name == "request_critique")
        assert await fresh_tool.on_invoke_tool(context, "{}") == "critique"
        assert owner._request_critique_impl.await_count == 2
    asyncio.run(check())
