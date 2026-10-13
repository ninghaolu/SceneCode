"""Regression tests for the interrupted Qwen room's orchestration failures."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from agents import Agent, Model, ModelBehaviorError, ModelResponse, RunConfig, SQLiteSession, Usage, function_tool
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from pydantic import BaseModel

from scenecode.agent_utils.bounded_critic import run_bounded_critic
from scenecode.agent_utils.saved_images import SavedImageGenerator


class Verdict(BaseModel):
    passed: bool
    reason: str


@pytest.mark.asyncio
async def test_looping_critic_finishes_with_existing_tool_evidence(tmp_path):
    @function_tool
    def observe_scene() -> str:
        return "sofa_back_missing"

    model = MagicMock(spec=Model)
    calls = []

    async def respond(**kwargs):
        calls.append(kwargs)
        if kwargs['tools']:
            output = [ResponseFunctionToolCall(type="function_call", name="observe_scene",
                      call_id=f"call_{len(calls)}", arguments="{}")]
        else:
            assert kwargs['model_settings'].tool_choice == 'none'
            assert 'sofa_back_missing' in json.dumps(kwargs['input'])
            output = [ResponseOutputMessage(id="verdict", type="message", role="assistant", status="completed",
                      content=[ResponseOutputText(type="output_text", annotations=[],
                               text='{"passed": false, "reason": "sofa_back_missing"}')])]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    model.get_response = AsyncMock(side_effect=respond)
    agent = Agent(name="critic", model=model, tools=[observe_scene], output_type=Verdict)
    session = SQLiteSession("test", str(tmp_path / "session.db"))
    result = await run_bounded_critic(agent=agent, instruction="Review", session=session,
                                      max_turns=2, run_config=RunConfig(tracing_disabled=True))
    assert len(calls) == 3
    assert result.final_output.passed is False
    assert len(agent.tools) == 1  # A later critique still has observation tools.
    session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("valid_finalizer", [True, False])
async def test_malformed_critic_uses_only_one_finalization_and_keeps_defects(tmp_path, valid_finalizer):
    @function_tool
    def observe_scene() -> str:
        return "sofa_back_missing"

    model = MagicMock(spec=Model)
    calls = []

    async def respond(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            output = [ResponseFunctionToolCall(type="function_call", name="observe_scene",
                      call_id="observation", arguments="{}")]
        else:
            if len(calls) == 3:
                assert not kwargs['tools']
                assert kwargs['model_settings'].tool_choice == 'none'
                assert 'sofa_back_missing' in json.dumps(kwargs['input'])
                assert 'Preserve its reported defects' in json.dumps(kwargs['input'])
            text = ('{"passed": false, "reason": "sofa_back_missing"}'
                    if len(calls) == 3 and valid_finalizer else '## Defects: sofa_back_missing')
            output = [ResponseOutputMessage(id=f"verdict_{len(calls)}", type="message",
                      role="assistant", status="completed", content=[ResponseOutputText(
                      type="output_text", annotations=[], text=text)])]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    model.get_response = AsyncMock(side_effect=respond)
    agent = Agent(name="critic", model=model, tools=[observe_scene], output_type=Verdict)
    session = SQLiteSession("malformed", str(tmp_path / "session.db"))
    try:
        kwargs = dict(agent=agent, instruction="Review", session=session,
                      max_turns=2, run_config=RunConfig(tracing_disabled=True))
        if valid_finalizer:
            result = await run_bounded_critic(**kwargs)
            assert result.final_output == Verdict(passed=False, reason="sofa_back_missing")
        else:
            with pytest.raises(ModelBehaviorError, match="Invalid JSON"):
                await run_bounded_critic(**kwargs)
        assert len(calls) == 3
        assert len(agent.tools) == 1
    finally:
        session.close()


@pytest.mark.asyncio
async def test_unrelated_model_behavior_error_is_not_retried(monkeypatch):
    run = AsyncMock(side_effect=ModelBehaviorError("Tool missing from agent"))
    monkeypatch.setattr("scenecode.agent_utils.bounded_critic.Runner.run", run)
    with pytest.raises(ModelBehaviorError, match="Tool missing"):
        await run_bounded_critic(agent=Agent(name="critic", output_type=Verdict),
                                 instruction="Review", session=None, max_turns=2,
                                 run_config=RunConfig(tracing_disabled=True))
    assert run.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('acceptable, expected_pass', [(False, True), (True, False)])
async def test_preflight_format_finalization_keeps_measured_gate(monkeypatch, acceptable, expected_pass):
    from scripts.qwen_preflight import sdk_tool_roundtrip
    monkeypatch.setattr('scenecode.utils.openai.create_run_config',
                        lambda: RunConfig(tracing_disabled=True))
    calls = []
    model = MagicMock(spec=Model)

    async def respond(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            output = [ResponseFunctionToolCall(type='function_call', name='observe_scene',
                      call_id='measured', arguments='{}')]
        else:
            if len(calls) == 3:
                assert not kwargs['tools']
                assert '0.04' in json.dumps(kwargs['input'])
                text = json.dumps({'category': 'table', 'acceptable': acceptable, 'gap_m': .04})
            else:
                text = '```json\n{"category":"table","acceptable":false,"gap_m":0.04}\n```\nThe gap is too large.'
            output = [ResponseOutputMessage(id=f'verdict_{len(calls)}', type='message',
                      role='assistant', status='completed', content=[ResponseOutputText(
                      type='output_text', annotations=[], text=text)])]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    model.get_response = AsyncMock(side_effect=respond)
    result = await sdk_tool_roundtrip(model)
    assert result['passed'] is expected_pass
    assert result['tool_calls'] == ['observe_scene']
    assert len(calls) == 3  # Within the unchanged four-call preflight ceiling.


@pytest.mark.asyncio
@pytest.mark.parametrize('budget', [1, 8])
async def test_floor_critic_reserves_finalization_without_increasing_budget(monkeypatch, budget):
    from scenecode.floor_plan_agents.stateful_floor_plan_agent import StatefulFloorPlanAgent

    class RunnerReached(Exception):
        pass

    bounded = AsyncMock(side_effect=RunnerReached)
    direct = AsyncMock(side_effect=RunnerReached)
    monkeypatch.setattr('scenecode.agent_utils.bounded_critic.run_bounded_critic', bounded)
    monkeypatch.setattr('scenecode.floor_plan_agents.stateful_floor_plan_agent.Runner.run', direct)
    fake = SimpleNamespace(
        prompt_registry=SimpleNamespace(get_prompt=lambda **kwargs: 'Review shell'),
        house_prompt='One room with four specified items', critic=object(), critic_session=object(),
        cfg=SimpleNamespace(agents=SimpleNamespace(critic_agent=SimpleNamespace(max_turns=budget))),
        _create_run_config=lambda: RunConfig(tracing_disabled=True),
    )
    with pytest.raises(RunnerReached):
        await StatefulFloorPlanAgent._request_critique_impl(fake)
    called = bounded if budget > 1 else direct
    assert called.await_count == 1
    assert called.call_args.kwargs['max_turns'] == (budget - 1 if budget > 1 else budget)
    assert (direct if budget > 1 else bounded).await_count == 0


def test_fixed_reference_overrides_invented_description(tmp_path):
    path = tmp_path / 'refs.json'
    path.write_text(json.dumps([{'aliases': ['coffee table'], 'description': 'coffee table with solid pedestal'}]))
    generator = SavedImageGenerator(path)
    assert generator.reference_description('coffee_table', 'coffee table with metal legs and lower shelf') == 'coffee table with solid pedestal'
    with pytest.raises(ValueError, match='matched 0'):
        generator.reference_description('lamp', 'a floor lamp')


@pytest.mark.asyncio
async def test_planner_cannot_exceed_critique_or_repair_budget():
    from scenecode.agent_utils.base_stateful_agent import BaseStatefulAgent
    from agents.tool_context import ToolContext
    fake = SimpleNamespace(cfg=SimpleNamespace(max_critique_rounds=1),
                           _is_placement_agent=False,
                           _request_critique_impl=AsyncMock(return_value="defects"),
                           _request_design_change_impl=AsyncMock(return_value="changed"),
                           _create_reset_checkpoint_tool=lambda: None)
    tools = BaseStatefulAgent._create_planner_tools(fake)
    for name, args, method in [('request_critique', '{}', fake._request_critique_impl),
                              ('request_design_change', '{"instruction":"fix"}', fake._request_design_change_impl)]:
        context = ToolContext(context=None, tool_name=name, tool_call_id=name, tool_arguments=args)
        tool = next(t for t in tools if t and t.name == name)
        assert tool.is_enabled is True
        await tool.on_invoke_tool(context, args)
        assert tool.is_enabled is True  # Exhaustion is handled by the tool guard.
        assert 'exhausted' in await tool.on_invoke_tool(context, args)
        assert method.await_count == 1
