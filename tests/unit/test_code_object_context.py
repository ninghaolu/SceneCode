"""Image planning must receive dimensions already supplied by the room agent."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scenecode.agent_utils.asset_router.router import AssetRouter


@pytest.mark.parametrize('articulated', [False, True])
def test_router_delivers_dimensions_description_and_retry_feedback(tmp_path, articulated):
    router = object.__new__(AssetRouter)
    router._prepare_code_object_inputs = MagicMock(return_value=(tmp_path/'ref.png', tmp_path, None))
    router._build_code_generated_geometry = MagicMock()
    runner = MagicMock()
    runner.generate_articulated_from_image.return_value.status = 'no_movable_parts'
    item = SimpleNamespace(description='Three-seat fabric sofa', dimensions=[2.2, .9, .85])
    generate = (router._generate_code_articulated_geometry if articulated
                else router._generate_code_object_geometry)
    generate(item=item, code_object_runner=runner, image_generator=MagicMock(),
             images_dir=tmp_path, code_object_dir=tmp_path, feedback='Keep both arms')
    called = runner.generate_articulated_from_image if articulated else runner.generate_from_image
    context = called.call_args.kwargs['feedback']
    assert 'Three-seat fabric sofa' in context
    assert '[2.2, 0.9, 0.85]' in context
    assert str(.85 / 2.2) in context
    assert 'Keep both arms' in context
    assert called.call_args.kwargs['image_path'] == tmp_path/'ref.png'


def test_context_without_dimensions_preserves_request():
    item = SimpleNamespace(description='wooden shelf', dimensions=[])
    assert AssetRouter._code_object_context(item, None) == 'Requested object description: wooden shelf'


@pytest.mark.parametrize('with_reference', [False, True])
def test_validation_labels_reference_separately_from_generated_evidence(tmp_path, monkeypatch, with_reference):
    module = 'scenecode.agent_utils.asset_router.router'
    monkeypatch.setattr(module + '.encode_image_to_base64', lambda path: path.name)
    monkeypatch.setattr(module + '.prompt_manager.get_prompt', lambda **kwargs: 'Generic validation')
    reference = tmp_path/'reference.png'
    reference.touch()
    router = object.__new__(AssetRouter)
    router.blender_server = MagicMock()
    router.blender_server.render_multiview_for_analysis.return_value = [tmp_path/'front.png', tmp_path/'rear.png']
    router.side_view_elevation_degrees = 20
    router.validation_taa_samples = 8
    router.cfg = SimpleNamespace(openai=SimpleNamespace(
        model='qwen3.7-plus', reasoning_effort=SimpleNamespace(asset_validation='low'),
        verbosity=SimpleNamespace(asset_validation='low'), vision_detail='high'))
    router.vlm_service = MagicMock()
    router.vlm_service.create_completion.return_value = '{"is_acceptable":false,"reason":"missing outward detail","suggestions":[]}'
    result = router.validate_asset(tmp_path/'mesh.glb', 'slatted cabinet', tmp_path/'views',
                                   reference_image=reference if with_reference else None)
    assert result.is_acceptable is False
    content = router.vlm_service.create_completion.call_args.kwargs['messages'][0]['content']
    images = [entry['image_url']['url'] for entry in content if entry['type'] == 'image_url']
    assert [url.split(',')[-1] for url in images] == (
        ['reference.png', 'front.png', 'rear.png'] if with_reference else ['front.png', 'rear.png'])
    if with_reference:
        assert 'override generic geometry/material leniency' in content[0]['text']
        assert content[1]['text'].startswith('REFERENCE IMAGE')
        assert content[3]['text'].startswith('GENERATED VIEW 0')
        assert content[5]['text'].startswith('GENERATED VIEW 1')
    else:
        assert content[0]['text'] == 'Generic validation'
