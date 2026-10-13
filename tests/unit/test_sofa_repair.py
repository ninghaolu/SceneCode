import copy

from scripts.sofa_repair_checks import check_sofa


def component(center, dimensions):
    return {'center':center,'dimensions':dimensions,
            'min':[c-d/2 for c,d in zip(center,dimensions)],
            'max':[c+d/2 for c,d in zip(center,dimensions)]}


def valid_case():
    groups = {
        'base_frame':[component([0,0,.15],[.96,.42,.12])],
        'legs':[component([x,y,.045],[.035,.035,.09]) for x in (-.42,.42) for y in (-.16,.16)],
        'armrests':[component([x,0,.235],[.1,.42,.29]) for x in (-.43,.43)],
        'seat_cushions':[component([x,-.04,.255],[.25,.34,.09]) for x in (-.255,0,.255)],
        'back_cushions':[component([x,.15,.365],[.25,.12,.27]) for x in (-.255,0,.255)]}
    return {name:{'components':values,'geometry_sha256':name} for name,values in groups.items()}


def test_valid_geometry_control_and_original_mirror_defect():
    good=valid_case()
    assert check_sofa(good,good)['passed']
    bad=copy.deepcopy(good)
    bad['armrests']['components'][0]=bad['armrests']['components'][1]
    result=check_sofa(bad,good)
    assert not result['passed']
    assert 'Armrests must occupy opposite outer sides' in result['errors']


def test_occluded_back_and_changed_protected_geometry_fail():
    good=valid_case()
    bad=copy.deepcopy(good)
    bad['seat_cushions']['components']=[component([x,-.04,.32],[.25,.34,.22]) for x in (-.255,0,.255)]
    assert any('exposed above' in x for x in check_sofa(bad,good)['errors'])
    bad=copy.deepcopy(good)
    bad['legs']['geometry_sha256']='changed'
    assert 'legs: protected geometry changed' in check_sofa(bad,good)['errors']


def test_tall_back_regression_from_first_repair_trial():
    case=valid_case()
    case['back_cushions']['components']=[component([x,.12,.42],[.24,.18,.34]) for x in (-.24,0,.24)]
    errors=check_sofa(case)['errors']
    assert 'Back cushion is too tall relative to its width' in errors
    assert 'Overall sofa height-to-width ratio exceeds 0.55 for this reference' in errors


def test_asset_validator_receives_reference_before_generated_views(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from omegaconf import OmegaConf
    from scenecode.agent_utils.asset_router import router as module
    router=object.__new__(module.AssetRouter)
    reference=tmp_path/'reference.png'
    reference.write_bytes(b'fixture')
    router.blender_server=MagicMock()
    router.blender_server.render_multiview_for_analysis.return_value=[tmp_path/'front.png',tmp_path/'back.png']
    router.side_view_elevation_degrees=20
    router.validation_taa_samples=8
    router.cfg=OmegaConf.create({'openai':{'model':'test','reasoning_effort':{'asset_validation':'low'},
                                        'verbosity':{'asset_validation':'low'},'vision_detail':'high'}})
    router.vlm_service=MagicMock()
    router.vlm_service.create_completion.return_value='{"is_acceptable":false,"reason":"Wrong proportions","suggestions":[]}'
    monkeypatch.setattr(module,'encode_image_to_base64',lambda p:p.name)
    monkeypatch.setattr(module.prompt_manager,'get_prompt',lambda **kw:'Check this asset')
    result=router.validate_asset(tmp_path/'mesh.glb','sofa',tmp_path/'debug',reference_image=reference)
    assert not result.is_acceptable
    content=router.vlm_service.create_completion.call_args.kwargs['messages'][0]['content']
    assert 'FIRST image is the generation reference' in content[0]['text']
    assert [c['image_url']['url'].split(',')[1] for c in content[1:]]==['reference.png','front.png','back.png']
