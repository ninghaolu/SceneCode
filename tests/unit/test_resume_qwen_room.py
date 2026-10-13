import json
from pathlib import Path

import pytest

from scripts.resume_qwen_room import copy_assets, digest_tree


def test_copied_registry_paths_are_isolated_and_source_preserved(tmp_path):
    source = tmp_path / 'source'
    assets = source / 'room/scene_000/room_living_room/generated_assets/furniture'
    assets.mkdir(parents=True)
    mesh = assets / 'mesh.glb'
    mesh.write_bytes(b'mesh')
    sdf = assets / 'mesh.sdf'
    sdf.write_text(f'<uri>{mesh}</uri>')
    image = assets / 'preview.png'
    image.write_bytes(b'image')
    (assets / 'asset_registry.json').write_text(json.dumps({'table_0': {
        'geometry_path': str(mesh), 'sdf_path': str(sdf), 'image_path': str(image)}}))
    before = digest_tree(source)
    output = tmp_path / 'output'
    output.mkdir()
    registry = copy_assets(source, output)
    copied = json.loads(registry.read_text())['table_0']
    for key in ('geometry_path', 'sdf_path', 'image_path'):
        assert Path(copied[key]).is_relative_to(output)
        assert Path(copied[key]).is_file()
    assert copied['geometry_path'] in Path(copied['sdf_path']).read_text()
    assert digest_tree(source) == before
    with pytest.raises(FileExistsError):
        copy_assets(source, output)
