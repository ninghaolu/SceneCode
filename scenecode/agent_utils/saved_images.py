"""Replay reference images from an explicit manifest without image API calls."""
import hashlib
import json
import re
import shutil
from pathlib import Path

from scenecode.agent_utils.image_generation import BaseImageGenerator


class SavedImageGenerator(BaseImageGenerator):
    def __init__(self, manifest):
        self.entries = json.loads(Path(manifest).read_text())

    def resolve_entry(self, name, description):
        for text in (name.replace("_", " "), description):
            matches = [entry for entry in self.entries if any(
                re.search(r"\b" + re.escape(alias) + r"\b", text, re.I)
                for alias in entry["aliases"])]
            if matches:
                break
        if len(matches) != 1:
            raise ValueError(f"Saved-reference manifest matched {len(matches)} entries for {description!r}")
        return matches[0]

    def reference_description(self, name, description):
        return self.resolve_entry(name, description).get("description", description)

    def generate_images(self, style_prompt, object_descriptions, output_paths, size=None, labels=None):
        if len(object_descriptions) != len(output_paths):
            raise ValueError("Descriptions and image destinations must have equal lengths")
        for description, destination in zip(object_descriptions, output_paths):
            destination = Path(destination)
            entry = self.resolve_entry(destination.stem, description)
            source = Path(entry["path"])
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"Saved reference changed: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            destination.with_suffix(".reference.json").write_text(json.dumps({
                "source": str(source), "sha256": digest, "description": description,
                "style_prompt": style_prompt, "note": "Reused saved reference; no image synthesis."}, indent=2))

    def generate_furniture_context_image(self, **kwargs):
        raise NotImplementedError("Saved-reference pilot does not synthesize context images")

    def generate_manipuland_context_image(self, **kwargs):
        raise NotImplementedError("Saved-reference pilot does not synthesize context images")
