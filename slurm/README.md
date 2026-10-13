# Living-room experiment

## Qwen saved-reference pilot

Previous submission: **19051280**, Qwen 3.7 Plus baseline, cancelled by UID 0
after 2h13m on 2026-10-02, before final export. Low GPU utilization is a
plausible cause, but the scheduler record does not confirm the policy reason.
See `/scratch/nl2752/scenecode_1003.md` and `outputs/qwen_19051280/status.json`.
No verified correct full-room result yet.

Previous execution: job **18970258** was stopped with a defective console assembly
and poor sofa fidelity. There is **no verified correct full-room result yet**.
Later fixes passed focused tests but have not completed another full room run.
See `/scratch/nl2752/scenecode_small_model.md` for failure evidence and run history.

`living_room_qwen.sbatch` runs API capability checks, generates and validates one
coffee table, then runs the four-item living room and checks the final exported
scene. It defaults to `qwen3.7-plus` at the official Beijing endpoint for every active
language/vision role, including summaries. Saved images are selected from
`qwen_saved_images.json` and verified by SHA256; image generation is disabled.
Floor and wall materials use the existing local material library. The fresh-run
prompt explicitly limits the scene to these four items. Furniture designer,
planner and critic calls are bounded at 100, 60 and 8 turns respectively.
At the critic limit, one additional tool-free structured verdict uses the saved
observations and must report unresolved issues. Critique and repair round counts
are enforced in code. The Qwen launcher has a two-hour allocation cap; this is
a budget limit, not a guarantee against utilization-based cancellation.
Saved-reference descriptions are included in the room prompt and used consistently
for asset generation and validation, with description replacements recorded.
Generated part code is checked against measured Blender bounds before export;
misplaced parts enter the same error-feedback repair loop as API/runtime failures.
Cached part code is rechecked when resuming. URDF mesh origins are derived from
zero-joint forward kinematics because exported OBJ vertices already encode world
positions; raw model output is preserved in the audit.

```bash
sbatch slurm/living_room_qwen.sbatch
SCENECODE_QWEN_MODEL=qwen3.5-plus sbatch slurm/living_room_qwen.sbatch
```

To restart furniture generation from an already completed floor plan, append
`experiment.pipeline.start_stage=furniture experiment.pipeline.resume_from_path=/absolute/path/to/previous/room`.
The new run copies the floor-plan checkpoint and retains its own output directory.

Supply `QWEN_API_KEY` in the submission environment, or put an export in the
private file `/scratch/nl2752/.config/scenecode/qwen.env` (mode 600). Credentials
must never be put in the launcher or manifest. `SCENECODE_QWEN_ENV` selects a
different private credential file. `QWEN_API_BASE` selects a different region.

Optional environment variables:

- `SCENECODE_QWEN_MODEL`: one model for all active roles; unexpected requested or
  returned model IDs stop the request. Default `qwen3.7-plus`.
- `SCENECODE_QWEN_REASONING`: default `low`; `SCENECODE_THINKING_BUDGET`: 2048 for Chat requests; `SCENECODE_MAX_TOKENS`: 16384 answer tokens. Responses uses the reasoning-effort setting.
- Forced tool-selection requests disable thinking because DashScope rejects that
  combination; the required tool is preserved and later automatic tool-selection
  calls restore low reasoning. The exception is recorded in each API audit entry.
- `SCENECODE_CONFIG_ONLY=1`: compose/print the room config without model calls or
  generation, e.g. `SCENECODE_CONFIG_ONLY=1 bash slurm/living_room_qwen.sbatch`.
- `SCENECODE_PILOT_ONLY=1`: stop after the single-asset pilot.
- `SCENECODE_ASSET_VALIDATION`: reuse a previously passing pilot's `validation.json`
  for the same model when starting a room job.
- `SCENECODE_RUN_DIR`: otherwise results go to `outputs/qwen_JOBID`.

The room's `independent_validation.json` checks that all four requested items
survive physics, have finite nonzero bounds and existing artifacts, have valid
joint limits, sample joint motion at five positions, pass Drake collision checks, provide connected walking space, and
orient the sofa toward the console. They also check door clearance, room height,
and articulated-link AABB connectivity at rest (2.5 cm tolerance). Nonzero exit status means the run is not a
validated success. These checks do not replace visual inspection or continuous
joint-motion collision testing.
Rigid-body inertia uses closed visual geometry when possible, with an explicitly
recorded approximation from convex collision pieces when visual mass properties
are invalid. The `.inertia.json` sidecar identifies that approximation.

Each run keeps resolved object configs, API request/response/usage records,
content-addressed input images, delivered feedback, and immutable intermediate
plan/code and execution histories. Provider billing is not inferred from tokens
or latency. The previous GPT run is left intact.

New runs also save `source_snapshot/manifest.json` with file hashes and explicit
model settings, and update `status.json` at each stage. An exit trap writes
`run_summary.json` and final status on normal completion or handled failure.
The preflight includes a real Agents SDK forced-tool → tool-result → structured
verdict round trip. Source is not edited during a running experiment. Completion
of numerical validation still requires visual review before a correctness claim.
The current execution plan is `/scratch/nl2752/scenecode_1003.md`.

## Recover an interrupted room without new model calls

`sbatch slurm/recover_qwen_room.sbatch` copies the saved four-item checkpoint
from `qwen_19051280`, applies the original physics configuration and saves the
settled house state. It requests 8 CPUs, 32 GB and 45 minutes,
with no GPU. Outputs go to `outputs/qwen_recovery_JOBID/`. The original run stays
unchanged. This is a recovery diagnostic, not a new successful baseline; the
known sofa visual mismatch remains for separate repair and review.

Then run `sbatch slurm/export_qwen_recovery.sbatch /absolute/path/to/CPU/output`.
This copies the settled room into `outputs/qwen_recovery_export_JOBID/`, exports
Blender, and runs independent final validation. It requests one L40S for at most
15 minutes and makes no API calls. CPU job 19084424 completed physics, but its
attempted Blender export crashed in VTK EGL/OpenGL; its saved physics output is
reusable. CPU-only Blender export is therefore not supported by this environment.

Long API orchestration should use CPU resources. GPU allocations should contain
rendering work, not API waits; splitting fresh generation into CPU orchestration
and separate GPU rendering remains to be implemented and verified.

The remainder of this document describes the original launcher.

Their example prompt: “A living room with a TV console, sofa, bookshelf, and coffee table.” Runs floor plan → furniture, one worker. Script: `slurm/living_room_test.sbatch`.

Export credentials (`OPENAI_API_KEY`, optionally `OPENAI_API_BASE`) before submission:

```bash
sbatch slurm/living_room_test.sbatch
```

The script keeps your account, one L40S, 16 CPUs, 64 GB RAM, and 4-hour limit. Logs: `/scratch/nl2752/scenecode_logs/`; results: `outputs/`. No startup tests or custom run modes.

Optional settings:

- `SCENECODE_ROOM_MODEL`: defaults to `gpt-5.4`; summarization models remain as configured in the agent YAMLs.
- `SCENECODE_OBJECT_CONFIG`: path to an object-model YAML; defaults to `scenecode/code_object/agent_framework/config.yaml` (Gemini). Edit its `model` entries to change object models. The former `SCENECODE_OBJECT_MODEL` helper was removed.
- `SCENECODE_IMAGE_BACKEND`: `openai` (default, `gpt-image-2`) or `flux-klein`.

Flux requires a separate environment/checkpoint, not yet installed:

```bash
export SCENECODE_IMAGE_BACKEND=flux-klein
export FLUX_PYTHON=/path/to/flux/environment/bin/python
export FLUX_MODEL_PATH=/path/to/FLUX.2-klein-9B
sbatch slurm/living_room_test.sbatch
```

Flux replaces image generation; GPT/Gemini calls still use APIs. GPU work is intermittent: Blender rendering and, with Flux, image inference. Code construction and physics use CPU; API waits do not use the local GPU. The Flux implementation starts a subprocess and loads the checkpoint for each image request. Sustained GPU utilization is not guaranteed.

Your reported policy (roughly 50% cancellation / 70% warning over two hours, depending on GPU type) makes this API-heavy single-scene job a poor fit. No utilization measurement is available yet. Flux does not guarantee meeting the threshold; GPU memory occupancy is not compute utilization. Use an allocation that permits intermittent rendering, or separate CPU orchestration from GPU workers (requires pipeline changes).

Configuration-only check: `bash slurm/living_room_test.sbatch --cfg job`.

The previously observed Drake rendering crash remains unresolved; simplifying the launcher does not fix it.

GPU diagnosis: `slurm/render_gpu_test.sbatch` runs the failing room-render test on one L40S for up to 10 minutes, with no model API calls. Submitted as job **18571192**; initial state was `PENDING (QOSGrpGRES)`. Its logs will be `/scratch/nl2752/scenecode_logs/18571192.out` and `.err`.

For a later generation run, set credentials in the terminal from which you submit:

```bash
cd /scratch/nl2752/SceneCode
read -rsp 'API key: ' OPENAI_API_KEY; echo
export OPENAI_API_KEY
export OPENAI_API_BASE=https://YOUR_GATEWAY/v1
sbatch slurm/living_room_test.sbatch
```

The gateway must support the configured GPT/Gemini/image models. Environment activation happens inside the job script.

Generation output root: `outputs/YYYY-MM-DD/HH-MM-SS/`. A successful furniture-stage run saves `scene_000/combined_house_after_furniture/house.blend`, `house.dmd.yaml`, and `house_state.json`, plus per-room generated object code, meshes, SDF assets, and renders. The exact output root is printed in the job log.
