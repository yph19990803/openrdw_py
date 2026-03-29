# OpenRdw_vbdi Agent Harness

This directory adapts the long-running agent harness pattern from Anthropic's article to the current Unity project.

## Goal

Provide a stable handoff surface so multiple sessions or specialized agents can work without relying on hidden context. The shared state for every session is:

- `agent-harness/feature_list.json`
- `agent-harness/progress.md`
- `agent-harness/init.sh`

## Project facts captured here

- Unity editor version: `2022.3.46f1`
- Main scene: `Assets/OpenRDW/Scenes/OpenRDW Scene.unity`
- Results output folder: `Experiment Results/`
- Existing implementation plan: `RDW_Aware_Finetune_Plan.md`
- Current limitation: project root is not a git repository

## Recommended roles

- `coordinator`: chooses the next task, enforces single-task sessions, checks that progress and feature status agree.
- `bootstrap`: maintains harness files, setup scripts, and shared conventions.
- `implementer`: makes one bounded code change tied to a single feature id.
- `verifier`: runs batchmode, editor smoke checks, data-shape checks, or experiment validation for the current feature.
- `cleanup`: improves comments, naming, or structure only after a feature is already passing.

## Operating rules

- Every session starts by running `agent-harness/init.sh`.
- Every session reads `progress.md` and `feature_list.json` before editing.
- A session owns exactly one feature id at a time.
- Do not mark a feature `done` until the verification evidence field is updated.
- If validation cannot run, record the exact blocker instead of claiming completion.
- If git is enabled later, commit once per completed feature so the next session has a durable checkpoint.

## Suggested session loop

1. Run `bash agent-harness/init.sh`.
2. Inspect the highest-priority feature whose status is not `done`.
3. Implement only that feature.
4. Validate with the narrowest meaningful check.
5. Update `progress.md`.
6. Update `feature_list.json`.

## Verification ladder

Use the smallest check that can legitimately prove the current feature.

- `doc`: file exists, schema is correct, instructions are internally consistent.
- `editor smoke`: project opens or batchmode completes without import/compiler errors.
- `play mode/manual`: the target Unity scene behavior is observed in editor.
- `data`: generated tensors/csv/json have expected shapes and metadata.
- `training`: a short smoke epoch or preprocessing dry run finishes.
- `experiment`: ADE/FDE or other metrics are produced on the agreed split.

## Git note

The harness assumes git history is useful, but this project currently has no `.git` directory. The harness still works without git by treating `progress.md` and `feature_list.json` as the primary source of truth. Initializing git is still recommended.
