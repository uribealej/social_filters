# Refactor Loop Policy

Purpose

Keep work moving through a whole ownership slice instead of stopping after one superficial cleanup in a large notebook-driven workflow.

Use this file when

- The task is a refactor or repeated cleanup in one owner area.
- You need to know when to keep going and when to stop.
- You are handing off partial work.

## Default working unit
- One ownership slice centered on one public helper, one writer stage, or one tightly related notebook-to-module behavior.

## Required sub-pass loop
1. Identify the owning module or writer stage.
2. Trace the immediate caller and the first downstream consumer.
3. Make the owner-layer change before touching notebook orchestration.
4. Run the smallest practical validation for that slice.
5. Update the relevant reference doc or recent-changes log if public behavior or handoff status changed.

## Keep-going rules
- Keep going if the remaining work is in the same owner module and validation surface.
- Keep going if a downstream notebook is still compensating for the same upstream issue you are already fixing.
- Keep going if the public helper changed but its router docs or output docs still describe the old behavior.

## Valid stop conditions
- The owning slice is coherent, validated, and the first downstream consumer still works.
- The remaining work moves into a different owner layer, different workflow profile, or materially different validation surface.
- You hit a real external blocker such as missing data, unavailable environment dependency, or conflicting user edits.

## Invalid stop conditions
- One cleaned-up cell in a notebook while the duplicate helper still exists in the owner module.
- A writer-stage change without checking the first consumer.
- A partial rename or output-contract tweak without updating the relevant docs.

## Handoff requirements
- If stopping mid-slice, append the workflow-specific recent-changes log with what changed, what remains broken, and the next likely breakpoint.
- Note rerun implications whenever a cache, file contract, or public helper changed.

## Validation expectations
- Use the smallest rerun or smoke check that exercises the edited owner and its first downstream consumer.
- Do not declare success from static reasoning alone.
