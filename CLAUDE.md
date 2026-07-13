# Core rules

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Verify before calling something done — but verifying and
committing a permanent test are different things.**

- "Fix the bug" → write a regression test that reproduces it, keep it in the suite.
  This is the case that pays for itself: without it, the bug can silently come back.
- "Add a feature/validation" → verify it actually works before calling it done. Only
  add a permanent test for it if the check is cheap and offline (unit-level), or if
  asked for one.
- Verification that needs network calls, live services, or other ongoing cost →
  fine to run once to confirm the change works, but ask before adding it as a
  permanent test in the suite. Default to not persisting it.
- Don't add a test because "the file already has one like this for every other
  case" — match existing density only when the new test independently earns its
  keep, not to fill a pattern.

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently within a single task. Weak criteria ("make it work") require constant clarification.

## 5. Never Delete a File
Never delete a file. I won't ask you to, if you think I did, you misunderstood. Tell me if you think I said a file needs to be deleted, or you determine a file should be deleted. I will delete what I don't want, don't do it for me.

## 6. Git
You will not be pushing code. I will deal with Git myself.

## 7. Pushback
When I propose an approach, do not assume it's correct just because I stated it confidently.
- If you see a flaw, tradeoff, or simpler alternative, say so before implementing — even if I sound sure.
- If my request conflicts with something else in this codebase (existing pattern, prior decision, performance constraint), flag the conflict explicitly rather than quietly reconciling it.
- Distinguish "I disagree" from "I have a clarifying question" — don't dress disagreement up as a question.
- It's fine to implement what I asked after pushing back once if I still want it. Don't keep relitigating.

## 8. Subagents
When using subagents (at my direction or for tasks that genuinely benefit from isolation), use Haiku by default. Suggest Sonnet if the task requires significant reasoning; never use Opus or higher.

## 9. Docs Before and After Every Change

**Keep docs in sync with code. Always.**

Before touching code:
- Identify which docs are affected (CLAUDE.md, status.md, others?).
- Update docs to reflect findings and the planned change — phantom entries removed, new modules added, descriptions corrected.

After completing a change:
- Update CLAUDE.md if current behavior changed (architecture, invariants, CLI flags, formula logic).
- Update status.md only to: tick off a completed backlog item, add a new backlog item, or add/remove a module line. Do not append bugfix history — that lives in git commits.

The test: if someone cloned the repo and read only the docs, would they have an accurate picture of what's in it right now?

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

# s2fetch — Project Guide

s2fetch is a standalone Sentinel-2 L2A acquisition utility. Domain-agnostic core:

    AOI + date window + bands + cloud filter  ->  cloud-masked xarray

It also tiles results to GeoTIFF patches for downstream ML dataloaders.

It is deliberately independent of any consumer project. **No PyTorch, no ML stack, no
burn/water/mineral domain logic.** Consumers import s2fetch; s2fetch never imports them.
The standalone env with no ML deps is the honest test that this coupling never creeps in.

**This repo has not been built yet. The full build plan is in [BUILD.md](BUILD.md). Read it first.**

---

## Why this exists

Different domains that consume Sentinel-2 (burn scar segmentation, aquatic optics,
mineral mapping) all need the same "cloud-masked S2 scene for this AOI and date"
capability. Rather than duplicate that acquisition logic per-project, it's a shared,
domain-agnostic utility that each consumer imports.

Consumer categories, present and planned:
- Burn scar segmentation — S2 L2A patches
- Aquatic optics — S2 broadband baseline scenes
- Mineral mapping — S2 context scenes

Domain logic for each (dNBR, water constituent indices, mineral absorption) lives in the
consumer and imports s2fetch. Never add it here.

---

## Design principles

- **Provider-swappable STAC backend behind one API.** Default Planetary Computer; Earth Search
  and CDSE selectable by a `provider=` string. See BUILD.md for the verified tradeoffs.
- **Bands keyed by canonical name + wavelength**, mapped per-provider to that provider's asset
  keys (PC uses `B02/B8A/B12/SCL`; Earth Search uses `blue/nir08/swir22/scl`). Do not hardcode
  provider asset names in fetch logic. This band/wavelength registry is the hook for
  wavelength-conditional models and future sensor extensibility.
- **Domain-agnostic core.** Return generic reflectance arrays. No spectral indices in this repo.

## Stack

`pystac-client` (STAC query) + `odc-stac` (STAC items -> xarray, windowed COG reads). Confirmed
current standard as of July 2026.

## Environment

Minimal conda env pins the compiled geo libs (gdal/rasterio/pyproj/shapely/geopandas) from
conda-forge for a robust Windows build, then `pip install -e .`. `pyproject.toml` is the
canonical dependency list, so `pip install s2fetch` also works standalone where wheels exist.
No torch, no terratorch. Env files are ready to paste in BUILD.md.

## Conventions

- src-layout package under `src/s2fetch/`.
- `fetch()` returns a lazy (dask-backed) xarray; the caller computes. Keep I/O windowed.
- Be precise about S2 processing: **L2A is atmospherically corrected surface reflectance
  (Sen2Cor, bottom-of-atmosphere), not raw.** Do not describe it as raw.

## Working preferences

- Concise and direct. State the result or decision, then stop. No trailing recap.
- Provider access terms and asset formats change. When a claim depends on current provider
  behavior, verify against live docs rather than asserting from memory.
