# Refactor Diagnosis — partimorph (2026-04-11)

## Scope

Full codebase sweep of `src/partimorph/` (~1,300 LOC, 18 files). Both MECE and Feynman lenses applied from the main agent (targeted mode — small enough to fit in context).

## Context read

- `CLAUDE.md` — functional style, src-layout, TypedDict schema
- `git log` — recent churn concentrated in `analyzer.py`, `metrics.py`, `fitting.py`, `wadell/*`
- `tests/test_partimorph.py` — 8 tests, baseline `8 passed in 1.76s` via `.venv/bin/pytest`
- `pyproject.toml`, `README.md` — no conflicts with findings

## Selected findings

### F-04: Replace Moore neighborhood tracer with `cv2.findContours`

- **Location**: `src/partimorph/wadell/boundary.py` (full file, 93 lines)
- **Category**: `complexity` / `perf-waste` (Feynman lens)
- **Observation**: `extract_boundary` uses `skimage.measure.label` + `regionprops` + a custom Moore neighborhood tracer (`moore_neighborhood`, `boundary_tracing`). Total: 93 LOC of hand-written boundary tracing.
- **Reconstruction attempt**: "cv2's contour output does not provide the boundary ordering or continuity the Wadell algorithm needs."
- **Failure point**: No evidence for this. `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_NONE)` returns every boundary pixel in order. Experiment confirmed.
- **Suggested direction**: Replace the entire file body with a thin wrapper around `cv2.findContours` that (a) picks the largest-area contour, (b) closes the polyline, (c) returns `(N, 2) float64`. Delete `moore_neighborhood` and `boundary_tracing`.
- **Axes**: Impact: high, Confidence: high, Effort: M
- **Verification**: In `/tmp/partimorph-refactor-exp` worktree, replaced full file with ~20-line cv2-based implementation. `pytest tests/test_partimorph.py -q` → `8 passed in 0.90s` (baseline 1.76s, ~2× faster). Worktree discarded.

### F-02: Remove redundant `to_binary` calls in metrics

- **Location**: `src/partimorph/metrics.py:28, 49, 71, 93` plus `src/partimorph/analyzer.py:85`
- **Category**: `duplication` (MECE lens)
- **Observation**: `analyze_mask` calls `to_binary(mask)` once, then passes the binary mask to each of the four `compute_*` functions, each of which calls `to_binary` again on the already-binary input. Total: 5 calls per invocation.
- **Reconstruction attempt**: "Each metric is an independent public function and must validate its own input."
- **Failure point**: `partimorph/__init__.py` only re-exports `analyze_mask` and `utils`. The metric functions are not part of the public API. The belt-and-suspenders validation has no caller that needs it.
- **Suggested direction**: Remove `to_binary(mask)` calls from all four `compute_*` functions in `metrics.py`. Parameter types should be `Mask` (already-validated uint8 binary). Optional: rename `metrics.py` → `_metrics.py` to signal private intent.
- **Axes**: Impact: med, Confidence: high, Effort: S
- **Verification**: In worktree, deleted `mask_binary: Mask = to_binary(mask)` from all four metric functions and replaced references with `mask`. `pytest tests/test_partimorph.py -q` → `8 passed in 2.24s`. Worktree discarded.

### F-08: Dead branch in `_rescale_results`

- **Location**: `src/partimorph/analyzer.py:62`
- **Category**: `dead-code` (Feynman lens)
- **Observation**: `if "bbox" in ellipse:` guards bbox rescaling, but `fitting._ellipse_payload` (the only constructor of `EllipseData`) always sets the `bbox` field. `EllipseData` TypedDict declares `bbox: Bbox` as required.
- **Reconstruction attempt**: "A historical EllipseData may have made bbox optional."
- **Failure point**: Current code never emits an `EllipseData` without `bbox`. The guard is unreachable.
- **Suggested direction**: Remove the `if "bbox" in ellipse:` guard. Rescale bbox unconditionally.
- **Axes**: Impact: low, Confidence: high, Effort: S
- **Verification**: Ran `fit_ellipse` on a sample mask in worktree → `'bbox' in result == True`. Schema confirmed. No conditional creation path exists.

### F-07: Extract oriented bbox helper from `fit_ellipse`

- **Location**: `src/partimorph/fitting.py:92-131`
- **Category**: `complexity` (Feynman lens)
- **Observation**: `fit_ellipse` uses `cv2.fitEllipse` to obtain only `angle`, then projects all points onto the rotated axes, takes min/max, and manually constructs 4 corners of an oriented bounding box (13 lines of coordinate arithmetic).
- **Reconstruction attempt**: "It rescales the ellipse dimensions."
- **Failure point**: The actual intent — "compute the tight oriented bounding box aligned with the fitted ellipse's principal angle, because `cv2.fitEllipse`'s own width/height are a Gaussian fit that underestimates the tight extent" — cannot be reconstructed from the code alone. The manual 4-corner construction obscures the intent.
- **Suggested direction**: Extract a helper `_oriented_bbox_from_angle(points: Points, angle: float) -> tuple[float, float, float, float, float, float, list[list[float]]]` returning `(center_x, center_y, width, height, min_w, min_h, bbox_corners)` — or whichever factoring keeps the caller readable. The helper's name carries the intent without needing a comment.
- **Axes**: Impact: med, Confidence: high, Effort: S
- **Verification**: not falsifiable by execution (behavior-preserving refactor; tests must still pass)

## Refactoring constraints

1. **All 8 tests in `tests/test_partimorph.py` must still pass.** Run via `.venv/bin/pytest tests/test_partimorph.py -q`.
2. **Public API stable**: `partimorph.analyze_mask` signature and `AnalysisResult` schema unchanged. `partimorph.utils.*` exports unchanged.
3. **`EllipseData` schema unchanged** (F-09 was explicitly skipped). `major`/`minor`/`w`/`h`/`bbox` all remain.
4. **Functional style preserved** per `CLAUDE.md` — no classes except TypedDicts, pure functions.
5. **`skimage` may remain a dependency** (used elsewhere — `metrics.compute_circularity` uses `perimeter_crofton`), but `skimage.measure.label`/`regionprops` usage in `wadell/boundary.py` should go away with F-04.
6. **Ordering**: F-04 first (largest change, sets up file state), then F-02, F-07, F-08 in any order. Each should be a separate commit so regressions are bisectable.

## Success criteria

After the refactor:

1. **F-04**: Reading `wadell/boundary.py` cold, a reader can reconstruct its purpose in one sentence — "extract the closed boundary polyline of the largest component as an (N, 2) float64 array, via cv2." The 93-line Moore tracer no longer exists.
2. **F-02**: Reading `metrics.py` cold, the reader sees each `compute_*` accepting a `Mask` and immediately doing its metric-specific work. There is no `to_binary` call in the file.
3. **F-07**: Reading `fit_ellipse`, the reader reaches a helper call whose name carries the intent ("oriented bbox at the ellipse angle") without needing to parse 13 lines of projection arithmetic.
4. **F-08**: `analyzer._rescale_results` contains no unreachable branches; every `if` gates a real case.
5. **Tests**: `8 passed`.
6. **Speed**: baseline test time improves (expected ~1.8s → ~0.9s from F-04 alone).
