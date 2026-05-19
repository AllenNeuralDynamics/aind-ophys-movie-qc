# aind-ophys-movie-qc: Schema v1 → v2 Migration

## Summary

Migrated from `aind-data-schema==1.2.0` (v1) to `aind-data-schema==2.6.0` (v2).
Replaced `argparse` with `pydantic-settings`, centralized schema imports into `utils/`,
and restructured QC output from per-evaluation JSON files to a single `quality_control.json`.

---

## Design Decisions

1. **Outputs:** Two metadata files:
   - `quality_control.json` — single `QualityControl` with all `QCMetric`s
   - `processing.json` — single `Processing` with one `DataProcess`

2. **Metadata loading:** `acquisition.json` loaded via `Acquisition.model_validate_json()`.

3. **CLI parsing:** `argparse` replaced with `pydantic-settings` `BaseSettings(cli_parse_args=True)`.

4. **Frame rate:** Exclusively from `acquisition.json` (`ImagingConfig.sampling_strategy.frame_rate`). Raises `ValueError` if not found. No CLI override, no upstream `data_process.json` fallback.

5. **FOV scale factor:** `PlanarImage.image_to_acquisition_transform → Scale.scale[0]`. Raises `ValueError` if not found.

6. **Tags format:** Operational QC metrics use `{"evaluation": "<name>", "Operational QC": "True"}`. Non-operational metrics use `{"evaluation": "<name>"}`.

7. **Legacy writers deleted:** `write_qc_evaluation()`, `save_qc_evaluation_to_file()`, `save_qc_metric_to_file()`, `write_legacy_movie_qc_metrics()`, `parse_args()` — all removed.

8. **Dockerfile:** Updated externally in Code Ocean remote environment.

---

## File Layout

```
code/
├── run                        # Bash entry point (unchanged)
├── run_capsule.py             # Orchestrator (slim — no schema imports)
├── image_utils.py             # PIL image combining (unchanged)
├���─ local_z_stack.py           # Z-drift analysis (session→acquisition)
└── utils/
    ├── __init__.py            # Empty
    ├── metadata_utils.py      # ALL schema imports, loaders, builders, output writers
    └── qc.py                  # QCMetric builders + combined image creation
```

---

## Changes Made

### New: `code/utils/metadata_utils.py`

All `aind-data-schema` imports centralized here.

| Function | Purpose |
|----------|---------|
| `load_acquisition(path)` | `Acquisition.model_validate_json()` |
| `load_data_description(path)` | `DataDescription.model_validate_json()` |
| `get_frame_rate(acquisition)` | From `ImagingConfig.sampling_strategy.frame_rate`. Raises `ValueError`. |
| `get_fov_scale_factor(acquisition)` | From `PlanarImage → Scale.scale[0]`. Raises `ValueError`. |
| `pending_qc_status(status)` | Creates timezone-aware `QCStatus` (Seattle TZ) |
| `write_quality_control(metrics, output_dir)` | Wraps in `QualityControl`, writes JSON |
| `build_data_process(parameters, start_time, end_time, ...)` | Constructs `DataProcess` with `ProcessName.ANALYSIS` |
| `write_processing(data_process, output_dir)` | Wraps in `Processing`, writes JSON |
| `build_and_write_quality_control(...)` | Builds all QC metrics + creates combined image + writes `quality_control.json` |
| `build_and_write_processing(...)` | Builds `DataProcess` + writes `processing.json` |

### New: `code/utils/qc.py`

| Function | Tags | Status Logic |
|----------|------|-------------|
| `create_combined_metric_image(seg, poisson, combined)` | — | Creates combined PIL image |
| `build_intensity_metric(unique_id, metrics)` | `{"evaluation": "Intensity Drift", "Operational QC": "True"}` | PASS/PENDING/FAIL by \|change\| thresholds |
| `build_epilepsy_metric(unique_id, metrics)` | `{"evaluation": "Epilepsy", "Operational QC": "True"}` | PASS if 0, else FAIL |
| `build_snr_dprime_metric(metrics)` | `{"evaluation": "Event detection statistics"}` | Always PASS |
| `build_pixel_saturation_metric(unique_id, metrics)` | `{"evaluation": "Pixel Saturation", "Operational QC": "True"}` | PASS/PENDING/FAIL by sat count |
| `build_photon_metric(metrics, combined_img_path)` | `{"evaluation": "Photon detection statistics"}` | Always PASS |
| `build_zdrift_metric(unique_id, metrics)` | `{"evaluation": "Z-drift", "Operational QC": "True"}` | PASS/FAIL by 10um threshold |

### Modified: `code/run_capsule.py`

- **Imports:** Removed `argparse`, v1 schema imports, `PIL.Image`, `combine_images_vertically`. Added `pydantic`/`pydantic_settings`, `utils.metadata_utils` wrappers.
- **`JobSettings`:** Replaces `parse_args()`. No `frame_rate` field (acquisition only).
- **`SCANIMAGE_MAXIMUM_PIXEL_BRIGHTNESS_LEVEL`:** Kept as named constant, referenced by `JobSettings.max_pixel_range` default.
- **Deleted functions:** `save_qc_evaluation_to_file`, `save_qc_metric_to_file`, `write_qc_evaluation`, `write_legacy_movie_qc_metrics`, `parse_args`
- **`__main__` block:** Uses `JobSettings()`, `load_acquisition()`, `get_frame_rate()`. Calls `build_and_write_quality_control()` and `build_and_write_processing()` at the end.
- **All computational/plotting functions unchanged** (zero schema dependency).

### Modified: `code/local_z_stack.py`

- `session_json_path` → `acquisition_json_path` (param, attribute, docstring)
- Raw JSON traversal (`ophys_fovs[0].fov_scale_factor`) → `load_acquisition()` + `get_fov_scale_factor()`

---

## Output File Changes

| v1 Output | v2 Output |
|-----------|-----------|
| 6× `*_evaluation.json` | **Removed** — merged into `quality_control.json` |
| 2× `*_metric.json` (legacy) | **Removed** |
| *(none)* | `quality_control.json` (new) |
| *(none)* | `processing.json` (new) |
| `*_metrics.json` (raw numbers) | Unchanged |
| All `*.png` plots | Unchanged |

---

## Verification

- All 4 modified files pass `ast.parse()` syntax check
- Full capsule test requires real data on Code Ocean
