# aind-ophys-movie-qc

Quality control for a motion-corrected ophys movie. One task processes one
plane; the pophys pipeline flattens motion correction's per-plane results into
this capsule.

This is a thin Code Ocean wrapper. All logic lives in
[aind-ophys-movie-qc-library](https://github.com/AllenNeuralDynamics/aind-ophys-movie-qc-library),
which the capsule environment installs pinned to an exact commit;
`code/run_capsule.py` only calls `aind_ophys_movie_qc_library.job.run`. Settings
come from `pydantic-settings`, so every parameter is an underscored
`--name=value` option, and `.codeocean/app-panel.json` is generated from the
library's `settings.py` with
[`auto-app-panel`](https://github.com/bjhardcastle/auto-app-panel) rather than
hand-written.

## Inputs

Mounted under `../data`:

- `<plane>/motion_correction/<plane>_registered.h5` - the registered movie, and
  motion correction's `processing.json` next to it.
- The raw asset's `acquisition.json` (aind-data-schema v2) **or**
  `session.json` (v1), plus optional `data_description.json`, `subject.json`
  and `platform.json`.
- Optionally `<plane>_z_stack_local.h5`, which enables the z-drift metrics.

The frame rate is resolved in this order, and never defaulted: motion
correction's own provenance next to the movie (both the v2
`processing.json` and the v1 `*_data_process.json` shapes are read), then the
acquisition metadata, then `--frame_rate`. It scales the epilepsy event-width
window and every per-second photon statistic, so an unresolvable frame rate
fails the run.

## Outputs

Everything is written to `../results/<plane>/movie_qc/`, never the results
root - the pipeline publishes with `saveAs` flattening to basename, so
root-level files from concurrent per-plane tasks would collide.

- `quality_control.json` - one full aind-data-schema v2 `QualityControl`, with
  `default_grouping = ["evaluation"]`.
- `processing.json` - one full `Processing`. Written even when the run fails,
  with the error recorded in `notes`.
- `<plane>_registered_metrics.json` - the raw metrics dump.
- Ten PNG figures, plus the registered z-stack when one was supplied.
