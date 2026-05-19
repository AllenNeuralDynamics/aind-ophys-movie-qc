"""Schema interaction layer — all aind-data-schema imports live here."""

import json
import os
from datetime import datetime as dt, timezone
from pathlib import Path
from typing import Optional

import pytz
from aind_data_schema.components.configs import ImagingConfig, PlanarImage
from aind_data_schema.components.coordinates import Scale
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.acquisition import Acquisition
from aind_data_schema.core.data_description import DataDescription
from aind_data_schema.core.processing import DataProcess, Processing, ProcessStage
from aind_data_schema.core.quality_control import (
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.process_names import ProcessName

SEATTLE_TZ = pytz.timezone("America/Los_Angeles")
CODE_URL = "https://github.com/AllenNeuralDynamics/aind-ophys-movie-qc"


# ── Loaders ──────────────────────────────────────────────────────────


def load_acquisition(path: Path) -> Acquisition:
    """Load and validate acquisition.json into Pydantic model.

    Parameters
    ----------
    path : Path
        Path to acquisition.json.

    Returns
    -------
    Acquisition
        Validated Acquisition model.
    """
    return Acquisition.model_validate_json(path.read_text())


def load_data_description(path: Path) -> DataDescription:
    """Load and validate data_description.json into Pydantic model.

    Parameters
    ----------
    path : Path
        Path to data_description.json.

    Returns
    -------
    DataDescription
        Validated DataDescription model.
    """
    return DataDescription.model_validate_json(path.read_text())


# ── Acquisition traversal helpers ────────────────────────────────────


def get_frame_rate(acquisition: Acquisition) -> float:
    """Extract frame rate from ImagingConfig.sampling_strategy.

    Parameters
    ----------
    acquisition : Acquisition
        Validated Acquisition model.

    Returns
    -------
    float
        Frame rate in Hz.

    Raises
    ------
    ValueError
        If no frame rate is found in any ImagingConfig.
    """
    for ds in acquisition.data_streams:
        for config in ds.configurations:
            if isinstance(config, ImagingConfig):
                sampling = getattr(config, "sampling_strategy", None)
                if sampling and hasattr(sampling, "frame_rate"):
                    return float(sampling.frame_rate)
    raise ValueError(
        "No frame rate found in acquisition.json. "
        "Check that ImagingConfig.sampling_strategy.frame_rate is set."
    )


def get_fov_scale_factor(acquisition: Acquisition) -> float:
    """Extract xy scale factor (um/pixel) from PlanarImage transform.

    In v2, fov_scale_factor moved from ophys_fovs[].fov_scale_factor
    to PlanarImage.image_to_acquisition_transform -> Scale.scale[0].

    Parameters
    ----------
    acquisition : Acquisition
        Validated Acquisition model.

    Returns
    -------
    float
        Scale factor in um/pixel.

    Raises
    ------
    ValueError
        If no Scale transform is found in any PlanarImage.
    """
    for ds in acquisition.data_streams:
        for config in ds.configurations:
            if isinstance(config, ImagingConfig):
                for image in config.images:
                    if isinstance(image, PlanarImage):
                        for transform in (
                            image.image_to_acquisition_transform or []
                        ):
                            if isinstance(transform, Scale):
                                return float(transform.scale[0])
    raise ValueError(
        "No FOV scale factor (Scale transform) found in acquisition.json. "
        "Check that PlanarImage.image_to_acquisition_transform contains a Scale."
    )


# ── QC helpers ───────────────────────────────────────────────────────


def pending_qc_status(status: Status = Status.PENDING) -> QCStatus:
    """Create a QCStatus with timezone-aware timestamp.

    Parameters
    ----------
    status : Status
        QC status value (PASS, FAIL, or PENDING).

    Returns
    -------
    QCStatus
        Status with Seattle-timezone-aware timestamp.
    """
    return QCStatus(
        evaluator="Automated",
        status=status,
        timestamp=dt.now(SEATTLE_TZ).isoformat(),
    )


def write_quality_control(metrics: list[QCMetric], output_dir: Path) -> Path:
    """Wrap metrics in QualityControl and write quality_control.json.

    Parameters
    ----------
    metrics : list[QCMetric]
        List of v2 QCMetric objects.
    output_dir : Path
        Directory to write quality_control.json.

    Returns
    -------
    Path
        Path to written file.
    """
    qc = QualityControl(
        metrics=metrics,
        default_grouping=["modality", "stage", ("evaluation",)],
        allow_tag_failures=[],
    )
    filepath = output_dir / "quality_control.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(json.loads(qc.model_dump_json()), f, indent=4)
    return filepath


# ── Processing helpers ───────────────────────────────────────────────


def build_data_process(
    parameters: dict,
    start_time: dt,
    end_time: dt,
    output_path: Optional[str] = None,
    output_parameters: Optional[dict] = None,
) -> DataProcess:
    """Build a v2 DataProcess for this capsule's QC run.

    Parameters
    ----------
    parameters : dict
        Capsule input parameters (from JobSettings).
    start_time : dt
        Timezone-aware start time.
    end_time : dt
        Timezone-aware end time.
    output_path : Optional[str]
        Relative path to output directory.
    output_parameters : Optional[dict]
        Output metrics and parameters summary.

    Returns
    -------
    DataProcess
        Constructed DataProcess object.
    """
    return DataProcess(
        process_type=ProcessName.ANALYSIS,
        name="Movie QC",
        stage=ProcessStage.PROCESSING,
        code=Code(
            url=CODE_URL,
            name="aind-ophys-movie-qc",
            version=os.getenv("CO_CAPSULE_VERSION", ""),
            parameters=parameters,
        ),
        experimenters=[],
        start_date_time=start_time.astimezone(SEATTLE_TZ),
        end_date_time=end_time.astimezone(SEATTLE_TZ),
        output_path=output_path,
        output_parameters=output_parameters,
    )


def write_processing(data_process: DataProcess, output_dir: Path) -> Path:
    """Wrap DataProcess in Processing container and write processing.json.

    Parameters
    ----------
    data_process : DataProcess
        The DataProcess to wrap.
    output_dir : Path
        Directory to write processing.json.

    Returns
    -------
    Path
        Path to written file.
    """
    processing = Processing(data_processes=[data_process])
    filepath = output_dir / "processing.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(json.loads(processing.model_dump_json()), f, indent=4)
    return filepath


# ── High-level capsule output ────────────────────────────────────────


def build_and_write_quality_control(
    unique_id: str,
    metrics: dict,
    seg_img_path: str,
    poisson_img_path: str,
    combined_img_path: str,
    z_drift: bool,
    output_dir: Path,
) -> Path:
    """Build all QC metrics and write quality_control.json.

    Parameters
    ----------
    unique_id : str
        Unique identifier for the recording plane.
    metrics : dict
        Dictionary of all computed metrics.
    seg_img_path : str
        Path to the segmentation image.
    poisson_img_path : str
        Path to the Poisson plot image.
    combined_img_path : str
        Path to save the combined segmentation + Poisson image.
    z_drift : bool
        Whether z-drift metrics were calculated.
    output_dir : Path
        Directory to write quality_control.json.

    Returns
    -------
    Path
        Path to written quality_control.json.
    """
    # Lazy import to avoid circular dependency (qc.py imports from this module)
    from utils.qc import (
        build_epilepsy_metric,
        build_intensity_metric,
        build_photon_metric,
        build_pixel_saturation_metric,
        build_snr_dprime_metric,
        build_zdrift_metric,
        create_combined_metric_image,
    )

    create_combined_metric_image(seg_img_path, poisson_img_path, combined_img_path)

    qc_metrics = [
        build_intensity_metric(unique_id, metrics),
        build_epilepsy_metric(unique_id, metrics),
        build_snr_dprime_metric(metrics),
        build_pixel_saturation_metric(unique_id, metrics),
        build_photon_metric(metrics, combined_img_path),
    ]
    if z_drift:
        qc_metrics.append(build_zdrift_metric(unique_id, metrics))

    return write_quality_control(qc_metrics, output_dir)


def build_and_write_processing(
    parameters: dict,
    start_time: dt,
    output_dir: Path,
) -> Path:
    """Build DataProcess and write processing.json.

    Parameters
    ----------
    parameters : dict
        Capsule input parameters (from JobSettings.model_dump()).
    start_time : dt
        Timezone-aware start time of the capsule run.
    output_dir : Path
        Directory to write processing.json.

    Returns
    -------
    Path
        Path to written processing.json.
    """
    data_process = build_data_process(
        parameters=parameters,
        start_time=start_time,
        end_time=dt.now(timezone.utc),
    )
    return write_processing(data_process, output_dir)
