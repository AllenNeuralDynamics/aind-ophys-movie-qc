import argparse
import json
import os
from datetime import datetime as dt
from pathlib import Path
from typing import Union
import shutil

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status, QCEvaluation, Modality, Stage
from oasis.functions import deconvolve as oasis_deconvolve

from scipy import ndimage
from scipy.linalg import LinAlgError
from scipy.stats import gaussian_kde
from skimage import filters, measure
from local_z_stack import LocalZStack

def save_qc_evaluation_to_file(evaluation: QCEvaluation, output_dir: Path, filename: str) -> None:
    """Save a QC evaluation to a JSON file.
    
    Parameters
    ----------
    evaluation : QCEvaluation
        The QC evaluation to save
    output_dir : Path
        The output directory
    filename : str
        The filename (without extension) to save to
        
    Returns
    -------
    None
    """
    
    filepath = output_dir / f"{filename}_evaluation.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(json.loads(evaluation.model_dump_json()), f, indent=4)


def save_qc_metric_to_file(metric: QCMetric, output_dir: Path, filename: str) -> None:
    """Save a QC metric to a JSON file."""
    filepath = output_dir / f"{filename}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)


def write_qc_evaluation(output_dir: Path, unique_id: str, metrics: dict) -> None:
    """Write QC evaluations grouped by functional purpose.

    Parameters
    ----------
    output_dir: Path
        output directory
    unique_id: str
        unique_id number
    metrics: dict
        dictionary containing all calculated metrics

    Returns
    -------
    None
    """

    # 1. Intensity Change Evaluation
    intensity_change = metrics.get("percent_change_intensity", 0.0)
    if abs(intensity_change) >= 20:
        status = Status.FAIL
    elif abs(intensity_change) >= 10:
        status = Status.PENDING
    else:
        status = Status.PASS
        
    intensity_metric = QCMetric(
        name=f"{unique_id} Intensity Change",
        description="Percent change in intensity from start to end of movie",
        reference=str(f"{unique_id}/movie_qc/{unique_id}_registered_physio_intensity_plot.png"),
        value=float(intensity_change)
    )
    
    intensity_evaluation = QCEvaluation(
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        name=f"{unique_id} Intensity Change",
        description="Analysis of intensity changes throughout the movie",
        allow_failed_metrics=False,
        metrics=[intensity_metric]
    )
    save_qc_evaluation_to_file(intensity_evaluation, output_dir, f"{unique_id}_intensity_change")

    # 2. Epilepsy Probability Evaluation
    epilepsy_prob = metrics.get("epilepsy_probability", 0)
    if epilepsy_prob > 0:
        epilepsy_status = Status.FAIL
    else:
        epilepsy_status = Status.PASS
        
    epilepsy_metric = QCMetric(
        name=f"{unique_id} Epilepsy Probability",
        description="Automated assessment of epileptic activity probability",
        reference=str(f"{unique_id}/movie_qc/{unique_id}_registered_epilepsy_probability.png"),
        status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=epilepsy_status)],
        value=float(epilepsy_prob), 
    )

    epilepsy_evaluation = QCEvaluation(
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        name=f"{unique_id} Epilepsy Probability",
        description="Detection of potential epileptic activity in the recording",
        allow_failed_metrics=False,
        metrics=[epilepsy_metric]
    )
    save_qc_evaluation_to_file(epilepsy_evaluation, output_dir, f"{unique_id}_epilepsy_probability")

    # 3. SNR (Signal-to-Noise Ratio) Evaluation
    snr_metrics = []
    
    snr_mean = metrics.get("simple_snr_mean", 0.0)
        
    snr_metrics.append(QCMetric(
        name=f"{unique_id} SNR Mean",
        description="Mean signal-to-noise ratio across frames",
        value=float(snr_mean),
        status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
    ))

    snr_median = metrics.get("simple_snr_med", 0.0)
        
    snr_metrics.append(QCMetric(
        name=f"{unique_id} SNR Median",
        description="Median signal-to-noise ratio across frames",
        value=float(snr_median),
        status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
    ))

    snr_metrics.append(QCMetric(
        name=f"{unique_id} SNR Standard Deviation",
        description="Standard deviation of signal-to-noise ratio across frames",
        value=float(metrics.get("simple_snr_std", 0.0)),
        status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
    ))
    
    snr_evaluation = QCEvaluation(
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        name=f"{unique_id} SNR Analysis",
        description="Signal-to-noise ratio metrics for image quality assessment",
        allow_failed_metrics=False,
        metrics=snr_metrics
    )
    save_qc_evaluation_to_file(snr_evaluation, output_dir, f"{unique_id}_snr_analysis")

    # 4. Saturated Pixels Evaluation
    saturated_pixels = metrics.get("nb_saturated_pixels", 0)
    if saturated_pixels >= 1000:
        sat_status = Status.FAIL
    elif saturated_pixels >= 800:
        sat_status = Status.PENDING
    else:
        sat_status = Status.PASS
        
    saturated_metric = QCMetric(
        name=f"{unique_id} Saturated Pixels",
        description="Number of saturated pixels in the movie",
        value=int(saturated_pixels),
        status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=sat_status)],
    )
    
    saturated_evaluation = QCEvaluation(
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        name=f"{unique_id} Saturated Pixels",
        description="Analysis of pixel saturation in the recording",
        allow_failed_metrics=False,
        metrics=[saturated_metric]
    )
    save_qc_evaluation_to_file(saturated_evaluation, output_dir, f"{unique_id}_saturated_pixels")

    # 5. Low Intensity Pixels Evaluation
    low_intensity_metrics = []
    
    low_pixels = metrics.get("nb_low_pixels", 0)
    low_intensity_metrics.append(QCMetric(
        name=f"{unique_id} Low Intensity Pixels",
        description="Number of pixels with low intensity values",
        value=int(low_pixels),
        status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
    ))

    low_pixels_perc = metrics.get("nb_low_pixels_perc", 0.0)

    low_intensity_metrics.append(QCMetric(
        name=f"{unique_id} Low Intensity Pixels Percentage",
        description="Percentage of pixels with low intensity values",
        value=float(low_pixels_perc),
        status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
    ))
    
    low_intensity_evaluation = QCEvaluation(
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        name=f"{unique_id} Low Intensity Pixels",
        description="Analysis of low intensity pixels in the recording",
        allow_failed_metrics=False,
        metrics=low_intensity_metrics
    )
    save_qc_evaluation_to_file(low_intensity_evaluation, output_dir, f"{unique_id}_low_intensity_pixels")

    # 6. Intensity Percentiles Evaluation
    percentile_metrics = []
    
    percentile_metrics.append(QCMetric(
        name=f"{unique_id} Intensity Percentile Lower",
        description="Lower percentile (5th) of pixel intensities",
        value=float(metrics.get("percentile_lower", 0.0)),
        status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
    ))

    percentile_metrics.append(QCMetric(
        name=f"{unique_id} Intensity Percentile Upper",
        description="Upper percentile (95th) of pixel intensities",
        value=float(metrics.get("percentile_upper", 0.0)),
        status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
    ))
    
    percentile_evaluation = QCEvaluation(
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        name=f"{unique_id} Intensity Percentiles",
        description="Statistical analysis of pixel intensity distribution",
        allow_failed_metrics=False,
        metrics=percentile_metrics
    )
    save_qc_evaluation_to_file(percentile_evaluation, output_dir, f"{unique_id}_intensity_percentiles")

    # 7. ROI Detection Evaluation
    roi_metrics = []
    
    nb_rois = metrics.get("nb_rois", 0)
    if nb_rois < 1:
        roi_status = Status.FAIL
    elif nb_rois < 5:
        roi_status = Status.PENDING
    else:
        roi_status = Status.PASS
        
    roi_metrics.append(QCMetric(
        name=f"{unique_id} Number of ROIs",
        description="Number of regions of interest detected by segmentation",
        value=int(nb_rois),
        reference=str(f"{unique_id}/movie_qc/{unique_id}_registered_basic_segmentation_image.png"),
        status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=roi_status)],
    ))

    roi_metrics.extend([
        QCMetric(
            name=f"{unique_id} Mean ROI Intensity",
            description="Mean intensity across all detected ROIs",
            value=float(metrics.get("mean_rois_intensity", 0.0)),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f"{unique_id} ROI Intensity Standard Deviation",
            description="Standard deviation of intensities across ROIs",
            value=float(metrics.get("std_rois_intensity", 0.0)),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f"{unique_id} ROI Segmentation Threshold",
            description="Otsu threshold used for ROI segmentation",
            value=float(metrics.get("rois_threshold", 0.0)),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f"{unique_id} Median ROI Sum Intensity",
            description="Median of summed intensities across all ROIs",
            value=float(metrics.get("median_sum_rois_intensity", 0.0)),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f"{unique_id} Median ROI Neuropil Sum",
            description="Median of summed neuropil intensities across all ROIs",
            value=float(metrics.get("median_sum_rois_neuropil", 0.0)),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        )
    ])
    
    roi_evaluation = QCEvaluation(
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        name=f"{unique_id} ROI Detection",
        description="Analysis of region of interest detection and segmentation quality",
        allow_failed_metrics=False,
        metrics=roi_metrics
    )
    save_qc_evaluation_to_file(roi_evaluation, output_dir, f"{unique_id}_roi_detection")

    # 8. Photon Properties Evaluation
    photon_props_metrics = []
    
    photon_gain = metrics.get("photon_gain")

    photon_props_metrics.extend([
        QCMetric(
            name=f"{unique_id} Photon Gain",
            description="Photon gain parameter from Poisson noise analysis",
            value=float(photon_gain),
            reference=str(f"{unique_id}/movie_qc/{unique_id}_registered_physio_poisson_plot.png"),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f"{unique_id} Photon Offset",
            description="Photon offset parameter from Poisson noise analysis",
            value=float(metrics.get("photon_offset")),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f"{unique_id} Background Noise",
            description="Background noise level in photon units",
            value=float(metrics.get("background_noise")),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f"{unique_id} Photon Flux Median",
            description="Median photon flux per pixel per frame",
            value=float(metrics.get("photon_flux_median_per_pixel_per_frame")),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        )
    ])
    
    photon_props_evaluation = QCEvaluation(
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        name=f"{unique_id} Photon Properties",
        description="Analysis of photon-related parameters and noise characteristics",
        allow_failed_metrics=False,
        metrics=photon_props_metrics,
    )
    save_qc_evaluation_to_file(photon_props_evaluation, output_dir, f"{unique_id}_photon_properties")

    # 9. Photon Counts per ROI Evaluation
    roi_photon_metrics = []
    
    mean_photons_per_roi = metrics.get("mean_photons_per_roi_per_frame")
    if mean_photons_per_roi < 50:
        roi_photon_status = Status.FAIL
    elif mean_photons_per_roi < 100:
        roi_photon_status = Status.PENDING
    else:
        roi_photon_status = Status.PASS
        
    roi_photon_metrics.extend([
        QCMetric(
            name=f"{unique_id} Mean Photons per ROI per Frame",
            description="Mean number of photons per ROI per frame",
            value=float(mean_photons_per_roi),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=roi_photon_status)],
        ),
        QCMetric(
            name=f"{unique_id} Std Photons per ROI per Frame",
            description="Standard deviation of photons per ROI per frame",
            value=float(metrics.get("std_photons_per_roi_per_frame")),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f"{unique_id} Mean Photons per ROI per Second",
            description="Mean number of photons per ROI per second",
            value=float(metrics.get("mean_photons_per_roi_per_s")),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f"{unique_id} Std Photons per ROI per Second",
            description="Standard deviation of photons per ROI per second",
            value=float(metrics.get("std_photons_per_roi_per_s")),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        )
    ])
    
    roi_photon_evaluation = QCEvaluation(
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        name=f"{unique_id} Photon Counts per ROI",
        description="Analysis of photon counts within regions of interest",
        allow_failed_metrics=False,
        metrics=roi_photon_metrics
    )
    save_qc_evaluation_to_file(roi_photon_evaluation, output_dir, f"{unique_id}_photon_counts_roi")

    # 10. Photon Counts per Neuropil Evaluation
    neuropil_photon_metrics = []
    
    neuropil_photons = metrics.get("mean_photons_per_neuropil_per_s")

    neuropil_photon_metrics.extend([
        QCMetric(
            name=f"{unique_id} Mean Photons per Neuropil per Second",
            description="Mean number of photons per neuropil region per second",
            value=float(neuropil_photons),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f"{unique_id} Std Photons per Neuropil per Second",
            description="Standard deviation of photons per neuropil region per second",
            value=float(metrics.get("std_photons_per_neuropil_per_s")),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        )
    ])
    
    neuropil_photon_evaluation = QCEvaluation(
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        name=f"{unique_id} Photon Counts per Neuropil",
        description="Analysis of photon counts in neuropil regions",
        allow_failed_metrics=False,
        metrics=neuropil_photon_metrics
    )
    save_qc_evaluation_to_file(neuropil_photon_evaluation, output_dir, f"{unique_id}_photon_counts_neuropil")

    # 11. Event Detection (D-prime) Evaluation
    dprime_metrics = []
    
    median_dprime = metrics.get("median_rois_dprime", 0.0)
        
    dprime_metrics.extend([
        QCMetric(
            name=f"{unique_id} Median ROI D-prime",
            description="Median d-prime value for event detection across all ROIs",
            value=float(median_dprime),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f"{unique_id} ROI D-prime Standard Deviation",
            description="Standard deviation of d-prime values for event detection across ROIs",
            value=float(metrics.get("std_rois_dprime", 0.0)),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        )
    ])
    
    dprime_evaluation = QCEvaluation(
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        name=f"{unique_id} Event Detection Performance",
        description="Analysis of event detection capability using d-prime metrics",
        allow_failed_metrics=False,
        metrics=dprime_metrics
    )
    save_qc_evaluation_to_file(dprime_evaluation, output_dir, f"{unique_id}_event_detection")

    print(f"Successfully created 11 QC evaluation groups with _aggregate.json suffix for: {unique_id}")

    # 12. Z-drift Evaluation
    zdrift_metrics = []
    zdrift = metrics.get("zdrift", 0)
    if zdrift == 0:
        zdrift_status = Status.PENDING
    else:
        zdrift_um = float(zdrift['z_drift_um'])
        if abs(zdrift_um) <= 10:  # Low drift TODO: expose this threshold somewhere
            zdrift_status = Status.PASS
        else:
            zdrift_status = Status.FAIL
    
    zdrift_metrics.append(QCMetric(
        name=f"{unique_id} Z-drift um",
        description="Analysis of z-drift in the recording",
        value=zdrift_um,
        status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=zdrift_status)],
    ))

    zdrift_metrics.extend([
        QCMetric(
            name=f'{unique_id} start_frame in local z-stack',
            description="# of frame in local z-stack matched to the start of the movie",
            value=int(zdrift['start_frame']),
            reference=str(f"{unique_id}/movie_qc/{unique_id}_registered_zdrift_start.png"),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f'{unique_id} end_frame in local z-stack',
            description="# of frame in local z-stack matched to the end of the movie",
            value=int(zdrift['end_frame']),
            reference=str(f"{unique_id}/movie_qc/{unique_id}_registered_zdrift_end.png"),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f'{unique_id} z-drift frame',
            description="Calculated z-drift in # of frames of the local z-stack",
            value=int(zdrift['z_drift_frame']),
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f'{unique_id} start correlation - peak',
            description="Peak of the correlation between the start image and the local z-stack frames",
            value=zdrift['start_frame_corr'],
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f'{unique_id} end correlation - peak',
            description="Peak of the correlation between the end image and the local z-stack frames",
            value=zdrift['end_frame_corr'],
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f'{unique_id} start correlation',
            description="Correlation between the start image and the local z-stack frames",
            value=zdrift['start_corr'],
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        ),
        QCMetric(
            name=f'{unique_id} end correlation',
            description="Correlation between the end image and the local z-stack frames",
            value=zdrift['end_corr'],
            status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)],
        )
    ])

    zdrift_evaluation = QCEvaluation(
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        name=f"{unique_id} Z-drift Analysis",
        description="Analysis of z-drift in the recording",
        allow_failed_metrics=False,
        metrics=zdrift_metrics
    )

    save_qc_evaluation_to_file(zdrift_evaluation, output_dir, f"{unique_id}_z_drift")


def get_and_plot_epilepsy_probability(
    cropped_video: np.ndarray,
    frame_rate: float,
    signal_threshold: float = 10.0,
    min_width: float = 0.1,
    max_width: float = 0.3,
):
    """Get the probability of epilepsy in this physio movie.

    Parameters:
    -----------
    cropped_video: np.ndarray
        The video to analyze
    frame_rate: float
        Frames per second of the signal
    signal_threshold: float
        A bound above which events are tagged as epileptic events
    min_width: float
        Lower bound on the widths within which events are tagged as epileptic
    max_width: float
        Upper bound on the widths within which events are tagged as epileptic


    Returns
    -------
    float
        the probability of epilepsy.
    matplotlib.figure.Figure
        A figure showing the epilepsy plot.
    """

    avg_frames = np.mean(cropped_video, axis=(1, 2)).astype("double")

    perc_diff = -100 * (1 - avg_frames / avg_frames.mean())
    denoised_signal, spike_events, _, _, _ = oasis_deconvolve(perc_diff, penalty=1)

    if set(denoised_signal) == {0.0}:  # No signal
        return 0

    prominence_data, widths = get_spike_prominence_and_width(
        denoised_signal, spike_events, frame_rate=frame_rate
    )
    nb_epileptic_events = len(
        prominence_data[
            (prominence_data > signal_threshold)
            & (widths > min_width)
            & (widths < max_width)
        ]
    )

    fig, axs = plt.subplots(2, 1)

    # Calculate the point density
    xy = np.vstack([widths, prominence_data])
    try:
        z = gaussian_kde(xy)(xy)
    except (LinAlgError, ValueError):
        raise Exception("Unable to form plot for epilepsy - check video for anomalies.")

    plt.sca(axs[0])
    plt.scatter(widths, prominence_data, c=z, s=10)
    plt.xlabel("Event Width (s)")
    plt.xlim(0, 1)
    plt.ylabel("Event Size")

    plt.sca(axs[1])
    plt.plot(perc_diff, "b-")
    plt.xlabel("Frame number")
    plt.ylabel("DFF")

    plt.tight_layout()
    return float(nb_epileptic_events) / len(prominence_data), fig


def get_spike_prominence_and_width(
    denoised_signal: np.ndarray,
    spike_events: np.ndarray,
    frame_rate: float,
    spike_threshold: float = 0.05,
):
    """Get the local prominence and width of each spike in a calcium trace.

    Local prominence is a measure of the strength of the spike relative to other spikes
    nearest to it.

    Parameters
    ----------
    denoised_signal: np.ndarray
        The denoised fluorescence signal
    spike_events: np.ndarray
        Discretized deconvolved neural activity (spikes)
    frame_rate: float
        Frames per second of the signal, used to convert width to seconds
    spike_threshold: float
        A bound below which spikes will be thrown out

    Returns
    -------
    np.ndarray
        An array of prominences
    np.ndarray
        An array of widths
    """

    event_idxs = np.where(spike_events > spike_threshold)[0]
    filtered_signal = denoised_signal[event_idxs]

    _prominences = []
    _widths = []

    for idx, event_idx in enumerate(event_idxs):
        # Prominence calculation
        local_amplitude = denoised_signal[event_idx]
        local_diff = local_amplitude - filtered_signal
        larger_spike_idxs = event_idxs[local_diff < 0]

        pre_spike_idxs = larger_spike_idxs[larger_spike_idxs < event_idx]
        if not pre_spike_idxs.any():
            pre_spike_idx = 0
        else:
            pre_spike_idx = np.max(
                pre_spike_idxs
            )  # Nearest spike larger than local spike on left

        # Minimum signal between prev large spike and current spike
        pre_spikes = denoised_signal[pre_spike_idx:event_idx]
        pre_min = np.min(pre_spikes)

        post_peak_idxs = larger_spike_idxs[larger_spike_idxs > event_idx]
        if not post_peak_idxs.any():
            post_peak_idx = len(denoised_signal)
        else:
            post_peak_idx = np.min(
                post_peak_idxs
            )  # Nearest spike larger than local spike on right

        # Minimum signal between post large spike and current spike
        post_spikes = denoised_signal[event_idx:post_peak_idx]
        post_min = np.min(post_spikes)

        local_min = np.max([pre_min, post_min])  # TODO: Should be min here?
        local_prominence = local_amplitude - local_min
        _prominences.append(local_prominence)

        # Width calculation
        pre_half_idxs = np.where(pre_spikes < local_amplitude - local_prominence / 2)[0]
        if not pre_half_idxs.any():
            pre_half_idx = len(pre_spikes)
        else:
            pre_half_idx = np.min(len(pre_spikes) - pre_half_idxs)

        post_half_idxs = np.where(post_spikes < local_amplitude - local_prominence / 2)[0]
        if not post_half_idxs.any():
            post_half_idx = len(post_spikes)
        else:
            post_half_idx = np.min(post_half_idxs)

        local_width = post_half_idx + pre_half_idx
        _widths.append(local_width)

    return np.array(_prominences), np.array(_widths) / frame_rate


def plot_intensity_histogram(
    cropped_video: np.ndarray, min_range: int, max_range: int, log_scale: bool = True
):
    """Obtain a plot of the intensity histogram.

    Parameters
    ----------
    cropped_video: np.ndarray
        The video to plot
    min_range: int
        The minimum pixel value to consider
    max_range: int
        The maximum pixel value to consider
    log_scale: bool
        Whether to plot the histogram on a log scale

    Returns
    -------
    matplotlib.figure.Figure
        A figure showing the intensity histogram
    """

    fig = plt.figure()
    cropped_video = cropped_video.flatten().copy()
    plt.hist(
        cropped_video,
        bins=100,
        range=(min_range, max_range),
        density=True,
        log=log_scale,
        histtype="bar",
        ec="black",
    )
    plt.xlabel("Pixel Value")
    plt.ylabel("pdf")
    return fig


def plot_projection_image(
    cropped_video: np.ndarray, min_range: int = 1, max_range: int = 99
):
    """Create a projection image of the video.

    Parameters
    ----------
    cropped_video: np.ndarray
        The video to plot
    min_range: int
        The minimum pixel value to consider
    max_range: int
        The maximum pixel value to consider

    Returns
    -------
    matplotlib.figure.Figure
        A figure showing the projection image
    """

    fig = plt.figure()
    image_project = np.mean(cropped_video, axis=0)
    list_pixel_limits = np.percentile(image_project.flatten(), [min_range, max_range])
    plt.imshow(
        image_project, cmap="gray", vmin=list_pixel_limits[0], vmax=list_pixel_limits[1]
    )
    plt.axis("off")
    return fig


def get_and_plot_basic_segmentation(
    cropped_video: np.ndarray,
    min_object_size: int = 100,
    max_object_size: int = 300,
    sigma_segmentation: int = 30,
):
    """Get basic segmentation data and plot the results.

    Parameters
    ----------
    cropped_video: np.ndarray
        The video to analyze
    min_object_size: int
        The minimum size of an object to consider
    max_object_size: int
        The maximum size of an object to consider
    sigma_segmentation: int
        The sigma value to use in the segmentation

    Returns
    -------
    dict
        A dictionary of ROI data
    matplotlib.figure.Figure
        A figure showing the basic segmentation
    """

    # We use a median to remove calcium events for F0
    background_image = np.median(cropped_video, axis=0)

    # We do a high pass to remove background fluctuations and flatten the image.
    sigma = sigma_segmentation
    neuropil = ndimage.gaussian_filter(background_image, sigma=sigma)
    neuropil_substracted = background_image - neuropil

    otsu_threshold = filters.threshold_otsu(neuropil_substracted)
    binary_image = neuropil_substracted > otsu_threshold

    figure = plt.figure()
    list_pixel_limits = np.percentile(neuropil_substracted.flatten(), [1, 99])
    plt.imshow(
        neuropil_substracted,
        cmap="gray",
        vmin=list_pixel_limits[0],
        vmax=list_pixel_limits[1],
    )
    plt.axis("off")

    # Label connected components and filter objects larger than 100 pixels
    label_image = measure.label(binary_image)
    roi_intensities = []
    roi_sum = []
    neuropil_sum = []

    roi_data = {}
    roi_data["rois_threshold"] = otsu_threshold

    for region in measure.regionprops(label_image):
        if region.area > min_object_size and region.area < max_object_size:
            y1, x1, y2, x2 = region.bbox
            segmented_area_pixels = neuropil_substracted[y1:y2, x1:x2]
            segmented_area_pixels_neuropil = neuropil[y1:y2, x1:x2]

            local_mean = np.mean(segmented_area_pixels)
            roi_intensities.append(local_mean)
            roi_sum.append(np.sum(segmented_area_pixels))
            neuropil_sum.append(np.sum(segmented_area_pixels_neuropil))

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=1
            )
            plt.gca().add_patch(rect)

    roi_data["nb_rois"] = len(roi_intensities)
    roi_data["mean_rois_intensity"] = np.mean(roi_intensities)
    roi_data["std_rois_intensity"] = np.std(roi_intensities)
    roi_data["all_rois_intensity"] = roi_intensities
    roi_data["sum_rois_intensity"] = roi_sum
    roi_data["median_sum_rois_intensity"] = np.median(roi_sum)
    roi_data["sum_rois_neuropil"] = neuropil_sum
    roi_data["median_sum_rois_neuropil"] = np.median(neuropil_sum)

    return roi_data, figure


def plot_poisson_curve(photon_gain_parameters: dict) -> plt.Figure:
    """
    Obtain a plot showing Poisson characteristics of the signal.

    Parameters
    ----------
    photon_gain_parameters : dict
        A dictionary of parameters related to the photon gain.

    Returns
    -------
    matplotlib.figure.Figure
        A figure showing the Poisson characteristics of the signal.
    """

    h, xedges, yedges = np.histogram2d(
        photon_gain_parameters["var"], photon_gain_parameters["mean"], bins=(200, 200)
    )
    extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

    fig = plt.figure()
    plt.imshow(h, origin="lower", extent=extent, aspect="auto", cmap="Blues")
    plt.colorbar()
    plt.xlabel("Mean")
    plt.ylabel("Variance")
    plt.xlim(photon_gain_parameters["mean"].min(), photon_gain_parameters["mean"].max())
    plt.ylim(photon_gain_parameters["var"].min(), photon_gain_parameters["var"].max())

    mean_range = np.linspace(0, photon_gain_parameters["mean"].max(), num=200)
    background_noise_mean = (
        -photon_gain_parameters["photon_offset"] / photon_gain_parameters["photon_gain"]
    )
    plt.tight_layout()
    plt.plot(
        mean_range,
        photon_gain_parameters["photon_gain"] * (mean_range - background_noise_mean),
        "r",
    )

    return fig


def plot_avg_intensity_progression(cropped_video: np.ndarray, binning: int) -> plt.Figure:
    """
    Obtain a plot showing the average intensity over time.

    Parameters
    ----------
    cropped_video : np.ndarray
        The video to plot
    binning : int
        The amount of binning associated with cropped_video

    Returns
    -------
    matplotlib.figure.Figure
        A figure showing the average intensity progression.
    """
    fig = plt.figure()

    y_data = np.mean(cropped_video, axis=(1, 2))
    x_vals = list(binning * np.arange(0, y_data.shape[0]))
    plt.plot(x_vals, y_data, "b-")
    plt.xlabel("Frame #")
    plt.ylabel("Fluorescence Value")
    plt.title("physio_intensity_plot")
    plt.tight_layout()

    return fig


def get_saturation_metrics(
    cropped_video: np.ndarray, min_pixel_range: int, max_pixel_range: int
) -> dict:
    """
    Return the number of pixels falling outside the usual dynamic range.

    Parameters
    ----------
    cropped_video : np.ndarray
        The video to analyze
    min_pixel_range : int
        The minimum pixel value to consider
    max_pixel_range : int
        The maximum pixel value to consider

    Returns
    -------
    dict
        A dictionary of saturation metrics.
    """
    max_plane = cropped_video.max(axis=0).flatten()
    nb_saturated_pixels = int((max_plane >= max_pixel_range).sum())

    nb_undersat_pixels = (cropped_video.flatten() <= min_pixel_range).sum()
    nb_undersat_pixels = nb_undersat_pixels / float(cropped_video.shape[0])
    nb_undersat_pixels_perc = (
        100 * nb_undersat_pixels / float(cropped_video.shape[1] * cropped_video.shape[2])
    )

    return {
        "nb_saturated_pixels": nb_saturated_pixels,
        "nb_low_pixels": int(nb_undersat_pixels),
        "nb_low_pixels_perc": nb_undersat_pixels_perc,
    }


def get_percent_change_intensity(
    start_section: np.ndarray, end_section: np.ndarray
) -> float:
    """
    Return the percent change in total intensity throughout the physio movie.

    Parameters
    ----------
    start_section : np.ndarray
        The beginning section of the movie to use.
    end_section : np.ndarray
        The end section of the movie to use.

    Returns
    -------
    float
        A float corresponding to the loss in total intensity as a percentage.
    """
    start_mean = np.mean(start_section, axis=(0, 1, 2))
    end_mean = np.mean(end_section, axis=(0, 1, 2))

    percent_change_intensity = 100 * (end_mean - start_mean) / start_mean
    return percent_change_intensity


def get_simple_snr_metrics(cropped_video: np.ndarray) -> dict:
    """
    Compute metrics related to the signal to noise level for the images.

    Parameters
    ----------
    cropped_video : np.ndarray
        The video to analyze

    Returns
    -------
    dict
        A dictionary of metrics
    """

    # Simple SNR - https://en.wikipedia.org/wiki/Signal-to-noise_ratio_(imaging)
    simple_snr_array = cropped_video.mean(axis=(1, 2)) / cropped_video.std(axis=(1, 2))

    return {
        "simple_snr_mean": simple_snr_array.mean(),
        "simple_snr_med": np.median(simple_snr_array),
        "simple_snr_std": simple_snr_array.std(),
    }


def get_percentile_metrics(
    cropped_video: np.ndarray, perc_min: int, perc_max: int
) -> dict:
    """
    Compute metrics related to the range of the signal.

    Parameters
    ----------
    cropped_video : np.ndarray
        The video to analyze
    perc_min : int
        Minimum percentile value for filtering
    perc_max : int
        Maximum percentile value for filtering

    Returns
    -------
    dict
        A dictionary of percentile parameters.
    """
    return {
        "percentile_lower": np.percentile(cropped_video, perc_min),
        "percentile_upper": np.percentile(cropped_video, perc_max),
    }


def subsample_and_crop_video(
    data_pointer: Union[h5py._hl.dataset.Dataset, np.ndarray],
    subsample: int,
    crop: tuple[int, int],
    start_frame: int = 0,
    end_frame: int = -1,
) -> np.ndarray:
    """
    Subsample and crop a video, cache results. Also functions as a data_pointer load.

    Parameters
    ----------
    data_pointer : h5py._hl.dataset.Dataset or np.ndarray
        The data pointer to the video data.
    subsample : int
        An integer specifying the amount of subsampling (1 = full movie).
    crop : tuple[int, int]
        A tuple (px_y, px_x) specifying the number of pixels to remove.
    start_frame : int, optional
        The index of the first desired frame, by default 0.
    end_frame : int, optional
        The index of the last desired frame, by default -1.

    Returns
    -------
    np.ndarray
        The resultant array.
    """

    if not isinstance(data_pointer, (h5py._hl.dataset.Dataset, np.ndarray)):
        raise NotImplementedError("Data pointer must be a h5py dataset or numpy array.")
    else:
        _shape = data_pointer.shape
        px_y_start, px_x_start = crop
        px_y_end = _shape[1] - px_y_start
        px_x_end = _shape[2] - px_x_start

        if start_frame == _shape[0] - 1 and (end_frame == -1 or end_frame == _shape[0]):
            cropped_video = data_pointer[
                start_frame::subsample, px_y_start:px_y_end, px_x_start:px_x_end
            ]
        else:
            cropped_video = data_pointer[
                start_frame:end_frame:subsample,
                px_y_start:px_y_end,
                px_x_start:px_x_end,
            ]

    return cropped_video


def convert_intensity_into_photon_flux(
    intensities: np.ndarray, photon_offset: float, photon_gain: float
) -> np.ndarray:
    """
    Convert intensity values into photon flux.

    Parameters
    ----------
    intensities : np.ndarray
        The intensity values to convert.
    photon_offset : float
        The offset value for the photon conversion.
    photon_gain : float
        The gain value for the photon conversion.

    Returns
    -------
    np.ndarray
        The converted photon flux values.
    """
    background_noise_mean = -photon_offset / photon_gain
    photon_flux = (intensities - background_noise_mean) / photon_gain

    return photon_flux


def get_photon_gain_parameters(
    cropped_video: np.ndarray,
    max_pixel_range: int,
    perc_min: int = 3,
    perc_max: int = 90
) -> dict:
    """Photon Gain.

    Compute a variety of parameters related to the physio signal's gain.

    Args:
        cropped_video (np.ndarray): The video to analyze.
        max_pixel_range (int): The maximum pixel value to consider.
        perc_min (int): Min value between 0-100 used in filtering based on percentile.
        perc_max (int): Max value between 0-100 used in filtering based on percentile.

    Returns:
        dict: A dictionary of parameters related to the physio signal.
        Useful in making plots and metrics.
    """

    # Remove saturated pixels
    idxs_not_saturated = np.where(cropped_video.max(axis=0).flatten() < max_pixel_range)

    _var = cropped_video.var(axis=0).flatten()[idxs_not_saturated]
    _mean = cropped_video.mean(axis=0).flatten()[idxs_not_saturated]

    # Remove pixels that deviate from Poisson stats
    _var_scale = np.percentile(_var, [perc_min, perc_max])
    _mean_scale = np.percentile(_mean, [perc_min, perc_max])

    # Remove outliers
    _var_bool = np.logical_and(_var > _var_scale[0], _var < _var_scale[1])
    _mean_bool = np.logical_and(_mean > _mean_scale[0], _mean < _mean_scale[1])
    _no_outliers = np.logical_and(_var_bool, _mean_bool)

    _var_filt = _var[_no_outliers]
    _mean_filt = _mean[_no_outliers]
    _mat = np.vstack([_mean_filt, np.ones(len(_mean_filt))]).T

    try:
        slope, offset = np.linalg.lstsq(_mat, _var_filt)[0]
    except LinAlgError:
        raise Exception("Unable to get photon metrics - check video for anomalies.")

    background_noise_mean = -offset / slope

    photon_per_pixel_per_frame = convert_intensity_into_photon_flux(
        cropped_video.flatten(), offset, slope
    )

    return {
        "var": _var_filt,
        "mean": _mean_filt,
        "photon_gain": slope,
        "photon_offset": offset,
        "photon_flux_median_per_pixel_per_frame": np.median(photon_per_pixel_per_frame),
        "all_pixels_photon_per_pixel_per_frame": photon_per_pixel_per_frame,
        "background_noise": background_noise_mean,
    }


def get_dprime_indicator(
    baseline_photon_flux: np.ndarray,
    neuropil_photon_flux: np.ndarray,
    decay_time: float,
    dff_single_event_size: float,
) -> np.ndarray:
    """
    Calculate the d-prime indicator for event detection.

    This original formula was introduced in Wilt et al, 2013.
    Default values of decay_time and dff_single_event_size are for Gcamp6f.
    We improved Wilt et al formula to correct for the presence of non-responsive
    background neuropil, assuming a fixed DF/F given by the indicator. Note that if
    neuropil is null we end up with dff_single_event_size as the correcting factor is 1.

    Parameters
    ----------
    baseline_photon_flux : np.ndarray
        The baseline photon flux values.
    neuropil_photon_flux : np.ndarray
        The neuropil photon flux values.
    decay_time : float
        The estimated 1/2 decay time of the event reporter (s).
    dff_single_event_size : float
        The estimated DF/F event size of the event reporter for single events (au).

    Returns
    -------
    np.ndarray
        The d-prime indicator values.
    """
    corrected_dff_single_event_size = (
        dff_single_event_size
        * baseline_photon_flux
        / (baseline_photon_flux + neuropil_photon_flux)
    )
    d_prime = (
        np.sqrt((baseline_photon_flux + neuropil_photon_flux) * decay_time / 2)
        * corrected_dff_single_event_size
    )

    return d_prime


def save_figure_to_storage(
    figure: plt.Figure,
    storage_path: Path,
    root_filename: str,
    image_name: str,
    dpi: int = 120,
) -> None:
    """Saves QC images to the file system.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to save.
    storage_path : pathlib.Path
        The directory where the figure will be saved.
    root_filename : str
        The root name of the file.
    image_name : str
        The name of the image.
    dpi : int, optional
        The resolution in dots per inch, by default 120.

    Returns
    -------
    None
    """
    if not storage_path.exists():
        storage_path.mkdir(parents=True)
    filepath = storage_path / f"{root_filename}_{image_name}.png"

    figure.savefig(filepath, dpi=dpi)
    plt.close(figure)


def make_output_directory(output_dir: Path, unique_id: str) -> str:
    """Creates the output directory if it does not exist

    Parameters
    ----------
    output_dir: Path
        output directory
    unique_id: str
        unique_id number

    Returns
    -------
    output_dir: str
        output directory
    """
    output_dir = output_dir / unique_id
    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir / "movie_qc"
    output_dir.mkdir(exist_ok=True)

    return output_dir


def parse_args() -> argparse.Namespace:
    """Parse command line arguments

    Returns
    -------
    argparse.Namespace
        parsed arguments
    """
    parser = argparse.ArgumentParser(description="Raw movie QC")
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        help="Regular expression to input hdf5 movie. The first one found is picked",
        default="../data",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="Output directory", default="/results/"
    )

    # This is to constrain the analysis to a subset of frames
    # to reduce cost and increase speed
    parser.add_argument(
        "--start_frame",
        type=int,
        default=1,
        help=("Start of movie block to use for main analysis"),
    )

    parser.add_argument(
        "--end_frame",
        type=int,
        default=10000,
        help=("End of movie block to use for main analysis"),
    )

    parser.add_argument(
        "--crop",
        type=list,
        default=(30, 30),
        help=(
            "cropped area of movie to use for analysis. Useful to remove" " edge effects"
        ),
    )

    parser.add_argument(
        "--max_pixel_range",
        type=int,
        default=8000,
        help=("This is the pixel value above which we consider saturation occurred."),
    )

    parser.add_argument(
        "--min_pixel_range",
        type=int,
        default=0,
        help=(
            "This is the pixel value below which we consider under-saturation occurred."
        ),
    )

    parser.add_argument(
        "--decay_time",
        type=float,
        default=0.2,
        help=("This is the estimated 1/2 decay time of your event reporter (s)."),
    )

    parser.add_argument(
        "--frame_rate",
        type=float,
        default=0,
        help=("Optional argument to provide frame rate if not available."),
    )

    parser.add_argument(
        "--dff_single_event_size",
        type=float,
        default=0.15,
        help=(
            "This is the estimated DF/F event size of your event"
            " reporter for single events (au)."
        ),
    )

    return parser.parse_args()


def write_legacy_movie_qc_metrics(output_dir: Path, unique_id: str, metrics: dict) -> None:
    """Write only the legacy QC metrics currently expected by the aggregator.
    
    This function writes only the 2 specific metrics that the current aggregator's
    create_movie_qc_evaluations function looks for:
    - epilepsy_probability_metric 
    - physio_intensity_plot_metric
    
    Parameters
    ----------
    output_dir: Path
        output directory
    unique_id: str
        unique_id number
    metrics: dict
        dictionary containing all calculated metrics

    Returns
    -------
    None
    """
    
    # Only create the 2 metrics that the current aggregator expects
    
    # 1. Epilepsy probability metric (matches current aggregator pattern)
    epilepsy_prob = metrics.get("epilepsy_probability", 0)
    if epilepsy_prob > 0:
        epilepsy_status = Status.FAIL
    else:
        epilepsy_status = Status.PASS
        
    metric = QCMetric(
        name=f"{unique_id} Epilepsy Probability",
        description="Automated assessment of epileptic activity probability",
        reference=str(f"{unique_id}/movie_qc/{unique_id}_registered_epilepsy_probability.png"),
        status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=epilepsy_status)],
        value=float(epilepsy_prob)
    )
    save_qc_metric_to_file(metric, output_dir, f"{unique_id}_registered_epilepsy_probability_metric")

    # 2. Physio intensity plot metric (matches current aggregator pattern)
    intensity_change = metrics.get("percent_change_intensity", 0.0)
    if abs(intensity_change) >= 20:
        status = Status.FAIL
    elif abs(intensity_change) >= 10:
        status = Status.PENDING
    else:
        status = Status.PASS
        
    metric = QCMetric(
        name=f"{unique_id} Physio Intensity Plot",
        description="Percent change in intensity from start to end of movie",
        reference=str(f"{unique_id}/movie_qc/{unique_id}_registered_physio_intensity_plot.png"),
        value=float(intensity_change),
        status_history=[QCStatus(evaluator="Automated", timestamp=dt.now(), status=status)],
    )
    save_qc_metric_to_file(metric, output_dir, f"{unique_id}_registered_physio_intensity_plot_metric")

    print(f"Successfully created 2 legacy QC metrics for aggregator compatibility: {unique_id}")


if __name__ == "__main__":  # pragma: nocover
    # Create an ArgumentParser object

    start_time = dt.now()

    # Parse command-line arguments
    args = parse_args()
    # General settings

    # name of the dataset in the hdf5 file
    dataset_name = "data"

    output_dir = Path(args.output_dir)
    h5_file = next(Path(args.input_dir).rglob("*registered.h5"))
    unique_id = "_".join(h5_file.name.split("_")[:-1])
    output_dir = make_output_directory(output_dir, unique_id)

    frame_rate = args.frame_rate

    if frame_rate == 0:
        processing_json_fp = next(h5_file.parent.glob("*data_process.json"))
        with open(processing_json_fp, "r") as j:
            data = json.load(j)
        frame_rate = data["parameters"]["movie_frame_rate_hz"]

    with h5py.File(h5_file, "r") as h5_pointer:
        data_pointer = h5_pointer[dataset_name]

        # in principle, this is done once to avoid loading the data multiple times
        # We generate a few smaller version of the big movie to use for metrics
        cropped_video = subsample_and_crop_video(
            data_pointer=data_pointer,
            subsample=1,
            crop=args.crop,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        full_length_binning = 500
        small_cropped_video = cropped_video[0:full_length_binning, :, :]
        full_length_cropped_video = subsample_and_crop_video(
            data_pointer=data_pointer,
            subsample=full_length_binning,
            crop=args.crop,
            start_frame=0,
            end_frame=-1,
        )

        # We extract a few frames from the beginning and end of the movie
        # to use for stability metrics
        ignore_frames = 300
        nb_border_frames_to_avg = 200
        start_section = subsample_and_crop_video(
            data_pointer=data_pointer,
            subsample=1,
            crop=args.crop,
            start_frame=ignore_frames,
            end_frame=nb_border_frames_to_avg + ignore_frames,
        )
        end_section = subsample_and_crop_video(
            data_pointer=data_pointer,
            subsample=1,
            crop=args.crop,
            start_frame=-(ignore_frames + nb_border_frames_to_avg + 1),
            end_frame=-ignore_frames,
        )
        shape = data_pointer.shape

    metrics = {}
    metrics["crops"] = args.crop
    metrics["shape"] = shape
    metrics["percent_change_intensity"] = get_percent_change_intensity(
        start_section, end_section
    )
    metrics["epilepsy_probability"], fig_epilespy = get_and_plot_epilepsy_probability(
        cropped_video, frame_rate=frame_rate
    )

    # We get the input filename without the path
    base_file = os.path.basename(h5_file)
    # We remove extension
    base_file = os.path.splitext(base_file)[0]

    save_figure_to_storage(fig_epilespy, output_dir, base_file, "epilepsy_probability")

    # The following add the dictionary returned by
    # get_simple_snr_metrics to the metrics dictionary
    metrics.update(get_simple_snr_metrics(cropped_video))

    photon_gain_parameters = get_photon_gain_parameters(
        cropped_video, max_pixel_range=args.max_pixel_range
    )

    metrics.update(
        get_saturation_metrics(
            small_cropped_video,
            min_pixel_range=args.min_pixel_range,
            max_pixel_range=args.max_pixel_range,
        )
    )

    metrics.update(get_percentile_metrics(small_cropped_video, perc_min=5, perc_max=95))

    save_figure_to_storage(
        plot_projection_image(small_cropped_video),
        output_dir,
        base_file,
        "mean_projection_image",
    )

    save_figure_to_storage(
        plot_projection_image(start_section),
        output_dir,
        base_file,
        "start_projection_image",
    )

    save_figure_to_storage(
        plot_projection_image(end_section),
        output_dir,
        base_file,
        "end_projection_image",
    )

    rois_data, roi_figure = get_and_plot_basic_segmentation(start_section)

    save_figure_to_storage(roi_figure, output_dir, base_file, "basic_segmentation_image")

    metrics.update(rois_data)
    metrics.update(photon_gain_parameters)

    # We use the photon gain and offset to convert segmented intensities to photon flux
    metrics["all_rois_photons_per_rois_per_frame"] = convert_intensity_into_photon_flux(
        metrics["sum_rois_intensity"], metrics["photon_offset"], metrics["photon_gain"]
    )
    metrics["all_neuropils_photons_per_rois_per_frame"] = (
        convert_intensity_into_photon_flux(
            metrics["sum_rois_neuropil"],
            metrics["photon_offset"],
            metrics["photon_gain"],
        )
    )
    metrics["photon_offset"] = metrics["photon_offset"]
    metrics["photon_gain"] = metrics["photon_gain"]
    metrics["mean_photons_per_roi_per_frame"] = np.mean(
        metrics["all_rois_photons_per_rois_per_frame"]
    )
    metrics["std_photons_per_roi_per_frame"] = np.std(
        metrics["all_rois_photons_per_rois_per_frame"]
    )
    metrics["mean_photons_per_roi_per_s"] = frame_rate * np.mean(
        metrics["all_rois_photons_per_rois_per_frame"]
    )
    metrics["std_photons_per_roi_per_s"] = frame_rate * np.std(
        metrics["all_rois_photons_per_rois_per_frame"]
    )
    metrics["mean_photons_per_neuropil_per_s"] = frame_rate * np.mean(
        metrics["all_neuropils_photons_per_rois_per_frame"]
    )
    metrics["std_photons_per_neuropil_per_s"] = frame_rate * np.std(
        metrics["all_neuropils_photons_per_rois_per_frame"]
    )

    # Convert to dprime for spike detection

    # We first convert photons into photons per second
    all_rois_photons_per_second = (
        metrics["all_rois_photons_per_rois_per_frame"] * frame_rate
    )
    all_neuropils_photons_per_second = (
        metrics["all_neuropils_photons_per_rois_per_frame"] * frame_rate
    )

    metrics["all_rois_dprime"] = get_dprime_indicator(
        all_rois_photons_per_second,
        all_neuropils_photons_per_second,
        args.decay_time,
        args.dff_single_event_size,
    )

    metrics["median_rois_dprime"] = np.median(metrics["all_rois_dprime"])
    metrics["std_rois_dprime"] = np.std(metrics["all_rois_dprime"])

    save_figure_to_storage(
        plot_avg_intensity_progression(
            full_length_cropped_video, binning=full_length_binning
        ),
        output_dir,
        base_file,
        "physio_intensity_plot",
    )

    save_figure_to_storage(
        plot_intensity_histogram(
            cropped_video,
            min_range=args.min_pixel_range,
            max_range=args.max_pixel_range,
        ),
        output_dir,
        base_file,
        "physio_intensity_hist",
    )

    save_figure_to_storage(
        plot_poisson_curve(photon_gain_parameters),
        output_dir,
        base_file,
        "physio_poisson_plot",
    )
    
    # z-drift metrics
    session_json_path = next(Path(args.input_dir).rglob("session.json"))
    zstack_filepath = next(Path(args.input_dir).rglob(f'{unique_id}_z_stack_local.h5'))
    local_zstack = LocalZStack(zstack_filepath=zstack_filepath,
                               physio_filepath=h5_file,
                               session_json_path=session_json_path)
    metrics["zdrift"], save_imgs = local_zstack.get_z_drift() #TODO: expose parameters

    image_segments = ['start', 'end']
    image_types = ['image', 'zstack_plane']
    for image_segment in image_segments:
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        for i, image_type in enumerate(image_types):
            key = f'{image_segment}_{image_type}'
            img = save_imgs[key]
            axes[i].imshow(img, cmap='gray',
                            vmin=np.percentile(img.flatten(), 1),
                            vmax=np.percentile(img.flatten(), 99))
            axes[i].set_title(key)
        save_figure_to_storage(
            fig,
            output_dir,
            base_file,
            f"zdrift_{image_segment}",
            dpi=300
        )
    
    # save z-stack files (both raw and registered)
    zstack_save_filepath = Path(args.output_dir) / unique_id / zstack_filepath.name
    shutil.copy(str(zstack_filepath), str(zstack_save_filepath))
    zstack_reg_save_filepath = Path(args.output_dir) / f'{unique_id}/{zstack_filepath.stem}_reg.h5'
    with h5py.File(zstack_reg_save_filepath, 'w') as h:
        h.create_dataset('data', data=local_zstack.zstack)

    # We remove stuff we don't need to save that would take space
    metrics.pop("mean")
    metrics.pop("var")
    metrics.pop("all_rois_intensity")
    metrics.pop("all_rois_dprime")
    metrics.pop("all_rois_photons_per_rois_per_frame")
    metrics.pop("sum_rois_intensity")
    metrics.pop("all_pixels_photon_per_pixel_per_frame")
    metrics.pop("sum_rois_neuropil")
    metrics.pop("all_neuropils_photons_per_rois_per_frame")

    # We save the metrics to a json file
    with open(os.path.join(output_dir, base_file + "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    write_qc_evaluation(output_dir, unique_id, metrics)
    
    # LEGACY: Keep only the 2 specific metrics needed by aggregator for backward compatibility
    # These are the ones currently recognized by the aggregator's create_movie_qc_evaluations function
    write_legacy_movie_qc_metrics(output_dir, unique_id, metrics)