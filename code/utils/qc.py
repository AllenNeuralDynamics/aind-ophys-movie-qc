"""QC metric construction for movie QC capsule.

Each builder function returns a single v2 QCMetric with modality, stage,
and tags. Collected into a list and wrapped by write_quality_control().
"""

from image_utils import combine_images_vertically
from PIL import Image

from utils.metadata_utils import (
    Modality,
    QCMetric,
    Stage,
    Status,
    pending_qc_status,
)


def create_combined_metric_image(
    seg_img_path: str, poisson_img_path: str, combined_img_path: str
) -> str:
    """Combine segmentation and Poisson plot images vertically.

    Parameters
    ----------
    seg_img_path : str
        Path to the segmentation image.
    poisson_img_path : str
        Path to the Poisson plot image.
    combined_img_path : str
        Path to save the combined image.

    Returns
    -------
    str
        Path to the saved combined image.
    """
    seg_img = Image.open(seg_img_path)
    poisson_img = Image.open(poisson_img_path)
    combined_img = combine_images_vertically([seg_img, poisson_img])
    combined_img.save(combined_img_path)
    return combined_img_path


def build_intensity_metric(unique_id: str, metrics: dict) -> QCMetric:
    """Build intensity stability QCMetric.

    Parameters
    ----------
    unique_id : str
        Unique identifier for the recording plane.
    metrics : dict
        Dictionary of computed metrics.

    Returns
    -------
    QCMetric
    """
    intensity_change = metrics.get("percent_change_intensity", 0.0)
    if abs(intensity_change) >= 20:
        status = Status.FAIL
    elif abs(intensity_change) >= 10:
        status = Status.PENDING
    else:
        status = Status.PASS

    return QCMetric(
        name="Intensity stability",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={
            "evaluation": "Intensity Drift",
            "Operational QC": "True",
        },
        description=(
            "This metric quantifies the percent change in mean pixel intensity from the start to the end of the movie. "
            "It is calculated as 100 * (end_mean - start_mean) / start_mean, where start_mean and end_mean are the average pixel values in the first and last 200 frames (ignoring the first and last 300 frames). "
            "A large negative value indicates photobleaching or instability."
            "Status is automatically assigned: PASS if |change| < 10%, PENDING if 10% <= |change| < 20%, FAIL if |change| >= 20%. "
            "If the status is not PASS, review the intensity progression plot for abrupt drops or trends."
        ),
        reference=str(
            f"{unique_id}/movie_qc/{unique_id}_registered_physio_intensity_plot.png"
        ),
        value=f"{float(intensity_change):.2f}%",
        status_history=[pending_qc_status(status)],
    )


def build_epilepsy_metric(unique_id: str, metrics: dict) -> QCMetric:
    """Build epilepsy probability QCMetric.

    Parameters
    ----------
    unique_id : str
        Unique identifier for the recording plane.
    metrics : dict
        Dictionary of computed metrics.

    Returns
    -------
    QCMetric
    """
    epilepsy_prob = metrics.get("epilepsy_probability", 0)
    epilepsy_status = Status.FAIL if epilepsy_prob > 0 else Status.PASS

    return QCMetric(
        name="Epilepsy probability",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={
            "evaluation": "Epilepsy",
            "Operational QC": "True",
        },
        description=(
            "This metric estimates the probability of epileptiform activity in the movie using deconvolution and event detection. "
            "The signal is denoised and events are detected; events with prominence > 10 and width between 0.1 and 0.3 seconds are considered epileptic. "
            "The metric is the fraction of such events among all detected events."
            "Status is automatically assigned: PASS if probability is 0, FAIL otherwise. "
            "If FAIL, inspect the event plot for clusters of large, brief events."
        ),
        reference=str(
            f"{unique_id}/movie_qc/{unique_id}_registered_epilepsy_probability.png"
        ),
        value=float(epilepsy_prob),
        status_history=[pending_qc_status(epilepsy_status)],
    )


def build_snr_dprime_metric(metrics: dict) -> QCMetric:
    """Build event detection statistics (SNR + d-prime) QCMetric.

    Parameters
    ----------
    metrics : dict
        Dictionary of computed metrics.

    Returns
    -------
    QCMetric
    """
    snr_dprime_values = {
        "SNR Mean": float(metrics.get("simple_snr_mean", 0.0)),
        "SNR Median": float(metrics.get("simple_snr_med", 0.0)),
        "SNR Standard Deviation": float(metrics.get("simple_snr_std", 0.0)),
        "Median ROI D-prime": float(metrics.get("median_rois_dprime", 0.0)),
        "ROI D-prime Standard Deviation": float(metrics.get("std_rois_dprime", 0.0)),
    }
    return QCMetric(
        name="Event detection statistics",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={"evaluation": "Event detection statistics"},
        description=(
            "This metric combines signal-to-noise ratio (SNR) and event detection (d-prime) statistics."
            "SNR is calculated as the mean pixel value divided by the standard deviation for each frame, summarized as mean, median, and standard deviation across the movie. "
            "Higher SNR indicates better image quality."
            "D-prime is calculated for each ROI using photon flux and neuropil estimates, following Wilt et al. (2013), and summarized as the median and standard deviation. "
            "Higher d-prime values indicate better event detectability."
            "Status is always PASS (no automatic threshold). Review low SNR or d-prime values for possible issues with signal quality or event detection."
        ),
        value=snr_dprime_values,
        status_history=[pending_qc_status(Status.PASS)],
    )


def build_pixel_saturation_metric(unique_id: str, metrics: dict) -> QCMetric:
    """Build pixel value distribution QCMetric.

    Parameters
    ----------
    unique_id : str
        Unique identifier for the recording plane.
    metrics : dict
        Dictionary of computed metrics.

    Returns
    -------
    QCMetric
    """
    saturated_pixels = metrics.get("nb_saturated_pixels", 0)
    if saturated_pixels >= 1000:
        sat_status = Status.FAIL
    elif saturated_pixels >= 800:
        sat_status = Status.PENDING
    else:
        sat_status = Status.PASS

    low_pixels = metrics.get("nb_low_pixels", 0)
    low_pixels_perc = metrics.get("nb_low_pixels_perc", 0.0)
    percentile_lower = float(metrics.get("percentile_lower", 0.0))
    percentile_upper = float(metrics.get("percentile_upper", 0.0))

    merged_pixel_values = {
        "Saturated Pixels": int(saturated_pixels),
        "Low Intensity Pixels": int(low_pixels),
        "Low Intensity Pixels Percentage": float(low_pixels_perc),
        "Intensity Percentile Lower (5th)": percentile_lower,
        "Intensity Percentile Upper (95th)": percentile_upper,
    }
    return QCMetric(
        name="Pixel value distribution",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={
            "evaluation": "Pixel Saturation",
            "Operational QC": "True",
        },
        description=(
            "This metric summarizes the distribution of pixel values in the movie, including the number of saturated pixels (at or above max range), low intensity pixels (at or below min range), and the 5th/95th percentiles. "
            "Saturated and low pixels are counted in the first 500 frames."
            "Status is automatically assigned: PASS if saturated pixels < 800, PENDING if 800 <= saturated < 1000, FAIL if >= 1000. "
            "If not PASS, review the intensity histogram for evidence of clipping or poor dynamic range."
        ),
        reference=str(
            f"{unique_id}/movie_qc/{unique_id}_registered_physio_intensity_hist.png"
        ),
        value=merged_pixel_values,
        status_history=[pending_qc_status(sat_status)],
    )


def build_photon_metric(metrics: dict, combined_img_path: str) -> QCMetric:
    """Build photon detection statistics QCMetric.

    Parameters
    ----------
    metrics : dict
        Dictionary of computed metrics.
    combined_img_path : str
        Path to the combined segmentation + Poisson plot image.

    Returns
    -------
    QCMetric
    """
    merged_values = {
        "Mean ROI Intensity": float(metrics.get("mean_rois_intensity", 0.0)),
        "ROI Segmentation Threshold": float(metrics.get("rois_threshold", 0.0)),
        "Median ROI Neuropil Sum": float(metrics.get("median_sum_rois_neuropil", 0.0)),
        "Photon Gain": float(metrics.get("photon_gain")),
        "Photon Offset": float(metrics.get("photon_offset")),
        "Background Noise": float(metrics.get("background_noise")),
        "Photon Flux Median": float(
            metrics.get("photon_flux_median_per_pixel_per_frame")
        ),
        "Mean Photons per ROI per Frame": float(
            metrics.get("mean_photons_per_roi_per_frame")
        ),
        "Mean Photons per ROI per Second": float(
            metrics.get("mean_photons_per_roi_per_s")
        ),
        "Mean Photons per Neuropil per Second": float(
            metrics.get("mean_photons_per_neuropil_per_s")
        ),
    }
    return QCMetric(
        name="Photon detection statistics",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={"evaluation": "Photon detection statistics"},
        description=(
            "This metric summarizes photon detection."
            "ROI segmentation is performed using Otsu thresholding on a high-pass filtered median image from the start of the movie. "
            "Mean ROI intensity, segmentation threshold, and median neuropil sum are reported."
            "Photon gain and offset are estimated by fitting the variance vs. mean relationship for non-saturated pixels. "
            "Photon flux and background noise are derived from these parameters. "
            "Photon statistics are reported as mean/median values per ROI and neuropil, per frame and per second."
            "Status is always PASS (no automatic threshold). Review for outliers or unexpected values."
        ),
        reference=combined_img_path,
        value=merged_values,
        status_history=[pending_qc_status(Status.PASS)],
    )


def build_zdrift_metric(
    unique_id: str, metrics: dict, zdrift_qc_threshold: int = 10
) -> QCMetric:
    """Build z-drift analysis QCMetric.

    Parameters
    ----------
    unique_id : str
        Unique identifier for the recording plane.
    metrics : dict
        Dictionary of computed metrics (must contain 'zdrift' key).
    zdrift_qc_threshold : int
        Threshold in um for PASS/FAIL.

    Returns
    -------
    QCMetric
    """
    zdrift = metrics.get("zdrift", -100)
    if zdrift == -100:
        zdrift_status = Status.PENDING
        zdrift_metrics_dict = {}
    else:
        zdrift_um = float(zdrift["z_drift_um"])
        zdrift_status = (
            Status.PASS if abs(zdrift_um) <= zdrift_qc_threshold else Status.FAIL
        )
        zdrift_metrics_dict = {
            "z_drift_um": zdrift_um,
            "start_frame": int(zdrift["start_frame"]),
            "end_frame": int(zdrift["end_frame"]),
            "z_drift_frame": int(zdrift["z_drift_frame"]),
            "start_frame_corr": round(zdrift["start_frame_corr"], 3),
            "end_frame_corr": round(zdrift["end_frame_corr"], 3),
            "nb_of_loops": int(zdrift["local_zstack_parameters"]["nb_of_loops"]),
            "nb_of_planes": int(
                zdrift["local_zstack_parameters"]["nb_of_planes"]
            ),
            "z_spacing_um": round(
                zdrift["local_zstack_parameters"]["z_spacing_um"], 2
            ),
            "total_z_distance": round(
                zdrift["local_zstack_parameters"]["total_z_distance"], 2
            ),
        }

    return QCMetric(
        name=f"{unique_id} Z-drift Analysis",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={
            "evaluation": "Z-drift",
            "Operational QC": "True",
        },
        description=(
            f"Z-drift analysis metrics (threshold {zdrift_qc_threshold} um)"
        ),
        value=zdrift_metrics_dict,
        reference=str(
            f"{unique_id}/movie_qc/{unique_id}_registered_zdrift.png"
        ),
        status_history=[pending_qc_status(zdrift_status)],
    )
