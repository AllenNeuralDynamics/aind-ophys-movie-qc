import argparse
import glob
import json
import os
from datetime import datetime as dt
from pathlib import Path

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from oasis.functions import deconvolve as oasis_deconvolve
from scipy import ndimage
from scipy.linalg import LinAlgError
from scipy.stats import gaussian_kde
from skimage import filters, measure
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status
from aind_qcportal_schema.metric_value import DropdownMetric


def write_qc_metrics(output_dir, unique_id):

    metric = QCMetric(
        name=f"{unique_id} Epilepsy Probability",
        description="",
        reference=str(f"{unique_id}/movie_qc/{unique_id}_registered_epilepsy_probability.png"),
        status_history=[
            QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)
        ],
        value=DropdownMetric(
            value="Reasonable",
            options=[
                "Reasonable",
                "Unreasonable",
            ],
            status=[
                Status.PASS,
                Status.FAIL,
            ],
        ),
    )

    with open(
        output_dir / f"{unique_id}_registered_epilepsy_probability_metric.json", "w"
    ) as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)

    # physio_intensity metric
    metric = QCMetric(
        name=f"{unique_id} Physio Intensity",
        description="",
        reference=str(f"{unique_id}/movie_qc/{unique_id}_registered_physio_intensity_plot.png"),
        status_history=[
            QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)
        ],
        value=DropdownMetric(
            value="Reasonable",
            options=[
                "Reasonable",
                "Unreasonable",
            ],
            status=[
                Status.PASS,
                Status.FAIL,
            ],
        ),
    )

    with open(
        output_dir / f"{unique_id}_registered_physio_intensity_plot_metric.json", "w"
    ) as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)


def get_and_plot_epilepsy_probability(
    cropped_video, frame_rate, signal_threshold=10.0, min_width=0.1, max_width=0.3
):
    """Get the probability of epilepsy in this physio movie.

    Args:
        signal_threshold:  A bound above which events are tagged as epileptic events
        min_width, max_width:  Bounds on the widths within which events are tagged as epileptic

    Returns:
        A float corresponding to the probability of epilepsy.
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
    denoised_signal, spike_events, frame_rate, spike_threshold=0.05
):
    """Get the local prominence and width of each spike in a calcium trace.

    Local prominence is a measure of the strength of the spike relative to other spikes nearest to it.

    Args:
        denoised_signal:  The denoised fluorescence signal
        spike_events:  Discretized deconvolved neural activity (spikes)
        spike_threshold:  A float below which spikes will be thrown out
        frame_rate:  Frames per second of the signal, used to convert width to seconds
    Returns:
        A tuple, (prominence array, width array)
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


def plot_intensity_histogram(cropped_video, min_range, max_range, log_scale=True):
    """Obtain a plot of the intensity histogram.

    Args:
        cropped_video:  The video to plot
        min_range, max_range:  The range of values to include in the histogram
        log_scale:  A boolean specifying whether or not to plot on a log scale
    Returns:
        A figure.
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


def plot_projection_image(cropped_video, min_range=1, max_range=99):
    fig = plt.figure()
    image_project = np.mean(cropped_video, axis=0)
    list_pixel_limits = np.percentile(image_project.flatten(), [min_range, max_range])
    plt.imshow(
        image_project, cmap="gray", vmin=list_pixel_limits[0], vmax=list_pixel_limits[1]
    )
    plt.axis("off")
    return fig


def get_and_plot_basic_segmentation(
    cropped_video, min_object_size=100, max_object_size=300, sigma_segmentation=30
):
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


def plot_poisson_curve(photon_gain_parameters):
    """Obtain a plot showing Poisson characteristics of the signal.

    Args:
        photon_gain_parameters:  A dictionary of parameters related to the photon gain
    Returns:
        A figure.
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


def plot_avg_intensity_progression(cropped_video, binning):
    """Obtain a plot showing the average intensity over time.

    Args:
        cropped_video:  The video to plot
        binning:  The amount of binning associated with cropped_video
    Returns:
        A figure.
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


def get_saturation_metrics(cropped_video, min_pixel_range, max_pixel_range):
    """Return the number of pixels falling outside the usual dynamic range.

    Args:
        cropped_video:  The video to analyze
        min_pixel_range, max_pixel_range:  The range of pixel values to consider
    Returns:
        An dictionary of saturation metrics.
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


def get_percent_change_intensity(start_section, end_section):
    """Return the percent change in total intensity throughout the physio movie.

    Args:
        start_section, end_section:  The beginning and end of the movie to use
    Returns:
        A float corresponding to the loss in total intensity as a percentage.
    """
    start_mean = np.mean(start_section, axis=(0, 1, 2))
    end_mean = np.mean(end_section, axis=(0, 1, 2))

    percent_change_intensity = 100 * (end_mean - start_mean) / start_mean
    return percent_change_intensity


def get_simple_snr_metrics(cropped_video):
    """SNR Metrics.

    Compute metrics related to the signal to noise level for the images.

    Args:
        cropped_video:  The video to analyze
    Returns:
        A dictionary of metrics
    """

    # Simple SNR - https://en.wikipedia.org/wiki/Signal-to-noise_ratio_(imaging)
    simple_snr_array = cropped_video.mean(axis=(1, 2)) / cropped_video.std(axis=(1, 2))

    return {
        "simple_snr_mean": simple_snr_array.mean(),
        "simple_snr_med": np.median(simple_snr_array),
        "simple_snr_std": simple_snr_array.std(),
    }


def get_percentile_metrics(cropped_video, perc_min, perc_max):
    """Signal Percentiles.

    Compute metrics related to the range of the signal.

    Args:
        perc_min, perc_max:  Min and max values between 0-100 used in filtering based on percentile
    Returns:
        A dictionary of percentile parameters.
    """
    return {
        "percentile_lower": np.percentile(cropped_video, perc_min),
        "percentile_upper": np.percentile(cropped_video, perc_max),
    }


def subsample_and_crop_video(data_pointer, subsample, crop, start_frame=0, end_frame=-1):
    """Subsample and crop a video, cache results. Also functions as a data_pointer load.

    Args:
        subsample:  An integer specifying the amount of subsampling (1 = full movie)
        crop:  A tuple (px_y, px_x) specifying the number of pixels to remove
        start_frame:  The index of the first desired frame
        end_frame:  The index of the last desired frame

    Returns:
        The resultant array.
    """

    if not isinstance(data_pointer, (h5py._hl.dataset.Dataset, np.ndarray)):
        cropped_video = _subsample_and_crop_video_cv(
            subsample, crop, start_frame=start_frame, end_frame=end_frame
        )
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
                start_frame:end_frame:subsample, px_y_start:px_y_end, px_x_start:px_x_end
            ]

    return cropped_video


def convert_intensity_into_photon_flux(intensities, photon_offset, photon_gain):
    background_noise_mean = -photon_offset / photon_gain
    photon_flux = (intensities - background_noise_mean) / photon_gain

    return photon_flux


def get_photon_gain_parameters(cropped_video, max_pixel_range, perc_min=3, perc_max=90):
    """Photon Gain.

    Compute a variety of parameters related to the physio signal's gain.

    Args:
        cropped_video:  The video to analyze
        max_pixel_range:  The maximum pixel value to consider
        perc_min, perc_max:  Min and max values between 0-100 used in filtering based on percentile

    Returns:
        A dictionary of parameters related to the physio signal.  Useful in making plots and metrics.
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
    baseline_photon_flux, neuropil_photon_flux, decay_time, dff_single_event_size
):
    """This original formula was introduced in Wilt et al, 2013
    Default values of decay_time and dff_single_event_size are for Gcamp6f.
    We improved Wilt et al formula to correct for the presence of non-responsive background neuropil, assuming a fixed DF/F given by the indicator. Note that if neuropil is null we end up with dff_single_event_size as the correcting factor is 1.
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


def save_figure_to_storage(figure, storage_path, root_filename, image_name, dpi=120):
    """Saves QC images to the file system.

    Args:

    Returns:
        None
    """
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    filepath = os.path.join(storage_path, "{}_{}.png".format(root_filename, image_name))

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
            "This is the estimated DF/F event size of your event reporter for single spikes (au)."
        ),
    )

    return parser.parse_args()


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

        # We extract a few frames from the beginning and end of the movie to use for stability metrics
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

    # The following add the dictionary returned by get_simple_snr_metrics to the metrics dictionary
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
        plot_projection_image(end_section), output_dir, base_file, "end_projection_image"
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
            metrics["sum_rois_neuropil"], metrics["photon_offset"], metrics["photon_gain"]
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
            cropped_video, min_range=args.min_pixel_range, max_range=args.max_pixel_range
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

    write_qc_metrics(output_dir, unique_id)