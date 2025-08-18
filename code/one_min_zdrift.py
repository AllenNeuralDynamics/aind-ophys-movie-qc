''' Calculating z-drift in finer temporal resolution. More robust to registration error than previous method'''
# This is copied and refactored from lamf_analysis.code_ocean.capsule_data_utils.zdrift

import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import json

from motion_border_utils import get_max_correction_from_df
import zstack as zs


def get_one_min_emf(data, frame_rate, threshold_sec=30):
    num_frames = data.shape[0]
    ny = data.shape[1]
    nx = data.shape[2]

    # get chunks in serial
    # first, use the quotient.
    # if the remainder is less than a threshold, attach it to the last chunk
    # otherwise, make it a separate chunk
    # as a result, the last chunk can be, e.g., 0.5 to 1.5 minutes
    dividers = np.arange(0, num_frames, one_minute_frames)
    if max(dividers) != num_frames:
        if (num_frames - max(dividers)) < last_minute_threshold:
            dividers[-1] = num_frames
        else:
            dividers = np.append(dividers, num_frames)
    emf = np.zeros((len(dividers)-1, ny, nx))
    for i in range(len(dividers)-1):
        emf[i, :, :] = np.mean(data[dividers[i]:dividers[i+1], :, :], axis=0)
    return emf


## Getting motion boundary for crop
# Codes are from lamf_analysis.utils, and depends on aind_ophys_utils.motion_border_utils (which is copied over to this capsule)
def get_motion_correction_crop_xy_range(plane_path: Union[Path, str], session_json_path) -> tuple:
    """Get x-y ranges to crop motion-correction frame rolling

    # TODO: validate in case where max < 0 or min > 0, which may exist (JK 2023)
    # TODO: use motion_border utils from aind_ophys_utils (04/2024)

    Parameters
    ----------
    plane_path : Path
        Path to the plane directory
    session_json_path : Path
        Path to the session.json file

    Returns
    -------
    list, list
        Lists of y range and x range, [start, end] pixel index
    """
    try:
        processing_json_fn = list((Path(plane_path) / 'motion_correction').glob(
            'processing.json'))[0]
        processing_json = json.load(open(processing_json_fn))
        max_shift_prop = processing_json['processing_pipeline']['data_processes'][0]['parameters']['suite2p_args']['maxregshift']
    except:
        processing_json_fn = list((Path(plane_path) / 'motion_correction').glob(
            '*_motion_correction_data_process.json'))[0]
        processing_json = json.load(open(processing_json_fn))
        max_shift_prop = processing_json['parameters']['suite2p_args']['maxregshift']
    
    motion_csv = list((Path(plane_path) / 'motion_correction').glob(
        '*_motion_transform.csv'))[0]
    motion_df = pd.read_csv(motion_csv)

    with open(session_json_path) as f:
        session_json = json.load(f)
    fov_info = session_json['data_streams'][0]['ophys_fovs'][0] # assume this data is the same for all fovs
    fov_height = fov_info['fov_height']
    fov_width = fov_info['fov_width']

    max_shift = max(fov_height, fov_width) * max_shift_prop
    motion_border = get_max_correction_from_df(motion_df, max_shift=max_shift)
    assert motion_border.down >= 0
    assert motion_border.up >= 0
    assert motion_border.left >= 0
    assert motion_border.right >= 0
    up = fov_height if motion_border.up == 0 else fov_height - motion_border.up
    right = fov_width if motion_border.right == 0 else fov_width - motion_border.right

    range_y = [int(motion_border.down), int(up)]
    range_x = [int(motion_border.left), int(right)]

    return range_y, range_x


## Getting filepaths from data assets
# Copied from lamf_analysis.code_ocean.capsule_data_utils
def get_decrosstalked_movie_file(plane_path):
    ''' Load decrosstalked movie for a given plane path
    It can be retrieved from extraction folder.
    Faster than loading COMB object.
    NOTE: Leaving here just in case we change the code to use decrosstalked movie
    '''
    if isinstance(plane_path, str):
        plane_path = Path(plane_path)
    if not os.path.isdir(plane_path):
        raise ValueError(f'Path not found ({plane_path})')
    plane_id = plane_path.name
    decrosstalk_path = plane_path / 'decrosstalk'
    decrosstalked_movie_fn = decrosstalk_path / f'{plane_id}_decrosstalk.h5'
    if not os.path.isfile(decrosstalked_movie_fn):
        raise ValueError(f'No decrosstalked movie found for {plane_id}')    
    return decrosstalked_movie_fn



#####################################################################
## Calculate z-drift from images and metadata
def calc_zdrift_from_images(ref_zstack_crop, episodic_mean_fovs_crop,
                             number_of_z_planes, z_step,
                             use_clahe=True, use_valid_pix=True):
    """ Calculating z-drift from images and metadata
    Temporary exposure to work with custom data structure
    """

    # Get preprocessed z-stack
    stack_pre = zs.med_filt_z_stack(ref_zstack_crop)
    stack_pre = zs.rolling_average_stack(stack_pre)

    # Run registration for each episodic mean FOVs
    matched_plane_indices = np.zeros(
        episodic_mean_fovs_crop.shape[0], dtype=int)
    corrcoef = []
    segment_reg_imgs = []
    shift_list = []
    for i in range(episodic_mean_fovs_crop.shape[0]):
        fov_reg_stack, cc, shift = fov_stack_register_phase_correlation(
            episodic_mean_fovs_crop[i], stack_pre, use_clahe=use_clahe,
            use_valid_pix=use_valid_pix)
        matched_plane_indices[i] = np.argmax(cc)
        corrcoef.append(cc)
        segment_reg_imgs.append(fov_reg_stack[np.argmax(cc)])
        shift_list.append(shift)
    corrcoef = np.asarray(corrcoef)

    center_z = number_of_z_planes // 2
    zdrift_um_each = z_step * (matched_plane_indices - center_z)
    total_zdrift_um = abs(zdrift_um.max() - zdrift_um.min())

    results = { 'z_drift_um': total_zdrift_um,
                'zdrift_um_each': zdrift_um_each,
                'matched_plane_indices': matched_plane_indices,
                'corrcoef': corrcoef,
                'shift': shift_list,
                'use_clahe': use_clahe,
                'use_valid_pix': use_valid_pix}
    return results


def fov_stack_register_phase_correlation(fov, stack, use_clahe=True, use_valid_pix=True):
    """ Reigster FOV to each plane in the stack

    Parameters
    ----------
    fov : np.ndarray (2d)
        FOV image
    stack : np.ndarray (3d)
        stack images
    use_clahe: bool, optional
        If to adjust contrast using CLAHE for registration, by default True
    use_valid_pix : bool, optional
        If to use valid pixels (non-blank pixels after transfromation)
        to calculate correlation coefficient, by default True

    Returns
    -------
    np.ndarray (3d)
        stack of FOV registered to each plane in the input stack
    np.array (1d)
        correlation coefficient between the registered fov and the stack in each plane
    list
        list of translation shifts (y,x)
    """
    assert len(fov.shape) == 2
    assert len(stack.shape) == 3
    assert fov.shape == stack.shape[1:]

    if use_clahe:
        fov_for_reg = zs.image_normalization(skimage.exposure.equalize_adapthist(
            fov.astype(np.uint16)))  # normalization to make it uint16
        stack_for_reg = np.zeros_like(stack)
        for pi in range(stack.shape[0]):
            stack_for_reg[pi, :, :] = zs.image_normalization(
                skimage.exposure.equalize_adapthist(stack[pi, :, :].astype(np.uint16)))
    else:
        fov_for_reg = fov.copy()
        stack_for_reg = stack.copy()

    fov_reg_stack = np.zeros_like(stack_for_reg)
    corrcoef_arr = np.zeros(stack_for_reg.shape[0])
    shift_list = []
    for pi in range(stack_for_reg.shape[0]):
        shift, _, _ = skimage.registration.phase_cross_correlation(
            stack_for_reg[pi, :, :], fov_for_reg, normalization=None)
        fov_reg = scipy.ndimage.shift(fov, shift)
        fov_reg_stack[pi, :, :] = fov_reg
        if use_valid_pix:
            valid_y, valid_x = np.where(fov_reg > 0)
            corrcoef_arr[pi] = np.corrcoef(stack[pi, valid_y, valid_x].flatten(
            ), fov_reg[valid_y, valid_x].flatten())[0, 1]
        else:
            corrcoef_arr[pi] = np.corrcoef(
                stack[pi, :, :].flatten(), fov_reg.flatten())[0, 1]
        shift_list.append(shift)    
    return fov_reg_stack, corrcoef_arr, shift_list


###############################################################
## QC plots for z-drift

# Save all 3 figures in one file
def plot_all(result)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    total_drift = abs(result['zdrift_um'].max() - result['zdrift_um'].min())

    ax = plot_session_zdrift(result, ax=axes[0])    
    ax = plot_shifts(result, ax=axes[1])
    ax = plot_correlation_coefficients(result, ax=axes[2])
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def plot_session_zdrift(result, ax=None, cc_threshold=0.65,
                        add_colorbar=True):
    """Plot z-drift for all the segments in a session
    Drift with peak correlation coefficient overlaid

    Parameters
    ----------
    result : dict
        Dictionary of z-drift results for each plane
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    else:
        fig = ax.get_figure()
    zdrift_um = result['zdrift_um']
    max_cc = np.array([max(cc) for cc in result['corrcoef']])
    
    ax.plot(zdrift_um, color='black', zorder=1)
    
    # split color by correlation coefficient
    h1 = ax.scatter(np.arange(len(max_cc)), zdrift_um, c=max_cc, s=50,
                cmap='binary', vmin=cc_threshold, vmax=1,
                edgecolors='black', linewidth=0.5, zorder=2)
    under_threshold_ind = np.where(max_cc < cc_threshold)[0]
    if len(under_threshold_ind) > 0:
        has_low_cc = 1
        h2 = ax.scatter(under_threshold_ind, zdrift_um[under_threshold_ind], s=50,
                        c=max_cc[under_threshold_ind], cmap='Reds_r', vmin=0, vmax=cc_threshold,
                        edgecolors='red', linewidth=0.5, zorder=3)
    else:
        ylim = ax.get_ybound()
        xlim = ax.get_xbound()
        h2 = ax.scatter(xlim[0] - 1, ylim[0] - 1, c=0,
                        cmap='Reds_r', vmin=0, vmax=cc_threshold)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

    # Add dual colorbar   
    cax1 = fig.add_axes([ax.get_position().x1 + 0.01,
                         ax.get_position().y0 + (ax.get_position().height) * cc_threshold,
                         0.02,
                         ax.get_position().height * (1 - cc_threshold)])
    bar1 = plt.colorbar(h1, cax=cax1)
    bar1.set_label('Correlation coefficient')
    
    # Position label relative to colorbar
    bar1.ax.yaxis.set_label_coords(6, -0.5)
    
    # Create a second colorbar for the lower part
    cax2 = fig.add_axes([ax.get_position().x1 + 0.01,
                     ax.get_position().y0,
                     0.02,
                     ax.get_position().height * cc_threshold])
    plt.colorbar(h2, cax=cax2)
    
    ax.set_xlabel('Segment')
    ax.set_ylabel('Z-drift (um)')
    return ax


def plot_shifts(result, ax=None):
    """Plot shifts at the matched depth for all the segments in a session
    Both in x-y

    Parameters
    ----------
    result : dict
        Dictionary of z-drift results for each plane
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    max_cc_inds = np.array([np.argmax(cc) for cc in result['corrcoef']])        
    shifts = [result['shift'][i][max_cc_inds[i]] for i in range(len(max_cc_inds))]
    y_shift = [shift[0] for shift in shifts]
    x_shift = [shift[1] for shift in shifts]
    ax.plot(y_shift, color='c', label='y-shift')
    ax.plot(x_shift, color='m', label='x-shift')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Shift (pix)')
    ax.set_ylim(-512, 512)
    ax.legend()
    # plt.show()
    return ax


def plot_correlation_coefficients(result, ax=None, downsample_factor=10):
    """Plot correlation coefficients for all the segments in a session

    Parameters
    ----------
    result : dict
        Dictionary of z-drift results for each plane
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    downsample_factor : int, optional
        Factor to downsample the segments for plotting, by default 10
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for i, cc in enumerate(result['corrcoef']):
        if i % downsample_factor != 0:
            continue
        ax.plot(cc, label=f'Seg #{i}')
    ax.set_xlabel('Zstack plane index')
    ax.set_ylabel('Correlation coefficient')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.show()
    return ax
