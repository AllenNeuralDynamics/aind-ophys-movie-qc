# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.ndimage
from skimage.measure import compare_ssim as ssim
from skimage.feature import register_translation

from mindscope_qc_metrics.datasets import (MotionCorrPhysio, OphysDataset,
                                     VideoDataset)
from mindscope_qc_metrics.exceptions import DataAcquisitionError, DataMissingError
from mindscope_qc_metrics.plotting import plt
from mindscope_qc_metrics.utils import rigs, util, logger


class LocalZStack(VideoDataset):
    """Local Z Stack

    Objects contain methods to generate QC metrics from the local z stack files
    captured at the Allen Institute.  These files are typically named '*_zstack_local.h5'
    or '*_zstack_local_dewarping.h5', depending on the rig.
    
    Args:  
        ophys_experiment:  A LimsOphysExperiment object associated with this video
        _filepath:  The path to the local z stack file
        _meta_filepath:  The path to the local z stack metadata
        _physio_filepath:  The path to the physio movie to use in calculating z-drift

    Attributes: 
        ophys_experiment:  A LimsOphysExperiment object associated with this video
        data_pointer:  A reference to the HDF5 file of interest
        meta: A LocalZStackMeta object corresponding to the metadata of this video
        _cache:  A cache that stores previously subsampled and cropped videos
        dataset_name:  A string used to refer to this QC set
        metrics:  A dict of parameters to save in the Mouse-Seeks database
        images:  A dict of image locations to save in the Mouse-Seeks database
        _physio_filepath:  The path to the physio movie to use in calculating z-drift
    """
    def __init__(self, ophys_experiment=None, 
                       _filepath=None, 
                       _meta_filepath=None, 
                       _physio_filepath=None,
                       _physio_meta_filepath=None):
        self.ophys_experiment = ophys_experiment
        self._physio_filepath = _physio_filepath
        self._physio_meta_filepath = _physio_meta_filepath

        if _filepath is None:
            if ophys_experiment is None:
                raise NotImplementedError('Must provide either ophys experiment object or path to local z stack file.')
            storage_directory = self.ophys_experiment.data_pointer['storage_directory']
            rig = self.ophys_experiment.data_pointer['rig']
            if rig in rigs.NIKON:
                self.data_pointer, self._filepath = self._get_file(storage_directory, '*local.h5', glob=True)
            elif rig in rigs.SCIENTIFICA:
                try:
                    self.data_pointer, self._filepath = self._get_file(storage_directory, '*zstack_local_dewarping.h5', glob=True)
                except DataMissingError: # For experiments that never underwent dewarping
                    self.data_pointer, self._filepath = self._get_file(storage_directory, '*zstack_local.h5', glob=True)
            elif rig in rigs.MESOSCOPE:
                try:
                    self.data_pointer, self._filepath= self._get_file(storage_directory, '*zstack*.tif', glob=True)[0] # TODO: Temporary for off-pipeline work
                except:
                    self.data_pointer, self._filepath = self._get_file(storage_directory, '*z_stack_local.h5', glob=True)
            elif rig in rigs.DEEPSCOPE:
                try:
                    self.data_pointer, self._filepath = self._get_file(storage_directory, '*stack*.h5', glob=True)
                except:
                    self.data_pointer, self._filepath = self._get_file(storage_directory, '*local*.h5', glob=True)
            else:
                raise NotImplementedError('Local z stack filename is not known for rig {}'.format(rig))
        else:
            self.data_pointer, self._filepath = self._get_file(_filepath, glob=True) # TODO: What if _filepath is h5 file with 'data' key?
        try:
            while 'data' in self.data_pointer:
                self.data_pointer = self.data_pointer['data']
        except TypeError: # Not an iterable
            pass

        self._cache = {}
        # self.meta = LocalZStackMeta(self.ophys_experiment, _filepath=_meta_filepath)
        scanimage_metadata, roi_groups = self.local_zstack_metadata(self._filepath)
        nb_of_loops = int(si_metadata['SI.hStackManager.actualNumVolumes'])
        nb_of_planes = int(si_metadata['SI.hStackManager.actualNumSlices'])
        z_spacing_um = float(si_metadata['SI.hStackManager.actualStackZStepSize'])
        
        # if self.meta._reshape:
        #     self._preprocess_stack()

        if self.meta.nb_of_loops * self.meta.nb_of_planes != self.data_pointer.shape[0]:
            raise DataAcquisitionError('Number of frames in local z stack different from metadata')

        self.images = {}
        self.dataset_name = 'local_z_stack'

    def _calculate_qc_metrics(self):
        """Calculates QC metrics and saves to the parameter self.metrics

        Returns:
            None
        """
        z_drift_corr = self.get_z_drift(gaussian_filter=True, use_meta=True, save_images=True, metric='corr')
        self.metrics = {
            'shape': self.data_pointer.shape,
            'z_drift_corr_start_frame': z_drift_corr['start_frame'],
            'z_drift_corr_end_frame': z_drift_corr['end_frame'],
            'z_drift_corr_frame_diff': z_drift_corr['end_frame'] - z_drift_corr['start_frame'],
            'z_drift_corr_um_diff': (z_drift_corr['end_frame'] - z_drift_corr['start_frame']) * self.meta.z_spacing_um,
            'z_drift_start_frame_match_method': 'algorithm',
            'z_drift_end_frame_match_method': 'algorithm',
            'filepath': self.get_video_filepath()
        }
        if 'start_frame_corr' in z_drift_corr:
            self.metrics['start_frame_corr'] = z_drift_corr['start_frame_corr']
            self.metrics['end_frame_corr'] = z_drift_corr['end_frame_corr']
        elif 'start_frame_ssim' in z_drift_corr:
            self.metrics['start_frame_ssim'] = z_drift_corr['start_frame_ssim']
            self.metrics['end_frame_ssim'] = z_drift_corr['end_frame_ssim']


        try:
            z_drift_ssim = self.get_z_drift(gaussian_filter=True, sigma=5, use_meta=True, save_images=True, metric='ssim')
            self.metrics.update({
                'z_drift_ssim_start_frame': z_drift_ssim['start_frame'],
                'z_drift_ssim_end_frame': z_drift_ssim['end_frame'],
                'z_drift_ssim_frame_diff': z_drift_ssim['end_frame'] - z_drift_ssim['start_frame'],
                'z_drift_ssim_um_diff': (z_drift_ssim['end_frame'] - z_drift_ssim['start_frame']) * self.meta.z_spacing_um,
                'z_drift_start_frame_match_method': 'algorithm',
                'z_drift_end_frame_match_method': 'algorithm'
            })
        except ValueError: # SSIM is optional
            logger.debug('Unable to compute SSIM for session {}'.format(self.ophys_experiment.lims_id))
            pass

    def _calculate_qc_images(self):
        """Calculates QC images and saves to the parameter self.metrics

        Returns:
            None
        """
        self._calculate_qc_metrics() # Most local z stack images saved as part of metric calculation process

        stack = self.process_stack()
        self.save_image_to_storage(stack[0],
                                   'local_z_stack_top',
                                   self.ophys_experiment.lims_id)
        self.save_image_to_storage(stack[int(self.meta.nb_of_planes / 2)],
                                   'local_z_stack_mid',
                                   self.ophys_experiment.lims_id)
        self.save_image_to_storage(stack[self.meta.nb_of_planes - 1],
                                   'local_z_stack_bot',
                                   self.ophys_experiment.lims_id)

        xz_aspect = (self.meta.total_z_distance / len(stack)) * (stack.shape[1] / self.meta.x_size_um)
        yz_aspect = (self.meta.total_z_distance / len(stack)) * (stack.shape[2] / self.meta.y_size_um)
        self.save_image_to_storage(self.plot_array(np.mean(stack, axis=1), xlabel='X', ylabel='Z', aspect=xz_aspect, is_image=True),
                                   'local_z_stack_xz_plot',
                                   self.ophys_experiment.lims_id)
        self.save_image_to_storage(self.plot_array(np.mean(stack, axis=2), xlabel='Y', ylabel='Z', aspect=yz_aspect, is_image=True),
                                   'local_z_stack_yz_plot',
                                   self.ophys_experiment.lims_id)
        self.save_image_to_storage(self.plot_array(np.mean(stack, axis=(1,2)), xlabel='Plane #', ylabel='Average Intensity'),
                                   'local_z_stack_intensity_plot',
                                   self.ophys_experiment.lims_id)

        self.save_video_to_storage(self._cache['local_z_stack'].astype('uint16'),
                                   'local_z_stack',
                                   self.ophys_experiment.lims_id)

    def get_video_filepath(self): 
        return os.path.join(util.IMAGE_STORAGE_DIR, 
                            str(self.ophys_experiment.lims_id),
                            self.dataset_name,
                            'local_z_stack.tif')

    def _preprocess_stack(self):
        """Reshapes and averages stack, when it isn't done on the rig.

            Returns:
                None
        """
        print(self.data_pointer.shape)
        _temp = np.reshape(self.data_pointer, self.meta._newshape)
        self.data_pointer = _temp.mean(axis=1)

    def process_stack(self):
        """Averages common z-planes across loops.
        #TODO: Need to correct for motion.

        Returns:
            An array of shape (self.meta.nb_of_planes - (ignore_top + ignore_bot), Y, X)
        """
        if self.dataset_name in self._cache:
            return self._cache[self.dataset_name]

        avg_z_planes = []
        if self.meta.piezo_mode:
            for z_plane_nb in range(self.meta.nb_of_planes):
                avg_z_plane = self.get_axis_mean(subsample=self.meta.nb_of_planes,
                                                 axis=0,
                                                 start_frame=z_plane_nb)
                avg_z_planes.append(avg_z_plane)
        else:
            for z_plane_nb in range(self.meta.nb_of_planes):
                avg_z_plane_zig = self.get_axis_mean(subsample=2*self.meta.nb_of_planes,
                                                     axis=0,
                                                     start_frame=z_plane_nb)

                if z_plane_nb != 0 and z_plane_nb != self.meta.nb_of_planes:
                    avg_z_plane_zag = self.get_axis_mean(subsample=2*self.meta.nb_of_planes,
                                                         axis=0,
                                                         start_frame=2*self.meta.nb_of_planes-z_plane_nb)
                else:
                    avg_z_plane_zag = avg_z_plane_zig

                avg_z_planes.append(np.mean([avg_z_plane_zag, avg_z_plane_zig], axis=0))

        stack = np.array(avg_z_planes)
        self._cache[self.dataset_name] = stack
        return stack

    def _input_process(self, input_image, crop=(0,0), gaussian_filter=False, sigma=3, shift=None):
        """Processes the local z stack and the input image.
        # Note: may not be used.

        Z-stack is cropped, shifted, and blurred.  Input image is blurred.

        Args:
            input_image:  2D array to compare to the local z stack planes
            crop:  A tuple (px_y, px_x) specifying the number of pixels to remove from stack
            gaussian_filter:  A boolean denoting whether or not to apply a 2D gaussian filter
            sigma:  The width of the 2D gaussian filter
            shift: A tuple denoting (0, px_y, px_x) the number of pixels to shift, or None

        Returns:
            The processed stack (also cached) and input image as numpy arrays.
        """
        if ('local_z_stack', crop, gaussian_filter, sigma, shift) in self._cache:
            z_stack = self._cache[('local_z_stack', crop, gaussian_filter, sigma, shift)]
        else:
            z_stack = self.process_stack()

            # Filter z stack
            if gaussian_filter:
                z_stack = scipy.ndimage.gaussian_filter(z_stack, (0, sigma, sigma))

            #deprecated when zstacks were moved to before timeseries
            # Shift z stack
            # if shift:
            #     z_stack = scipy.ndimage.shift(z_stack, shift)

            # Crop z stack
            _shape = z_stack.shape
            px_y_start, px_x_start = crop
            px_y_end = _shape[1] - px_y_start
            px_x_end = _shape[2] - px_x_start
            z_stack = z_stack[:, px_y_start:px_y_end, px_x_start:px_x_end]

            # Pad with zeroes to ensure shape is the same
            #y_diff = z_stack.shape[1:][0] - input_image.shape[0]
            #x_diff = z_stack.shape[1:][1] - input_image.shape[1]
            #if y_diff < 0: # stack needs y padding
            #    z_stack = np.pad(z_stack, [(0,0), (0,-y_diff), (0,0)], 'constant')
            #elif y_diff > 0: # input image needs y padding
            #    input_image = np.pad(input_image, [(0,0), (0,y_diff), (0,0)], 'constant')
            #elif x_diff < 0: # stack needs x padding
            #    z_stack = np.pad(z_stack, [(0,0), (0,0), (0,-x_diff)], 'constant')
            #elif x_diff > 0: # input image needs x padding
            #    input_image = np.pad(input_image, [(0,0), (0,0), (0,x_diff)], 'constant')

            self._cache[('local_z_stack', crop, gaussian_filter, sigma, shift)] = z_stack

        # Filter input image
        if gaussian_filter:
            input_image = scipy.ndimage.gaussian_filter(input_image, sigma=sigma)
        #shift input image
        if shift:
            input_image = scipy.ndimage.shift(input_image, shift)

        return z_stack, input_image

    def plot_z_drift_scores(self, start_image, end_image, crop=(0,0), gaussian_filter=False, sigma=3, shift=None, metric='corr'):
        """Generate a score vs. depth plot for a given z-drift metric.

        Args:
            start_image, end_image:  2D arrays to compare to the local z stack planes
            crop:  A tuple (px_y, px_x) specifying the number of pixels to remove from stack
            gaussian_filter:  A boolean denoting whether or not to apply a 2D gaussian filter
            sigma:  The width of the 2D gaussian filter
            shift: A tuple denoting (0, px_y, px_x) the number of pixels to shift, or None
            metric:  Metric choice, either pearson correlation (corr) or ssim (ssim)

        Returns:
            A figure.
        """
        if metric == 'corr':
            start_coefs = self.get_corr_z_planes(start_image, crop=crop, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift)
            end_coefs = self.get_corr_z_planes(end_image, crop=crop, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift)
        elif metric == 'ssim':
            start_coefs = self.get_ssim_z_planes(start_image, crop=crop, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift)
            end_coefs = self.get_ssim_z_planes(end_image, crop=crop, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift)
        else:
            raise NotImplementedError('Unhandled z-drift metric type {}'.format(metric))

        z_spacing = self.meta.z_spacing_um

        fig = plt.figure()
        plt.plot(np.arange(0, len(start_coefs)) * z_spacing,
                 start_coefs,
                 marker='.',
                 label='Start {}'.format(metric.upper()))
        plt.plot(np.arange(0, len(end_coefs)) * z_spacing,
                 end_coefs,
                 marker='.',
                 label='End {}'.format(metric.upper()))

        plt.axvline(start_coefs.argmax() * z_spacing, color='black')
        plt.axvline(end_coefs.argmax() * z_spacing, color='black')
        plt.legend()
        plt.ylabel('Score')
        plt.xlabel(r'Depth ($um$)')
        plt.tight_layout()
        return fig

    def get_corr_z_planes(self, input_image, crop=(0,0), gaussian_filter=False, sigma=3, shift=None, ignore_top=5, ignore_bot=5):
        """Compares input_image to each z-plane of the local z stack. Returns correlation coefficients.

        Args:
            input_image:  2D array to compare to the local z stack planes
            crop:  A tuple (px_y, px_x) specifying the number of pixels to remove from stack
            gaussian_filter:  A boolean denoting whether or not to apply a 2D gaussian filter
            sigma:  The width of the 2D gaussian filter
            shift: A tuple denoting (0, px_y, px_x) the number of pixels to shift, or None
            ignore_top:  Number of planes to ignore from the top
            ignore_bot:  Number of planes to ignore from the bottom

        Returns:
            An array of correlation coefficients.
        """
        z_stack, input_image = self._input_process(input_image,
                                                   crop=crop,
                                                   gaussian_filter=gaussian_filter,
                                                   sigma=sigma,
                                                   shift=shift)
        coeffs = [np.corrcoef(input_image.flatten(), z_plane.flatten())[0, 1] for z_plane in z_stack]

        # Set ignore planes to 0 correlation
        coeffs[:ignore_top] = np.ones(ignore_top) * np.min(coeffs)
        coeffs[-ignore_bot:] = np.ones(ignore_bot) * np.min(coeffs)
        return np.array(coeffs)

    def get_ssim_z_planes(self, input_image, crop=(0,0), gaussian_filter=False, sigma=5, shift=None, ignore_top=5, ignore_bot=5):
        """Compares input_image to each z-plane of the local z stack. Returns correlation coefficients.

        Args:
            input_image:  2D array to compare to the local z stack planes
            crop:  A tuple (px_y, px_x) specifying the number of pixels to remove from stack
            gaussian_filter:  A boolean denoting whether or not to apply a 2D gaussian filter
            sigma:  The width of the 2D gaussian filter
            shift: A tuple denoting (0, px_y, px_x) the number of pixels to shift, or None
            ignore_top:  Number of planes to ignore from the top
            ignore_bot:  Number of planes to ignore from the bottom

        Returns:
            An array of correlation coefficients.
        """
        z_stack, input_image = self._input_process(input_image,
                                                   crop=crop,
                                                   gaussian_filter=gaussian_filter,
                                                   sigma=sigma,
                                                   shift=shift)
        coeffs = [ssim(input_image, z_plane) for z_plane in z_stack]

        # Set ignore planes to 0 correlation
        coeffs[:ignore_top] = np.ones(ignore_top) * np.min(coeffs)
        coeffs[-ignore_bot:] = np.ones(ignore_bot) * np.min(coeffs)
        return np.array(coeffs)

    def get_z_drift(self, nb_frames_to_avg=500, crop=(0,0), gaussian_filter=True, sigma=3, shift=None, use_meta=False, save_images=False, metric='corr'):
        """Determine the amount of z-drift in the motion corrected physio movie..
        # Note: This code assumes stable motion. 

        Args:
            nb_frames_to_avg:  Number of frames to average in the physio movie
            crop:  A tuple (px_y, px_x) specifying the number of pixels to remove from stack
            gaussian_filter:  A boolean denoting whether or not to apply a 2D gaussian filter
            sigma:  The width of the gaussian filter
            shift: A tuple denoting (0, px_y, px_x) the number of pixels to shift, or None
            use_meta: A boolean specifying whether or not to use the physio metadata to determine crop and shift
            save_images: A boolean specifying whether or not to save the resultant images from this calculation
            metric:  Metric choice, either pearson correlation (corr) or ssim (ssim)

        Returns:
            A dict of z-drift information.
        """
        motion_corr_physio = MotionCorrPhysio(self.ophys_experiment,
                                              _filepath=self._physio_filepath,
                                              _meta_filepath=self._physio_meta_filepath)

        if self.data_pointer.shape[1:] != motion_corr_physio.data_pointer.shape[1:]:
            raise NotImplementedError('Shape mismatch. Cannot determine z-drift.')
        
        if use_meta:
            #register start image to z projected stack
            stack =self.process_stack()
            stack_projection = np.max(stack, axis = 0)
            start_image =  motion_corr_physio.get_axis_mean(end_frame=nb_frames_to_avg, axis=0)
            shift_xy, _, _ = register_translation(stack_projection, start_image)
            #3d shift used for zstack, 2d shift for start/end image
            #shift = (0, -shift_xy[0], -shift_xy[1])
            shift = (shift_xy[0], shift_xy[1])
            if crop == (0, 0):
                crop = (int(np.abs(np.floor(shift_xy[0]))), int(np.abs(np.floor(shift_xy[1]))))

        start_image = motion_corr_physio.get_axis_mean(end_frame=nb_frames_to_avg, axis=0, crop=crop)
        end_image = motion_corr_physio.get_axis_mean(start_frame=-(nb_frames_to_avg+1), axis=0, crop=crop)
        self._last_start_image = start_image
        self._last_end_image = end_image
        correlation_scores = {}
        if metric == 'corr':
            start_corr = self.get_corr_z_planes(start_image, crop=crop, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift)
            start_idx = int(np.argmax(start_corr))
            correlation_scores['start_frame_corr']= start_corr[start_idx]
            end_corr = self.get_corr_z_planes(end_image, crop=crop, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift)
            end_idx = int(np.argmax(end_corr))
            correlation_scores['end_frame_corr'] = end_corr[end_idx]
        elif metric == 'ssim':
            start_corr = self.get_ssim_z_planes(start_image, crop=crop, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift)
            start_idx = int(np.argmax(start_corr))
            correlation_scores['start_frame_ssim'] = start_corr[start_idx]
            end_corr = self.get_ssim_z_planes(end_image, crop=crop, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift)
            end_idx = int(np.argmax(end_corr))
            correlation_scores['end_frame_ssim'] = end_corr[end_idx]
        else:
            raise NotImplementedError('Unhandled z-drift metric type {}'.format(metric))

        if save_images:
            self.save_image_to_storage(scipy.ndimage.shift(start_image, shift), 'motion_corr_physio_start', self.ophys_experiment.lims_id)
            self.save_image_to_storage(scipy.ndimage.shift(end_image, shift), 'motion_corr_physio_end', self.ophys_experiment.lims_id)

            try:
                # Temporary for vgg_project
                self.save_video_to_storage(start_image.astype('uint16'), 'motion_corr_physio_start_ml', self.ophys_experiment.lims_id)
                #self.save_video_to_storage(end_image.astype('uint16'), 'motion_corr_physio_end_ml', self.ophys_experiment.lims_id)
            except ValueError:
                pass

            _score_fig = self.plot_z_drift_scores(start_image, end_image, crop=crop, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift, metric=metric)
            self.save_image_to_storage(_score_fig, '{}_z_drift_score_plot'.format(metric), self.ophys_experiment.lims_id)

            stack = self.process_stack()
            _shape = stack.shape
            px_y_start, px_x_start = crop
            px_y_end = _shape[1] - px_y_start
            px_x_end = _shape[2] - px_x_start
            stack = stack[:, px_y_start:px_y_end, px_x_start:px_x_end]
            self.save_image_to_storage(stack[start_idx], 'local_z_stack_start_match_{}'.format(metric), self.ophys_experiment.lims_id)
            self.save_image_to_storage(stack[end_idx], 'local_z_stack_end_match_{}'.format(metric), self.ophys_experiment.lims_id)
            z_drift_metrics = {
                'start_frame': start_idx,
                'end_frame': end_idx,
                'z_drift_frame': end_idx - start_idx,
                'z_drift_um': (end_idx - start_idx) * self.meta.z_spacing_um}
            z_drift_metrics.update(correlation_scores)
            return z_drift_metrics


    def local_zstack_metadata(zstack_path: Union[Path, str]) -> tuple:
        """Get scanimage metadata and ROI groups from a local z-stack

        Parameters
        ----------
        zstack_path : Union[Path, str]
            Path to the local z-stack

        Returns
        -------
        dict
            Scanimage metadata
        """
        zstack_path = Path(zstack_path)
        with h5py.File(zstack_path, 'r') as f:
            if 'scanimage_metadata' not in f:
                raise ValueError("scanimage_metadata not found in the h5 file")
            si = f["scanimage_metadata"][()]
        si = si.decode()
        si = json.loads(si)
        scanimage_metadata = si[0]
        roi_groups = si[1]

        # number_of_z_planes= int(si_metadata['SI.hStackManager.actualNumSlices'])
        # number_of_repeats = int(si_metadata['SI.hStackManager.actualNumVolumes'])
        # z_step = float(si_metadata['SI.hStackManager.actualStackZStepSize'])
        return scanimage_metadata, roi_groups


class LocalZStackMeta(OphysDataset):
    """Local Z Stack Metadata

    Objects contain methods to generate QC metrics from the local z stack metadata files
    captured at the Allen Institute.  These files are typically named '*_zstack_local.json'
    or '*_local_XYT*.ini', depending on the rig.

    Attributes:
        ophys_experiment:  A LimsOphysExperiment object associated with this metadata
        data_pointer:  A reference to the metadata file of interest
        piezo_mode:  True == sawtooth, False == zig-zag
        nb_of_loops:  The total number of z-stack loops performed - unidirectional
        nb_of_planes: The number of planes in a single z-stack loop
        z_spacing_um:  The spacing between z-planes in microns
        total_z_distance:  The total distance traveled in z during a single z-stack loop
        dataset_name:  A string used to refer to this QC set
        metrics:  A dict of parameters to save in the Mouse-Seeks database
    """
    def __init__(self, ophys_experiment=None, _filepath=None):
        self.ophys_experiment = ophys_experiment

        if _filepath is None:
            if ophys_experiment is None:
                raise NotImplementedError('Must provide either ophys experiment object or path to local z stack meta file.')
            rig = self.ophys_experiment.data_pointer['rig']
            if rig in rigs.NIKON:
                storage_directory = self.ophys_experiment.data_pointer['storage_directory']
                self.data_pointer = self._get_file(storage_directory, '*zstack_local.json', glob=True)[0]
                self._parse_nikon()
            elif rig in rigs.SCIENTIFICA:
                storage_directory = self.ophys_experiment.data_pointer['ophys_session']['storage_directory']
                storage_directory = util.file_path_replace(storage_directory)
                self.data_pointer = self._get_file(storage_directory, '*local_XYT*.ini', glob=True)[0]
                self._parse_scientifica()
            elif rig in rigs.MESOSCOPE:
                try:
                    storage_directory = self.ophys_experiment.data_pointer['storage_directory']
                    try:
                        self.data_pointer = self._get_file(storage_directory, '*zstack*.json', glob=True)[0] # TODO: Temporary for off-pipeline work
                    except:
                        self.data_pointer = 'None'
                    self._parse_mesoscope()
                except:
                    raise NotImplementedError('Local z stack meta filename is not known for rig {}'.format(rig))

            elif rig in rigs.DEEPSCOPE:
                try:
                    storage_directory = self.ophys_experiment.data_pointer['storage_directory']
                    self.data_pointer = self._get_file(storage_directory, '*stack*.h5', glob=True)[0]['scanimage_metadata']
                    self._parse_deepscope()
                except:
                    storage_directory = self.ophys_experiment.data_pointer['storage_directory']
                    self.data_pointer = self._get_file(storage_directory, '*local*.h5', glob=True)[0]['scanimage_metadata']
                    self._parse_deepscope()
            else:
                raise NotImplementedError('Local z stack meta filename is not known for rig {}'.format(rig))   
        else:
            self.data_pointer = self._get_file(_filepath, glob=True)[0] # TODO:  How to specify which parse function to call?
            try:
                self._parse_nikon()
            except:
                pass
            try:
                self._parse_scientifica()
            except:
                pass
            
        self.dataset_name = 'local_z_stack_meta'

    def _calculate_qc_metrics(self):
        """Calculates QC metrics and saves to the parameter self.metrics

        Returns: 
            None
        """
        self.metrics = {
            'piezo_mode': self.piezo_mode,
            'nb_of_loops': self.nb_of_loops,
            'nb_of_planes': self.nb_of_planes,
            'z_spacing_um': self.z_spacing_um,
            'total_z_distance': self.total_z_distance,
            'x_size_um': self.x_size_um,
            'y_size_um': self.y_size_um
        }

    def _parse_nikon(self):
        """Parses the description string from the Nikon metadata for key information.

        Returns: 
            None
        """
        self._reshape = False
        try:
            meta = {k: v for k, v in [line.replace(' ', '').split(':') 
                for line in self.data_pointer['description'].split('\n') 
                if len(line.split(':')) == 2]}
            self.piezo_mode = True
            self.nb_of_loops = int(meta['TimeLoop'])
            self.nb_of_planes = int(float(meta['ZStackLoop']))
            self.z_spacing_um = float(meta['-Step'].replace('_m', ''))
            self.total_z_distance = (self.nb_of_planes - 1) * self.z_spacing_um
            self.x_size_um = float(self.data_pointer['calibration']) * int(self.data_pointer['width'])
            self.y_size_um = float(self.data_pointer['calibration']) * int(self.data_pointer['height'])
        except KeyError as e:
            raise DataMissingError('Local z stack metadata missing key {}'.format(e))

    def _parse_scientifica(self):
        """Parses the scientifica INI file for key information.

        Returns:  
            None
        """
        self._reshape = False
        try:
            if self.data_pointer['piezo.mode'] == 'TRUE':
                self.piezo_mode = True
                self.nb_of_loops = int(float(self.data_pointer['no.of.cycles.to.scan']))
                self.nb_of_planes = int(float(self.data_pointer['frames.per.z.cycle']))
            else:
                self.piezo_mode = False
                self.nb_of_loops = int(2 * float(self.data_pointer['no.of.cycles.to.scan']))
                self.nb_of_planes = int(float(self.data_pointer['frames.per.z.cycle']) / 2)
            self.x_size_um = float(self.data_pointer['x.pixels']) * float(self.data_pointer['x.pixel.sz']) * 1e6
            self.y_size_um = float(self.data_pointer['y.pixels']) * float(self.data_pointer['y.pixel.sz']) * 1e6
            self.total_z_distance = float(self.data_pointer['total.z.distance'])
            self.z_spacing_um = self.total_z_distance / self.nb_of_planes
        except KeyError as e:
            raise DataMissingError('Local z stack metadata missing key {}'.format(e))

    def _parse_mesoscope(self):
        """Parses the mesoscope metadata for key information. 

        Temporary for off-pipeline work.

        Returns:  
            None
        """
        self._reshape = False
        try:
            if self.data_pointer == 'None':
                self.piezo_mode = True
                self.nb_of_loops = int(20)
                self.nb_of_planes = int(81) 
                self.z_spacing_um = float(0.75)
                self.total_z_distance = (self.nb_of_planes - 1) * self.z_spacing_um 
                self.x_size_um = 1
                self.y_size_um = 1              
            else:
                if self.data_pointer['hFastZ']['waveformType'] == 'sawtooth':
                    self.piezo_mode = True
                else:
                    raise NotImplementedError('Unhandled scan pattern: {}.'.format(self.data_pointer['hFastZ']['waveformType'])) 
                self.nb_of_loops = int(float(self.data_pointer['hFastZ']['numVolumes']))
                self.nb_of_planes = int(float(self.data_pointer['hFastZ']['numFramesPerVolume'])) 
                self.z_spacing_um = float(self.data_pointer['hStackManager']['stackZStepSize'])
                self.total_z_distance = (self.nb_of_planes - 1) * self.z_spacing_um 

                # TODO: The following is based on temporary metadata files, self-made
                roi = self.data_pointer['imagingRoiGroup']['roi']['scanfields']
                self.x_size_um = roi['pixelResolutionXY'][0] * 1.26
                self.y_size_um = roi['pixelResolutionXY'][1] * 1.26
        except KeyError as e:
            raise DataMissingError('Local z stack metadata missing key {}'.format(e))

    def _parse_deepscope(self):
        """Parses the deepscope metadata for key information.

        Note: deepscope meta not currently being saved.

        Returns:  
            None
        """
        from numpy import inf, nan
        if self.data_pointer == 'None' or 'FrameData' not in eval(self.data_pointer.value).keys():
            #missing metadata
            self._reshape = False
            self.piezo_mode = True
            self.nb_of_loops = int(45)
            self.nb_of_planes = int(62) 
            self.z_spacing_um = float(0.75)
            self.total_z_distance = (self.nb_of_planes - 1) * self.z_spacing_um 
            self.x_size_um = float(400/512)
            self.y_size_um = float(400/512) 
            # self._newshape = (self.nb_of_planes,
            #                 int(1), #framesPerSlice
            #                 int(512), #linesPerFrame
            #                 int(512)) #pixelsPerLine
        else:
            self.data_pointer = eval(self.data_pointer.value)

            self.piezo_mode = True
            self._reshape = False  # 20180917 - implemented averaging on the rig, change to True for off rig averaging
            if self.data_pointer['FrameData']['SI.hScan2D.logAverageFactor'] == 1:
                self.nb_of_loops = int(self.data_pointer['FrameData']['SI.hFastZ.numVolumes'])
            else:
                self.nb_of_loops = 1
            self.nb_of_planes = int(self.data_pointer['FrameData']['SI.hStackManager.numSlices'])
            self.z_spacing_um = float(self.data_pointer['FrameData']['SI.hStackManager.stackZStepSize'])

            self.total_z_distance = (self.nb_of_planes - 1) * self.z_spacing_um

            self._newshape = (self.nb_of_planes,
                            int(self.data_pointer['FrameData']['SI.hStackManager.framesPerSlice']),
                            int(self.data_pointer['FrameData']['SI.hRoiManager.linesPerFrame']),
                            int(self.data_pointer['FrameData']['SI.hRoiManager.pixelsPerLine']))


            self.x_size_um = 400.0
            self.y_size_um = 400.0

