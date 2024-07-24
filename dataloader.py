import numpy as np
import torch

import os

import math
import random as rand
from tqdm import tqdm

class ExtractROI():
    def __init__(self, extract_dir, save_dir):
        '''
        Extracts ROIs from the image data and saves them as npz files in the save_dir. This dataloader was buit for
        the Allen Institute Brain Atlas. We separate the data into training and testing data based on the hemisphere.

        Args:
        extract_dir (str): 
            The directory containing the npz files with the image data and label data of slices
        save_dir (str): 
            The directory to save the extracted ROIs as npz files
        '''
        self.extract_dir = extract_dir
        self.save_dir = save_dir
        self.ROI_size = None
        self.sampling_rate = None

    def extract_ROI(self, ROI_size, sampling_rate):
        '''
        Args:
        ROI_size (tuple): 
            The size of the ROI to be extracted from the image
        sampling_rate (float):
            The sampling rate to be used for the extraction of the ROI for a given slice
        '''
        # The size of the ROI to be extracted from the image
        self.ROI_size = ROI_size

        # The sampling rate to be used for the extraction of the ROI for a given slice
        self.sampling_rate = sampling_rate

        for npz_file in tqdm(os.listdir(self.extract_dir)):
            if not npz_file.endswith('.npz'):
                continue

            # I is the image data, L is the label data
            I, L = np.load(os.path.join(self.extract_dir, npz_file)).values()
            
            # Get the shape of the image data and label data
            Ishape = I.shape
            Lshape = L.shape

            # I is 501 x rows x cols: get the number of ROIs that can be extracted from the image on rows x cols
            nrows = Ishape[1]//self.ROI_size[0]
            ncols = Ishape[2]//self.ROI_size[1]

            # Calculate the total number of ROIs that can be extracted from the image and the number of ROIs to be extracted
            total_rois = nrows * ncols
            num_samples = math.ceil(total_rois * self.sampling_rate)

            # Check if there are ROIs to be extracted from the image and if the number of ROIs to be extracted is less than the total ROIs
            assert num_samples > 0, f'No ROIs to be extracted from image {npz_file}'
            assert num_samples <= total_rois, f'Number of ROIs to be extracted {num_samples} is greater than total ROIs {total_rois}'
            assert nrows > 0 and ncols > 0, f'ROI size {self.ROI_size} is too large for image size {Ishape[1:]}'

            # get all possible coordinates for the ROIs and sample from them
            all_coords = [(i, j) for i in range(nrows) for j in range(ncols)]
            ROI_coords = rand.sample(all_coords, num_samples)

            for row, col in ROI_coords:

                # get the ROI and label for the given row and column
                rois = I[:, row*self.ROI_size[0]:(row+1)*self.ROI_size[0], col*self.ROI_size[1]:(col+1)*self.ROI_size[1]]
                lbls = L[:, row*self.ROI_size[0]:(row+1)*self.ROI_size[0], col*self.ROI_size[1]:(col+1)*self.ROI_size[1]]
                
                # get center pixel coordinates for the ROI
                x = row*self.ROI_size[0] + self.ROI_size[0]//2
                y = col*self.ROI_size[1] + self.ROI_size[1]//2

                # save the extracted ROI as a npz file
                self._save_npz(npz_file, rois, lbls, (x, y))

    def _save_npz(self, npz_file, rois, lbls, coords):
        npz_name = npz_file

        # if is left brain, save to train folder
        if npz_name.split('_')[2] == '1.npz':
            sub_dir = os.path.join(self.save_dir, 'train')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

        # if is right brain, save to test folder
        elif npz_name.split('_')[2] == '0.npz':
            sub_dir = os.path.join(self.save_dir, 'test')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            # Save the extracted ROI as a npz file
            np.savez(os.path.join(sub_dir, f'{npz_file.split(".")[0]}_{coords[0]}_{coords[1]}.npz'), I=rois, L=lbls)