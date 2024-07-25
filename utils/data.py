import numpy as np
import torch
from torch.utils.data import Dataset

import os
import re

import csv
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
        self.label_mode = None

    def extract_ROI(self, ROI_size, sampling_rate, label_mode='center'):
        '''
        Args:
        ROI_size (tuple): 
            The size of the ROI to be extracted from the image
        sampling_rate (float):
            The sampling rate to be used for the extraction of the ROI for a given slice
        label_mode (str):
            The mode to be used for the label data extraction. Options include 'center' and 'all'. 
            - If 'center' is selected, the label data for the center pixel of the ROI is extracted. 
            - If 'mode' is selected, the mode of the labels data is extracted (most frequent label).
            - If 'all' is selected, the labels remain an n x n matrix
            where n is the size of the ROI. Default is 'center'.
        '''
        # The size of the ROI to be extracted from the image
        self.ROI_size = ROI_size
        self.label_mode = label_mode

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
            assert label_mode in ['center', 'all', 'mode'], f'Invalid label assignment {label_mode}'
            assert num_samples > 0, f'No ROIs to be extracted from image {npz_file}'
            assert num_samples <= total_rois, f'Number of ROIs to be extracted {num_samples} is greater than total ROIs {total_rois}'
            assert nrows > 0 and ncols > 0, f'ROI size {self.ROI_size} is too large for image size {Ishape[1:]}'

            # get all possible patch coordinates for the ROIs and sample from them (in units of patch size)
            all_coords = [(i, j) for i in range(nrows) for j in range(ncols)]
            ROI_coords = rand.sample(all_coords, num_samples)

            for row, col in ROI_coords:

                # get the ROI and label for the given row and column
                rois = I[:, row*self.ROI_size[0]:(row+1)*self.ROI_size[0], col*self.ROI_size[1]:(col+1)*self.ROI_size[1]]
                lbls = L[:, row*self.ROI_size[0]:(row+1)*self.ROI_size[0], col*self.ROI_size[1]:(col+1)*self.ROI_size[1]]
                
                # get center pixel coordinates for the ROI
                x = row*self.ROI_size[0] + self.ROI_size[0]//2
                y = col*self.ROI_size[1] + self.ROI_size[1]//2

                #NOTE: could do mode of all labels, or take all the labels, (do x,y for labels at center of ROI)
                # label chosen as center coordinates for the ROI
                if label_mode == 'center':
                    lbls = lbls[:, self.ROI_size[0]//2, self.ROI_size[1]//2]
                if label_mode == 'all':
                    lbls = lbls
                if label_mode == 'mode':
                    lbls = np.array([np.argmax(np.bincount(lbls.flatten()))])


                # save the extracted ROI as a npz file
                self._save_npz(npz_file, rois, lbls, (x, y))
        print(f"All ROIs extracted from {self.extract_dir} and saved to {self.save_dir} under train and test subfolders")

    def _save_npz(self, npz_file, rois, lbls, coords):
        # strip npz at the end of the file name: C57BL6J-638850.13_lr_0.npz -> C57BL6J-638850.13_lr_0
        npz_name = npz_file.split('.')[0] + npz_file.split('.')[1]

        # if is left brain, save to train folder
        if npz_file.split('_')[-1].split('.')[0] == '1':
            sub_dir = os.path.join(self.save_dir, 'train')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

        # if is right brain, save to test folder
        elif npz_file.split('_')[-1].split('.')[0] == '0':
            sub_dir = os.path.join(self.save_dir, 'test')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

        # Save the extracted ROI as a npz file
        np.savez(os.path.join(sub_dir, f'{npz_name}._{coords[0]}_{coords[1]}_{self.label_mode}.npz'), I=rois, L=lbls)

class AtlasDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None, label_map_file=None or str):
        '''
        Args:
        data_dir (str): 
            The directory containing the npz files with the image data and label data of slices
        transform (callable, optional):
            A function/transform that takes in the image data and returns a transformed version
        target_transform (callable, optional):
            A function/transform that takes in the label data and returns a transformed version
        label_map_file (str, optional):
            The path to the csv file containing the level label data mapped to the corresponding string name and index
            These should be divisions.csv, organs.csv, structures.csv, substructures.csv, categories.csv
            and contain | number  name, indices | (e.g. 1, 'brain', [0])
        '''
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_map_file = label_map_file
        self.label_map = self._map_label(self.label_map_file)

        # Check if the label map file exists and is a csv file
        if self.label_map_file is not None:
            assert os.path.exists(self.label_map_file), f'Label map file {self.label_map_file} does not exist'
            assert os.path.isfile(self.label_map_file), f'Path {self.label_map_file} is not a file'
            assert self.label_map_file.endswith('.csv'), f'Label map file {self.label_map_file} is not a csv file'

        # Check if the data directory exists and is a directory, and if it is not empty
        assert os.path.exists(self.data_dir), f'Directory {self.data_dir} does not exist'
        assert os.path.isdir(self.data_dir), f'Path {self.data_dir} is not a directory'
        assert len(self) > 0, f'Directory {self.data_dir} is empty'

        self.data_dirs = os.listdir(self.data_dir)

    def _map_label(self, file):
        '''
        Maps the label data to the corresponding string name and index
        
        Args:
        file (str):
            The path to the csv file containing the label data
            
        Returns:
        label_map (dict):
            A dictionary containing the label data mapped to the corresponding string name and index
            Our label_map will consists of {label: {name: str, index: int}} where label is the parcellation value, 
            name is the string name, and index is the index that we will use as label
        '''
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            label_map = {}

            # Remove any unwanted characters from the label data and convert to a list of integers
            for row in reader:
                rep = {'\n': '', '\t': '', ']': '', '[': ''}
                rep = dict((re.escape(k), v) for k, v in rep.items())
                pattern = re.compile("|".join(rep.keys()))
                row[2] = row[2].strip()
                row[2] = pattern.sub(lambda m: rep[re.escape(m.group(0))], row[2])
                row[2] = row[2].split(' ')
                row[2] = [i for i in row[2] if i not in ['']]
                row[2] = [int(i) for i in row[2]]
                for value in row[2]:
                    label_map[value] = {'name': row[1], 'index': int(row[0])}
        return label_map

    def __len__(self):
        '''
        Returns the number of npz files in the data directory
        '''
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        '''
        Returns the image data and label data of the npz file at the given index
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        I, L = np.load(os.path.join(self.data_dir, self.data_dirs[idx])).values()
        
        assert len(I.shape) == 3, f'Image data has incorrect shape {I.shape}'

        # Any transformation to the image data (normalization, to tensor, etc...)
        # Since train and test data are saved in separate folders, it is appropriate
        # to apply augmentation at this stage already (so long as any further splits 
        # into val & test are done on test data - not train).
        if self.transform:
            I = self.transform(I)

        if self.label_map is not None:
            for label in np.unique(L):
                L[L == label] = self.label_map[label]['index']

        if self.target_transform:
            L = self.target_transform(L)

        return I, L