import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, parent_dir)

from utils.data import ExtractROI, AtlasDataset
from models.moment_unet import *
from models.unet import *
import moment_kernels as mk

import torch # type: ignore
from torch import nn # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import torchvision.transforms as transforms # type: ignore

from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassAUROC # type: ignore
import json
from tqdm import tqdm # type: ignore
from comet_ml import Experiment # type: ignore


