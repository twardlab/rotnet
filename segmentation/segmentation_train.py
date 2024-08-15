import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, parent_dir)

import argparse
from utils.data import ExtractROI, AtlasDataset
from models.unet import *
import moment_kernels as mk

import torch # type: ignore
import numpy as np # type: ignore
from torch import nn # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import torchvision.transforms as transforms # type: ignore
from torch.optim import Adam # type: ignore
from monai.losses import DiceCELoss, DiceLoss, MaskedDiceLoss # type: ignore
from monai.metrics import HausdorffDistanceMetric, DiceMetric, MeanIoU # type: ignore
from monai.transforms import AsDiscrete, EnsureType # type: ignore
import torch.nn.functional as F # type: ignore
from torch.nn.functional import one_hot # type: ignore
from sklearn.decomposition import IncrementalPCA # type: ignore

import json
from tqdm import tqdm # type: ignore
from comet_ml import Experiment # type: ignore

PRESETS = {
    'unet': UNet
    # TODO: include trivial moment unet implementation. Should be saved and imported from models/moment_unet.py -> 'moment_unet': MomentUNet
}

#NOTE: for anyone else eventually using this, you'll need to replace these with your own comet_ml API key, project name, and workspace or remove experiment logging to comet_ml entirely
experiment = Experiment(
    api_key="API_KEY", # NOTE - replace with your own comet_ml API key
    project_name="PROJECT_NAME", # NOTE - replace with your own comet_ml project name
    workspace="WORKSPACE" # NOTE - replace with your own comet_ml workspace,
)

class Metrics():
    def __init__(self):
        '''
        This class is used to store metrics for a given model. It can be updated with a single value or appended with a list of values.
        '''
        self.metrics = {}

    def update(self, **kwargs):
        '''Updates for a single value
        
        Args:
            **kwargs: key-value pairs to update
        '''

        for key, value in kwargs.items():
            self.metrics[key] = value

    def append(self, **kwargs):
        '''Appends for a list of values
        
        Args:
            **kwargs: key-value pairs to append
        '''

        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = [value]
            else:
                self.metrics[key].append(value)
    
    def __getitem__(self, key):
        if key not in self.metrics:
            raise KeyError(f"{key} not in metrics")
        return self.metrics[key]

def load_data(args):
    assert args.data_dir is not None, "Please provide a data flag"
    assert args.label_flag is not None, "Please provide a label flag"
    assert os.path.exists(args.data_dir), f"Data flag {args.data_dir} does not exist"
    assert os.path.exists(args.label_flag), f"Label flag {args.label_flag} does not exist"
    assert os.path.isfile(args.label_flag), f"Label flag {args.label_flag} is not a file"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Optional transforms to apply to either the data or the labels. Default is no transforms
    train_transforms = transforms.Compose([])
    test_transforms = transforms.Compose([])
    label_transforms = transforms.Compose([])

    # Load the data
    train_data = AtlasDataset(data_dir = os.path.join(args.data_dir, "train"), label_map_file = args.label_flag)
    test_data = AtlasDataset(data_dir = os.path.join(args.data_dir, "test"), label_map_file = args.label_flag)
    num_classes = train_data.num_classes

    # Prepare the data loaders with batching
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False)
    return train_loader, test_loader, device, num_classes

# soft hausdorff distance (directly with porbability masks instead of the labels)
# randomly pick 1 point from backroung and from foreground, what is the expected value of the square distance between them
# random means chosen on the probability in our map: BAD IDEA??

class CustomHausdorffDistanceMetric(HausdorffDistanceMetric):
    def __init__(self, include_background: bool = True, distance_metric: str = "euclidean"):
        '''
        NOTE: This class is not working as intended. There seems to be some weird edge case I am not accounting for where suddenly a tensor is not being passed in as expected on some unknown iteration.
        TODO: Reproduce issue, troubleshoot, and fix this class to work as intended.

        This class is used to calculate the Hausdorff distance between two tensors. It is a subclass of the HausdorffDistanceMetric class from MONAI.
        The intent with this custom class for hausdorff distance is to account for scenarios where labels are missing in one or both tensors of the prediction and the truth.
        Hausdorff is not defined when the labels are missing, so we default to the max hausdorff distance in this case.

        Args:
            include_background: 
                whether to include the background class in the calculation
            distance_metric: 
                the distance metric to use    
        '''
        super().__init__(include_background=include_background, distance_metric=distance_metric)

    def __call__(self, y_pred, y): # y_pred is a tensor of shape (batch_size, num_classes, height, width)
        distances = []
        for batch_idx in range(y_pred.shape[0]):
            for class_idx in range(y_pred.shape[1]):
                pred = y_pred[batch_idx, class_idx, :, :]
                true = y[batch_idx, class_idx, :, :]

                # if both truth and prediction are empty, then the distance is 0
                if torch.sum(true) == 0 and torch.sum(pred) == 0:
                    distance = 0
                    distances.append(distance)
                elif torch.sum(true) == 0 or torch.sum(pred) == 0:

                    # default to the max hausdorff distance if class is missing in either the prediction xor the truth
                    height, width = pred.shape[0], pred.shape[1]
                    distance = np.sqrt((height - 1)**2 + (width - 1)**2)
                    distances.append(distance)
                elif torch.sum(true) > 0 and torch.sum(pred) > 0: 
                    distance = super().__call__(y_pred=pred.unsqueeze(0).unsqueeze(0), y=true.unsqueeze(0).unsqueeze(0))
                    distances.append(distance.item())
        return np.mean(distances)

def calculate_metrics(model, test_loader, device):
    '''
    This function is used to calculate the metrics for a given model on a test set.
    '''
    model.eval()
    criterion = DiceLoss(to_onehot_y=True, softmax=True)
    as_discrete = AsDiscrete(argmax=True, to_onehot=test_loader.dataset.num_classes, dim=1)
    ensure_type = EnsureType()
    #hausdorff_distance = CustomHausdorffDistanceMetric() # NOTE - this is not working
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    IOU_metric = MeanIoU(include_background=True)
    batch_losses = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            outputs = ensure_type(outputs)
            labels = ensure_type(labels)
            outputs = as_discrete(outputs)
            labels = as_discrete(labels)

            batch_losses.append(loss.item())

            # Compute the metrics
            dice_metric(y_pred = outputs, y = labels)
            IOU_metric(y_pred = outputs, y = labels)
            #hausdorff_distance(y_pred = outputs, y = labels) # NOTE - this is not working

    # Get the metrics
    hausdorff_metric = 0 #hausdorff_distance.aggregate().item() # NOTE - this is not working
    dice_metric = dice_metric.aggregate().item()
    IOU_metric = IOU_metric.aggregate().item()

    test_loss = np.mean(batch_losses)
    return test_loss, hausdorff_metric, dice_metric, IOU_metric

def main(args):
    train_loader, test_loader, device, num_classes = load_data(args)
    model = PRESETS[args.model](args.img_channels, args.num_channels, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr = args.lr)
    criterion = DiceLoss(to_onehot_y=True, softmax=True)
    as_discrete = AsDiscrete(argmax=True, to_onehot=train_loader.dataset.num_classes, dim=1)
    metrics = Metrics()
    batch_losses = []
    best_DiceCE_loss = np.inf

    for epoch in range(args.epochs):
        model.train()

        for i, (images, labels) in enumerate(train_loader):
            batch_losses.clear()

            images = images.to(device)
            labels = labels.to(device)

            assert torch.is_tensor(images), "Image is not a tensor"
            assert torch.is_tensor(labels), "Labels are not a tensor"

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)

        # for every epoch where we want to validate or the last epoch
        if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()

            test_loss, hausdorff_distance, dice_metric, IOU_metric = calculate_metrics(model, test_loader, device)
            metrics.append(test_loss=test_loss,
                           train_loss=train_loss,
                           hausdorff_distance=hausdorff_distance,
                           dice_metric=dice_metric,
                           IOU_metric=IOU_metric)
            
            experiment.log_metric("train Dice loss", train_loss)
            experiment.log_metric("test Dice loss", test_loss)
            experiment.log_metric("test Hausdorff distance", hausdorff_distance)
            experiment.log_metric("test Dice metric", dice_metric)
            experiment.log_metric("test IOU metric", IOU_metric)

        if test_loss < best_DiceCE_loss:
            best_DiceCE_loss = test_loss
            torch.save(model.state_dict(), os.path.join("results", f"run{args.run}", f"segmentation", f"{args.model}", f"{args.model}{args.run}.pt"))

    with open(os.path.join("results", f"run{args.run}", f"segmentation", f"{args.model}", f"metrics_{args.model}{args.run}.json"), "w") as f:
        json.dump(metrics.metrics, f)     
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, action="store", dest="data_dir", default="/nafs/dtward/allen/rois/64x64_sample_rate_0_5_all", help="The dataset path to use: could be .../NxN_sample_rate_X_Z_all, etc...")
    parser.add_argument("--label_flag", type=str, action="store", dest="label_flag", default="/nafs/dtward/allen/rois/divisions.csv", help="The label mapping path to use: ../categories.csv, ../divisions.csv, ../organs.csv, ../structures.csv, ../substructures.csv")
    parser.add_argument("--model", type=str, action="store", dest="model", default="unet", help="The model to use: unet")
    parser.add_argument("--batch_size", type=int, action="store", dest="batch_size", default=8, help="The batch size to use")
    parser.add_argument("--img_channels", type=int, action="store", dest="img_channels", default=501, help="The number of image channels")
    parser.add_argument("--num_channels", type=int, action="store", dest="num_channels", default=16, help="The number of channels to use out of the first layer")
    parser.add_argument("--epochs", type=int, action="store", dest="epochs", default=100, help="The number of epochs to use")
    parser.add_argument("--lr", type=float, action="store", dest="lr", default=1e-3, help="The learning rate to use")
    parser.add_argument("--val_interval", type=int, action="store", dest="val_interval", default=1, help="The validation interval to use")
    parser.add_argument("--run", type=int, action="store", dest="run", default=0, help="The run number to use")
    args = parser.parse_args()

    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)
    os.chdir(parent_dir)
    path = os.path.join(parent_dir, "results", f"run{args.run}", f"segmentation", f"{args.model}")
    print(f"Path: {path}")
    if not os.path.exists(path):
        os.makedirs(path)

    main(args)
