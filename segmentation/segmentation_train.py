import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, parent_dir)

import argparse
from utils.data import ExtractROI, AtlasDataset
from models.moment_unet import *
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
from monai.transforms import AsDiscrete # type: ignore
import torch.nn.functional as F # type: ignore

import json
from tqdm import tqdm # type: ignore
from comet_ml import Experiment # type: ignore

PRESETS = {
    'moment_unet': MomentUNet,
    'unet': UNet
}

#NOTE: for anyone else eventually using this, you'll need to replace these with your own comet_ml API key, project name, and workspace or remove experiment logging to comet_ml entirely
experiment = Experiment(
    api_key="Fl7YrvyVQDhLRYuyUfdHS3oE8",
    project_name="segmentation_benchmark",
    workspace="joeshmoe03",
)

class Metrics():
    def __init__(self):
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
    
def to_onehot(y, num_classes):
    segmentation_map = y.long()
    segmentation_map = segmentation_map.squeeze(1)

    batch_size, height, width = segmentation_map.size()
    segmentation_map = segmentation_map.view(batch_size, -1)

    one_hot_map = F.one_hot(segmentation_map, num_classes)
    one_hot_map = one_hot_map.view(batch_size, height, width, num_classes)
    one_hot_map = one_hot_map.permute(0, 3, 1, 2)
    return one_hot_map

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

def calculate_metrics(model, test_loader, device):
    model.eval()
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    hausdorff_distance = HausdorffDistanceMetric(include_background=True, distance_metric="euclidean")
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    IOU_metric = MeanIoU(include_background=True)
    batch_losses = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            labels = to_onehot(labels, test_loader.dataset.num_classes)
            batch_losses.append(loss.item())

            # Compute the metrics
            hausdorff_distance(y_pred = outputs, y = labels)
            dice_metric(y_pred = outputs, y = labels)
            IOU_metric(y_pred = outputs, y = labels)

    # Get the metrics
    hausdorff_distance = hausdorff_distance.aggregate().item()
    dice_metric = dice_metric.aggregate().item()
    IOU_metric = IOU_metric.aggregate().item()

    test_loss = np.mean(batch_losses)
    return test_loss, hausdorff_distance, dice_metric, IOU_metric

def main(args):
    train_loader, test_loader, device, num_classes = load_data(args)
    model = PRESETS[args.model](args.img_channels, args.num_channels, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr = args.lr)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
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
            
        experiment.log_metric("train DiceCE loss", train_loss)
        experiment.log_metric("test DiceCE loss", test_loss)
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
    parser.add_argument("--model", type=str, action="store", dest="model", default="moment_unet", help="The model to use: moment_unet or unet")
    parser.add_argument("--batch_size", type=int, action="store", dest="batch_size", default=8, help="The batch size to use")
    parser.add_argument("--img_channels", type=int, action="store", dest="img_channels", default=501, help="The number of image channels")
    parser.add_argument("--num_channels", type=int, action="store", dest="num_channels", default=16, help="The number of channels to use out of the first layer")
    parser.add_argument("--epochs", type=int, action="store", dest="epochs", default=10, help="The number of epochs to use")
    parser.add_argument("--lr", type=float, action="store", dest="lr", default=1e-4, help="The learning rate to use")
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
