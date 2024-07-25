import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, parent_dir)

from models.benchmark_models2 import *
import torchvision.transforms as transforms
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassAUROC
import torch.utils.data as data
import torch
from medmnist import INFO
import medmnist
import argparse
import numpy as np
import time
from tqdm import tqdm
import json
from comet_ml import Experiment
from torch import profiler

PRESETS ={
    "VanillaCNN" : VanillaCNN,
    "TrivialECNN": TrivialECNN,
    "TrivialIrrepECNN": TrivialIrrepECNN,
    "RegularECNN": RegularECNN,
    "TrivialMoment": TrivialMoment,
    "TrivialIrrepMoment": TrivialIrrepMoment,
}

experiment = Experiment(
    api_key="Fl7YrvyVQDhLRYuyUfdHS3oE8",
    project_name="equiv_benchmark",
    workspace="joeshmoe03",
)

def calculate_metrics(model, loader, device, num_classes):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    batch_losses = []
    acc = MulticlassAccuracy(num_classes=num_classes, device=device)
    precision = MulticlassPrecision(num_classes=num_classes, device=device)
    recall = MulticlassRecall(num_classes=num_classes, device=device)
    auroc = MulticlassAUROC(num_classes=num_classes, device=device)
    n_batches = len(loader)
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).squeeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_losses.append(loss.item())
            acc.update(outputs, labels)
            precision.update(outputs, labels)
            recall.update(outputs, labels)
            auroc.update(outputs, labels)

    valid_loss = np.mean(batch_losses)
    valid_acc = acc.compute().item()
    valid_precision = precision.compute().item()
    valid_recall = recall.compute().item()
    valid_auroc = auroc.compute().item()

    acc.reset()
    precision.reset()
    recall.reset()
    auroc.reset()

    return valid_loss, valid_acc, valid_precision, valid_recall, valid_auroc

class CircleCrop(object):
    def __init__(self, h, w, center=None, radius=None):
        self.h = h
        self.w = w
        self.center = center
        self.radius = radius

    def __call__(self, img):
        if self.center is None:
            self.center = (self.h // 2, self.w // 2)
        if self.radius is None:
            self.radius = min(self.h, self.w) // 2

        y, x = np.ogrid[:self.h, :self.w]
        dist_from_center = np.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2)
        mask = dist_from_center <= self.radius
        img[:, ~mask] = 0
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + f"(h={self.h}, w={self.w}, center={self.center}, radius={self.radius})"

def load_data(args):

    data_flag = args.data_flag
    download = True
    info = INFO[data_flag]
    task = info['task']
    img_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms to convert from image to normalized tensor (or more if augmentation, but not for testing purposes)
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        #CircleCrop(28, 28, radius = 14, center = (13.5, 13.5)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # separate transforms for test
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        #CircleCrop(28, 28, radius = 14, center = (13.5, 13.5)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = DataClass(split="train", transform=train_transforms, download=download)
    valid_dataset = DataClass(split="val", transform=test_transforms, download=download)
    test_dataset = DataClass(split="test", transform=test_transforms, download=download)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader, img_channels, n_classes, device

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

def main(args):
    train_loader, valid_loader, test_loader, img_channels, n_classes, device = load_data(args)
    model = PRESETS[args.model](img_channels, args.n_channels, n_classes, args.kernel_size, args.padding, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = Metrics()

    # initialize the metrics
    metrics.update(best_acc=0, best_auc=0, auc_on_best_acc=0, acc_on_best_auc=0, final_acc=0, final_auc=0)

    # training loop
    for epoch in range(args.epochs):
        model.train()

        # start timing the epoch
        start_time = time.time()

        # loop through the batches
        for i, (images, labels) in enumerate(train_loader):
            batch_losses = []

            # images to configured device. NOTE: labels squeezed for every dataset?
            images = images.to(device)
            labels = labels.to(device).squeeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass and backward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        # end timing the epoch save the duration
        end_time = time.time()
        total_time = end_time - start_time
        train_loss = np.mean(batch_losses)

        # validation loop: evaluate at given interval or at the end
        if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()

            valid_loss, valid_acc, valid_prec, valid_rec, valid_auc = calculate_metrics(model, valid_loader, device, n_classes)
            metrics.append(train_losses=train_loss, 
                           valid_losses=valid_loss, 
                           valid_accuracies=valid_acc, 
                           valid_precision=valid_prec, 
                           valid_recall=valid_rec, 
                           valid_AUCROC=valid_auc, 
                           epoch_times=total_time)

            if valid_acc > metrics.metrics["best_acc"]:
                metrics.update(best_acc=valid_acc, 
                               auc_on_best_acc=valid_auc)
                if args.save_on_best in ["acc", "acc_auc", "all"]:
                    path = os.path.join("results", f"{args.data_flag}", f"run{args.run}", f"best_acc_{args.model}_{args.data_flag}{args.run}.pt")
                    torch.save(model.state_dict(), path)

            if valid_auc > metrics.metrics["best_auc"]:
                metrics.update(best_auc=valid_auc, 
                               acc_on_best_auc=valid_acc)
                if args.save_on_best in ["auc", "acc_auc", "all"]:
                    path = os.path.join("results", f"{args.data_flag}", f"run{args.run}", f"best_auc_{args.model}_{args.data_flag}{args.run}.pt")
                    torch.save(model.state_dict(), path)

            if epoch == args.epochs - 1:
                metrics.update(final_acc=valid_acc, 
                               final_auc=valid_auc)
                if args.save_on_best in ["final", "all"]:
                    path = os.path.join("results", f"{args.data_flag}", f"run{args.run}", f"final_{args.model}_{args.data_flag}{args.run}.pt")
                    torch.save(model.state_dict(), path)

            # log the metrics to comet
            experiment.log_metric("train_loss", train_loss)
            experiment.log_metric("val_loss", valid_loss)
            experiment.log_metric("val_acc", valid_acc)
            experiment.log_metric("val_auc", valid_auc)
            experiment.log_metric("val_prec", valid_prec)
            experiment.log_metric("val_rec", valid_rec)

    # Save the metrics to a file
    path = os.path.join("results", f"{args.data_flag}", f"run{args.run}", f"metrics_{args.model}_{args.data_flag}{args.run}.json")
    with open(path, "w") as f:
        json.dump(metrics.metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_flag", type=str, action="store", dest="data_flag", default="bloodmnist", help="The dataset to use: pathmnist, dermamnist, bloodmnist, etc...")
    parser.add_argument("--model", type=str, action="store", dest="model", default="VanillaCNN", help="The model to use")
    parser.add_argument("--batch_size", type=int, action="store", dest="batch_size", default=128, help="The batch size to use")
    parser.add_argument("--n_channels", type=int, action="store", dest="n_channels", default=32, help="The number of channels to use")
    parser.add_argument("--kernel_size", type=int, action="store", dest="kernel_size", default=3, help="The kernel size to use")  
    parser.add_argument("--padding", type=int, action="store", dest="padding", default=1, help="The padding to use")
    parser.add_argument("--num_layers", type=int, action="store", dest="num_layers", default=5, help="The number of layers to use")
    parser.add_argument("--epochs", type=int, action="store", dest="epochs", default=100, help="The number of epochs to use")
    parser.add_argument("--lr", type=float, action="store", dest="lr", default=1e-4, help="The learning rate to use")
    parser.add_argument("--val_interval", type=int, action="store", dest="val_interval", default=1, help="The validation interval to use")
    parser.add_argument("--save_on_best", type=str, action="store", dest="save_on_best", default="acc", help="Whether to save on best: acc, auc, acc_auc, final, all")
    parser.add_argument("--run", type=int, action="store", dest="run", default=0, help="The run number to use")
    args = parser.parse_args()
    
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)
    os.chdir(parent_dir)
    path = os.path.join(parent_dir, "results", f"{args.data_flag}", f"run{args.run}")
    print(f"Path: {path}")
    if not os.path.exists(path):
        os.makedirs(path)

    main(args)