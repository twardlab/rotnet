from benchmark_models import *
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
import sys
import os

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
    acc = MulticlassAccuracy()
    precision = MulticlassPrecision()
    recall = MulticlassRecall()
    auroc = MulticlassAUROC(num_classes = num_classes)
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

    return np.mean(batch_losses), acc.compute().item(), precision.compute().item(), recall.compute().item(), auroc.compute().item()

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
        transforms.Normalize(mean = [0.5], std = [0.5]),
    ])

    # separate transforms for test
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5])
    ])

    train_dataset = DataClass(split = "train", transform = train_transforms, download = download)
    valid_dataset = DataClass(split = "val", transform = test_transforms, download = download)
    test_dataset = DataClass(split = "test", transform = test_transforms, download = download)

    train_loader = data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
    valid_loader = data.DataLoader(dataset = valid_dataset, batch_size = args.batch_size, shuffle = False)
    test_loader = data.DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False)

    return train_loader, valid_loader, test_loader, img_channels, n_classes, device

metrics = {
    "train_losses": [],
    "valid_losses": [],
    "valid_accuracies": [],
    "valid_precision": [],
    "valid_recall": [],
    "valid_AUCROC": [],
    "best_acc": 0,
    "best_auc": 0,
    "best_acc_on_best_auc": 0,
    "best_auc_on_best_acc": 0,
    "epoch_times": [],
    "final_acc": 0,
    "final_auc": 0
}

def main(args):
    train_loader, valid_loader, test_loader, img_channels, n_classes, device = load_data(args)
    model = PRESETS[args.model](img_channels, args.n_channels, n_classes, args.kernel_size, args.padding, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    prof = profiler.profile(
        activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        profile_memory = True,
        with_flops = True,
    )

    # training loop and profiling
    prof.start()
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
        metrics["epoch_times"].append(end_time - start_time)
        metrics["train_losses"].append(np.mean(batch_losses))

        # validation loop: evaluate at given interval or at the end
        if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            valid_loss, valid_acc, valid_prec, valid_rec, valid_auc = calculate_metrics(model, valid_loader, device, n_classes)

            if valid_acc > metrics["best_acc"]:
                metrics["best_acc"] = valid_acc
                metrics["best_auc_on_best_acc"] = valid_auc
                if args.save_on_best in ["acc", "acc_auc", "all"]:
                    path = os.path.join(f"run{args.run}", f"best_acc_{args.model}_{args.data_flag}{args.run}.pt")
                    torch.save(model.state_dict(), path)
            if valid_auc > metrics["best_auc"]:
                metrics["best_auc"] = valid_auc
                metrics["best_acc_on_best_auc"] = valid_acc
                if args.save_on_best in ["auc", "acc_auc", "all"]:
                    path = os.path.join(f"run{args.run}", f"best_auc_{args.model}_{args.data_flag}{args.run}.pt")
                    torch.save(model.state_dict(), path)

            metrics['valid_losses'].append(valid_loss)
            metrics['valid_accuracies'].append(valid_acc)
            metrics['valid_precision'].append(valid_prec)
            metrics['valid_recall'].append(valid_rec)
            metrics['valid_AUCROC'].append(valid_auc)

            if epoch == args.epochs - 1:
                metrics["final_acc"] = valid_acc
                metrics["final_auc"] = valid_auc
                if args.save_on_best in ["final", "all"]:
                    path = os.path.join(f"run{args.run}", f"final_{args.model}_{args.data_flag}{args.run}.pt")
                    torch.save(model.state_dict(), path)

            # log the metrics to comet
            experiment.log_metric("train_loss", np.mean(metrics["train_losses"]))
            experiment.log_metric("val_loss", valid_loss)
            experiment.log_metric("val_acc", valid_acc)
            experiment.log_metric("val_auc", valid_auc)
            experiment.log_metric("val_prec", valid_prec)
            experiment.log_metric("val_rec", valid_rec)

    # Stop the profiler
    prof.stop()

    # Save the metrics to a file
    path = os.path.join(f"run{args.run}", f"metrics_{args.model}_{args.data_flag}{args.run}.json")
    with open(path, "w") as f:
        json.dump(metrics, f)

    # Save the profile to a file
    path = os.path.join(f"run{args.run}", f"profile_{args.model}_{args.data_flag}{args.run}.json")
    prof.export_chrome_trace(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_flag", type = str, action = "store", dest = "data_flag", default = "bloodmnist", help = "The dataset to use: pathmnist, dermamnist, bloodmnist, etc...")
    parser.add_argument("--model", type = str, action = "store", dest = "model", default = "VanillaCNN", help = "The model to use")
    parser.add_argument("--batch_size", type = int, action = "store", dest = "batch_size", default = 128, help = "The batch size to use")
    parser.add_argument("--n_channels", type = int, action = "store", dest = "n_channels", default = 32, help = "The number of channels to use")
    parser.add_argument("--kernel_size", type = int, action = "store", dest = "kernel_size", default = 3, help = "The kernel size to use")  
    parser.add_argument("--padding", type = int, action = "store", dest = "padding", default = 1, help = "The padding to use")
    parser.add_argument("--num_layers", type = int, action = "store", dest = "num_layers", default = 5, help = "The number of layers to use")
    parser.add_argument("--epochs", type = int, action = "store", dest = "epochs", default = 100, help = "The number of epochs to use")
    parser.add_argument("--lr", type = float, action = "store", dest = "lr", default = 1e-4, help = "The learning rate to use")
    parser.add_argument("--val_interval", type = int, action = "store", dest = "val_interval", default = 1, help = "The validation interval to use")
    parser.add_argument("--save_on_best", type = str, action = "store", dest = "save_on_best", default = "acc", help = "Whether to save on best: acc, auc, acc_auc, final, all")
    parser.add_argument("--run", type = int, action = "store", dest = "run", default = 0, help = "The run number to use")
    args = parser.parse_args()
    
    cwd = os.getcwd()
    path = os.path.join(cwd, f"run{args.run}")
    if not os.path.exists(path):
        os.makedirs(path)

    main(args)