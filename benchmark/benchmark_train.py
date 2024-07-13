from benchmark_models import *
import torchvision.transforms as transforms
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassAUROC
import torch.utils.data as data
import torch
from medmnist import INFO
import medmnist
import argparse
import time
from tqdm import tqdm
import json
from comet_ml import Experiment

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

def calculate_metrics(model, loader, criterion, device, num_classes):
    model.eval()
    acc_loss = 0
    acc = MulticlassAccuracy()
    precision = MulticlassPrecision()
    recall = MulticlassRecall()
    auroc = MulticlassAUROC(num_classes = num_classes)
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).squeeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc_loss += loss.item()
            acc.update(outputs, labels)
            precision.update(outputs, labels)
            recall.update(outputs, labels)
            auroc.update(outputs, labels)

    return acc_loss / len(loader), acc.compute().item(), precision.compute().item(), recall.compute().item(), auroc.compute().item()

def main(args):

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
    n_batches = len(train_loader)

    # model can be any of the models in the benchmark_models.py file
    model = PRESETS[args.model](img_channels, args.n_channels, n_classes, args.kernel_size, args.padding, args.num_layers).to(device)

    # our optimizer is Adam
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    # our loss function is CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss()

    epoch_times = []
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    valid_AUCROC = []
    valid_precision = []
    valid_recall = []
    best_acc = 0
    best_auc = 0
    final_acc = 0
    final_auc = 0
    best_acc_on_best_auc = 0
    best_auc_on_best_acc = 0

    # training loop
    for epoch in range(args.epochs):
        model.train()

        # start timing the epoch
        start_time = time.time()
        pbar = tqdm(train_loader, unit = "batches")

        #tqdm is a progress bar with units in batches
        for i, (images, labels) in enumerate(pbar):
            acc_loss = 0

            # images to configured device. NOTE: labels squeezed for every dataset?
            images = images.to(device)
            labels = labels.to(device).squeeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass and backward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # update progress bar
            pbar.set_postfix({"loss": loss.item()})
            pbar.update()
            acc_loss += loss.item()
        
        # end timing the epoch save the duration
        end_time = time.time()
        epoch_times.append(end_time - start_time)
        train_losses.append(acc_loss / n_batches)

        # validation loop: evaluate at given interval or at the end
        if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            valid_loss, valid_acc, valid_prec, valid_rec, valid_auc = calculate_metrics(model, valid_loader, criterion, device, n_classes)

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_auc_on_best_acc = valid_auc
                torch.save(model.state_dict(), f"best_acc_{args.model}_{data_flag}.pt")
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_acc_on_best_auc = valid_acc
                torch.save(model.state_dict(), f"best_auc_{args.model}_{data_flag}.pt")

            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_acc)
            valid_precision.append(valid_prec)
            valid_recall.append(valid_rec)
            valid_AUCROC.append(valid_auc)

            pbar.set_postfix({"loss": loss.item(), "val_loss": valid_loss, "val_acc": valid_acc, "val_auc": valid_auc})
            pbar.update()

            if epoch == args.epochs - 1:
                final_acc = valid_acc
                final_auc = valid_auc
                torch.save(model.state_dict(), f"final_{args.model}_{data_flag}.pt")

        # log the metrics to comet
        experiment.log_metric("train_loss", acc_loss / n_batches)
        experiment.log_metric("val_loss", valid_loss)
        experiment.log_metric("val_acc", valid_acc)
        experiment.log_metric("val_auc", valid_auc)
        experiment.log_metric("val_prec", valid_prec)
        experiment.log_metric("val_rec", valid_rec)
        
    # Save all the metrics as a dictionary
    metrics = {
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "valid_accuracies": valid_accuracies,
        "valid_precision": valid_precision,
        "valid_recall": valid_recall,
        "valid_AUCROC": valid_AUCROC,
        "best_acc": best_acc,
        "best_auc": best_auc,
        "best_acc_on_best_auc": best_acc_on_best_auc,
        "best_auc_on_best_acc": best_auc_on_best_acc,
        "epoch_times": epoch_times,
        "final_acc": final_acc,
        "final_auc": final_auc
    }

    # Save the metrics to a file
    with open(f"metrics_{args.model}_{data_flag}.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_flag", type = str, action = "store", dest = "data_flag", default = "dermamnist", help = "The dataset to use")
    parser.add_argument("--model", type = str, action = "store", dest = "model", default = "VanillaCNN", help = "The model to use")
    parser.add_argument("--batch_size", type = int, action = "store", dest = "batch_size", default = 128, help = "The batch size to use")
    parser.add_argument("--n_channels", type = int, action = "store", dest = "n_channels", default = 32, help = "The number of channels to use")
    parser.add_argument("--kernel_size", type = int, action = "store", dest = "kernel_size", default = 3, help = "The kernel size to use")  
    parser.add_argument("--padding", type = int, action = "store", dest = "padding", default = 1, help = "The padding to use")
    parser.add_argument("--num_layers", type = int, action = "store", dest = "num_layers", default = 5, help = "The number of layers to use")
    parser.add_argument("--epochs", type = int, action = "store", dest = "epochs", default = 100, help = "The number of epochs to use")
    parser.add_argument("--lr", type = float, action = "store", dest = "lr", default = 1e-4, help = "The learning rate to use")
    parser.add_argument("--val_interval", type = int, action = "store", dest = "val_interval", default = 1, help = "The validation interval to use")
    args = parser.parse_args()
    main(args)