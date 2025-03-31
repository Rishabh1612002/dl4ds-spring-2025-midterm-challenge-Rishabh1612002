#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from tqdm.auto import tqdm
import wandb
import json
import os

################################################################################
# Model Definition (Using ResNet50 from torchvision)
################################################################################

# ResNet50CNN defines a ResNet-50 model without pretrained weights.
# The final fully connected layer is replaced to output predictions 
# for 100 CIFAR-100 classes. This setup is useful for training from scratch.

class ResNet50CNN(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet50CNN, self).__init__()
        self.model = models.resnet50(weights=None)  # No pretrained weights
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

################################################################################
# Training Function
################################################################################

# The train function trains the model for one epoch on the training set.
# For each batch: performs forward pass, computes loss, backpropagates gradients, 
# and updates model weights. Tracks running loss and accuracy across the epoch.
# Returns average training loss and accuracy.


def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    return running_loss / len(trainloader), 100. * correct / total

################################################################################
# Validation Function
################################################################################

# The below function is used to evaluate the model on the validation set.
# Disables gradient calculation for efficiency. For each batch, it performs a 
# forward pass, computes the loss, and tracks accuracy. Returns average 
# validation loss and accuracy over the entire validation set.


def validate(model, valloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    return running_loss / len(valloader), 100. * correct / total

################################################################################
# Main Function
################################################################################

# The main function sets up the training pipeline for CIFAR-100 using ResNet50.
# It initializes data transformations, model, optimizer, loss function, and scheduler.
# Performs training and validation over multiple epochs, logs metrics to Weights & Biases,
# and saves the best-performing model based on validation accuracy.

def main():
    CONFIG = {
        "model": "ResNet50", 
        "batch_size": 64,
        "learning_rate": 0.01,
        "epochs": 30,
        "num_workers": 4,
        "device": "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }

    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True, transform=transform_train)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    model = ResNet50CNN().to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)

    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

    wandb.finish()

    ################################################################################
    # Evaluation & Submission
    ################################################################################
    import eval_cifar100
    import eval_ood

    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood2.csv", index=False)
    print("submission_ood2.csv created successfully.")

if __name__ == '__main__':
    main()


# In[ ]:




