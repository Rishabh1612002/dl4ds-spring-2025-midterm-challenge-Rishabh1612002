#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json

################################################################################
# Model Definition (Simple Example - You need to complete)
################################################################################

# SimpleCNN is a lightweight architecture for CIFAR-100.
# It uses two convolutional layers with ReLU and 2x2 max pooling
# to extract spatial features, increasing depth from 3 → 32 → 64.
# The output is flattened and passed through a fully connected layer (256 units),
# followed by a final layer that outputs predictions for 100 classes.

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 100)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

################################################################################
# Define a one epoch training function
################################################################################

# The below function trains the model for one epoch on the training dataset.
# For each batch: moves data to device, performs a forward pass,
# computes loss, backpropagates gradients, and updates weights.
# Tracks and returns average training loss and accuracy for the epoch.


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
# Define a validation function
################################################################################

# The validate function evaluates the model on the validation set.
# Disables gradient computation for efficiency.
# For each batch: performs forward pass, computes loss, and tracks predictions.
# Returns the average validation loss and accuracy.

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


# The main function sets up configuration, data loaders, model, loss function, 
# optimizer, and training loop. It trains the model, validates after each epoch, 
# logs results to Weights & Biases (wandb), and saves the best-performing model.

def main():
    CONFIG = {
        "model": "SimpleCNN",   
        "batch_size": 8, 
        "learning_rate": 0.1,
        "epochs": 5,  
        "num_workers": 4, 
        "device": "cpu",  # Force using CPU
        "data_dir": "./data",  
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }
    
    print("\nCONFIG Dictionary:")
    print(json.dumps(CONFIG, indent=4))
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_size = int(0.8 * len(trainset))  
    val_size = len(trainset) - train_size     
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    model = SimpleCNN().to(CONFIG["device"])
    print("\nModel summary:")
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
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
    
    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################
    
    # After training, evaluate the model on the CIFAR-100 test set and OOD data.
    # Generates predictions, computes clean test accuracy, and prepares the
    # submission file in the required format for OOD evaluation.

    import eval_cifar100
    import eval_ood

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()


# In[ ]:




