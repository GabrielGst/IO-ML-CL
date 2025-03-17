####################################
### Useful imports
############################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.utils import make_grid
from torchvision import transforms, datasets
import torchvision.models as models
from torchvision.transforms import v2
import copy

import numpy as np
import random
import time, os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Useful if you want to store intermediate results on your drive
# from google.colab import drive

# # Useful if you want to store intermediate results on your drive from google.colab import drive

# drive.mount('/content/gdrive/')
# DATA_DIR =  '/content/gdrive/MyDrive/teaching/ENSTA/2024'


# Check if GPU is available
#if torch.cuda.is_available():
#  !nvidia-smi

# Define transformations
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)) # GTSRB stats
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    #transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
])

transform_train = v2.Compose([
    #v2.Grayscale(),
    #v2.RandomResizedCrop(32),
    v2.Resize((32,32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    #v2.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)) # GTSRB stats
])

transform_test = v2.Compose([
    #v2.Grayscale(),
    v2.Resize((32,32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    #v2.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)) # GTSRB stats
])

# Define dataset and dataloader
def get_dataset(root_dir, transform, train=True):
    dataset = datasets.GTSRB(root=root_dir, split='train' if train else 'test', download=True, transform=transform)
    target = [data[1] for data in dataset]
    return dataset, target

def create_dataloader(dataset, targets, current_classes, batch_size, shuffle):
    indices = [i for i, label in enumerate(targets) if label in current_classes]
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# Loads datasets (on your local computer)
root_dir = '/home/stephane/Documents/Onera/Cours/ENSTA/2025/data'

# Loads datasets (on Colab local computer)
root_dir = './data'

train_dataset = datasets.GTSRB(root=root_dir, split='train', download=True, transform=transform_train)
test_dataset = datasets.GTSRB(root=root_dir, split='test', download=True, transform=transform_test)

print(f"Train dataset contains {len(train_dataset)} images")
print(f"Test dataset contains {len(test_dataset)} images")

# Loads target id lists and class names (not in torchvision dataset)
import csv
data = pd.read_csv('https://raw.githubusercontent.com/stepherbin/teaching/refs/heads/master/IOGS/projet/test_target.csv', delimiter=',', header=None)
test_target = data.to_numpy().squeeze().tolist()

data = pd.read_csv('https://raw.githubusercontent.com/stepherbin/teaching/refs/heads/master/IOGS/projet/train_target.csv', delimiter=',', header=None)
train_target = data.to_numpy().squeeze().tolist()

data = pd.read_csv('https://raw.githubusercontent.com/stepherbin/teaching/refs/heads/master/IOGS/projet/signnames.csv')
class_names = data['SignName'].tolist()

nclasses = len(np.unique(train_target))
all_classes = list(range(nclasses))
#random.shuffle(all_classes)
classes_per_task = 8
current_classes = []

task = 0
task_classes = all_classes[task * classes_per_task : (task + 1) * classes_per_task]
current_classes.extend(task_classes)
batch_size = 64

# Create data for first task
train_loader = create_dataloader(train_dataset, train_target, current_classes, batch_size, shuffle = True)
test_loader = create_dataloader(train_dataset, train_target, current_classes, batch_size, shuffle = True)

# Displays a few examples
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

sample,targets = next(iter(train_loader))
show(make_grid(sample))
plt.show()

print(sample.shape)     ## 64 is the batch
                        ## 1 for grey values --  3 for RGB
                        ## 32x32 for mage size (small here)


test_loader = create_dataloader(train_dataset, train_target, all_classes, batch_size, shuffle = True)

# Get the data from the test set and computes statistics
# gtsrbtest_gt = []
# for _, targets in test_loader:
#   gtsrbtest_gt += targets.numpy().tolist()
# print(len(gtsrbtest_gt))

from collections import Counter

label_counts = Counter(test_target).most_common()
for l, c in label_counts:
    print(c, '\t', l, '\t', class_names[l])

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self,n_out=10, n_in=1):
        super().__init__()

        # Put the layers here
        self.conv1 = nn.Conv2d(n_in, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc = nn.Linear(4096, n_out)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x)) ## l'image 1x32x32 devient 32x32x32
        x = F.max_pool2d(x, kernel_size=2, stride=2) ## puis 32x16x16
        x = F.leaky_relu(self.conv2(x)) ## puis devient 64x16x16
        x = F.max_pool2d(x, kernel_size=2, stride=2) ## puis devient 64x8x8
        x = F.leaky_relu(self.conv3(x)) ## pas de changement

        x = x.view(-1,4096) ## 64x8x8 devient 4096

        x = self.fc(x) ## on finit exactement de la même façon

        return x

# Another simple model (compare them using torchinfo below)
class SimpleCNN2(nn.Module):
    def __init__(self, n_out=10, n_in=1):
        super(SimpleCNN2, self).__init__()
        self.conv1 = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc = nn.Linear(128, n_out)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc(x)
        return x

#!pip install torchinfo
#from torchinfo import summary

import subprocess

# Install torchinfo if not installed
try:
    import torchinfo
except ImportError:
    subprocess.run(["pip", "install", "torchinfo"])
    import torchinfo  # Import again after installation

from torchinfo import summary


model = SimpleCNN2(n_out=10, n_in=3)
model.to(device)
print(summary(model, input_size=(batch_size, 3, 32, 32)))

model = SimpleCNN(n_out=10, n_in=3)
model.to(device)
print(summary(model, input_size=(batch_size, 3, 32, 32)))

#print(model)

from torch.optim import lr_scheduler
import torch.nn.init as init

# Evaluation
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, ncols=80):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Simple Training loop
def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()

    for images, labels in tqdm(train_loader, ncols=80,  desc="Epoch {}".format(epoch)):   #ajouter d'autres données, techniques memory buffer,
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def initialize_weights(module):
    """Initializes the weights of a PyTorch module using Xavier/Glorot initialization."""
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):  # Check for relevant layers
        init.xavier_uniform_(module.weight) #Xavier uniform initialization
        if module.bias is not None:
            init.zeros_(module.bias)  # Initialize bias to zero
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)): #Initialize normalization layers
        if module.weight is not None:
            init.ones_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)


# Main training loop for incremental learning
def incremental_learning(model, train_dataset, train_target, test_dataset, test_target,
                         num_tasks, classes_per_task, batch_size, num_epochs, lr, device):
    nclasses = len(np.unique(train_target))
    all_classes = list(range(nclasses))
    criterion = nn.CrossEntropyLoss() #peut être changé comme critère
    current_classes = []
    accuracies = []

    for task in range(num_tasks):
        task_classes = all_classes[task * classes_per_task : (task + 1) * classes_per_task]
        current_classes.extend(task_classes)

        train_loader = create_dataloader(train_dataset, train_target, task_classes, batch_size, shuffle = True)
        test_loader = create_dataloader(test_dataset, test_target, current_classes, batch_size, shuffle = False)

        if task == 0:
            model.fc = nn.Linear(model.fc.in_features, len(current_classes)).to(device)
        else:
            # Expand the output layer for new classes
            old_weight = model.fc.weight.data
            old_bias = model.fc.bias.data
            model.fc = nn.Linear(model.fc.in_features, len(current_classes)).to(device)
            model.fc.weight.data[:len(old_weight)] = old_weight
            model.fc.bias.data[:len(old_bias)] = old_bias

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        print(f"Starting Task {task+1} - Training on classes: {task_classes}")
        for epoch in range(num_epochs): # Adjust number of epochs as needed
            train(model, train_loader, optimizer, criterion, device, epoch)
            scheduler.step()
            accuracy = evaluate(model, train_loader, device)
            print(f"Task {task+1}, Epoch {epoch+1}: Accuracy Train = {accuracy:.2f}%")
        accuracy = evaluate(model, test_loader, device)
        accuracies.append(accuracy)
        print(f"Task {task+1}: Accuracy Test = {accuracy:.2f}%")

    return accuracies



#Memory Buffer Class
class MemoryBuffer:
    """
    A memory buffer to store previous examples for rehearsal.
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.images = []
        self.labels = []
        self.class_indices = {}  # To keep track of examples per class
        
    def update(self, images, labels, new_classes):
        """Update buffer with new examples, maintaining balanced classes."""
        # Convert tensors to numpy for storage if needed
        if isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy()
        else:
            images_np = images
            
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = labels
        
        # Initialize class indices for new classes
        for cls in new_classes:
            if cls not in self.class_indices:
                self.class_indices[cls] = []
        
        # Add new examples to buffer
        for i, (image, label) in enumerate(zip(images_np, labels_np)):
            if label in new_classes:
                self.images.append(image)
                self.labels.append(label)
                self.class_indices[label].append(len(self.images) - 1)
        
        # Ensure buffer doesn't exceed size limit by randomly removing samples
        self._balance_buffer()
    
    def _balance_buffer(self):
        """Balance the buffer to ensure equal representation of classes."""
        if len(self.images) <= self.buffer_size:
            return
        
        # Calculate how many samples per class we should keep
        num_classes = len(self.class_indices)
        samples_per_class = self.buffer_size // num_classes
        
        # Select indices to keep
        indices_to_keep = []
        for cls, idx_list in self.class_indices.items():
            if len(idx_list) > samples_per_class:
                # Randomly select samples for this class
                selected = random.sample(idx_list, samples_per_class)
                indices_to_keep.extend(selected)
            else:
                # Keep all samples for this class
                indices_to_keep.extend(idx_list)
        
        # If we still haven't filled the buffer, add more samples randomly
        remaining_slots = self.buffer_size - len(indices_to_keep)
        if remaining_slots > 0:
            remaining_indices = [i for i in range(len(self.images)) if i not in indices_to_keep]
            if remaining_indices:
                additional_indices = random.sample(remaining_indices, min(remaining_slots, len(remaining_indices)))
                indices_to_keep.extend(additional_indices)
        
        # Keep only the selected indices
        new_images = [self.images[i] for i in indices_to_keep]
        new_labels = [self.labels[i] for i in indices_to_keep]
        
        # Update the buffer
        self.images = new_images
        self.labels = new_labels
        
        # Update class indices
        self.class_indices = {}
        for i, label in enumerate(self.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(i)
    
    def get_batch(self, batch_size):
        """Get a random batch from the buffer."""
        if len(self.images) == 0:
            return None, None
        
        indices = random.sample(range(len(self.images)), min(batch_size, len(self.images)))
        batch_images = [self.images[i] for i in indices]
        batch_labels = [self.labels[i] for i in indices]
        
        return torch.tensor(batch_images), torch.tensor(batch_labels)
    
    def __len__(self):
        return len(self.images)


#Knowledge Distillation Loss
def distillation_loss(outputs, targets, old_outputs, T=2.0, alpha=0.5):
    """
    Compute the knowledge distillation loss.
    
    Args:
        outputs: Current model outputs
        targets: Ground truth labels
        old_outputs: Outputs from the old model (before learning new classes)
        T: Temperature for softening the distributions
        alpha: Weight for balancing classification loss and distillation loss
    
    Returns:
        Combined loss: classification loss + distillation loss
    """
    # Standard cross entropy loss
    ce_loss = F.cross_entropy(outputs, targets)
    
    # Knowledge distillation loss
    if old_outputs is not None:
        # Apply temperature scaling
        scaled_outputs = outputs[:, :old_outputs.size(1)] / T
        scaled_old_outputs = old_outputs / T
        
        # Compute KL divergence loss
        kd_loss = F.kl_div(
            F.log_softmax(scaled_outputs, dim=1),
            F.softmax(scaled_old_outputs, dim=1),
            reduction='batchmean'
        ) * (T * T)
        
        # Combine losses
        return alpha * ce_loss + (1 - alpha) * kd_loss
    else:
        return ce_loss


#Training with Rehearsal and Knowledge Distillation
def train_rehearsal_kd(model, old_model, train_loader, memory_buffer, optimizer, device, epoch, 
                       alpha=0.5, T=2.0, rehearsal_ratio=0.3):
    """
    Training loop with rehearsal and knowledge distillation.
    
    Args:
        model: Current model being trained
        old_model: Model from previous task (for knowledge distillation)
        train_loader: DataLoader for current task data
        memory_buffer: Memory buffer containing examples from previous tasks
        optimizer: Optimizer for model parameters
        device: Device to run computation on
        epoch: Current epoch number
        alpha: Weight for balancing classification and distillation loss
        T: Temperature for knowledge distillation
        rehearsal_ratio: Ratio of rehearsal batch size to current batch size
    """
    model.train()
    if old_model is not None:
        old_model.eval()
    
    for images, labels in tqdm(train_loader, ncols=80, desc=f"Epoch {epoch}"):
        images, labels = images.to(device), labels.to(device)
        
        # Add rehearsal data if available
        if memory_buffer is not None and len(memory_buffer) > 0:
            rehearsal_batch_size = max(1, int(images.size(0) * rehearsal_ratio))
            mem_images, mem_labels = memory_buffer.get_batch(rehearsal_batch_size)
            
            if mem_images is not None and mem_images.size(0) > 0:
                mem_images, mem_labels = mem_images.to(device), mem_labels.to(device)
                combined_images = torch.cat([images, mem_images], dim=0)
                combined_labels = torch.cat([labels, mem_labels], dim=0)
            else:
                combined_images, combined_labels = images, labels
        else:
            combined_images, combined_labels = images, labels
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(combined_images)
        
        # Compute knowledge distillation loss if old model exists
        if old_model is not None:
            with torch.no_grad():
                old_outputs = old_model(combined_images)
            loss = distillation_loss(outputs, combined_labels, old_outputs, T=T, alpha=alpha)
        else:
            loss = F.cross_entropy(outputs, combined_labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()


#Incremental Learning with Rehearsal and KD
def incremental_learning_rehearsal_kd(model, train_dataset, train_target, test_dataset, test_target,
                                     num_tasks, classes_per_task, batch_size, num_epochs, lr, device,
                                     buffer_size=200, alpha=0.5, T=2.0):
    """
    Incremental learning with rehearsal and knowledge distillation.
    
    Args:
        model: Model to be trained
        train_dataset, train_target: Training data and targets
        test_dataset, test_target: Test data and targets
        num_tasks: Number of tasks to learn sequentially
        classes_per_task: Number of classes in each task
        batch_size: Batch size for training
        num_epochs: Number of epochs per task
        lr: Learning rate
        device: Device to run computation on
        buffer_size: Size of memory buffer for rehearsal
        alpha: Weight for balancing classification and distillation loss
        T: Temperature for knowledge distillation
    """
    nclasses = len(np.unique(train_target))
    all_classes = list(range(nclasses))
    current_classes = []
    accuracies = []
    
    # Initialize memory buffer
    memory_buffer = MemoryBuffer(buffer_size)
    
    for task in range(num_tasks):
        # Define classes for the current task
        task_classes = all_classes[task * classes_per_task : (task + 1) * classes_per_task]
        current_classes.extend(task_classes)
        
        # Create data loaders
        train_loader = create_dataloader(train_dataset, train_target, task_classes, batch_size, shuffle=True)
        test_loader = create_dataloader(test_dataset, test_target, current_classes, batch_size, shuffle=False)
        
        # Create a copy of the current model before task learning (for knowledge distillation)
        old_model = copy.deepcopy(model) if task > 0 else None
        if old_model is not None:
            old_model.eval()  # Set to evaluation mode
        
        # Modify the model's output layer for new classes
        if task == 0:
            model.fc = nn.Linear(model.fc.in_features, len(current_classes)).to(device)
        else:
            # Expand the output layer for new classes
            old_weight = model.fc.weight.data
            old_bias = model.fc.bias.data
            model.fc = nn.Linear(model.fc.in_features, len(current_classes)).to(device)
            model.fc.weight.data[:len(old_weight)] = old_weight
            model.fc.bias.data[:len(old_bias)] = old_bias
        
        # Set up optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        print(f"Starting Task {task+1} - Training on classes: {task_classes}")
        
        # Training loop for current task
        for epoch in range(num_epochs):
            # Train with rehearsal and knowledge distillation
            train_rehearsal_kd(model, old_model, train_loader, memory_buffer, optimizer, device, epoch, alpha=alpha, T=T)
            scheduler.step()
            
            # Evaluate on training data
            train_accuracy = evaluate(model, train_loader, device)
            print(f"Task {task+1}, Epoch {epoch+1}: Accuracy Train = {train_accuracy:.2f}%")
        
        # After training, update memory buffer with examples from current task
        for images, labels in train_loader:
            memory_buffer.update(images, labels, task_classes)
            break  # Just use one batch to update memory (for efficiency)
        
        # Evaluate on test data (all classes seen so far)
        accuracy = evaluate(model, test_loader, device)
        accuracies.append(accuracy)
        print(f"Task {task+1}: Accuracy Test = {accuracy:.2f}%")
    
    return accuracies


#Incremental Learning with Rehearsal and KD (plus de wandb ici)
def incremental_learning_rehearsal_kd_no_wandb(model, train_dataset, train_target, test_dataset, test_target,
                                     num_tasks, classes_per_task, batch_size, num_epochs, lr, device,
                                     buffer_size=200, alpha=0.5, T=2.0):
    nclasses = len(np.unique(train_target))
    all_classes = list(range(nclasses))
    current_classes = []
    accuracies = []
    
    # Initialize memory buffer
    memory_buffer = MemoryBuffer(buffer_size)
    
    for task in range(num_tasks):
        # Define classes for the current task
        task_classes = all_classes[task * classes_per_task : (task + 1) * classes_per_task]
        current_classes.extend(task_classes)
        
        # Create data loaders
        train_loader = create_dataloader(train_dataset, train_target, task_classes, batch_size, shuffle=True)
        test_loader = create_dataloader(test_dataset, test_target, current_classes, batch_size, shuffle=False)
        
        # Create a copy of the current model before task learning (for knowledge distillation)
        old_model = copy.deepcopy(model) if task > 0 else None
        if old_model is not None:
            old_model.eval()
            
        # Modify the model's output layer for new classes
        if task == 0:
            model.fc = nn.Linear(model.fc.in_features, len(current_classes)).to(device)
        else:
            # Expand the output layer for new classes
            old_weight = model.fc.weight.data
            old_bias = model.fc.bias.data
            model.fc = nn.Linear(model.fc.in_features, len(current_classes)).to(device)
            model.fc.weight.data[:len(old_weight)] = old_weight
            model.fc.bias.data[:len(old_bias)] = old_bias
            
        # Set up optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        print(f"Starting Task {task+1} - Training on classes: {task_classes}")
        
        # Training loop for current task
        for epoch in range(num_epochs):
            # Train with rehearsal and knowledge distillation
            train_rehearsal_kd(model, old_model, train_loader, memory_buffer, optimizer, device, epoch, alpha=alpha, T=T)
            scheduler.step()
            
            # Evaluate on training data
            train_accuracy = evaluate(model, train_loader, device)
            print(f"Task {task+1}, Epoch {epoch+1}: Accuracy Train = {train_accuracy:.2f}%")
            
        # After training, update memory buffer with examples from current task
        for images, labels in train_loader:
            memory_buffer.update(images, labels, task_classes)
            break  # Just use one batch to update memory
            
        # Evaluate on test data (all classes seen so far)
        accuracy = evaluate(model, test_loader, device)
        accuracies.append(accuracy)
        print(f"Task {task+1}: Accuracy Test = {accuracy:.2f}%")
        
    return accuracies

###################################
##### For using Weight & Biases
###############

#!pip install wandb -qU

#import wandb

#wandb.login()
#import wandb

#wandb.login(key="425b3742039350921379a12e9ef4f7c33878e2d2") #Nada API key


import math
# Simple Training loop
def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, ncols=80, desc=f"Epoch {epoch}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


# Main training loop for incremental learning without a wandb login ici
def incremental_learning_no_wandb(model, train_dataset, train_target, test_dataset, test_target,
                         num_tasks, classes_per_task, batch_size, num_epochs, lr, device, non_incremental=False):
    nclasses = len(np.unique(train_target))
    all_classes = list(range(nclasses))
    criterion = nn.CrossEntropyLoss()
    current_classes = []
    accuracies = []
    
    for task in range(num_tasks):
        if non_incremental == True:  # Learn from all available data
            task_classes = all_classes[0 : (task + 1) * classes_per_task]
            current_classes = task_classes
            model.apply(initialize_weights)
        else:
            task_classes = all_classes[task * classes_per_task : (task + 1) * classes_per_task]
            current_classes.extend(task_classes)
            
        train_loader = create_dataloader(train_dataset, train_target, task_classes, batch_size, shuffle=True)
        test_loader = create_dataloader(test_dataset, test_target, current_classes, batch_size, shuffle=False)
        
        if task == 0 or non_incremental == True:
            model.fc = nn.Linear(model.fc.in_features, len(current_classes)).to(device)
        else:
            # Expand the output layer for new classes
            old_weight = model.fc.weight.data
            old_bias = model.fc.bias.data
            model.fc = nn.Linear(model.fc.in_features, len(current_classes)).to(device)
            model.fc.weight.data[:len(old_weight)] = old_weight
            model.fc.bias.data[:len(old_bias)] = old_bias
            
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        print(f"Starting Task {task+1} - Training on classes: {task_classes}")
        for epoch in range(num_epochs):
            train(model, train_loader, optimizer, criterion, device, epoch)
            scheduler.step()
            
            train_accuracy = evaluate(model, train_loader, device)
            print(f"Task {task+1}, Epoch {epoch+1}: Accuracy Train = {train_accuracy:.2f}%")
            
        accuracy = evaluate(model, test_loader, device)
        accuracies.append(accuracy)
        print(f"Task {task+1}: Accuracy Test = {accuracy:.2f}%")
        
    return accuracies



# Hyperparameters
root_dir = './data'  # Path to GTSRB dataset
num_tasks = 5
numclasses = len(np.unique(train_target))
classes_per_task = numclasses // num_tasks  # 43/5 ~ 8
batch_size = 64
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
buffer_size = 200  # Adjust rehearsal set size
alpha = 0.5  # Weight for balancing CE and KD loss
T = 2.0  # Temperature for knowledge distillation
num_epochs = 4

# The name of the network
tag = "simpleCNN_GTSRB_pretrained"
netname = os.path.join(root_dir, 'network_{:s}.pth'.format(tag))

#################################################
## Pre-training
####
# Read the last learned network (if stored)
if (os.path.exists(netname)):
    print('Load pre-trained network')
    model = SimpleCNN(n_in=3, n_out=classes_per_task)
    model.load_state_dict(torch.load(netname, weights_only=True))
    model = model.to(device)
else:
    print('Pretrain')
    model = SimpleCNN(n_in=3, n_out=1)
    model.apply(initialize_weights)
    model.to(device)
    accu = incremental_learning(model, train_dataset, train_target, test_dataset, test_target,
                            1, classes_per_task, batch_size, num_epochs, lr, device)
    print(f"!!!!! Pre-training on first task = {accu[0]:.2f}%")
    # Save last learned model
    torch.save(model.state_dict(), netname)

## Copy model to have the same initialization
copy_model = copy.deepcopy(model)  # Copy model to start from the same initialization

# For faster experiments
num_epochs = 1

#############################################
## Fine tuning
####
print("\n------ RUNNING FINE TUNING EXPERIMENT ------\n")
model = copy.deepcopy(copy_model)
ft_accuracies = incremental_learning_no_wandb(model, train_dataset, train_target, test_dataset, test_target,
                                   num_tasks, classes_per_task, batch_size, num_epochs, lr, device)
print("Fine tuning accuracies:", ft_accuracies)

#############################################
## Rehearsal + Knowledge Distillation
####
print("\n------ RUNNING REHEARSAL + KD EXPERIMENT ------\n")
model = copy.deepcopy(copy_model)
rkd_accuracies = incremental_learning_rehearsal_kd_no_wandb(model, train_dataset, train_target, test_dataset, test_target,
                                                 num_tasks, classes_per_task, batch_size, num_epochs, lr, device,
                                                 buffer_size=buffer_size, alpha=alpha, T=T)
print("Rehearsal + KD accuracies:", rkd_accuracies)

#################################################
## Global upper bound (all data, all classes)
####
print("\n------ CALCULATING GLOBAL UPPER BOUND ------\n")
model = copy.deepcopy(copy_model)
accu = incremental_learning(model, train_dataset, train_target, test_dataset, test_target,
                        1, (numclasses // num_tasks) * num_tasks, batch_size, 5, lr, device)
print(f"!!!!! Upper bound of accuracy = {accu[0]:.2f}%")

########################################
## Upper bound for each task
####
print("\n------ CALCULATING UPPER BOUND PER TASK ------\n")
model = copy.deepcopy(copy_model)
upper_bound_accuracies = incremental_learning_no_wandb(model, train_dataset, train_target, test_dataset, test_target,
                                            num_tasks, classes_per_task, batch_size, num_epochs, lr, device, 
                                            non_incremental=True)
print("Upper bound accuracies:", upper_bound_accuracies)

# Plot the results
plt.figure(figsize=(10, 6))
tasks = list(range(1, num_tasks + 1))
plt.plot(tasks, ft_accuracies, 'bo-', label='Fine Tuning')
plt.plot(tasks, rkd_accuracies, 'ro-', label='Rehearsal + KD')
plt.plot(tasks, upper_bound_accuracies, 'go-', label='Upper Bound')
plt.xlabel('Task')
plt.ylabel('Accuracy (%)')
plt.title('Incremental Learning Performance')
plt.legend()
plt.grid(True)
plt.savefig('incremental_learning_results.png')
plt.show()