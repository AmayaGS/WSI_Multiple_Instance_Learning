# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:18:12 2022

@author: AmayaGS
"""

import time
import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import itertools
import sys
import random

import copy
from collections import defaultdict
from collections import Counter
import pickle

from PIL import Image
from PIL import ImageFile

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import ticker as tc
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch.utils.data import Dataset, Subset, IterableDataset
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.ion()  

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
device = torch.device("cuda:0")

import gc 
gc.enable()


# %%

# Define random seed
torch.manual_seed(42)

# %%

train_transform = transforms.Compose([
        transforms.Resize((224, 224)),                            
        #transforms.ColorJitter(brightness=0.005, contrast=0.005, saturation=0.005, hue=0.005),
        transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.001),
        transforms.ColorJitter(contrast=0.001), 
        transforms.ColorJitter(saturation=0.001),
        transforms.ColorJitter(hue=0.001)]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))      
    ])

test_transform = transforms.Compose([
        transforms.Resize((224, 224)),                            
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))      
    ])

# %%

patch_labels = r"C:/Users/Amaya/Documents/PhD/NECCESITY/all_slides_patch_labels.csv"
df = pd.read_csv(patch_labels, header=0)
df["Group label"] = df["Group label"].astype('Int64')
df["Patient ID"] = df["Patient ID"].astype('str')
df["Binary disease"] = df["Binary disease"].astype('Int64')

# df_qmul = pd.read_csv(qmul_csv_file, header=0)
# df_qmul["Group ID"] = df_qmul["Group ID"].astype('Int64')

# df_birm = pd.read_csv(birm_csv_file, header=0)
# df_birm["Group ID"] = df_birm["Group ID"].astype('Int64')

# %%

df = df[df["CENTER"] == "Bicetre"]

# %%

train_fraction = .7

# %%

pathssai_ids  = df['Patient ID'].tolist()
file_ids = sorted(set(pathssai_ids))

# %%

train_ids, test_ids = train_test_split(file_ids, test_size=1-train_fraction, random_state=42)
train_subset_ids = random.sample(train_ids, 10)
test_subset_ids = random.sample(test_ids, 6)

# %%

df_train = df[df['Patient ID'].isin(train_ids)].reset_index(drop=True)
df_test = df[df['Patient ID'].isin(test_ids)].reset_index(drop=True)
subset_df_train = df[df['Patient ID'].isin(train_subset_ids)].reset_index(drop=True)
subset_df_test = df[df['Patient ID'].isin(test_subset_ids)].reset_index(drop=True)

# %%

class histoDataset(Dataset):

    def __init__(self, df, transform, label):
        
        self.transform = transform 
        self.labels = df[label].tolist()
        self.filepaths = df['Location'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = Image.open(self.filepaths[idx])
        image_tensor = self.transform(image)
        image_label = self.labels[idx]
            
        return image_tensor, image_label             
    
# %%

train_df = histoDataset(df_train, train_transform, label="Binary disease")
test_df = histoDataset(df_test, test_transform, label="Binary disease")  
subset_train_df = histoDataset(subset_df_train, train_transform, label="Binary disease")
subset_test_df = histoDataset(subset_df_test, test_transform, label="Binary disease")

# %%

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# %%

##########################################

# %%

# weights for over sampling minority classes

# count = Counter(train_df.labels)
# class_count=np.array([count[0],count[1], count[2]]) # add len to count the number of axis for class weight
# weight=1./class_count
# samples_weight = np.array([weight[t] for t in train_df.labels])
# samples_weight=torch.from_numpy(samples_weight)
# sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

# %%

count = Counter(train_df.labels)
class_count= np.array([count[0],count[1]]) # add len to count the number of axis for class weight
weight= 1. / class_count
samples_weight = np.array([weight[t] for t in train_df.labels])
samples_weight = torch.from_numpy(samples_weight)
sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

# %%

train_loader = torch.utils.data.DataLoader(train_df, batch_size=10, shuffle=False, num_workers=0, drop_last=False, sampler=sampler, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_df, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
subset_train_loader = torch.utils.data.DataLoader(subset_train_df, batch_size=10, shuffle=False, num_workers=0, drop_last=True, sampler=sampler, collate_fn=collate_fn)
subset_test_loader = torch.utils.data.DataLoader(subset_test_df, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

###############################################
# %%

from sklearn.metrics import roc_auc_score

# %%

def train_model(vgg, train_loader, test_loader, criterion, optimizer, num_epochs=1):
    
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    history = defaultdict(list)
    
    train_batches = len(train_loader)
    val_batches = len(test_loader)
        
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        total_train = 0
        total_test= 0
        
        labels_train = []
        labels_test = []
        probs_train = []
        probs_test = []
        
        vgg.train(True)

        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)
                
            inputs, labels = data
            
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            else:
                inputs, labels = inputs, labels
            
            optimizer.zero_grad()
            
            outputs = vgg(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            probs = (F.softmax(outputs, dim=1))[:, 1]
            np_probs = list(probs.detach().to('cpu').numpy())
            probs_train.append(np_probs)
            
            loss.backward()
            optimizer.step()
            
            labels_train.append(list(labels.detach().to('cpu').numpy()))
            
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data)
            total_train += 1
            
            del inputs, labels, outputs, preds, probs, np_probs
            torch.cuda.empty_cache()

        avg_loss = loss_train  / (train_batches * 10)
        avg_acc = acc_train / (train_batches * 10)
        avg_auc = roc_auc_score(sum(labels_train, []), sum(probs_train, []))
        
        vgg.train(False)
        
        vgg.eval()

        for i, data in enumerate(test_loader):
            if i % 10 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                
            inputs, labels = data
            
            with torch.no_grad():
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                else:
                    inputs, labels = inputs, labels
            
            optimizer.zero_grad()
            
            outputs = vgg(inputs)
                
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            probs = (F.softmax(outputs, dim=1))[:, 1]
            np_probs = list(probs.detach().to('cpu').numpy())
            probs_test.append(np_probs)
            
            labels_test.append(list(labels.detach().to('cpu').numpy()))
            
            loss_val += loss.item()
            acc_val += torch.sum(preds == labels.data)    
            total_test += 1
                        
            del inputs, labels, outputs, preds, probs, np_probs
            torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / total_test
        avg_acc_val = acc_val / total_test
        avg_auc_val = roc_auc_score(sum(labels_test, []), sum(probs_test, []), average="weighted")
        
        history['train_acc'].append(avg_acc)
        history['train_auc'].append(avg_auc)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(avg_acc_val)
        history['val_auc'].append(avg_auc_val)
        history['val_loss'].append(avg_loss_val)
                
        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg auc (train): {:.4f}".format(avg_auc))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print("Avg auc (val): {:.4f}".format(avg_auc_val))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print('-' * 10)
        print()
        
        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())
            
    elapsed_time = time.time() - since
    
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Lowest loss: {:.2f}".format(best_acc))
    
    vgg.load_state_dict(best_model_wts)
    
    return vgg, history

# %%

model = models.vgg16_bn(pretrained=True)
                
# Freeze training for all layers
for param in model.parameters():
    param.require_grad = False

# Newly created modules have require_grad=True by default
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, len(count))]) # Add our layer with n outputs
model.classifier = nn.Sequential(*features) # Replace the model classifier
# #print(vgg16)

if use_gpu:
    model.cuda() #.cuda() will move everything to the GPU side
    
    
# %%

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
#optimizer_ft = optim.Adam(vgg16.parameters())~
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# %%

model, history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=3)

# %%

torch.save(model.state_dict(), r"C:/Users/Amaya/Documents/PhD/NECCESITY/Slides" + "/vgg16_BICETRE_Binary_12.pt") # rerun for three epochs as accuracy is better. 
