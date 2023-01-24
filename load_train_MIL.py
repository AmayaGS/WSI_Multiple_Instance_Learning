# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:52:02 2022

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
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

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

df = df[df["CENTER"] == "Birmingham"]

# %%

train_fraction = .7

# %%

pathssai_ids  = df['Patient ID'].tolist()
file_ids = sorted(set(pathssai_ids))

# %%

train_ids, test_ids = train_test_split(file_ids, test_size=1-train_fraction, random_state=42)
subset_train_ids = random.sample(train_ids, 5)
subset_test_ids = random.sample(test_ids, 17)

# %%

df_train = df[df['Patient ID'].isin(train_ids)].reset_index(drop=True)
df_test = df[df['Patient ID'].isin(test_ids)].reset_index(drop=True)
subset_df_train = df[df['Patient ID'].isin(subset_train_ids)].reset_index(drop=True)
subset_df_test = df[df['Patient ID'].isin(subset_test_ids)].reset_index(drop=True)

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

# %%

from attention_models import VGG_embedding, GatedAttention
from auxiliary_functions import Accuracy_Logger

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_net = VGG_embedding(center="QMUL")
gated_net = GatedAttention()

# %%

if use_gpu:
    embedding_net.cuda()
    gated_net.cuda()

# %%

##########################################

# TRAIN dict
# TEST dict

train_subsets = {}
file_indices = []

for i, file in enumerate(train_ids):
    file_indices.append(np.where(df_train["Patient ID"] == file))
    train_subsets['subset_%02d' % i] = histoDataset(df_train[file_indices[i][0][0]: file_indices[i][0][-1] + 1], train_transform, label="Binary disease")

train_loaded_subsets = {}

for i, value in enumerate(train_subsets.values()):
    train_loaded_subsets['subset_%02d' % i] = torch.utils.data.DataLoader(value, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    
    
# %%

# TEST dict

test_subsets = {}
file_indices = []

for i, file in enumerate(test_ids):
    file_indices.append(np.where(df_test["Patient ID"] == file))
    test_subsets['subset_%02d' % i] = histoDataset(df_test[file_indices[i][0][0]: file_indices[i][0][-1] + 1], test_transform, label="Binary disease")

test_loaded_subsets = {}

for i, value in enumerate(test_subsets.values()):
    test_loaded_subsets['subset_%02d' % i] = torch.utils.data.DataLoader(value, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    

# %%

def train_model(embedding_net, classification_net, train_loaded_subsets, test_loaded_subsets, loss_fn, optimizer, n_classes, bag_weight,  num_epochs=1):
    
    since = time.time()
    #best_model_embedding_wts = copy.deepcopy(embedding_net.state_dict())
    best_model_classification_wts = copy.deepcopy(classification_net.state_dict())
    best_auc = 0.

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        ##################################
        # TRAIN
        
        acc_logger = Accuracy_Logger(n_classes=n_classes)
        inst_logger = Accuracy_Logger(n_classes=n_classes)
        
        train_loss = 0 # train_loss
        train_error = 0 
        train_inst_loss = 0.
        inst_count = 0
        
        train_acc = 0
        
        ###################################
        # TEST
        
        val_acc_logger = Accuracy_Logger(n_classes)
        val_inst_logger = Accuracy_Logger(n_classes)
        val_loss = 0.
        val_error = 0.
    
        val_inst_loss = 0.
        val_inst_count= 0
        
        val_acc = 0

        ###################################
        # TRAIN
        
        embedding_net.train(False)
        classification_net.train(True)
        
        for batch_idx, loader in enumerate(train_loaded_subsets.values()):

            print("\rTraining batch {}/{}".format(batch_idx, len(train_loaded_subsets)), end='', flush=True)
            
            optimizer.zero_grad()
            
            patient_embedding = []
            for data in loader:
                
                inputs, label = data
                
                if use_gpu:
                    inputs, label = inputs.cuda(), label.cuda()
                else:
                    inputs, label = inputs, label
                
                embedding = embedding_net(inputs)
                
                embedding = embedding.detach().to('cpu')
                embedding = embedding.squeeze(0)
                patient_embedding.append(embedding)
                
            patient_embedding = torch.stack(patient_embedding)
            patient_embedding = patient_embedding.cuda()
            
            logits, Y_prob, Y_hat, _, instance_dict = classification_net(patient_embedding, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)
            loss_value = loss.item()
            
            train_acc += torch.sum(Y_hat == label.data)
            
            instance_loss = instance_dict['instance_loss']
            inst_count+=1
            instance_loss_value = instance_loss.item()
            train_inst_loss += instance_loss_value
            
            total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)
            
            train_loss += loss_value
            if (batch_idx + 1) % 20 == 0:
                print('- batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                    'label: {}, bag_size: {}'.format(label.item(), patient_embedding.size(0)))
     
            error = classification_net.calculate_error(Y_hat, label)
            train_error += error
            
            # backward pass
            total_loss.backward()
            # step
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss /= len(train_loaded_subsets)
        train_error /= len(train_loaded_subsets)
        train_accuracy =  train_acc / len(train_loaded_subsets)
        
        if inst_count > 0:
            train_inst_loss /= inst_count
            print('\n')
            for i in range(2):
                acc, correct, count = inst_logger.get_summary(i)
                print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
         
            
        print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}, train_accuracy: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error, train_accuracy))
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
                  
        embedding_net.train(False)
        classification_net.train(False)
        
        
        ################################
        # TEST
        
        embedding_net.eval()
        classification_net.eval()
        
        prob = np.zeros((len(test_loaded_subsets), n_classes))
        labels = np.zeros(len(test_loaded_subsets))
        #sample_size = classification_net.k_sample

        for batch_idx, loader in enumerate(test_loaded_subsets.values()):
            
            print("\rValidation batch {}/{}".format(batch_idx, len(test_loaded_subsets)), end='', flush=True)
            
            patient_embedding = []
            for data in loader:
                
                inputs, label = data
                
                with torch.no_grad():
                    if use_gpu:
                        inputs, label = inputs.cuda(), label.cuda()
                    else:
                        inputs, label = inputs, label
                
                embedding = embedding_net(inputs)
                embedding = embedding.detach().to('cpu')
                embedding = embedding.squeeze(0)
                patient_embedding.append(embedding)
                
            patient_embedding = torch.stack(patient_embedding)
            patient_embedding = patient_embedding.cuda()
            
            logits, Y_prob, Y_hat, _, instance_dict = classification_net(patient_embedding, label=label, instance_eval=True)
            val_acc_logger.log(Y_hat, label)
            
            val_acc += torch.sum(Y_hat == label.data)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            val_inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            val_inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.detach().to('cpu').numpy()
            labels[batch_idx] = label.item()
            
            error = classification_net.calculate_error(Y_hat, label)
            val_error += error
            
            
        val_error /= len(test_loaded_subsets)
        val_loss /= len(test_loaded_subsets)
        val_accuracy = val_acc / len(test_loaded_subsets)
        
        if n_classes == 2:
            val_auc = roc_auc_score(labels, prob[:, 1])
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
        
            val_auc = np.nanmean(np.array(aucs))
    
        clsf_report = pd.DataFrame(classification_report(labels, np.argmax(prob, axis=1), output_dict=True, zero_division=1)).transpose()
        conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))
        sensitivity = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0]) # TP / (TP + FN)
        specificity = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1]) # TN / (TN + FP) 
                    
        print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_error, val_auc, val_accuracy))
        if val_inst_count > 0:
            val_inst_loss /= val_inst_count
            for i in range(2):
                acc, correct, count = val_inst_logger.get_summary(i)
                print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    
        for i in range(n_classes):
            acc, correct, count = val_acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
            
        print(clsf_report)
        print(conf_matrix)
        print('Sensitivity: ', sensitivity) 
        print('Specificity: ', specificity) 
        #avg_loss_val = loss_val / len(test_loaded_subsets)
        #avg_acc_val = acc_val / len(test_loaded_subsets)
        
        # history['train_acc'].append(avg_acc)
        # history['train_loss'].append(avg_loss)
        # history['val_acc'].append(avg_acc_val)
        # history['val_loss'].append(avg_loss_val)
        
        # print()
        # print("Epoch {} result: ".format(epoch))
        # print("Avg loss (train): {:.4f}".format(avg_loss))
        # print("Avg acc (train): {:.4f}".format(avg_acc))
        # print("Avg loss (val): {:.4f}".format(avg_loss_val))
        # print("Avg acc (val): {:.4f}".format(avg_acc_val))
        # print('-' * 10)
        # print()

        if val_auc > best_auc:
            best_model_classification_wts = copy.deepcopy(classification_net.state_dict())
            best_auc = val_auc
            
    elapsed_time = time.time() - since
    
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    #print("Best acc: {:.4f}".format(best_acc))
    
    #embedding_net.load_state_dict(best_model_embedding_wts)
    classification_net.load_state_dict(best_model_classification_wts)
        
    return embedding_net, classification_net

# %%

loss_fn = nn.CrossEntropyLoss()

#all_params = itertools.chain(embedding_net.parameters(), classification_net.parameters())

#optimizer_ft = optim.SGD(all_params, lr=0.0001, momentum=0.9, weight_decay=0)
#optimizer_ft = optim.Adam(classification_net.parameters(), lr=0.0001)
optimizer_ft = optim.Adam(gated_net.parameters(), lr=0.0001)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.01)

# %%

embedding_weights, classification_weights = train_model(embedding_net, gated_net, train_loaded_subsets, test_loaded_subsets, loss_fn, optimizer_ft, n_classes=2, bag_weight=0.7, num_epochs=10)

#embedding_net, classification_net, train_loaded_subsets, test_loaded_subsets, n_classes, bag_weight, loss_fn, optimizer, num_epochs=1

# %%

center = "BIRM"

#torch.save(embedding_weights.state_dict(), r"C:/Users/Amaya/Documents/PhD/NECCESITY/Slides/embedding_QMUL_Binary_12.pt")
torch.save(classification_weights.state_dict(), r"C:/Users/Amaya/Documents/PhD/NECCESITY/Slides/classification_" + center + "_Binary_12.pt")

# %%

def test_model(embedding_net, classification_net, test_loaded_subsets, loss_fn, n_classes): 
               
    since = time.time()
    
    ###################################
    # TEST
    
    val_acc_logger = Accuracy_Logger(n_classes)
    val_inst_logger = Accuracy_Logger(n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_count=0
    
    val_acc = 0
    
    embedding_net.eval()
    classification_net.eval()
    
    prob = np.zeros((len(test_loaded_subsets), n_classes))
    labels = np.zeros(len(test_loaded_subsets))
    #sample_size = classification_net.k_sample

    for batch_idx, loader in enumerate(test_loaded_subsets.values()):
        
        print("\rValidation batch {}/{}".format(batch_idx, len(test_loaded_subsets)), end='', flush=True)
        
        patient_embedding = []
        for data in loader:
            
            inputs, label = data
            
            with torch.no_grad():
                if use_gpu:
                    inputs, label = inputs.cuda(), label.cuda()
                else:
                    inputs, label = inputs, label
            
            embedding = embedding_net(inputs)
            embedding = embedding.detach().to('cpu')
            embedding = embedding.squeeze(0)
            patient_embedding.append(embedding)
            
        patient_embedding = torch.stack(patient_embedding)
        patient_embedding = patient_embedding.cuda()
        
        logits, Y_prob, Y_hat, _, instance_dict = classification_net(patient_embedding, label=label, instance_eval=True)
        val_acc_logger.log(Y_hat, label)
        
        val_acc += torch.sum(Y_hat == label.data)
        
        loss = loss_fn(logits, label)

        val_loss += loss.item()

        instance_loss = instance_dict['instance_loss']
        
        val_inst_count+=1
        instance_loss_value = instance_loss.item()
        val_inst_loss += instance_loss_value

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        val_inst_logger.log_batch(inst_preds, inst_labels)

        prob[batch_idx] = Y_prob.detach().to('cpu').numpy()
        labels[batch_idx] = label.item()
        
        error = classification_net.calculate_error(Y_hat, label)
        val_error += error
        
        
    val_error /= len(test_loaded_subsets)
    val_loss /= len(test_loaded_subsets)
    val_accuracy = val_acc / len(test_loaded_subsets)
    
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
    
        auc = np.nanmean(np.array(aucs))

    clsf_report = pd.DataFrame(classification_report(labels, np.argmax(prob, axis=1), output_dict=True, zero_division=1)).transpose()
    conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))
    sensitivity = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0]) # TP / (TP + FN)
    specificity = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1]) # TN / (TN + FP) 

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_error, auc, val_accuracy))
    if val_inst_count > 0:
        val_inst_loss /= val_inst_count
        for i in range(2):
            acc, correct, count = val_inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))


    for i in range(n_classes):
        acc, correct, count = val_acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    print(clsf_report)
    print(conf_matrix)
    print('Sensitivity: ', sensitivity) 
    print('Specificity: ', specificity)

    elapsed_time = time.time() - since
    
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        
    return val_error, auc, val_accuracy, val_acc_logger, clsf_report, conf_matrix, sensitivity, specificity


# %%
    
from attention_models import VGG_embedding, GatedAttention

center = "BIRM"

gated_net = GatedAttention()

# load pre trained models
embedding_net = VGG_embedding(center="BIRM")
gated_net.load_state_dict(torch.load(r"C:/Users/Amaya/Documents/PhD/NECCESITY/Slides/classification_" + center + "_Binary_12.pt"))

if use_gpu:
    embedding_net.cuda()
    gated_net.cuda()

# %%

test_error, test_auc, test_accuracy, test_acc_logger, clsf_report, conf_matrix, sensitivity, specificity =  test_model(embedding_net, gated_net, test_loaded_subsets, loss_fn, n_classes=2)

 # %%
