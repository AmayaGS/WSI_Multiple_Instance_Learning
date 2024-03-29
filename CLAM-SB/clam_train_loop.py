# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:58:10 2023

@author: AmayaGS
"""

import time
import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc

import torch
from auxiliary_functions import Accuracy_Logger

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

import gc
gc.enable()


def train_clam_multi_wsi(clam_net, train_loader, test_loader, loss_fn, optimizer, n_classes=2, bag_weight=0.7, num_epochs=30, checkpoint=False, checkpoint_path="PATH_checkpoints"):

    since = time.time()
    best_acc = 0.
    best_AUC = 0.

    train_loss_list = []
    train_accuracy_list = []

    val_loss_list = []
    val_accuracy_list = []
    val_auc_list = []

    sensitivity_list = []
    specificity_list = []

    for epoch in range(num_epochs):

        ###################################
        # TRAIN
        acc_logger = Accuracy_Logger(n_classes=n_classes)
        inst_logger = Accuracy_Logger(n_classes=n_classes)

        train_loss = 0 # train_loss
        train_error = 0
        train_inst_loss = 0.
        inst_count = 0

        train_acc = 0
        train_count = 0

        clam_net.train()

        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        for batch_idx, (patient_ID, embedding) in enumerate(train_loader.dataset.items()):

            data, label,_,_ = embedding

            if use_gpu:
                data, label = data.cuda(), label.cuda()
            else:
                data, label = data, label

            logits, Y_prob, Y_hat, _, instance_dict,_ = clam_net(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)
            loss_value = loss.item()

            train_acc += torch.sum(Y_hat == label.data)
            train_count += 1

            instance_loss = instance_dict['instance_loss']
            inst_count+=1
            instance_loss_value = instance_loss.item()
            train_inst_loss += instance_loss_value

            total_loss = bag_weight * loss + (1-bag_weight) * instance_loss

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            train_loss += loss_value
            error = clam_net.calculate_error(Y_hat, label)
            train_error += error

            # backward pass
            total_loss.backward()
            # step
            optimizer.step()
            optimizer.zero_grad()

        train_loss /= train_count
        train_error /= train_count
        train_accuracy =  train_acc / train_count

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy.to('cpu').detach().numpy())


        if inst_count > 0:
            train_inst_loss /= inst_count
            print('\n')

        print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}, train_accuracy: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error, train_accuracy))
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count), flush=True)

        ###################################
        # TEST

        clam_net.eval()

        val_acc_logger = Accuracy_Logger(n_classes)
        val_inst_logger = Accuracy_Logger(n_classes)
        val_loss = 0.
        val_error = 0.

        val_inst_loss = 0.
        val_inst_count= 0

        val_acc = 0
        val_count = 0

        prob = []
        labels = []

        for batch_idx, (patient_ID, embedding) in enumerate(test_loader.dataset.items()):

            data, label,_,_ = embedding

            with torch.no_grad():
                if use_gpu:
                    data, label = data.cuda(), label.cuda()
                else:
                    data, label = data, label

            logits, Y_prob, Y_hat, _, instance_dict,_ = clam_net(data, label=label, instance_eval=True)

            val_acc_logger.log(Y_hat, label)
            val_acc += torch.sum(Y_hat == label.data)
            val_count +=1
            loss = loss_fn(logits, label)
            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            val_inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value
            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            val_inst_logger.log_batch(inst_preds, inst_labels)

            prob.append(Y_prob.detach().to('cpu').numpy())
            labels.append(label.item())

            error = clam_net.calculate_error(Y_hat, label)
            val_error += error

        val_error /= val_count
        val_loss /= val_count
        val_accuracy = val_acc / val_count

        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy.item())

        if n_classes == 2:
            prob =  np.stack(prob, axis=1)[0]
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

        val_auc_list.append(val_auc)

        conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))

        print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_error, val_auc, val_accuracy))
        if val_inst_count > 0:
            val_inst_loss /= val_inst_count

        print(conf_matrix)
        if n_classes == 2:
            sensitivity = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) # TP / (TP + FN)
            specificity = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
            print('Sensitivity: ', sensitivity)
            print('Specificity: ', specificity, flush=True)

        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

        if val_accuracy >= best_acc:
            best_acc = val_accuracy

            if checkpoint:
                checkpoint_weights = checkpoint_path + str(epoch) + ".pth"
                torch.save(clam_net.state_dict(), checkpoint_weights)

        if val_auc >= best_AUC:
            best_AUC = val_auc

            if checkpoint:
                checkpoint_weights = checkpoint_path + str(epoch) + ".pth"
                torch.save(clam_net.state_dict(), checkpoint_weights)

    #if checkpoint:
    #    clam_net.load_state_dict(torch.load(checkpoint_weights), strict=True)

    results_dict = {'train_loss': train_loss_list,
                    'val_loss': val_loss_list,
                    'train_accuracy': train_accuracy_list,
                    'val_accuracy': val_accuracy_list,
                    'val_auc': val_auc_list,
                    'sensitivity': sensitivity_list,
                    'specificity': specificity_list
                    }

    elapsed_time = time.time() - since

    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    return clam_net, results_dict, best_acc, best_AUC


def test_clam_slides(clam_net, test_loader, loss_fn, optimizer_ft, embedding_vector_size, n_classes=2):

    # TEST

    since = time.time()

    test_acc_logger = Accuracy_Logger(n_classes)
    test_inst_logger = Accuracy_Logger(n_classes)
    test_loss = 0.
    test_error = 0.

    test_inst_loss = 0.
    test_inst_count= 0

    test_acc = 0
    test_count = 0

    test_loss_list = []
    test_accuracy_list = []
    test_auc_list = []

    prob = []
    labels = []

    clam_net.eval()

    for batch_idx, (patient_ID, embedding) in enumerate(test_loader.items()):

        data, label = embedding

        with torch.no_grad():
            if use_gpu:
                data, label = data.cuda(), label.cuda()
            else:
                data, label = data, label

        logits, Y_prob, Y_hat, _, instance_dict,_ = clam_net(data, label=label, instance_eval=True)

        test_acc_logger.log(Y_hat, label)
        test_acc += torch.sum(Y_hat == label.data)
        test_count +=1
        loss = loss_fn(logits, label)
        test_loss += loss.item()

        instance_loss = instance_dict['instance_loss']
        test_inst_count+=1
        instance_loss_value = instance_loss.item()
        test_inst_loss += instance_loss_value
        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        test_inst_logger.log_batch(inst_preds, inst_labels)

        prob.append(Y_prob.detach().to('cpu').numpy())
        labels.append(label.item())

        error = clam_net.calculate_error(Y_hat, label)
        test_error += error

    test_error /= test_count
    test_loss /= test_count
    test_accuracy = test_acc / test_count

    test_loss_list.append(test_loss)
    test_accuracy_list.append(test_accuracy.item())

    if n_classes == 2:
        prob =  np.stack(prob, axis=1)[0]
        test_auc = roc_auc_score(labels, prob[:, 1])
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

        test_auc = np.nanmean(np.array(aucs))

    test_auc_list.append(test_auc)

    conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_error, test_auc, test_accuracy))
    if test_inst_count > 0:
        test_inst_loss /= test_inst_count

    print(conf_matrix)
    if n_classes == 2:
        sensitivity = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) # TP / (TP + FN)
        specificity = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
        print('Sensitivity: ', sensitivity)
        print('Specificity: ', specificity, flush=True)

    elapsed_time = time.time() - since

    print()
    print("Testing completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    return test_loss_list, test_accuracy_list, test_auc_list, labels, prob, conf_matrix, sensitivity, specificity