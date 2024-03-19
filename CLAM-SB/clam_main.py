# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:35:37 2024

@author: AmayaGS
"""
import os
import os.path
from pathlib import Path
import numpy as np
import pandas as pd
import statistics
from collections import Counter
import pickle
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

# KRAG functions
from auxiliary_functions import seed_everything

# CLAM functions
from clam_train_loop import train_clam_multi_wsi
from CLAM_SB_model import GatedAttention

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")



def minority_sampler(train_graph_dict):

    # calculate weights for minority oversampling
    count = []
    for k, v in train_graph_dict.items():
        count.append(v[1].item())
    counter = Counter(count)
    class_count = np.array(list(counter.values()))
    weight = 1 / class_count
    samples_weight = np.array([weight[t] for t in count])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), num_samples=len(samples_weight),  replacement=True)

    return sampler


def arg_parse():

    parser = argparse.ArgumentParser(description="self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level")

    # Command line arguments
    parser.add_argument("--dataset_name", type=str, default="RA", choices=['RA', 'LUAD', 'LSCC'], help="Dataset name")
    parser.add_argument("--embedding_vector_size", type=int, default=1000, help="Embedding vector size")
    parser.add_argument("--embedding_net", type=str, default="vgg16", choices=['resnet18', 'vgg16', 'convnext'], help="feature extraction network used")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--train_fraction", type=float, default=0.7, help="Train fraction")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--batch_size", type=int, default=1, help="Graph batch size for training")
    parser.add_argument("--checkpoint", action="store_false", default=True, help="Enable checkpointing of GNN weights. Set to False if you don't want to store checkpoints.")

    return parser.parse_args()


def main(args):

    # Parameters
    seed = args.seed
    seed_everything(seed)
    num_workers = args.num_workers
    batch_size = args.batch_size
    embedding_vector_size = args.embedding_vector_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs

    dataset_name = args.dataset_name
    embedding_net = args.embedding_net
    n_classes= args.n_classes

    checkpoint = args.checkpoint
    current_directory = "/data/scratch/wpw030/MUSTANGv2_scratch" # change dictionary loading and results directory here.
    run_results_folder = f"clam_{dataset_name}_{embedding_net}_{seed}_{learning_rate}"
    results = os.path.join(current_directory, "results/" + run_results_folder)
    checkpoints = results + "/checkpoints"
    os.makedirs(results, exist_ok = True)
    os.makedirs(checkpoints, exist_ok = True)


    with open(current_directory + f"/embedding_dict_{dataset_name}_{embedding_net}.pkl", "rb") as file:
    # Load the dictionary from the file
        embedding_dict = pickle.load(file)

    with open(current_directory + f"/train_test_strat_splits_{dataset_name}.pkl", "rb") as splits:
        sss_folds = pickle.load(splits)

    mean_best_acc = []
    mean_best_AUC = []

    training_folds = []
    testing_folds = []
    for folds, splits in sss_folds.items():
        for i, (split, patient_ids) in enumerate(splits.items()):
            if i == 0:
                train_dict = dict(filter(lambda i:i[0] in patient_ids, embedding_dict.items()))
                training_folds.append(train_dict)
            if i ==1:
                test_dict = dict(filter(lambda i:i[0] in patient_ids, embedding_dict.items()))
                testing_folds.append(test_dict)

    for fold_idx, (train_fold, test_fold) in enumerate(zip(training_folds, testing_folds)):

        clam_net = GatedAttention(embedding_vector_size)
        loss_fn = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(clam_net.parameters(), lr=learning_rate)
        if use_gpu:
            clam_net.cuda()

        sampler = minority_sampler(train_fold)

        train_loader = torch.utils.data.DataLoader(train_fold, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_fold, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        _, results_dict, best_acc, best_AUC = train_clam_multi_wsi(clam_net, train_loader, test_loader, loss_fn, optimizer_ft, n_classes=n_classes, num_epochs=num_epochs, checkpoint=checkpoint, checkpoint_path= checkpoints + "/checkpoint_fold_" + str(fold_idx) + "_epoch_")

        mean_best_acc.append(best_acc.item())
        mean_best_AUC.append(best_AUC.item())

        df_results = pd.DataFrame.from_dict(results_dict)
        df_results.to_csv(results + "/" + run_results_folder + "_fold_" + str(fold_idx) + ".csv", index=False)

    average_best_acc = sum(mean_best_acc) / len(mean_best_acc)
    std_best_acc = statistics.pstdev(mean_best_acc)
    mean_best_acc.append(average_best_acc)
    mean_best_acc.append(std_best_acc)

    average_best_AUC = sum(mean_best_AUC) / len(mean_best_AUC)
    std_best_AUC = statistics.pstdev(mean_best_AUC)
    mean_best_AUC.append(average_best_AUC)
    mean_best_AUC.append(std_best_AUC)

    summary =[mean_best_acc] + [mean_best_AUC]
    summary_df = pd.DataFrame(summary, index=['val_accuracy', 'val_AUC']).transpose()
    summary_df.to_csv(results + "/" + run_results_folder + "_summary_best_scores.csv", index=0)


# %%

if __name__ == "__main__":
    args = arg_parse()
    args.checkpoint = False
    args.embedding_net = 'vgg16'
    main(args)