import numpy as np
import torch
from torch import nn
import numpy as np
import pandas as pd
import math


def create_train_val_test(train_folds, val_fold, test_fold):
    df = pd.read_csv('/zhome/be/1/138857/DL_project/data/dataframe_ids_y_len_prot_fold.csv')

    # Added just to test smaller dataset, delete after
    #df = df.sort_values('len_prot')
    #df = df.iloc[:300]

    # Training set
    train_set = df[df['fold'].isin(train_folds)]

    # Validation set
    val_set = df[df.fold == val_fold]

    # Test setv
    test_set = df[df.fold == test_fold]

    return train_set, val_set, test_set

def create_batches(data_set, batch_size, device):
    X_path = '/zhome/be/1/138857/DL_project/data/protein_encodings/'

    # Sorting the sequences by length
    sorted_df = data_set.sort_values('len_prot')

    # Calculating the number of batches
    n_batches = math.ceil(len(data_set) / batch_size)

    # Initiating the result arrays
    batches_X, batches_y, batches_len = [], [], []

    # Creating the batches
    for batch in range(n_batches):

        X_batch = []
        y_batch = []
        max_len = []

        # Calculate the start and end indices for the batch
        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size

        # Finding the longest protein in the batch
        max_len = max(sorted_df[start_idx:end_idx]['len_prot'].values)

        # Loading in the relevant X.pt files in the order of the ids of the batch
        ids_batch = sorted_df[start_idx:end_idx]['ids'].values

        for i in range(len(ids_batch)):

            # Load ESM-IF1 encoding
            ESM_encoding = torch.load(f'{X_path}{ids_batch[i]}.pt')

            # 0 pad so all proteins in the batch have the same length
            pad = nn.ConstantPad2d((0,0,0,(max_len - ESM_encoding.size(0))), 0)
            tmp_x = pad(ESM_encoding)
            X_batch.append(tmp_x)

        # Turn into tensors and collect each batch in a list
        X_batch = torch.stack(X_batch, dim=0).to(device)
        batches_X.append(X_batch)

        # Define labels from the df that belongs to the batch
        y_batch_string = sorted_df[start_idx:end_idx]['y'].values

        # -1 pad all labels
        for i in range(len(y_batch_string)):
            # Convert string to a list of integers
            int_list = [int(x) for x in y_batch_string[i]]
            padded_list = int_list + [-1] * (max_len - len(int_list))
            y_batch.append(padded_list)

        # Turn into tensors and collect each batch in a list
        y_batch = torch.tensor(y_batch, dtype=torch.float, device = device)
        batches_y.append(y_batch)

        # Saving the size of the batch
        batches_len.append(len(ids_batch))

    return batches_X, batches_y, batches_len
