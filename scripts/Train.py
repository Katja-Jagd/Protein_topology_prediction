import numpy as np
import torch.optim as optim
import wandb
import torch.nn.functional as F
import torch.nn as nn
import torch
from model import Model1, Model2, Model3, Model4
from batches import create_train_val_test, create_batches
from bio_accuracy import *


def train(args):
    device = torch.device('cpu')
    model = Model4()			# Change model here 
    model.to(device)

    # Function to initialize weights
    def custom_weights_init(layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    # Initializing wandb setup
    wandb.init(project = "DL project")

    # List of configurations for splitting the data
    splits = [([0, 1, 2], 3, 4),
              ([1, 2, 3], 4, 0),
              ([2, 3, 4], 0, 1),
              ([3, 4, 0], 1, 2),
              ([4, 0, 1], 2, 3)]
    # 5-fold CV
    k = 0
    for split in splits:
        k += 1
        train_set, val_set, test_set = create_train_val_test(*split)
        batches_X_train, batches_y_train, batches_len_train = create_batches(train_set, args.batch_size, device)
        batches_X_val, batches_y_val, batches_len_val = create_batches(val_set, args.batch_size, device)
        batches_X_test, batches_y_test, batches_len_test = create_batches(test_set, args.batch_size, device)

        # Initialization
        model.apply(custom_weights_init)
        optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = 0.0001)
        criterion = nn.CrossEntropyLoss(ignore_index = -1)
        num_batches_train = len(batches_X_train)
        num_batches_val = len(batches_X_val)
        epochs = 100

        train_losses, val_losses, train_acc, val_acc = [], [], [], []
        best_acc = 0
        c = 0 
        for epoch in range(args.epochs):
            model.train()
            train_losses_batches, train_predictions = [], []
            for i in range(num_batches_train):

                # Forward pass.
                output = model(batches_X_train[i])

                # Reshape output and targets and compute loss.
                reshaped_output = output.permute(0, 2, 1)
                targets = batches_y_train[i].long()
                loss = criterion(reshaped_output , targets)
                #loss = -1 * crf(output, targets)
                train_losses_batches.append(loss.item())

                # Clean up gradients from the model.
                optimizer.zero_grad()

                # Compute gradients based on the loss from the current batch (backpropagation).
                loss.backward()

                # Take one optimizer step using the gradients computed in the previous step.
                optimizer.step()

                # Save prediction for batch in list for evaluation over epoch
                softmax_output = F.softmax(output, dim = 2)
                prediction = softmax_output.argmax(axis = 2)
                train_predictions.append(prediction)

            # Append average training loss to list.
            train_losses.append(np.mean(train_losses_batches))

            # Defining and logging training metrics onto wandb
            train_loss_metric = {"Train loss": train_losses[-1]}
            wandb.log(train_loss_metric)

            # Resetting lists for next batch
            train_losses_batches = []

            # Compute loss on validation set.
            val_losses_batches, val_predictions = [], []
            with torch.no_grad():
                model.eval()
                for j in range(num_batches_val):
                    output = model(batches_X_val[j])

                    # Reshape output and targets and compute loss.
                    reshaped_output = output.permute(0, 2, 1)
                    targets = batches_y_val[j].long()
                    loss = criterion(reshaped_output , targets)
                    #loss = -1 * crf(output, targets)
                    val_losses_batches.append(loss.item())

                    # Save prediction for batch in list for evaluation over epoch
                    softmax_output = F.softmax(output, dim = 2)
                    prediction = softmax_output.argmax(axis = 2)
                    val_predictions.append(prediction)

            _, acc = bio_acc(val_predictions, batches_y_val)
            val_acc.append(acc)
            val_acc_metric = {"Validation accuracy": val_acc[-1]}
            wandb.log(val_acc_metric)

            # Save best model per fold
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), '/zhome/be/1/138857/DL_project/models/model.'+str(k)+'.pt')

            # Append average validation loss to list.
            val_losses.append(np.mean(val_losses_batches))

            # Defining and logging validation metrics onto wandb
            val_loss_metric = {"Validation loss": val_losses[-1]}
            wandb.log(val_loss_metric)

            _, acc = bio_acc(train_predictions, batches_y_train)
            train_acc.append(acc)

            train_acc_metric = {"Train accuracy": train_acc[-1]}
            wandb.log(train_acc_metric)
            c += 1
            print(c)
    print("Finished training.")
    wandb.finish()
