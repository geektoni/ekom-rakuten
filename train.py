from model import DefaultLSTM
from dataset_loader import RakutenLoader

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import fasttext

import numpy as np
import pandas as pd

from argparse import ArgumentParser
import os
import datetime

import tqdm

if __name__ == "__main__":

    # Set up the argument parser
    parser = ArgumentParser()
    parser.add_argument("--save-model", type=str, help="Location where to save the trained model", default="./models")
    parser.add_argument("--dataset", type=str, help="Location where to search for/create the dataset",
                        default="./dataset")
    parser.add_argument("--seed", type=int, help="Set the random number generator seed to ensure reproducibility",
                        default=2020)
    parser.add_argument("--epochs", type=int, help="Training epochs", default=1000)
    parser.add_argument("--gpu", help="If True, the training will procede on a GPU if it is available",
                        default=False, action="store_true")
    parser.add_argument("--skip-validation", help="If True, the training will procede on a GPU if it is available",
                        default=False, action="store_true")

    # Parse the arguments
    args = parser.parse_args()
    dataset_directory = args.dataset
    model_directory = args.save_model
    seed = args.seed
    epochs = args.epochs
    gpu = args.gpu
    do_val = args.skip_validation

    # Set seed and deterministic behaviour
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_deterministic(True)

    # Model name
    timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    model_save = os.path.join(model_directory,
                              "rakuten_{}.pth".format(timestamp))

    # Load the complete dataset
    train = pd.read_pickle("./dataset/train_dataset.tsv.gzip", compression="gzip")
    test = pd.read_pickle("./dataset/train_dataset.tsv.gzip", compression="gzip")

    # Embeddings (these are learnt using only the train dataset)
    embedding = fasttext.load_model("./dataset/fasttext_embeddings.bin")

    # Generate categories
    categories = dict([(s, i) for i, s in enumerate(train["category"].unique())])

    # Create the dataset and convert the games into something
    # more usable (one-hot encoded version)
    dataset_train = RakutenLoader(train["product"], train["category"], embedding, categories)
    dataset_test = RakutenLoader(test["product"], train["category"], embedding, categories)

    # Create the DataLoader object used for training
    dataloader_train = DataLoader(dataset_train, batch_size=100,
                                shuffle=True, num_workers=4,
                                pin_memory=True)

    dataloader_test = DataLoader(dataset_test, batch_size=100,
                                  shuffle=True, num_workers=4,
                                 pin_memory=True)

    # Create the model we will use
    rakuten_model = DefaultLSTM(len(categories))

    # Define the optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(rakuten_model.parameters(), lr=0.001)

    # Check if we are on GPU, then move everything to GPU
    if torch.cuda.is_available() and gpu:
        device = "cuda:0"
    else:
        device = "cpu"

    # Move model to device
    rakuten_model.to(device)

    # Train the model
    for epoch in range(epochs):

        train_loss = 0.0
        for i_batch, sample_batched in enumerate(tqdm.tqdm(dataloader_train)):

            inputs, labels = sample_batched
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = rakuten_model(inputs)
            loss = criterion(outputs, labels.flatten())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # After each epoch return the validation results
        validation_loss = 0.0
        t_batch = 1
        if not do_val:
            for t_batch, sample_batched_test in enumerate(dataloader_test):

                inputs, labels = sample_batched_test
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = rakuten_model(inputs)
                loss_validation = criterion(outputs, labels.flatten())

                validation_loss += loss_validation.item()

        # Print the validation loss
        print("{} - Train/Validation Loss: {:.3f} {:.3f}".format(epoch,
                                                                 train_loss/i_batch,
                                                                 validation_loss/t_batch))

        # After each batch, we checkpoint the model
        torch.save(rakuten_model.state_dict(), model_save)
