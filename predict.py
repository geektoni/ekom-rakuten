import torch
from torch.utils.data import DataLoader

from model import BidLSTM
from dataset_loader import RakutenLoader

import fasttext

import pandas as pd
from tqdm import tqdm

from argparse import ArgumentParser

if __name__ == "__main__":

    # Set up the argument parser
    parser = ArgumentParser()
    parser.add_argument("--load-model", help="Load a pre-trained model to continue training", default=None, type=str)

    # Parse the arguments
    args = parser.parse_args()
    load_model = args.load_model

    # Load the complete dataset
    train = pd.read_pickle("./dataset/train_dataset.tsv.gzip", compression="gzip")
    test = pd.read_pickle("./dataset/test_dataset.tsv.gzip", compression="gzip")

    # Embedding (these are learnt using only the train dataset)
    embedding = fasttext.load_model("./dataset/fasttext_embeddings.bin")

    # Generate categories
    categories = dict([(s, i) for i, s in enumerate(train["category"].unique())])

    # Load test data
    dataset = RakutenLoader(test["product"],
                            test["category"],
                            embedding, categories)
    dataloader_test = DataLoader(dataset, batch_size=100,
                                 shuffle=False, num_workers=4,
                                 pin_memory=True)

    # Build model
    model = BidLSTM(len(categories))
    model.load_state_dict(torch.load(load_model))

    correct_result = []
    predicted_result = []

    for t_batch, sample_batched_test in enumerate(tqdm(dataloader_test)):

        inputs, labels = sample_batched_test

        labels = labels.flatten()
        outputs = model(inputs)
        outputs = torch.argmax(outputs, axis=1).flatten()

        for p, g in zip(outputs.numpy(), labels.numpy()):
            predicted_result.append(("empty", p))
            correct_result.append(("empty", g))

    # Convert the result into dataframes and save those dataframes
    predicted_result = pd.DataFrame(predicted_result, columns=["product", "category"])
    correct_result = pd.DataFrame(correct_result, columns=["product", "category"])

    predicted_result.to_csv("predictions.tsv", header=False, index=False, sep="\t")
    correct_result.to_csv("gold.tsv", header=False, index=False, sep="\t")
