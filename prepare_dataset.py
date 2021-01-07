import argparse
import string
import csv
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Punctuation table
table = str.maketrans(dict.fromkeys(string.punctuation))

def strip_punctuation(product_title):
    """
    Given a string, it removes the punctuation signs as defined
    in string.punctuation
    :param product_title: string
    :return: string without punctuation
    """
    return product_title.translate(table)


def convert_number_to_token(product_tokenized):
    """
    Given a tokenized string, it will convert all the numeric occurrences
    into a custom token (@NUMBER@).
    :param product_tokenized: the tokenized product title
    :return: sanitized product (list)
    """
    return list(map(lambda x: "@NUMBER@" if x.isnumeric() else x, product_tokenized))


def stratified_sampling(df, N):
    """
    Perform stratified sampling
    :param df: original dataset
    :param N: size of the final dataset
    :return: reduced dataset
    """
    return df.groupby('category', group_keys=False).apply(
        lambda x: x.sample(int(np.rint(N * len(x) / len(df))))).sample(
        frac=1).reset_index(drop=True)


if __name__ == "__main__":

    # Build the command-line parser
    parser = argparse.ArgumentParser("prepare_dataset.py",
                                     description="Perform some basic cleaning on Rakuten dataset.")
    parser.add_argument("--data", help="Path to the .tsv file containing the data",
                        type=str, default="./dataset/rdc-catalog-train.tsv")
    parser.add_argument("--output-train", help="Path where to save the polished data",
                        type=str, default="./dataset/train_dataset.tsv.gzip")
    parser.add_argument("--output-test", help="Path where to save the polished data",
                        type=str, default="./dataset/test_dataset.tsv.gzip")
    parser.add_argument("--fasttext", help="Prepare the dataset for Fasttext",
                        action="store_true", default=False)
    parser.add_argument("--size", help="Specify the size of the final dataset",
                        type=int, default=8000000)
    parser.add_argument("--stratified", help="Perform stratified sampling to reduce the size of the"
                                             "dataset.", action="store_true", default=False)

    # Parse and get the arguments
    args = parser.parse_args()
    data_path = args.data
    output_train = args.output_train
    output_test = args.output_test
    fasttext = args.fasttext
    size = args.size
    stratified = args.stratified

    # Read the data inside a dataframe
    df = pd.read_csv(data_path, header=None, sep="\t", names=["product", "category"])
    if not stratified:
        df = df[0:size]
    else:
        df = stratified_sampling(df, size)

    # We polish the data given, namely the "product" column. The step we perform are
    # the following:
    # 1) Put everything to lowercase
    # 2) Remove punctuation marks
    # 3) Tokenize the string
    # 4) Convert numbers with a specific token @NUMBER@. We keep unchanged
    #    alphanumeric entries (e.g, 3343als)
    df["product"] = df["product"].apply(lambda x: x.lower())
    df["product"] = df["product"].apply(lambda x: strip_punctuation(x))
    df["product"] = df["product"].apply(lambda x: x.split())
    df["product"] = df["product"].apply(lambda x: convert_number_to_token(x))

    # Split the dataset into train/test
    train, test = train_test_split(df, test_size=0.2)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Save the train and test sets into pickles
    train.to_pickle(output_train, compression="gzip")
    test.to_pickle(output_test, compression="gzip")

    if fasttext:
        # Fasttext dataset format expects lines in the following fashion:
        # __label__<category>  <sentence>
        train["category"] = train["category"].apply(lambda x: "__label__" + x)
        train[['category', 'product']].to_csv(output_train + ".fasttext.csv",
                                           index=False,
                                           sep=' ',
                                           header=None,
                                           quoting=csv.QUOTE_NONE,
                                           quotechar="",
                                           escapechar=" ")
