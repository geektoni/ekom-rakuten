import argparse
import string
import csv
import pandas as pd

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


if __name__ == "__main__":

    # Build the command-line parser
    parser = argparse.ArgumentParser("prepare_dataset.py",
                                     description="Perform some basic cleaning on Rakuten dataset.")
    parser.add_argument("--data", help="Path to the .tsv file containing the data",
                        type=str, default="./dataset/rdc-catalog-train.tsv")
    parser.add_argument("--output", help="Path where to save the polished data",
                        type=str, default="./dataset/rdc-catalog-train-polished.tsv")
    parser.add_argument("--fasttext", help="Prepare the dataset for Fasttext",
                        action="store_true", default=False)

    # Parse and get the arguments
    args = parser.parse_args()
    data_path = args.data
    output_path = args.output
    fasttext = args.fasttext

    # Read the data inside a dataframe
    df = pd.read_csv(data_path, header=None, sep="\t", names=["product", "category"])

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

    # Save the polished dataset on disk
    if not fasttext:
        df.to_pickle(output_path, compression="gzip")
    else:
        # Fasttext dataset format expects lines in the following fashion:
        # __label__<category>  <sentence>
        df["category"] = df["category"].apply(lambda x: "__label__" + x)
        df[['category', 'product']].to_csv(output_path + ".fasttext.csv",
                                           index=False,
                                           sep=' ',
                                           header=None,
                                           quoting=csv.QUOTE_NONE,
                                           quotechar="",
                                           escapechar=" ")
