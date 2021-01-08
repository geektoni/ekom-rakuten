# Large-scale taxonomy classification of product titles

## Method

The methods used were inspired by [1] (FastText embeddings) and [2]. See the reference section for the original papers.

### Data preprocessing

The data analysis was not reported here. Please have a look at [1] for a brief
explanation (especially sections 2 and 2.1).

Before feeding the data to our model, we perform several standard actions to improve the quality of our dataset. The steps taken are:
- Place everything to lowercase.
- Remove punctuation marks.
- Tokenize the given product title.
- Convert numbers with a specific token *@NUMBER@*. We keep unchanged alphanumeric entries (e.g, 3343als) since they could provide meaningful information specific to certain categories.
- Add *@EOS@* token at the end of the product title.

No further improvements are made. It is hinted in [2] that character-level tokenization may be better since there is a high incidence of alphanumeric product codes.

The original dataset was then split into two separate entries (`train_dataset.tsv.gzip` and `test_dataset.tsv.gzip`) which are used to train and then validate our model (80% training, 20% test).
The splitting was done by random sampling (as an improvement, stratified sampling could be used instead).

### Embeddings and Categories

We use the FastText utility to produce suitable embeddings given the training dataset only. We also created a dictionary to match all the 3001 categories of
the training set with an index. We also set an index for unknown categories (which may be present in the test set but not in the training set).

### Model

The model is a **Bidirectional LSTM** with a Linear layer to predict the target
categories. Dropout is employed to reduce overfitting and to improve final
performances. We use the Cross-Entropy loss and Adam as the optimizer. More complex strategies such as cross-validation
to validate the models were not used because of time constraints. The same applies for hyperparameters search.


The training process is the following:
- We fetch a batch of product titles and the corresponding categories.
- We add padding (*@PAD@* symbol) to some of the product titles to have a uniform size batch. The padding symbol's embedding is an empty tensor (all zeros);
- For each product, we convert the title in its embedding form;
- We feed the result to our network and we compute the Cross-Entropy loss. We then perform a learning iteration.
- We start over with a new batch.

## Installation

This repository was tested on Ubuntu 16.04 with Python 3.7. You can use conda
to install all the dependencies needed for the project. Once you have cloned
the repository, please follow these steps to install everything and to start
the environment
```bash
git clone https://github.com/geektoni/ekom-rakuten
cd ekom-rakuten
conda create --name ekom python=3.7
conda activate ekom
pip install -r  requirements.txt
```
Besides standard requirements, we make use of FastText library of Facebook to
generate the word embeddings and also to train a baseline model.
Follow the steps below to obtain the code (you will need to have a suitable g++
compiler).
```bash
cd ekom-rakuten
conda activate ekom
git clone https://github.com/facebookresearch/fastText.git
cd fastText
make
pip install .
```  

## Usage

### Prepare the data

**Note: you will need to place the data called rdc-catalog-train.tsv.gz
into the dataset directory before running any script**

Before training or running a pre-trained model, we need first to parse the original
dataset and we need to create the word embeddings by using the FastText command-line
tool. We first preprocess the dataset with the `prepare_dataset.py` script. The option `--fasttext` generates a
suitable file to be parsed with FastText.
```bash
cd ekom-rakuten
conda activate ekom
python prepare_dataset.py --fasttext
```

We then compute the FastText embeddings by taking as inputs the file generated in the `./dataset` directory.
We use the skipgram models and we generate 100 dimensions embeddings.
```bash
cd fastText
./fasttext skipgram -input ../dataset/train_dataset.tsv.gzip.fasttext.csv -output ../dataset/fasttext_embeddings -dim 100
```

### Train the model
After having built the embeddings and the training and test dataset, we can move forward to training the model itself. Since we load most of the data inside the main
memory, make sure to have at least around 6GB of free RAM.

```bash
cd ekom-rakuten
conda activate ekom
python train.py
```
To speed up training, it is possible also to skip the validation by passing an additional keyword.
```bash
python train.py --skip-validation
```

### Predict and evaluate
Once the model is trained, we can easily compute our predictions and evaluate our model by running the following script.
```bash
cd ekom-rakuten
conda activate ekom
python predict.py --load-models ./dataset/<model_of_you_choice>.pth
python eval.py prediction.csv gold.csv
```
If you want to supply the pre-trained model the commands are:
```bash
python predict.py --load-models ./dataset/rakuten_pretrained.pth
python eval.py prediction.csv gold.csv
```

## References

[1] Lin, Yiu-Chang, Pradipto Das, and Ankur Datta. "Overview of the SIGIR 2018 eCom Rakuten Data Challenge." eCOM@ SIGIR. 2018.

[2] Skinner, Michael. "Product categorization with LSTMs and balanced pooling views." eCOM@ SIGIR. 2018.
