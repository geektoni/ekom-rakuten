# Large-scale taxonomy classification of product titles

## Method

### Data preprocessing

Before feeding the data to our model, we performed several standard actions to
improve the quality of our dataset. The steps taken are:
- Place everything to lowercase.
- Remove punctuation marks.
- Tokenize the given product title.
- Convert numbers with a specific token *@NUMBER@*. We keep unchanged alphanumeric entries (e.g, 3343als) since they could provide meaningful information specific to certain categories.
- Add *@EOS@* token at the end of the product title.

The original dataset was then split into two separate entries (`train_dataset.tsv.gzip` and `test_dataset.tsv.gzip`) which are used to train and then validate our model. More complex strategies such as cross-validation where not used because of time constraints.

### Model

The model is a Bidirectional LSTM with a Linear layer to predict the target
categories. Dropout is employed to reduce overfitting and to improve final
performances.

The pipeline is the following:
- We fetch a batch of product titles
- We add padding, the *@PAD@* string, to some of the product titles to have an uniform size batch. The padding symbol's embedding is an empty tensor (all zeros);
- For each product, we convert the title in its embedding form;
- We feed the result to our network and we compute the Cross Entropy loss.
- We start over with a new batch.

## Installation

This repository was tested on Ubuntu 16.04 with Python 3.6. You can use conda
to install all the dependencies needed for the project. Once you have cloned
the repository, please follow these steps to install everything and to start
the environment
```bash
cd ekom-rakuten
conda create --name ekom --file requirements.txt
conda activate ekom
```
Beside standard requirements, we make use of FastText library of Facebook to
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

Before training or running a pre-trained model, we need first to parse the original
dataset and we need to create the word embeddings by using the FastText command line
tool. The command sequence is the following.
```bash
cd ekom-rakuten
conda activate ekom
python prepare_dataset.py --fasttext
```

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

### Predict and evaluate
Once the model is trained, we can easily compute our predictions and evaluate our model by running the following script.
```bash
cd ekom-rakuten
conda activate ekom
python predict.py
python eval.py prediction.csv gold.csv
```
