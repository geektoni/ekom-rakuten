# Large-scale taxonomy classification of product titles

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
