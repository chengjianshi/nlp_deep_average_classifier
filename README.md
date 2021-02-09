# README.md

> Python - 3.8.5 
> PyTorch - 1.7.0 
> PyTorch-lighting - 1.1.4 
> nltk - 3.5.0

## 1. Word2Vect

**FeatureGenerator.py** is the module to build vocaburary and feature space of training file in the **{local_path}/features/**. The availabe word grams are uni/bi/trigram depend on user settings. The output file of 
vocaburary will be stored in *{local_path}/features/{type}_vocab.json*. The module can generate text features from 
given vocaburary, the available features space are **binary** and **tfidf** features in given uni/bi/trigram vocaburary. 
The labels and features output are saved in **{local_path}_{type}_{feature_type}_{feature/label}.npz**


>example: {sst2.train} is the training file for movie classification

```{python}

from featureGenerator import *

local_path = "./data"
gen = featureGenerator(local_path)
gen.getNgramVocab(local_path+"/sst2.train", ngram = 1)

# output #
# data/features/unigram_vocab.json size: 4949 #

gen.binaryFeature(local_path+"/sst2.train", ngram = 1)
gen.tfidfFeature(local_path+"/sst2.train", ngram=1)

# output
# labels saved in data/features/train_label.npz
features saved in data/features/train_unigram_binary_feature.npz
features space (6920, 4949)

labels saved in data/features/train_label.npz
features saved in data/features/train_unigram_binary_feature.npz
features space (6920, 4949) #

```

## 2. Deep Average Model

The deep average model are built by Pytorch-lighting, the base model of **pl** is written in **pl_base.py**. Then both model can specify customized model params in **dan.py**. The output with hyperparameters are saved in **{local_path}/{lr/dan}** dir. 

> training data
```
pip install gdown
gdown --id 1thWkUj7uGOApr_dXRvMr9TsEHpo_H_2q -O sst2.zip
mkdir data
unzip sst2.zip -d data
```

> training using DAN
```{python}

import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
import time
import argparse
import glob
import os
from featureGenerator import *
from pl_base import *

logger = logging.getLogger(__name__)

gen =  featureGenerator("./")

gen.getNgramVocab("./data/sst2.train", ngram = 1)

def main(lr, hidden_layers, optim, drop_out):

    DATA_DIR = "./data/"
    mock_args = f"--word_embedding_size 300 --data_dir {DATA_DIR} --output_dir dan --optimizer {optim} \
    --vocab_filename features/unigram_vocab.json --learning_rate {lr} --hidden_layers {hidden_layers} --drop_out {drop_out} --max_epochs 10 --do_predict"

    # load hyperparameters
    parser = argparse.ArgumentParser()
    BaseModel.add_generic_args(parser, os.getcwd())
    parser = DAN_PL.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args(mock_args.split())
    print(args)
    # fix random seed to make sure the result is reproducible
    pl.seed_everything(args.seed)

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./results",
            f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(args.output_dir)
    dict_args = vars(args)
    model = DAN_PL(**dict_args)
    trainer = generic_train(model, args)

```
