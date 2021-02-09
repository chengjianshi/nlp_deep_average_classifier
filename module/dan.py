import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import argparse
import os
from pathlib import Path
from torch.optim import SGD, Adam
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from datetime import datetime
from pathlib import Path
from pytorch_lightning import loggers as pl_loggers
import time
from argparse import Namespace
import json
import shutil
from pl_base import *
tokenizer = WordPunctTokenizer()


class SST2Dataset(Dataset):
    """
    Using dataset to process input text on-the-fly
    """

    def __init__(self, vocab, data):
        self.data = data
        self.vocab = vocab
        self.max_len = 50  # assigned based on length analysis of training set

    def __getitem__(self, index):
        note = []
        label, text = int(self.data[index][0]), self.data[index][1]
        tokens = list(nltk.ngrams(tokenizer.tokenize(text.strip().lower()), 1))
        # if word does not exist, give <unk> token id
        token_ids = [self.vocab.get(repr(t), 1) for t in tokens]
        # in case token length exceed max length
        length = min(len(token_ids), self.max_len)
        # truncate or pad to max length
        padded_token_ids = token_ids[:50] + [0] * (self.max_len - length)
        return padded_token_ids, label, length

    def collate_fn(self, batch_data):
        padded_token_ids, labels, lengths = list(zip(*batch_data))
        return (torch.LongTensor(padded_token_ids).view(-1, self.max_len),
                torch.FloatTensor(labels).view(-1, 1), torch.FloatTensor(lengths).view(-1, 1))

    def __len__(self):
        return len(self.data)


class DAN_PL(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_model(self):
        return DAN(self.hparams.vocab, self.hparams.vocab_size, self.hparams.word_embedding_size, self.hparams.hidden_layers, self.hparams.drop_out, self.hparams.use_glove)

    def get_dataloader(self, type_path, batch_size, shuffle=False):

        datapath = os.path.join(self.hparams.data_dir, f"sst2.{type_path}")
        data = open(datapath).readlines()
        data = [d.strip().split(" ", maxsplit=1)
                for d in data]  # list of [label, text] pair
        dataset = SST2Dataset(self.hparams.vocab, data)

        logger.info(f"Loading {type_path} data and labels from {datapath}")
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            collate_fn=dataset.collate_fn
        )

        return data_loader

    def configure_optimizers(self):

        model = self.model
        if self.hparams.optimizer == "sgd":
            optimizer = SGD(model.parameters(), lr=self.hparams.learning_rate)
        else:
            optimizer = Adam(model.parameters(), lr=self.hparams.learning_rate)

        self.opt = optimizer
        return [optimizer]

    def batch2input(self, batch):
        return {"input_ids": batch[0], "labels": batch[1], "lengths": batch[2]}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--vocab_filename",
            default=None,
            type=str,
            required=True,
            help="Pretrained tokenizer name or path",
        )
        parser.add_argument(
            "--optimizer",
            default="adam",
            type=str,
            required=True,
            help="Whether to use SGD or not",
        )
        parser.add_argument(
            "--word_embedding_size",
            default=300,
            type=int,
            help="Pretrained tokenizer name or path",
        )
        parser.add_argument("--hidden_layers", default=3,
                            type=int, help="choose number of hidden layer in NN")
        parser.add_argument("--drop_out", default=.0, type=float,
                            help="percentage of dropout between hidden layer during training")
        parser.add_argument("--use_glove", action="store_true",
                            help="Whether to run predictions on the test set.")

        return parser


class DAN(torch.nn.Module):
    """
    BagOfWords classification model
    """

    def __init__(self, vocab, vocab_size, word_embedding_size, hidden_layers=3, drop_out=.0, use_glove=None):
        """
        @param vocab_size: size of the vocabulary.
        """
        super(DAN, self).__init__()
        self.vocab = vocab
        self.embeds = torch.nn.Embedding(vocab_size, word_embedding_size)
        self.hidden = [torch.nn.Linear(
            word_embedding_size, 300), torch.nn.Tanh()]
        for i in range(hidden_layers - 1):
            self.hidden.append(torch.nn.Dropout(drop_out))
            self.hidden.append(torch.nn.Linear(300, 300))
            self.hidden.append(torch.nn.Tanh())
        self.hidden = torch.nn.Sequential(*self.hidden)
        self.fc = torch.nn.Linear(300, 1)

    def forward(self, input_ids, labels, lengths):
        """
        @return loss: loss should be a scalar averaged accross batches
        @return predicted_labels : model predictions. Should be either 0 or 1 based on a threshold (usually 0.5).

        @param data: matrix of size (batch_size, feature_length). Each row in data represents a sequence of token ids coming from tokenzied input text and vocabulary. 
        @param label: matrix of size (batch_size,). Ground truth labels.
        @param lengths: matrix of size (batch_size, 1). Token length of input text. Help you to compute average word embedding
        """

        out = self.embeds(input_ids).sum(dim=1, keepdim=False)
        out = torch.mul(out, 1. / lengths)
        out = self.hidden(out)
        pred_prob = torch.sigmoid(self.fc(out))
        LF = torch.nn.BCELoss()
        predicted_labels = (pred_prob > 0.5).float()
        loss = LF(pred_prob.view(-1), labels.view(-1))

        return loss, predicted_labels
