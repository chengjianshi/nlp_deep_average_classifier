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
import json, shutil

logger = logging.getLogger(__name__)

class BaseModel(pl.LightningModule):
    def __init__(
        self,
        **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        logger.info("Initilazing BaseModel")
        super().__init__()
        self.save_hyperparameters() #save hyperparameters to checkpoint
        self.step_count = 0
        self.output_dir = Path(self.hparams.output_dir)
        self.hparams.vocab = json.load(open(Path(self.hparams.vocab_filename)))
        self.hparams.vocab_size = len(self.hparams.vocab)
        self.model = self._load_model()

        self.accuracy = Accuracy()

    def _load_model(self):
        raise NotImplementedError

    def forward(self, **inputs):
        return self.model(**inputs)

    def batch2input(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        input = self.batch2input(batch)
        labels = input['labels']
        loss, pred_labels = self(**input)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.accuracy(pred_labels.view(-1), labels.view(-1)), prog_bar=True)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input = self.batch2input(batch)
        labels = input['labels']
        loss, pred_labels = self(**input)

        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(pred_labels.view(-1), labels.view(-1)))

    def test_step(self, batch, batch_nb):
        input = self.batch2input(batch)
        labels = input['labels']
        loss, pred_labels = self(**input)
        self.log('test_loss', loss)
        self.log('test_acc', self.accuracy(pred_labels.view(-1), labels.view(-1)))

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        # optimizer = SGD(model.parameters(), lr=self.hparams.learning_rate)
        optimizer = Adam(model.parameters(), lr=self.hparams.learning_rate)

        self.opt = optimizer
        return [optimizer]

    def train_dataloader(self):
        return self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    @staticmethod
    def add_generic_args(parser, root_dir) -> None:
        parser.add_argument(
            "--max_epochs",
            default=10,
            type=int,
            help="The number of epochs to train your model.",
        )
        ############################################################
        ## WARNING: set --gpus 0 if you do not have access to GPUS #
        ############################################################
        parser.add_argument(
            "--gpus",
            default=1,
            type=int,
            help="The number of GPUs allocated for this, it is by default 0 meaning none",
        )
        parser.add_argument(
            "--output_dir",
            default=None,
            type=str,
            required=True,
            help="The output directory where the model predictions and checkpoints will be written.",
        )
        parser.add_argument("--do_train", action="store_true", default=True, help="Whether to run training.")
        parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        parser.add_argument(
            "--data_dir",
            default="./",
            type=str,
            help="The input data dir. Should contain the training files.",
        )
        parser.add_argument("--learning_rate", default=1e-2, type=float, help="The initial learning rate for training.")
        parser.add_argument("--num_workers", default=16, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
    
def generic_train(
    model: BaseModel,
    args: argparse.Namespace,
    early_stopping_callback=False,
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs
):
    
    # init model
    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)
    log_dir = Path(os.path.join(model.hparams.output_dir, 'logs'))
    log_dir.mkdir(exist_ok=True)

    # Tensorboard logger
    pl_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir,
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
        default_hp_metric=True
    )

    # add custom checkpoints
    ckpt_path = os.path.join(
        args.output_dir, pl_logger.version, "checkpoints",
    )
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path, filename="{epoch}-{val_acc:.2f}", monitor="val_acc", mode="max", save_top_k=1, verbose=True
        )

    train_params = {}

    train_params["max_epochs"] = args.max_epochs

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        callbacks= extra_callbacks,
        logger=pl_logger,
        checkpoint_callback=checkpoint_callback,
        **train_params,
    )

    if args.do_train:
        trainer.fit(model)
        # track model performance under differnt hparams settings in "Hparams" of TensorBoard
        pl_logger.log_hyperparams(params=model.hparams, metrics={'hp_metric': checkpoint_callback.best_model_score.item()})
        pl_logger.save()
        
        # save best model to `best_model.ckpt`
        target_path = os.path.join(ckpt_path, 'best_model.ckpt')
        logger.info(f"Copy best model from {checkpoint_callback.best_model_path} to {target_path}.")
        shutil.copy(checkpoint_callback.best_model_path, target_path)

    
    # Optionally, predict on test set and write to output_dir
    if args.do_predict:
        best_model_path = os.path.join(ckpt_path, "best_model.ckpt")
        model = model.load_from_checkpoint(best_model_path)
        return trainer.test(model)
    
    return trainer

