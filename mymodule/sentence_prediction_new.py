# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np

from fairseq.data import (
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    StripTokenDataset,
    TruncateDataset,
)

from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask

from .utils import *

import copy

import torch
import torch.nn.functional as F


@register_task('sentence_prediction_new')
class SentencePredictionNEWTask(SentencePredictionTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')
        parser.add_argument('--regression-target', action='store_true', default=False)
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--truncate-sequence', action='store_true', default=False,
                            help='Truncate sequence to max_sequence_length')

        ## historical model
        parser.add_argument('--hist_model',
            default=True,
            type=boolean_string,
            help="False (default), use historical model.")
        parser.add_argument('--hist_lambda',
            default=1.0,
            type=float,
            help="1.0 (default), trade off parameters")


    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args, data_dictionary, label_dictionary)
        self.hist_model = None
        self.global_trainstep = 0

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, 'Must set --num-classes'

        args.tokens_per_sample = args.max_positions

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, 'input0', 'dict.txt'),
            source=True,
        )
        print('| [input] dictionary: {} types'.format(len(data_dict)))

        label_dict = None
        if not args.regression_target:
            # load label dictionary
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, 'label', 'dict.txt'),
                source=False,
            )
            print('| [label] dictionary: {} types'.format(len(label_dict)))
        else:
            label_dict = data_dict
        return cls(args, data_dict, label_dict)

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample, save_all=True)
        return loss, sample_size, logging_output

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        args = self.args

        if ignore_grad:
            loss, sample_size, logging_output = criterion(model, sample)
            loss *= 0
            optimizer.backward(loss)
            return loss, sample_size, logging_output

        if self.global_trainstep == 0:
            if self.args.hist_model:
                self.hist_model = copy.deepcopy(model)
                self.hist_model.train()

        self.global_trainstep = self.global_trainstep+1
        hist_model = self.hist_model


        loss, sample_size, logging_output = criterion(model, sample)

        # historical model
        if self.args.hist_model:
            loss_hist, _, _ = criterion(model, sample, save_all=False, hist_model=hist_model)
            loss = loss+loss_hist*self.args.hist_lambda

        optimizer.backward(loss)
        return loss, sample_size, logging_output

def get_logit(model, sample, padding_mask):
    features, extra = model(**sample['net_input'], features_only=True, return_all_hiddens=True)
    features = features.transpose(0, 1)
    logits = model.classification_heads['sentence_classification_head'](
        features,
        padding_mask=padding_mask,
    )
    return logits, extra

def embed_forward(model, embed, padding_mask):
    encoder = model.decoder # RobertaEncoder
    encoder = encoder.sentence_encoder # TransformerSentenceEncoder

    if not padding_mask.any():
        _padding_mask = None
    else:
        _padding_mask=padding_mask
    x=embed
    for layer in encoder.layers:
        x, _ = layer(x, self_attn_padding_mask=_padding_mask)

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)
    logits = model.classification_heads['sentence_classification_head'](
        x,
        padding_mask=padding_mask,
    )

    return logits
