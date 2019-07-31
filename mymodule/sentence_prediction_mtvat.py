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


@register_task('sentence_prediction_mtvat')
class SentencePredictionMTVATTask(SentencePredictionTask):
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

        ## Teacher
        parser.add_argument('--teacher_class',
            default="smart",
            type=none_string,
            help="smart (default) | logit, maxmargin, teach what.")

        ## Mean Teacher
        parser.add_argument('--mean_teacher',
            default=True,
            type=boolean_string,
            help="False (default), use mean teacher.")
        parser.add_argument('--mean_teacher_avg',
            default="exponential",
            type=none_string,
            help="exponential (default) | simple | double_ema, moving average method.")
        parser.add_argument('--mean_teacher_lambda',
            default=1,
            type=float,
            help="False (default), trade off parameter from the consistent loss using mean teacher.")
        parser.add_argument('--mean_teacher_rampup',
            default=4000,
            type=int,
            help="4000 (default), rampup iteration.")
        parser.add_argument('--mean_teacher_alpha1',
            default=0.99,
            type=float,
            help="False (default), moving average parameter of mean teacher  (for the exponential moving average).")
        parser.add_argument('--mean_teacher_alpha2',
            default=0.999,
            type=float,
            help="False (default), moving average parameter of mean teacher  (for the exponential moving average).")

        ## Virtual Adversarial Training
        parser.add_argument('--use_vat',
            default=True,
            type=boolean_string,
            help="False (default), use virtual adversarial training.")
        parser.add_argument('--vat_eps',
            default=1e-3,
            type=float,
            help="1 (default), perturbation size for virtual adversarial training.")
        parser.add_argument('--vat_lambda',
            default=1,
            type=float,
            help="1 (default), trade off parameter for virtual adversarial training.")

        ## Use Noisy Copy
        parser.add_argument('--use_noisycopy',
            default=False,
            type=boolean_string,
            help="False (default), use noisy copy training.")
        parser.add_argument('--noisycopy_eps',
            default=1e-2,
            type=float,
            help="1e-4 (default), eps for noisy copy training.")

        ## Use Adversarial Copy
        parser.add_argument('--use_advcopy',
            default=True,
            type=boolean_string,
            help="False (default), use adversarial copy training.")
        parser.add_argument('--advcopy_eps',
            default=1e-2,
            type=float,
            help="1e-3 (default), eps for adversarial copy training.")


    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args, data_dictionary, label_dictionary)
        self._meanteacher_model = None
        self.noisycopy_model  = None
        self.global_trainstep = 0
        if args.regression_target:
            self.args.teacher_class = 'logit'

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
            if self.args.mean_teacher:
                self.teacher_model = copy.deepcopy(model)
                self.teacher_model.train()
            if self.args.use_noisycopy or self.args.use_advcopy:
                self.noisycopy_model = create_noisycopy_model(model,args)
                self.noisycopy_model.train()

        self.global_trainstep = self.global_trainstep+1
        teacher_model = self.teacher_model
        noisycopy_model = self.noisycopy_model

        if args.mean_teacher:
            if self.global_trainstep < args.mean_teacher_rampup:
                _alpha = args.mean_teacher_alpha1
            else:
                _alpha = args.mean_teacher_alpha2
            update_meanteacher(teacher_model.named_parameters(), model.named_parameters(), average=args.mean_teacher_avg, alpha=_alpha, step=self.global_trainstep)

        padding_mask = sample['net_input']['src_tokens'].eq(criterion.padding_idx)
        if self.args.mean_teacher:
            with torch.no_grad():
                teacher_logits,_ = get_logit(teacher_model, sample, padding_mask)
            teacher_logits = teacher_logits.detach()

        if args.use_vat:
            loss, sample_size, logging_output, logits, embed = criterion_forward(criterion, model, sample)
            prob_orig = F.log_softmax(logits.view(-1, logits.size(-1)).float(), 1).exp()
            newembed = (embed.data.detach()+embed.data.new(embed.size()).normal_(0, 1)*1e-5).detach()
            newembed.requires_grad_()
            logits_vat = embed_forward(model, newembed, padding_mask)
            vat_loss = teach_class(logits_vat,logits.detach(),args.teacher_class,1)
            optimizer.backward(vat_loss)
            norm = newembed.grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                print("Hit nan gradient in embed vat")
                model.zero_grad()
                optimizer.backward(loss)
                return loss, sample_size, logging_output
            adv_direct = newembed.grad/(  newembed.grad.abs().max(-1,keepdim=True)[0]  +1e-4)
            newembed = newembed+adv_direct*self.args.vat_eps
            newembed = newembed.detach()
            if self.args.use_advcopy:
                _model,is_nan = update_advcopy_model(copy.deepcopy(self.noisycopy_model),model)
                if is_nan:
                    print("Hit nan gradient in advcopy")
                    model.zero_grad()
                    optimizer.backward(loss)
                    return loss, sample_size, logging_output
                logits_vat = embed_forward(_model, newembed, padding_mask)
            elif self.args.use_noisycopy:
                _model = update_noisycopy_model(copy.deepcopy(self.noisycopy_model),model)
                logits_vat = embed_forward(_model, newembed, padding_mask)
            else:
                logits_vat = embed_forward(model, newembed, padding_mask)
            model.zero_grad()
            # cross_entropy | double way
            vat_loss = teach_class(logits_vat,logits.detach(),self.args.teacher_class,args.vat_lambda)
            vat_loss += teach_class(logits,logits_vat.detach(),self.args.teacher_class,args.vat_lambda)

            logging_output.update(
                vat_loss=vat_loss.item()
            )
            loss = loss+vat_loss
        else:
            loss, sample_size, logging_output = criterion(model, sample)

        if self.args.mean_teacher and self.args.teacher_class is not None:
            if self.args.teacher_class == 'smart':
                _lambda = self.args.mean_teacher_lambda # 1, important
            else:
                _lambda = self.args.mean_teacher_lambda * min(1,math.exp(-5*(1-acc_steps/self.args.mean_teacher_rampup)**2))
            mt_loss = teach_class(logits,teacher_logits,self.args.teacher_class,_lambda)
            logging_output.update(
                mt_loss=mt_loss.item()
            )
            loss += mt_loss

        optimizer.backward(loss)
        return loss, sample_size, logging_output

def get_logit(model, sample, padding_mask):
    features, extra = model(**sample['net_input'], features_only=True, return_all_hiddens=True)
    logits = model.classification_heads['sentence_classification_head'](
        features,
        padding_mask=padding_mask,
    )
    return logits, extra

def criterion_forward(self, model, sample, reduce=True):
    """Compute the loss for the given sample.

    Returns a tuple with three elements:
    1) the loss
    2) the sample size, which is used as the denominator for the gradient
    3) logging outputs to display while training
    """
    features, extra = model(**sample['net_input'], features_only=True, return_all_hiddens=True)
    padding_mask = sample['net_input']['src_tokens'].eq(self.padding_idx)

    assert hasattr(model, 'classification_heads') and \
        'sentence_classification_head' in model.classification_heads, \
        "model must provide sentence classification head for --criterion=sentence_prediction"

    logits = model.classification_heads['sentence_classification_head'](
        features,
        padding_mask=padding_mask,
    )

    targets = model.get_targets(sample, [logits]).view(-1)
    sample_size = targets.numel()

    if not self.args.regression_target:
        loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            targets,
            reduction='sum',
        )
    else:
        logits = logits.squeeze().float()
        targets = targets.float()
        loss = F.mse_loss(
            logits,
            targets,
            reduction='sum',
        )

    logging_output = {
        'loss': utils.item(loss.data) if reduce else loss.data,
        'ntokens': sample['ntokens'],
        'nsentences': sample_size,
        'sample_size': sample_size,
    }

    if not self.args.regression_target:
        preds = logits.max(dim=1)[1]
        logging_output.update(
            ncorrect=(preds == targets).sum().item()
        )
    return loss, sample_size, logging_output, logits, extra['inner_states'][0]

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
