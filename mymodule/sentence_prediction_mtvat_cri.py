# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

import numpy as np

from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('sentence_prediction_mtvat')
class SentencePredictionMTVATCriterion(FairseqCriterion):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        # fmt: on

    def forward(self, model, sample, reduce=True, returnfull = False, save_all = True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        assert hasattr(model, 'classification_heads') and \
            'sentence_classification_head' in model.classification_heads, \
            "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, extra = model(**sample['net_input'], features_only=True,
                classification_head_name='sentence_classification_head', return_all_hiddens=returnfull)
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
            if save_all:
                logging_output.update(
                    preds_ids=np.copy(sample['id'].cpu().squeeze().numpy()).astype(int),
                    preds=np.copy(preds.detach().cpu().squeeze().numpy()).astype(int),
                    targets=np.copy(targets.detach().cpu().squeeze().numpy()).astype(int)
                )
        else:
            if save_all:
                logging_output.update(
                    preds_ids=np.copy(sample['id'].cpu().squeeze().numpy()),
                    preds=np.copy(logits.detach().cpu().squeeze().numpy()),
                    targets=np.copy(targets.detach().cpu().squeeze().numpy())
                )
                if sample_size==1:
                    import pdb;pdb.set_trace()
        if returnfull:
            return loss, sample_size, logging_output, logits, extra['inner_states'][0]
        else:
            return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        if len(logging_outputs) > 0:
            if 'ncorrect' in logging_outputs[0]:
                ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
                agg_output.update(accuracy=ncorrect/nsentences)
            if 'preds' in logging_outputs[0]:
                agg_output.update(preds=np.concatenate([log.get('preds', np.empty(0)).reshape(-1) for log in logging_outputs]))
                agg_output.update(targets=np.concatenate([log.get('targets', np.empty(0)).reshape(-1) for log in logging_outputs]))
                agg_output.update(pred_ids=np.concatenate([log.get('pred_ids', np.empty(0)).reshape(-1) for log in logging_outputs]))

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
