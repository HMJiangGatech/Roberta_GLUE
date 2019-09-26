# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)

from fairseq.models.roberta import (
    RobertaEncoder,
    RobertaClassificationHead,
)

from ..roberta_modules import (
    TransformerSentenceEncoderLayer_v2,
)

from fairseq.modules.transformer_sentence_encoder import init_bert_params

@register_model('roberta_v2')
class RobertaModel_v2(FairseqLanguageModel):

    @classmethod
    def hub_models(cls):
        return {
            'roberta.base': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz',
            'roberta.large': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz',
            'roberta.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz',
            'roberta.large.wsc': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz',
        }

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()
        self.roberta_decoder = RobertaDecoder(args)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--decoder-attn-stable-init-ratio', type=float, 
                            help='the ratio of attn heads to copy from previous layer of attn heads.')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        roberta_v2_base_archiecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def load_state_dict(self, state_dict, strict=False):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.
        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        return super().load_state_dict(state_dict, False)

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, classification_head_name=None, **kwargs):
        if classification_head_name is not None:
            features_only = True

        # encoder
        x, extra = self.decoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        # decoder
        x = self.roberta_decoder(x)

        # classification heads
        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)

        return x, extra

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                print(
                    'WARNING: re-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

        self.classification_heads[name] = RobertaClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    @property
    def supported_targets(self):
        return {'self'}

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='gpt2', **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return RobertaHubInterface(x['args'], x['task'], x['models'][0])

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    print('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v

class RobertaDecoder(nn.Module):
    """
    Implementation of Roberta v2 decoder.
    """

    def __init__(self, args, add_bias_kv = False, add_zero_attn = False, export = False):

        super().__init__()

        self.layers = nn.ModuleList(
            
            [
                TransformerSentenceEncoderLayer_v2(
                    embedding_dim=args.decoder_embed_dim,
                    ffn_embedding_dim=args.decoder_ffn_embed_dim,
                    num_attention_heads=args.decoder_attention_heads,
                    dropout=args.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    stable_init_ratio=args.decoder_attn_stable_init_ratio,
                    export=export,
                )

                for _ in range(args.decoder_layers)
            ]
        )


    def forward(self, x: torch.Tensor, last_state_only: bool = True) -> torch.Tensor:
        """
        Args: x (LongTensor): input size of `(batch_size, src_len, embed_dim)`
        """

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        attn = None
        for layer in self.layers:
            x, attn = layer(x = x, prev_weight = attn)
            if not last_state_only:
                inner_states.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if last_state_only:
            inner_states = [x]


        return inner_states[-1]


@register_model_architecture('roberta_v2', 'roberta_v2')
def roberta_v2_base_archiecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)

    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)

    args.decoder_attn_stable_init_ratio = getattr(args, 'decoder_attn_stable_init_ratio', 0.0)

@register_model_architecture('roberta_v2', 'roberta_v2_base_1')
def roberta_v2_base_1(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    roberta_v2_base_archiecture(args)

@register_model_architecture('roberta_v2', 'roberta_v2_base_2')
def roberta_v2_base_2(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    roberta_v2_base_archiecture(args)

@register_model_architecture('roberta_v2', 'roberta_v2_base_4')
def roberta_v2_base_4(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 4)
    roberta_v2_base_archiecture(args)

@register_model_architecture('roberta_v2', 'roberta_v2_base_6')
def roberta_v2_base_6(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    roberta_v2_base_archiecture(args)

@register_model_architecture('roberta_v2', 'roberta_v2_base_8')
def roberta_v2_base_8(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 8)
    roberta_v2_base_archiecture(args)

@register_model_architecture('roberta_v2', 'roberta_v2_base_10')
def roberta_v2_base_10(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 10)
    roberta_v2_base_archiecture(args)

@register_model_architecture('roberta_v2', 'roberta_v2_base_12')
def roberta_v2_base_12(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    roberta_v2_base_archiecture(args)

@register_model_architecture('roberta_v2', 'roberta_v2_large')
def roberta_v2_large(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    roberta_v2_base_archiecture(args)