#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>, Melika Behjati <melika.behjati@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# Transformer model with NVIB in self attention

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nvib_sa_transformer_encoder import (
    NVIBTransformerEncoder,
    NVIBTransformerEncoderLayer,
)
from models.seq2seq_lightning import Seq2SeqLightning
from models.transformer import *
from models.transformer_encoder import (
    CustomTransformerEncoder,
    CustomTransformerEncoderLayer,
)

# Note:
# B: Batch size
# Ns: Source length
# Nt: Target length
# Nl: Latent length (typically = Ns)
# E: Embedding dimension
# H: Model dimension
# V: Vocab dimension


class NVIBSaTransformer(Transformer):
    """
    A vanilla Transformer Encoder-Decoder in Pytorch

    Data format:
    SRC: ... [EOS]
    TGT: ... [EOS]
    Encoder_input(SRC): ... [EOS]
    Decoder_input(TGT): [SOS] ...

    For an autoencoder x -> x (SRC = TGT)
        The loss function requires SRC and logits.
    For different models x -> y (Eg: translation SRC != TGT)
        The loss function requires TGT and logits.

    If we keep this format the attention masks for padding are identical for autoencoder's encoder + decoder .
    """

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)

        self.d_model = kwargs["d_model"]
        self.nhead = kwargs["nhead"]
        self.dim_feedforward = kwargs["dim_feedforward"]
        self.dropout = kwargs["dropout"]
        self.num_encoder_layers = kwargs["num_encoder_layers"]
        self.num_nvib_encoder_layers = kwargs["num_nvib_encoder_layers"]
        self.num_decoder_layers = kwargs["num_decoder_layers"]
        self.kappa = kwargs["kappa"]
        self.delta = kwargs["delta"]

        # Transformer encoder layer
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="relu",
        )

        # NVIB Transformer encoder layer
        nvib_transformer_encoder_layer = NVIBTransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="relu",
            kappa=self.kappa,
            delta=self.delta,
            norm_first=True,
        )
        encoder_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.encoder = CustomTransformerEncoder(
            encoder_layer, self.num_encoder_layers, encoder_norm
        )
        self.nvib_transformer_encoder = NVIBTransformerEncoder(
            nvib_transformer_encoder_layer, self.num_nvib_encoder_layers, encoder_norm
        )

        # Transformer decoder
        decoder_layer = CustomTransformerDecoderLayer(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            self.dropout,
        )
        decoder_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.decoder = CustomTransformerDecoder(
            decoder_layer, self.num_decoder_layers, decoder_norm
        )

        self.pad_token_id = tokenizer.pad_token_id
        self.decoder_start_token_id = tokenizer.cls_token_id
        self.args = kwargs
        self.embedding = nn.Embedding(tokenizer.vocab_size, self.d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.output_proj = nn.Linear(self.d_model, tokenizer.vocab_size)
        self.drop = nn.Dropout(self.dropout)

    def encode(self, src, src_key_padding_mask):
        """
        Encode the input ids to embeddings and then pass to the transformer encoder
        :param src: source token ids [Ns, B]
        :param src_key_padding_mask: Trues where to mask [B,Ns]
        :return: memory: [Ns,B,H]
        """
        assert not torch.isnan(src).any(), (
            f"NaN values detected in src at {self.__class__.__name__}. "
            f"NaN indices: {torch.nonzero(torch.isnan(src), as_tuple=True)}. "
            f"Shape: {src.shape}, Max value: {torch.max(src)}"
            f"Src: {src}"
        )
        assert not torch.isnan(self.embedding.weight).any(), (
            f"NaN values detected in embedding weights at {self.__class__.__name__}. "
            f"NaN indices: {torch.nonzero(torch.isnan(self.embedding.weight), as_tuple=True)}. "
            f"Shape: {self.embedding.weight.shape}, Max value: {torch.max(self.embedding.weight)}"
            f"Embedding weights: {self.embedding.weight}"
        )
        temp = self.embedding(src)
        assert not torch.isnan(temp).any(), (
            f"NaN values detected in temp at {self.__class__.__name__}. "
            f"NaN indices: {torch.nonzero(torch.isnan(temp), as_tuple=True)}. "
            f"Shape: {temp.shape}, Max value: {torch.max(temp)}"
            f"Temp: {temp}"
        )
        embedded = self.drop(temp)  # [Ns,B,H]
        assert not torch.isnan(embedded).any(), (
            f"NaN values detected in embedded at {self.__class__.__name__}. "
            f"NaN indices: {torch.nonzero(torch.isnan(embedded), as_tuple=True)}. "
            f"Shape: {embedded.shape}, Max value: {torch.max(embedded)}"
            f"Embedded: {embedded}"
        )
            
        src = self.positional_encoding(embedded)  # [Ns,B,H]
        
        # Transformer encoder
        memory1, attention1 = self.encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )  # [Ns,B,H]
        # print("Number of layers in memory1:", len(memory1))
        # for i, layer_output in enumerate(memory1):
        #     print(f"Layer {i} output shape:", layer_output.shape)
        # NVIB Transformer encoder
        # take memory[-1] since memory1 store all output of each layer
        memory2, attention2, klg, kld, latent_dict = self.nvib_transformer_encoder(
            memory1[-1], src_key_padding_mask=src_key_padding_mask
        )  # [Ns,B,H]
        # Concatenate the attention lists
        attention = attention1 + attention2
        return memory2, attention, klg, kld, latent_dict, memory1

    def decode(
        self, tgt, z, memory_key_padding_mask, tgt_key_padding_mask, *args, **kwargs
    ):
        """

        :param tgt: target token ids [Nt,B]
        :param z: output from the latent layer [Nl,B,H]
        :param memory_key_padding_mask: mask for latent layer [B, Nl] (typically Ns = Nl)
        :param tgt_key_padding_mask: target mask [B,Nt]
        :param args:
        :param kwargs:
        :return: logits over the vocabulary [Nt,B,V]
        """

        # Add position encodings + Embeddings
        tgt = self.positional_encoding(self.drop(self.embedding(tgt)))  # [Nt,B,H]
        
        # Normalize and use tanh for smooth, differentiable value bounding
        tgt = F.layer_norm(tgt, [tgt.size(-1)])
        tgt = 100 * torch.tanh(tgt / 100)  # Smooth bound to [-100, 100]
        
        # Generate target teacher forcing mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(
            tgt.device
        )  # [Nt, Nt]
        
        # Normalize and bound memory input
        z = F.layer_norm(z, [z.size(-1)])
        z = 100 * torch.tanh(z / 100)  # Smooth bound to [-100, 100]

        assert not torch.isnan(z).any(), (
            f"NaN values detected in z before decoder layer at {self.__class__.__name__}. "
            f"NaN indices: {torch.nonzero(torch.isnan(z), as_tuple=True)}. "
            f"Shape: {z.shape}, Max value: {torch.max(z)}"
            f"Z: {z}"
        )
        
        output, attention = self.decoder(
            tgt=tgt,  # [Nt,B,H]
            memory=z,  # [Nt,B,H]
            tgt_mask=tgt_mask,  # [Nt,Nt]
            tgt_key_padding_mask=tgt_key_padding_mask,  # [B,Nt]
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B,Nl]
        assert not torch.isnan(output).any(), (
            f"NaN values detected in output after decoder layer at {self.__class__.__name__}. "
            f"NaN indices: {torch.nonzero(torch.isnan(output), as_tuple=True)}. "
            f"Shape: {output.shape}, Max value: {torch.max(output)}"
            f"Output: {output}"
        )
        # Normalize decoder output
        output = F.layer_norm(output, [output.size(-1)])
        
        # Differentiable gradient norm scaling
        norm = torch.norm(output, p=2, dim=-1, keepdim=True)
        scale = torch.min(
            torch.ones_like(norm),
            5 / (norm + 1e-6)  # Target norm of 5
        )
        output = output * scale
        
        # Smooth value bounding
        output = 100 * torch.tanh(output / 100)
        
        # Apply output projection
        logits = self.output_proj(output)  # [Nt,B,V]
        
        # Final smooth bounding on logits
        logits = 100 * torch.tanh(logits / 100)
        
        return logits, attention

    def generate(self, input_ids, max_new_tokens, *args, **kwargs):
        """
        Generate autoregressively without teacher forcing
        :param z: output from the latent layer [Nl,B,H]
        :param memory_key_padding_mask: mask from the latent layer [B,Nl]
        :param max_len: maximum generation length
        :param tokenizer: tokenizer
        :param args:
        :param kwargs:
        :return: logits [Nt,B,V] and list of predictions
        """
        # Encode
        src_key_padding_mask = ~(input_ids.bool()).transpose(0, 1)  # [B,Ns]
        memory, _, _, _, self_attention_latent, _ = self.encode(
            input_ids, src_key_padding_mask=src_key_padding_mask
        )  # [Ns,B,H]

        # Mask the src_key_padding_mask with the final latent layer's pi for cross attention
        src_key_padding_mask = src_key_padding_mask + self_attention_latent[-1][
            "alpha"
        ].squeeze(-1).transpose(0, 1)[:, 1:].le(0.1)

        # Soft weighting of vectors
        # memory = memory * self_attention_latent[-1]["pi"][1:, :, :]

        # latent layer
        latent_output_dict = self.latent_layer(memory, src_key_padding_mask)
        memory_key_padding_mask = latent_output_dict["memory_key_padding_mask"]
        z = latent_output_dict["z"]

        # Initialise target ids with BOS token
        target_ids = (
            torch.tensor([[self.decoder_start_token_id]])
            .expand(memory_key_padding_mask.shape[0], -1)
            .T.to(memory_key_padding_mask.device)
        )  # [1, B]
        # For each token in length
        for token_idx in range(max_new_tokens):
            # Decode the target ids regressively
            logits, _ = self.decode(
                target_ids,
                z,
                memory_key_padding_mask,
                None,
                # latent_dict=self_attention_latent[-1]
            )  # [token_idx, B, V]
            # Select only the final set of logits
            prediction = logits[-1, :, :].unsqueeze(0)  # [target_ids1,B,V]
            # Get prediction over vocabulary and return index
            prediction = prediction.argmax(-1)  # [1,B]
            # Concatenate the predictions to form next token_ids
            target_ids = torch.cat((target_ids, prediction), dim=0)  # [token_index, B]

        # Decode into a sentence
        # predictions = [tokenizer.decode(encoded) for encoded in target_ids[1:, :].T]  # list [B]
        return target_ids[1:, :]

    def forward(
        self,
        input_ids,
        decoder_input_ids,
        labels,
        attention_mask,
        **kwargs,
    ):
        """
        Forward pass for all transformer models

        :param src: the sequence to the encoder (required). [Ns,B]
        :param tgt: the sequence  nce to the decoder (required). [Nt,B]
        :param src_mask: the additive mask for the src sequence (optional). [Ns, Ns]
        :param tgt_mask: the additive mask for the tgt sequence (optional). [Nt, Nt]
        :param memory_mask: the additive mask for the encoder output (optional). [Nt,Ns]
        :param src_key_padding_mask: the ByteTensor mask for src keys per batch (optional). [B,Ns]
        :param tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional). [B,Nt]
        :param memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).[B,Nl]
        :return: logits and latent dimension dictionary

        Check out here for more info masks on https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask
        The memory ones are interesting. I use memory_key_padding_mask to mask the tokens in the latent space.

        """
        # Reformat the attention mask
        src_key_padding_mask = ~(attention_mask.bool())
        tgt_key_padding_mask = decoder_input_ids.transpose(0, 1) == self.pad_token_id
        assert not torch.isnan(input_ids).any(), (
            f"NaN values detected in input_ids at {self.__class__.__name__}. "
            f"NaN indices: {torch.nonzero(torch.isnan(input_ids), as_tuple=True)}. "
            f"Shape: {input_ids.shape}, Max value: {torch.max(input_ids)}"
            f"Input ids: {input_ids}"
        )
        # Encode
        (
            memory,
            encoder_attention,
            klg,
            kld,
            self_attention_latent,
            old_memory,
        ) = self.encode(
            input_ids, src_key_padding_mask=src_key_padding_mask
        )  # [Ns,B,H]
        assert not torch.isnan(memory).any(), (
            f"NaN values detected in memory after encode at {self.__class__.__name__}. "
            f"Shape: {memory.shape}, Max value: {torch.max(memory)}"
        )
        # Mask the src_key_padding_mask with the final latent layer's pi for cross attention
        src_key_padding_mask = src_key_padding_mask + self_attention_latent[-1][
            "alpha"
        ].squeeze(-1).transpose(0, 1)[:, 1:].le(0.1)

        # Soft weighting of vectors
        # memory = memory * self_attention_latent[-1]["pi"][1:, :, :]

        # latent layer
        latent_output_dict = self.latent_layer(memory, src_key_padding_mask)
        # Decode
        assert not torch.isnan(latent_output_dict["z"]).any(), (
            f"NaN values detected in latent_output_dict['z'] at {self.__class__.__name__}. "
            f"Shape: {latent_output_dict['z'].shape}, Max value: {torch.max(latent_output_dict['z'])}"
        )
        output, decoder_attention = self.decode(
            tgt=decoder_input_ids,  # [Nt,B,H]
            z=latent_output_dict["z"],  # [Nl,B,H]
            tgt_key_padding_mask=tgt_key_padding_mask,  # [B,Nt]
            memory_key_padding_mask=latent_output_dict["memory_key_padding_mask"],
            # latent_dict=self_attention_latent[-1],
        )  # [B,Nl]
        assert not torch.isnan(output).any(), (
            f"NaN values detected in output decode at {self.__class__.__name__}. "
            f"Shape: {output.shape}, Max value: {torch.max(output)}"
        )

        return {
            "logits": output,  # [Nt, B, V]
            "encoder_attentions": encoder_attention,  # Self attention
            "cross_attentions": decoder_attention,  # Cross attention
            "kl_gaussian": klg,
            "kl_dirichlet": kld,
            "latent_dict_list": self_attention_latent,
            "old_memory": old_memory,
            "old_memory_mask": ~(attention_mask.bool()),
        }

class NVIBSaTransformerLightning(Seq2SeqLightning):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        # Tokenizer
        self.tokenizer = CharizardTokenizer(model_max_length=args.max_length)
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Model
        self.model = NVIBSaTransformer(tokenizer=self.tokenizer, **vars(args))

        # Nvib
        self.lambda_klg = args.klg_lambda
        self.lambda_kld = args.kld_lambda

        # Logging metrics
        self.log_bleu = True
        self.log_chrf = True
        self.plot_encoder_attention = True
        self.plot_cross_attention = True
        self.model_type = "NVIBSaTransformer"
        self.is_nvib = True
        self.weighted_kl = args.weighted_kl

        # Initialization
        init_weights(self.model)

        self.save_hyperparameters()
