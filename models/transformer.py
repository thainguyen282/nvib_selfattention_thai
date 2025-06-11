#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>, Melika Behjati <melika.behjati@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#


import math

import torch
import torch.nn as nn

from data_modules.CharizardTokenizer import CharizardTokenizer
from models.seq2seq_lightning import Seq2SeqLightning
from models.transformer_decoder import (
    CustomTransformerDecoder,
    CustomTransformerDecoderLayer,
)
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
# P: Compressed dimension


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def init_weights(model):
    for name, param in model.named_parameters():
        if "bias" in name:
            if "alpha_proj" in name:
                # Initialize alpha projection bias to positive values
                # This ensures log_alpha starts positive, so alpha = exp(log_alpha) > 1.0
                torch.nn.init.constant_(param, 1.0)  # exp(1.0) ≈ 2.7 >> 0.1
            else:
                torch.nn.init.zeros_(param)
        elif "weight" in name:
            if param.dim() > 1:
                if "embedding" in name:
                    # Use normal distribution for embeddings
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
                elif "nvib_layer" in name and "alpha_proj" in name:
                    # Initialize alpha projection weights to reasonable scale
                    # Use Xavier but with positive bias toward larger values
                    torch.nn.init.xavier_uniform_(param)
                    # Add small positive bias to weights to encourage larger alpha
                    with torch.no_grad():
                        param += 0.1  # Small positive bias
                elif "mu_proj" in name:
                    # Initialize mu projection with identity-like matrix  
                    torch.nn.init.eye_(param)
                elif "logvar_proj" in name:
                    # Initialize logvar projection with small values for stable variances
                    torch.nn.init.xavier_uniform_(param)
                    torch.nn.init.constant_(param, 0.0)  # Start with unit variance (exp(0) = 1)
                else:
                    # Xavier uniform for other weights
                    torch.nn.init.xavier_uniform_(param)
            else:
                # For 1D parameters (like LayerNorm), use normal distribution
                torch.nn.init.normal_(param, mean=0.0, std=0.02)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=1000, mul_by_sqrt=True):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.mul_by_sqrt = mul_by_sqrt

    def forward(self, x):
        x = x.permute(1, 0, 2)
        if self.mul_by_sqrt:
            x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = self.pe[:, 1 : seq_len + 1]
        pe = pe.expand_as(x)
        x = x + pe
        x = x.permute(1, 0, 2)
        return x


class Transformer(nn.Transformer):
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
        super().__init__(
            d_model=kwargs["d_model"],
            nhead=kwargs["nhead"],
            num_encoder_layers=kwargs["num_encoder_layers"],
            num_decoder_layers=kwargs["num_decoder_layers"],
            dim_feedforward=4 * kwargs["d_model"],
            dropout=kwargs["dropout"],
            batch_first=False,
            norm_first=False,
        )

        self.d_model = kwargs["d_model"]
        self.nhead = kwargs["nhead"]
        self.dim_feedforward = kwargs["dim_feedforward"]
        self.dropout = kwargs["dropout"]
        self.num_encoder_layers = kwargs["num_encoder_layers"]
        self.num_decoder_layers = kwargs["num_decoder_layers"]

        # Transformer encoder
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=kwargs["d_model"],
            nhead=kwargs["nhead"],
            dim_feedforward=kwargs["dim_feedforward"],
            dropout=kwargs["dropout"],
            activation="relu",
        )
        encoder_norm = nn.LayerNorm(kwargs["d_model"], eps=1e-5)
        self.encoder = CustomTransformerEncoder(
            encoder_layer, kwargs["num_encoder_layers"], encoder_norm
        )

        # Transformer decoder
        decoder_layer = CustomTransformerDecoderLayer(
            kwargs["d_model"],
            kwargs["nhead"],
            kwargs["dim_feedforward"],
            kwargs["dropout"],
        )
        decoder_norm = nn.LayerNorm(kwargs["d_model"], eps=1e-5)
        self.decoder = CustomTransformerDecoder(
            decoder_layer, kwargs["num_decoder_layers"], decoder_norm
        )

        self.pad_token_id = tokenizer.pad_token_id
        self.decoder_start_token_id = tokenizer.cls_token_id
        self.args = kwargs
        self.embedding = nn.Embedding(
            tokenizer.vocab_size, kwargs["d_model"], padding_idx=0
        )
        self.positional_encoding = PositionalEncoding(kwargs["d_model"])
        self.output_proj = nn.Linear(kwargs["d_model"], tokenizer.vocab_size)
        self.drop = nn.Dropout(kwargs["dropout"])

    def encode(self, src, src_key_padding_mask):
        """
        Encode the input ids to embeddings and then pass to the transformer encoder
        :param src: source token ids [Ns, B]
        :param src_key_padding_mask: Trues where to mask [B,Ns]
        :return: memory: [Ns,B,H]
        """
        # Add position encodings + Embeddings
        src = self.positional_encoding(self.drop(self.embedding(src)))  # [Ns,B,H]

        # Transformer encoder
        memory, attention = self.encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )  # [Ns,B,H]
        return memory, attention

    def latent_layer(self, encoder_output, src_key_padding_mask):
        """
        Latent layer for child classes like VAE

        :param encoder_output: encoder bov output [Ns,B,H]
        :param src_key_padding_mask: Trues where to mask [B,Nl] (typically encoder mask)
        :return: Z from the latent layer [Nl,B,H]
        """
        z = encoder_output  # [Ns,B,H]
        return {"z": z, "memory_key_padding_mask": src_key_padding_mask}  # [B,Nl]

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
        
        # Generate target teacher forcing mask and convert to float
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)  # [Nt, Nt]
        tgt_mask = tgt_mask.float()
        
        output, attention = self.decoder(
            tgt=tgt,  # [Nt,B,H]
            memory=z,  # [Nt,B,H]
            tgt_mask=tgt_mask,  # [Nt,Nt]
            tgt_key_padding_mask=tgt_key_padding_mask,  # [B,Nt]
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B,Nl]
        logits = self.output_proj(output)  # [Nt,B,V]
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
        outputs, _ = self.encode(
            input_ids, src_key_padding_mask=src_key_padding_mask
        )  # [Ns,B,H]
        memory = outputs[-1]
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
                target_ids, z, memory_key_padding_mask, None
            )  # [token_idx, B, V]
            # Select only the final set of logits
            prediction = logits[-1, :, :].unsqueeze(0)  # [target_ids1,B,V]
            # Get prediction over vocabulary and return index
            prediction = prediction.argmax(-1)  # [1,B]
            # Concatenate the predictions to form next token_ids
            target_ids = torch.cat((target_ids, prediction), dim=0).to(
                memory_key_padding_mask.device
            )  # [token_index, B]

        # Decode into a sentence
        # predictions = [tokenizer.decode(encoded) for encoded in target_ids[1:, :].T]  # list [B]
        return target_ids[1:, :]  # removes the BOS token

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.pad_token_id, self.decoder_start_token_id
        )

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
        tgt_key_padding_mask = (decoder_input_ids.transpose(0, 1) == self.pad_token_id)

        # Convert masks to float tensors to ensure consistent types
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.float()
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.float()

        # Encode
        outputs, encoder_attention = self.encode(
            input_ids, src_key_padding_mask=src_key_padding_mask
        )  # [Ns,B,H]
        memory = outputs[-1]
        # latent layer
        latent_output_dict = self.latent_layer(memory, src_key_padding_mask)
        # Decode
        output, decoder_attention = self.decode(
            tgt=decoder_input_ids,  # [Nt,B,H]
            z=latent_output_dict["z"],  # [Nl,B,H]
            tgt_key_padding_mask=tgt_key_padding_mask,  # [B,Nt]
            memory_key_padding_mask=latent_output_dict["memory_key_padding_mask"],
        )  # [B,Nl]

        return {
            "logits": output,  # [Nt, B, V]
            "encoder_attentions": encoder_attention,  # Self attention
            "cross_attentions": decoder_attention,  # Cross attention
            **latent_output_dict,
            "outputs": outputs,
        }

class TransformerLightning(Seq2SeqLightning):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        # Tokenizer
        self.tokenizer = CharizardTokenizer(model_max_length=args.max_length)
        # self.tokenizer = AutoTokenizer.from_pretrained("google/canine-c")
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Model
        self.model = Transformer(tokenizer=self.tokenizer, **vars(args))

        # Logging metrics
        self.log_bleu = True
        self.log_chrf = True
        self.plot_encoder_attention = True
        self.plot_cross_attention = True
        self.model_type = "Transformer"

        # Initialization
        init_weights(self.model)

        self.save_hyperparameters()
