import torch.nn.functional as F
from fastspeech2.model.blocks import *
from fastspeech2.model.variance_adaptors import *
from fastspeech2.utils.utils import *


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config, mel_config, train_config):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(model_config)
        self.length_regulator = LengthRegulator(
            model_config, train_config.device)

        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(
            model_config.decoder_dim, mel_config.num_mels)

        self.energy_bin_min = (
            train_config.energy_min - train_config.energy_mean) / train_config.energy_std
        self.energy_bin_max = (
            train_config.energy_max - train_config.energy_mean) / train_config.energy_std
        self.pitch_bin_min = (
            train_config.pitch_min - train_config.pitch_mean) / train_config.pitch_std
        self.pitch_bin_max = (
            train_config.pitch_max - train_config.pitch_mean) / train_config.pitch_std

        self.energy_adaptor = VarianceAdaptor(
            model_config, train_config, train_config.device, self.energy_bin_min, self.energy_bin_max)
        self.pitch_adaptor = VarianceAdaptor(
            model_config, train_config, train_config.device, self.pitch_bin_min, self.pitch_bin_max)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mel_mask = ~mask
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.), mel_mask

    def forward(
            self, src_seq, src_pos, mel_pos=None, mel_max_length=None,
            length_target=None, energy_target=None, pitch_target=None,
            alpha=1.0, e_param=1.0, p_param=1.0):
        x, src_mask = self.encoder(src_seq, src_pos)
        src_mask = src_mask.bool().squeeze()

        if self.training:
            output, duration_predictor_output = self.length_regulator(
                x, alpha, length_target, mel_max_length)

            pitch_embedding, pitch_predictor_output = self.pitch_adaptor(
                output, pitch_target)

            energy_embedding, energy_predictor_output = self.energy_adaptor(
                output, energy_target)

            output = output + pitch_embedding
            output = output + energy_embedding

            output = self.decoder(output, mel_pos)

            output, mel_mask = self.mask_tensor(
                output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return output, duration_predictor_output, energy_predictor_output, pitch_predictor_output, mel_mask, src_mask
        else:

            output, mel_pos = self.length_regulator(x, alpha)
            energy_embedding = self.energy_adaptor(output, param=e_param)
            pitch_embedding = self.pitch_adaptor(output, param=p_param)

            output = output + pitch_embedding
            output = output + energy_embedding

            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return output


class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder, self).__init__()

        len_max_seq = model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.encoder_n_layer

        self.src_word_emb = nn.Embedding(
            model_config.vocab_size,
            model_config.encoder_dim,
            padding_idx=model_config.PAD
        )

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config.encoder_dim,
            model_config.encoder_conv1d_filter_size,
            model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.fft_conv1d_kernel,
            model_config.fft_conv1d_padding,
            dropout=model_config.dropout
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, model_config):

        super(Decoder, self).__init__()

        len_max_seq = model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.decoder_n_layer

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config.encoder_dim,
            model_config.encoder_conv1d_filter_size,
            model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.fft_conv1d_kernel,
            model_config.fft_conv1d_padding,
            dropout=model_config.dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output
