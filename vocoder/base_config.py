import torch
from dataclasses import dataclass


@dataclass
class MelSpectrogramConfig:
    num_mels = 80


@dataclass
class FastSpeechConfig:
    vocab_size = 300
    max_seq_len = 3000

    encoder_dim = 256
    encoder_n_layer = 4
    encoder_head = 2
    encoder_conv1d_filter_size = 1024

    decoder_dim = 256
    decoder_n_layer = 4
    decoder_head = 2
    decoder_conv1d_filter_size = 1024

    fft_conv1d_kernel = (9, 1)
    fft_conv1d_padding = (4, 0)

    duration_predictor_filter_size = 256
    duration_predictor_kernel_size = 3
    dropout = 0.1

    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    PAD_WORD = "<blank>"
    UNK_WORD = "<unk>"
    BOS_WORD = "<s>"
    EOS_WORD = "</s>"


@dataclass
class TrainConfig:
    checkpoint_path = "./checkpoints"
    logger_path = "./logger"

    mel_ground_truth = "./data/melspecs"
    pitch_ground_truth = "./data/pitch"
    energy_ground_truth = "./data/energy"

    audio_ground_truth = "./data/LJSpeech-1.1/wavs"
    alignment_path = "./data/mfa_alignments"
    data_path = "./data/train_phones.txt"

    wandb_project = "fastspeech_example"

    text_cleaners = ["english_cleaners"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_mfa = True

    batch_size = 48
    epoch_len = -1
    epochs = 300
    n_warm_up_step = 3000

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [500000, 1000000, 2000000]

    save_step = 500
    log_step = 80
    clear_Time = 20

    batch_expand_size = 48

    hop_length = 256
    win_length = 1024
    filter_length = 1024
    sampling_rate = 22_500
    n_mel_channels = 80

    energy_mean = 21.832
    energy_min = 0.018
    energy_max = 314.962
    energy_std = 19.784

    pitch_mean = 129.851
    pitch_min = 0.000
    pitch_max = 788.677
    pitch_std = 111.120

    alignment_min = 0
    alignment_max = 4.304  # log
    alignment_mean = 5.669
    alignment_std = 4.940
