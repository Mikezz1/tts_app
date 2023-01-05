import streamlit as st
import os
from acoustic_model.fastspeech import FastSpeech
from vocoder.generator import Generator
from acoustic_model.base_config import *


st.title("Text to speech")
text = st.text_input("Enter text")

os.makedirs("audios", exists_ok=True)

mel_config = MelSpectrogramConfig()
model_config = FastSpeechConfig()
train_config = TrainConfig()

acoustic_model = FastSpeech(model_config, mel_config, train_config)

vocoder = Generator(
    k_u=[16, 16, 4, 4],
    upsample_first=512,
    kernels=[3, 7, 11],
    dilation=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
)

acoustic_model.load_state_dict(
    torch.load("./data/checkpoint_new_last_1.pth.tar", map_location="cpu")["model"]
)
acoustic_model.eval()
acoustic_model.to("cpu")

generator.load_state_dict(
    torch.load("./data/hifigan_generator.pth", map_location="cpu")
)
generator.eval()
generator.to("cpu")


def text_to_mel(
    model,
    text,
    train_config,
    alpha=1.0,
    energy=1.0,
    pitch=1.0,
):
    text = np.stack([text])
    src_pos = np.array([i + 1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)
    model.eval()
    with torch.no_grad():
        mel = model.forward(
            sequence, src_pos, alpha=alpha, e_param=energy, p_param=pitch
        )
    return mel[0].cpu().transpose(0, 1)


def text_to_speech(prompt):
    melspec = text_to_mel(acoustic_model, prompt, train_config)
    waveform = vocoder(melspec)
    # torchaudio.save(f"audios/{fname}.wav", waveform, 22050)
    return waveform


if st.button("convert"):
    waveform = text_to_speech(text)
    st.audio(waveform, format="audio/wav", start_time=0, sample_rate=22050)
