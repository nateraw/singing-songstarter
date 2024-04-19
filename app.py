###########################################
# For fast downloads from Hugging Face Hub
# **Requires the hf_transfer package**
###########################################
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
###########################################

import json
import random
import typing as tp
from datetime import datetime
from pathlib import Path
from functools import partial

import gradio as gr
import torch
import torchaudio
import numpy as np

from audiocraft.models import musicgen
from audiocraft.data.audio import audio_write
from audiocraft.utils.notebook import display_audio

from pitch_correction_utils import autotune, closest_pitch, aclosest_pitch_from_scale


def ta_to_librosa_format(waveform):
    """
    Convert an audio tensor from torchaudio format to librosa format.

    Args:
    waveform (torch.Tensor): Audio tensor from torchaudio with shape (n_channels, n_samples).

    Returns:
    np.ndarray: Audio array in librosa format with shape (n_samples,) or (2, n_samples).
    """
    # Ensure waveform is in CPU and convert to numpy
    waveform_np = waveform.numpy()

    # Check if audio is mono or stereo and transpose if necessary
    if waveform_np.shape[0] == 1:
        # Remove the channel dimension for mono
        waveform_np = waveform_np.squeeze(0)
    else:
        # Transpose to switch from (n_channels, n_samples) to (n_samples, n_channels)
        waveform_np = waveform_np.transpose()

    # Normalize to [-1, 1] if not already
    if waveform_np.dtype in [np.int16, np.int32]:
        waveform_np = waveform_np / np.iinfo(waveform_np.dtype).max

    return waveform_np


def librosa_to_ta_format(waveform_np):
    """
    Convert an audio array from librosa format to torchaudio format.

    Args:
    waveform_np (np.ndarray): Audio array from librosa with shape (n_samples,) or (2, n_samples).

    Returns:
    torch.Tensor: Audio tensor in torchaudio format with shape (n_channels, n_samples).
    """
    # Ensure it is a float32 array normalized to [-1, 1]
    waveform_np = np.array(waveform_np, dtype=np.float32)

    if waveform_np.ndim == 1:
        # Add a channel dimension for mono
        waveform_np = waveform_np[np.newaxis, :]
    else:
        # Transpose to switch from (n_samples, n_channels) to (n_channels, n_samples)
        waveform_np = waveform_np.transpose()

    # Convert numpy array to PyTorch tensor
    waveform = torch.from_numpy(waveform_np)
    return waveform


def run_autotune(y, sr, correction_method="closest", scale=None):
    # Only mono-files are handled. If stereo files are supplied, only the first channel is used.
    if y.ndim > 1:
        y = y[0, :]

    # Pick the pitch adjustment strategy according to the arguments.
    correction_function = closest_pitch if correction_method == 'closest' else \
        partial(aclosest_pitch_from_scale, scale=scale)

    # Torchaudio -> librosa
    y = ta_to_librosa_format(y)
    # Autotune
    pitch_corrected_y = autotune(y, sr, correction_function, plot=False)
    # Librosa -> torchaudio
    pitch_corrected_y = librosa_to_ta_format(pitch_corrected_y)

    return pitch_corrected_y


def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _preprocess_audio(
    audio_path, model: musicgen.MusicGen, duration: tp.Optional[int] = None
):
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)

    # Calculate duration in seconds if not provided
    if duration is None:
        duration = wav.shape[1] / model.sample_rate

    # Check if duration is more than 30 seconds
    if duration > 30:
        raise ValueError("Duration cannot be more than 30 seconds")

    end_sample = int(model.sample_rate * duration)
    wav = wav[:, :end_sample]

    assert wav.shape[0] == 1
    assert wav.shape[1] == model.sample_rate * duration

    wav = wav.cuda()
    wav = wav.unsqueeze(1)

    with torch.no_grad():
        gen_audio = model.compression_model.encode(wav)

    codes, scale = gen_audio

    assert scale is None

    return codes


def _get_stemmed_wav_patched(wav, sample_rate):
    print("Skipping stem separation!")
    return wav


class Pipeline:
    def __init__(self, model_id, max_batch_size=4, do_skip_demucs=True):
        self.model = musicgen.MusicGen.get_pretrained(model_id)
        self.max_batch_size = max_batch_size
        self.do_skip_demucs = do_skip_demucs

        if self.do_skip_demucs:
            self.model.lm.condition_provider.conditioners.self_wav._get_stemmed_wav = _get_stemmed_wav_patched

    def __call__(
        self,
        prompt,
        input_audio=None,
        scale=None,
        continuation=False,
        batch_size=1,
        duration=15,
        use_sampling=True,
        temperature=1.0,
        top_k=250,
        top_p=0.0,
        cfg_coef=3.0,
        output_dir="./samples",  # change to google drive if you'd like
        normalization_strategy="loudness",
        seed=-1,
        continuation_start=0,
        continuation_end=None,
    ):
        print("Prompt:", prompt)
        if scale == "closest":
            scale = None

        set_generation_params = lambda duration: self.model.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=cfg_coef,
        )

        if not seed or seed == -1:
            seed = torch.seed() % 2 ** 32 - 1
            set_all_seeds(seed)
        set_all_seeds(seed)
        print(f"Using seed {seed}")
        if not input_audio:
            set_generation_params(duration)
            wav, tokens = self.model.generate([prompt] * batch_size, progress=True, return_tokens=True)
        else:
            input_audio, sr = torchaudio.load(input_audio)
            # Save a copy of the original input audio
            original_input_audio = input_audio.clone()
            print("Input audio shape:", input_audio.shape)
            if scale is None:
                print("Running pitch correction for 'closest' pitch")
                input_audio = run_autotune(input_audio, sr, correction_method="closest")
            else:
                print("Running pitch correction for 'scale' pitch")
                input_audio = run_autotune(input_audio, sr, correction_method="scale", scale=scale)
            print(f"...Done running pitch correction. Shape after is {input_audio.shape}.\n")
            input_audio = input_audio[None] if input_audio.dim() == 2 else input_audio

            continuation_start = 0 if not continuation_start else continuation_start
            if continuation_end is None or continuation_end == -1:
                continuation_end = input_audio.shape[2] / sr

            if continuation_start > continuation_end:
                raise ValueError(
                    "`continuation_start` must be less than or equal to `continuation_end`"
                )

            input_audio_wavform = input_audio[
                ..., int(sr * continuation_start) : int(sr * continuation_end)
            ]
            input_audio_wavform = input_audio_wavform.repeat(batch_size, 1, 1)
            # TODO - not using this - is that wrong??
            input_audio_duration = input_audio_wavform.shape[-1] / sr

            if continuation:
                set_generation_params(duration)  # + input_audio_duration)  # SEE TODO above
                print("Continuation wavform shape!", input_audio_wavform.shape)
                wav, tokens = self.model.generate_continuation(
                    prompt=input_audio_wavform,
                    prompt_sample_rate=sr,
                    descriptions=[prompt] * batch_size,
                    progress=True,
                    return_tokens=True
                )
            else:
                print("Melody wavform shape!", input_audio_wavform.shape)
                set_generation_params(duration)
                wav, tokens = self.model.generate_with_chroma(
                    [prompt] * batch_size, input_audio_wavform, sr, progress=True, return_tokens=True
                )
        wav, tokens = wav.cpu(), tokens.cpu()
        # Write to files
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if input_audio is not None:
            outfile_path = output_dir / f"{dt_str}_input_raw"
            audio_write(
                outfile_path,
                original_input_audio,
                sr,
                strategy=normalization_strategy,
            )
            outfile_path = output_dir / f"{dt_str}_input_pitch_corrected"
            audio_write(
                outfile_path,
                input_audio_wavform[0],
                sr,
                strategy=normalization_strategy,
            )

        for i in range(batch_size):
            outfile_path = output_dir / f"{dt_str}_{i:02d}"
            audio_write(
                outfile_path,
                wav[i],
                self.model.sample_rate,
                strategy=normalization_strategy,
            )
        json_out_path = output_dir / f"{dt_str}.json"
        json_out_path.write_text(json.dumps(dict(
            prompt=prompt,
            batch_size=batch_size,
            duration=duration,
            use_sampling=use_sampling,
            temperature=temperature,
            top_k=top_k,
            cfg_coef=cfg_coef,
        )))

        to_return = [None] * (self.max_batch_size + 1)
        if input_audio is not None:
            print(f"trying to return input audio wavform of shape: {input_audio_wavform.shape}")
            to_return[0] = (sr, input_audio_wavform[0].T.numpy())

        for i in range(batch_size):
            to_return[i + 1] = (self.model.sample_rate, wav[i].T.numpy())
            print(wav[i].shape)
        return to_return


def main(model_id="nateraw/musicgen-songstarter-v0.2", max_batch_size=4, share=False, debug=False):
    pipeline = Pipeline(model_id, max_batch_size)
    interface = gr.Interface(
        fn=pipeline.__call__,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
            gr.Audio(
                sources=["microphone"],
                waveform_options=gr.WaveformOptions(
                    waveform_color="#01C6FF",
                    waveform_progress_color="#0066B4",
                    skip_length=2,
                    show_controls=False,
                ),
                type="filepath",
            ),
            gr.Dropdown(["closest", "A:maj", "A:min", "Bb:maj", "Bb:min", "B:maj", "B:min", "C:maj", "C:min", "Db:maj", "Db:min", "D:maj", "D:min", "Eb:maj", "Eb:min", "E:maj", "E:min", "F:maj", "F:min", "Gb:maj", "Gb:min", "G:maj", "G:min", "Ab:maj", "Ab:min"], label="Scale for pitch correction.", value="closest"),
            gr.Checkbox(label="Is Continuation", value=False),
            gr.Slider(label="Batch Size", value=1, minimum=1, maximum=pipeline.max_batch_size, step=1),
            gr.Slider(label="Duration", value=15, minimum=4, maximum=30),
            gr.Checkbox(label="Use Sampling", value=True),
            gr.Slider(label="Temperature", value=1.0, minimum=0.0, maximum=2.0),
            gr.Slider(label="Top K", value=250, minimum=0, maximum=1000),
            gr.Slider(label="Top P", value=0.0, minimum=0.0, maximum=1.0),
            gr.Slider(label="CFG Coef", value=3.0, minimum=0.0, maximum=10.0),
            gr.Textbox(label="Output Dir", value="./samples"),
            gr.Dropdown(["loudness", "clip", "peak", "rms"], value="loudness", label="Strategy for normalizing audio."),
            gr.Slider(label="random seed", minimum=-1, maximum=9e8),
        ],
        outputs=[gr.Audio(label=("Input " if i == 0 else "") + f"Audio {i}") for i in range(pipeline.max_batch_size + 1)],
        title="🎶 Generate song ideas with musicgen-songstarter-v0.2 🎶",
        description="Check out the repo [here](https://huggingface.co/nateraw/musicgen-songstarter-v0.2)",
        examples=[
            ["hip hop, soul, piano, chords, jazz, neo jazz, G# minor, 140 bpm", None, "closest", False, 1, 8, True, 1.0, 250, 0.0, 3.0, "./samples", "loudness", -1],
            ["acoustic, guitar, melody, rnb, trap, E minor, 85 bpm", None, "closest", False, 1, 8, True, 1.0, 250, 0.0, 3.0, "./samples", "loudness", -1],
            ["synth, dark, hip hop, melody, trap, Gb minor, 140 bpm", "./nate_is_singing_Gb_minor.wav", "Gb:min", False, 1, 7, True, 1.0, 250, 0.0, 3.0, "./samples", "loudness", -1],
            ["drill, layered, melody, songstarters, trap, C# minor, 130 bpm", None, "closest", False, 1, 8, True, 1.0, 250, 0.0, 3.0, "./samples", "loudness", -1],
            ["hip hop, soul, rnb, neo soul, songstarters, B minor, 140 bpm", None, "closest", False, 1, 8, True, 1.0, 250, 0.0, 3.0, "./samples", "loudness", -1],
            ["music, mallets, bells, melody, dancehall, african, afropop & afrobeats", "./nate_is_singing_Gb_minor.wav", "Gb:min", False, 1, 7, True, 1.0, 250, 0.0, 4.5, "./samples", "loudness", -1],
        ]
    )
    interface.launch(share=share, debug=debug)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)

    # For testing

    # pipe = Pipeline("nateraw/musicgen-songstarter-v0.2", max_batch_size=4)
    # example_input = (
    #     "hip hop, soul, piano, chords, jazz, neo jazz, G# minor, 140 bpm",
    #     "nate_is_humming.wav",
    #     "closest",
    #     False,
    #     1,
    #     8,
    #     True,
    #     1.0,
    #     250,
    #     0.0,
    #     3.0,
    #     "./samples",
    #     "loudness",
    #     -1,
    #     0,
    #     None
    # )
    # out = pipe(*example_input)