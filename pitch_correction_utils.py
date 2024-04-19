from functools import partial
from pathlib import Path
import argparse
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as sig
import psola


SEMITONES_IN_OCTAVE = 12


def degrees_from(scale: str):
    """Return the pitch classes (degrees) that correspond to the given scale"""
    degrees = librosa.key_to_degrees(scale)
    # To properly perform pitch rounding to the nearest degree from the scale, we need to repeat
    # the first degree raised by an octave. Otherwise, pitches slightly lower than the base degree
    # would be incorrectly assigned.
    degrees = np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE]))
    return degrees


def closest_pitch(f0):
    """Round the given pitch values to the nearest MIDI note numbers"""
    midi_note = np.around(librosa.hz_to_midi(f0))
    # To preserve the nan values.
    nan_indices = np.isnan(f0)
    midi_note[nan_indices] = np.nan
    # Convert back to Hz.
    return librosa.midi_to_hz(midi_note)


def closest_pitch_from_scale(f0, scale):
    """Return the pitch closest to f0 that belongs to the given scale"""
    # Preserve nan.
    if np.isnan(f0):
        return np.nan
    degrees = degrees_from(scale)
    midi_note = librosa.hz_to_midi(f0)
    # Subtract the multiplicities of 12 so that we have the real-valued pitch class of the
    # input pitch.
    degree = midi_note % SEMITONES_IN_OCTAVE
    # Find the closest pitch class from the scale.
    degree_id = np.argmin(np.abs(degrees - degree))
    # Calculate the difference between the input pitch class and the desired pitch class.
    degree_difference = degree - degrees[degree_id]
    # Shift the input MIDI note number by the calculated difference.
    midi_note -= degree_difference
    # Convert to Hz.
    return librosa.midi_to_hz(midi_note)


def aclosest_pitch_from_scale(f0, scale):
    """Map each pitch in the f0 array to the closest pitch belonging to the given scale."""
    sanitized_pitch = np.zeros_like(f0)
    for i in np.arange(f0.shape[0]):
        sanitized_pitch[i] = closest_pitch_from_scale(f0[i], scale)
    # Perform median filtering to additionally smooth the corrected pitch.
    smoothed_sanitized_pitch = sig.medfilt(sanitized_pitch, kernel_size=11)
    # Remove the additional NaN values after median filtering.
    smoothed_sanitized_pitch[np.isnan(smoothed_sanitized_pitch)] = \
        sanitized_pitch[np.isnan(smoothed_sanitized_pitch)]
    return smoothed_sanitized_pitch


def autotune(audio, sr, correction_function, plot=False):
    # Set some basis parameters.
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    # Pitch tracking using the PYIN algorithm.
    f0, voiced_flag, voiced_probabilities = librosa.pyin(audio,
                                                         frame_length=frame_length,
                                                         hop_length=hop_length,
                                                         sr=sr,
                                                         fmin=fmin,
                                                         fmax=fmax)

    # Apply the chosen adjustment strategy to the pitch.
    corrected_f0 = correction_function(f0)

    if plot:
        # Plot the spectrogram, overlaid with the original pitch trajectory and the adjusted
        # pitch trajectory.
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        time_points = librosa.times_like(stft, sr=sr, hop_length=hop_length)
        log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(log_stft, x_axis='time', y_axis='log', ax=ax, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.plot(time_points, f0, label='original pitch', color='cyan', linewidth=2)
        ax.plot(time_points, corrected_f0, label='corrected pitch', color='orange', linewidth=1)
        ax.legend(loc='upper right')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [M:SS]')
        plt.savefig('pitch_correction.png', dpi=300, bbox_inches='tight')

    # Pitch-shifting using the PSOLA algorithm.
    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)


def main(
    vocals_file,
    plot=False,
    correction_method="closest",
    scale=None
):
    """Run autotune-like pitch correction on the given audio file.

    Args:
        vocals_file (str): Filepath to the audio file to be pitch-corrected.
        plot (bool, optional): Whether to plot the results. Defaults to False.
        correction_method (str, optional): The pitch correction method to use. Defaults to `"closest"`. If set to "closest", the pitch will be rounded to the nearest MIDI note.
            If set to "scale", the pitch will be rounded to the nearest note in the given `scale`.
        scale (str, optional): The scale to use for pitch correction. ex. `"C:min"` / `"A:maj"`. Defaults to None.
    """    
    
    # Parse the command line arguments.
    # ap = argparse.ArgumentParser()
    # ap.add_argument('vocals_file')
    # ap.add_argument('--plot', '-p', action='store_true', default=False,
    #                 help='if set, will produce a plot of the results')
    # ap.add_argument('--correction-method', '-c', choices=['closest', 'scale'], default='closest')
    # ap.add_argument('--scale', '-s', type=str, help='see librosa.key_to_degrees;'
    #                                                 ' used only for the \"scale\" correction'
    #                                                 ' method')
    # args = ap.parse_args(args=args)

    filepath = Path(vocals_file)

    # Load the audio file.
    y, sr = librosa.load(str(filepath), sr=None, mono=False)

    # Only mono-files are handled. If stereo files are supplied, only the first channel is used.
    if y.ndim > 1:
        y = y[0, :]

    # Pick the pitch adjustment strategy according to the arguments.
    correction_function = closest_pitch if correction_method == 'closest' else \
        partial(aclosest_pitch_from_scale, scale=scale)

    # Perform the auto-tuning.
    pitch_corrected_y = autotune(y, sr, correction_function, plot)

    # Write the corrected audio to an output file.
    filepath = filepath.parent / (filepath.stem + '_pitch_corrected' + filepath.suffix)
    sf.write(str(filepath), pitch_corrected_y, sr)
    return pitch_corrected_y


if __name__=='__main__':
    # main("./singing_music_idea.wav --plot -c closest".split())
    # python pitch_correction_utils.py --vocals_file "./nate_is_humming.wav" --plot -c closest
    from fire import Fire
    Fire(main)

