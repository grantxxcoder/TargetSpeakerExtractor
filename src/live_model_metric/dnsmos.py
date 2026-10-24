"""DNSMOS: neural predictors of how a human listener would rate the audio, from 1 to 5, higher is better.

p808_mos is the ITU-T P.808 overall score. sig, bak and ovrl are the P.835 speech, background and overall scores.
Ported from the reference implementation in microsoft/DNS-Challenge, DNSMOS/dnsmos_local.py, retrieved 2026-09-01. Reddy et al. (2021) for P.808 and Reddy et al. (2022) for P.835. Borrowed, nothing here is ours.
It needs only the processed audio, no clean reference, and it is reported as the exhibit of a score that can be gamed, not as a metric we trust (metric-definitions.md section 4).
"""

from pathlib import Path

import librosa
import numpy as np
import onnxruntime
import soundfile

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 9.01
SEGMENT_SAMPLES = int(SEGMENT_SECONDS * SAMPLE_RATE)
HOP_SECONDS = 1.0

MODEL_DIRECTORY = Path(__file__).parent / "dnsmos_onnx"
P808_MODEL = MODEL_DIRECTORY / "model_v8.onnx"
# The personalised P.835 model, because target speaker extraction removes a speaker on purpose, which the standard model can score as damage to the speech.
P835_MODEL = MODEL_DIRECTORY / "p_sig_bak_ovr.onnx"

# The raw P.835 outputs are mapped onto the 1 to 5 scale with the reference's personalised polynomials, copied verbatim. P.808 is used raw.
POLYNOMIALS = {
    "sig": [-0.01019296, 0.02751166, 1.19576786, -0.24348726],
    "bak": [-0.04976499, 0.44276479, -0.1644611, 0.96883132],
    "ovrl": [-0.00533021, 0.005101, 1.18058466, -0.11236046],
}

# One ONNX session per model file, opened on first use and kept for the process.
SESSIONS = {}


# this opens an ONNX model once and reuses it
def session(model_path):
    key = str(model_path)
    if key not in SESSIONS:
        SESSIONS[key] = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return SESSIONS[key]


# this gives the mel spectrogram the P.808 model expects for one segment
def mel_spectrogram(segment):
    # n_fft is one more than the frame size in the reference, which with a 160-sample hop gives exactly the 900 frames the model expects.
    mel = librosa.feature.melspectrogram(y=segment, sr=SAMPLE_RATE, n_fft=321, hop_length=160, n_mels=120)
    return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T


# this packs the four scores and the number of segments behind them into one dictionary
def dnsmos_scores(p808_mos, sig, bak, ovrl, segments):
    return {"p808_mos": p808_mos, "sig": sig, "bak": bak, "ovrl": ovrl, "segments": segments}


# this scores one 16 kHz clip: 9.01 s segments with a 1 s hop, each scored and then averaged
def score_waveform(waveform):
    """A clip shorter than one segment is repeated until it fills one, as the reference does, because silence padding would be scored as bad audio."""
    waveform = np.asarray(waveform, dtype=np.float64).squeeze()
    while len(waveform) < SEGMENT_SAMPLES:
        waveform = np.append(waveform, waveform)

    hop_samples = int(HOP_SECONDS * SAMPLE_RATE)
    num_segments = int(np.floor(len(waveform) / SAMPLE_RATE) - SEGMENT_SECONDS) + 1
    p835 = session(P835_MODEL)
    p808 = session(P808_MODEL)

    p808_values = []
    sig_values = []
    bak_values = []
    ovrl_values = []
    for index in range(max(num_segments, 0)):
        segment = waveform[index * hop_samples:index * hop_samples + SEGMENT_SAMPLES]
        if len(segment) < SEGMENT_SAMPLES:
            continue
        p835_input = segment.astype(np.float32)[np.newaxis, :]
        # The last 160 samples are dropped so the spectrogram has exactly the 900 frames the P.808 model was built for.
        p808_input = mel_spectrogram(segment[:-160]).astype(np.float32)[np.newaxis, :, :]
        sig_raw, bak_raw, ovrl_raw = p835.run(None, {"input_1": p835_input})[0][0]
        p808_values.append(float(p808.run(None, {"input_1": p808_input})[0][0][0]))
        sig_values.append(float(np.poly1d(POLYNOMIALS["sig"])(sig_raw)))
        bak_values.append(float(np.poly1d(POLYNOMIALS["bak"])(bak_raw)))
        ovrl_values.append(float(np.poly1d(POLYNOMIALS["ovrl"])(ovrl_raw)))

    if not p808_values:
        return dnsmos_scores(None, None, None, None, 0)
    return dnsmos_scores(float(np.mean(p808_values)), float(np.mean(sig_values)), float(np.mean(bak_values)), float(np.mean(ovrl_values)), len(p808_values))


# this reads a 16 kHz wav file and scores it. Resampling here would change the score, so another rate is refused
def score_audio_file(audio_file_path):
    waveform, sample_rate = soundfile.read(str(audio_file_path))
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{audio_file_path} is {sample_rate} Hz, but DNSMOS needs {SAMPLE_RATE} Hz.")
    return score_waveform(waveform)


# this averages the scores over trials, leaving out any clip too short to score
def mean_scores(score_list):
    scored = []
    for scores in score_list:
        if scores["p808_mos"] is not None:
            scored.append(scores)
    if not scored:
        return dnsmos_scores(None, None, None, None, 0)
    p808_values = []
    sig_values = []
    bak_values = []
    ovrl_values = []
    for scores in scored:
        p808_values.append(scores["p808_mos"])
        sig_values.append(scores["sig"])
        bak_values.append(scores["bak"])
        ovrl_values.append(scores["ovrl"])
    return dnsmos_scores(float(np.mean(p808_values)), float(np.mean(sig_values)), float(np.mean(bak_values)), float(np.mean(ovrl_values)), len(scored))
