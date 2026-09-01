"""DNSMOS: neural predictors of what a human listener would rate the audio.

Returns four scores per clip, all 1-5 and higher-is-better:

    p808_mos   ITU-T P.808 protocol, one overall score. The metric the REAL-TSE
               Challenge switched TO, after OVRL was gamed.
    sig        P.835 speech signal quality  -- degraded by artefacts
    bak        P.835 background intrusiveness -- improved by suppression
    ovrl       P.835 overall -- THE SCORE THAT GOT GAMED

Ported from the reference implementation in microsoft/DNS-Challenge,
DNSMOS/dnsmos_local.py, retrieved 2026-09-01. Reddy et al. (2021) for P.808 and
Reddy et al. (2022) for P.835. Cited as borrowed; nothing here is ours.

NON-INTRUSIVE: needs only the degraded audio, no clean reference and no
transcript. It is therefore the only quality metric in this project that can be
run on real recordings such as AMI.

REPORTED AS THE EXHIBIT, NOT AS A METRIC WE TRUST. It is the only score in this
project with a gradient, so the only one that can be attacked by optimisation --
which is exactly what metric-definitions.md 4 is about.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 9.01
SEGMENT_SAMPLES = int(SEGMENT_SECONDS * SAMPLE_RATE)      # 144160, the model's input width
HOP_SECONDS = 1.0

MODEL_DIRECTORY = Path(__file__).parent / "dnsmos_onnx"
P808_MODEL = MODEL_DIRECTORY / "model_v8.onnx"
P835_MODEL_PERSONALISED = MODEL_DIRECTORY / "p_sig_bak_ovr.onnx"
P835_MODEL_STANDARD = MODEL_DIRECTORY / "sig_bak_ovr.onnx"

# Target speaker extraction IS personalised speech enhancement, so the
# personalised model and its coefficients are the correct calibration. The
# distinction exists because in this task the correct output REMOVES a speaker,
# and the standard model can score that removal as damage to the speech.
PERSONALISED = True

# The raw P.835 outputs are not MOS scores. These map them onto the 1-5 scale and
# are part of the measuring instrument, copied verbatim from the reference. The
# model file and the coefficient set must match: personalised model with
# personalised coefficients, standard with standard.
POLYNOMIALS = {
    True: {                                               # personalised
        "sig":  [-0.01019296, 0.02751166, 1.19576786, -0.24348726],
        "bak":  [-0.04976499, 0.44276479, -0.1644611, 0.96883132],
        "ovrl": [-0.00533021, 0.005101, 1.18058466, -0.11236046],
    },
    False: {                                              # standard
        "sig":  [-0.08397278, 1.22083953, 0.0052439],
        "bak":  [-0.13166888, 1.60915514, -0.39604546],
        "ovrl": [-0.06766283, 1.11546468, 0.04602535],
    },
}

# P.808 is used RAW. Only the P.835 scores are polynomial-corrected.

_sessions = {}


def _session(model_path):
    import onnxruntime
    key = str(model_path)
    if key not in _sessions:
        _sessions[key] = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
    return _sessions[key]


def _mel_spectrogram(segment):
    import librosa
    # n_fft is frame_size + 1 in the reference, which is deliberate: with
    # hop_length 160 over a segment trimmed by 160 samples it yields exactly the
    # 900 frames the P.808 model expects.
    mel = librosa.feature.melspectrogram(
        y=segment, sr=SAMPLE_RATE, n_fft=321, hop_length=160, n_mels=120)
    return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T


@dataclass
class DnsmosScores:
    p808_mos: float = None
    sig: float = None
    bak: float = None
    ovrl: float = None
    segments: int = 0
    personalised: bool = PERSONALISED

    def __str__(self):
        if self.p808_mos is None:
            return "DNSMOS: undefined, clip too short to score"
        return (f"P808 {self.p808_mos:.2f}  SIG {self.sig:.2f}  "
                f"BAK {self.bak:.2f}  OVRL {self.ovrl:.2f}  "
                f"({self.segments} segments)")


def score_waveform(waveform, personalised=PERSONALISED):
    """Score one clip. `waveform` is 1-D at 16 kHz.

    Segments of 9.01 s with a 1 s hop, scored independently and averaged. A clip
    shorter than one segment is repeated until it fills one, which is what the
    reference does -- looping rather than zero-padding, since silence would be
    scored as bad audio.
    """
    waveform = np.asarray(waveform, dtype=np.float64).squeeze()
    original_length = len(waveform)
    while len(waveform) < SEGMENT_SAMPLES:
        waveform = np.append(waveform, waveform)

    hop_samples = int(HOP_SECONDS * SAMPLE_RATE)
    n_segments = int(np.floor(len(waveform) / SAMPLE_RATE) - SEGMENT_SECONDS) + 1

    p835 = _session(P835_MODEL_PERSONALISED if personalised else P835_MODEL_STANDARD)
    p808 = _session(P808_MODEL)
    coefficients = POLYNOMIALS[bool(personalised)]

    p808_values, sig_values, bak_values, ovrl_values = [], [], [], []
    for index in range(max(n_segments, 0)):
        segment = waveform[index * hop_samples:
                           index * hop_samples + SEGMENT_SAMPLES]
        if len(segment) < SEGMENT_SAMPLES:
            continue

        p835_input = segment.astype(np.float32)[np.newaxis, :]
        # The trailing 160 samples are dropped so the spectrogram is exactly the
        # 900 frames the P.808 model was built for.
        p808_input = _mel_spectrogram(segment[:-160]).astype(np.float32)[np.newaxis, :, :]

        sig_raw, bak_raw, ovrl_raw = p835.run(None, {"input_1": p835_input})[0][0]
        p808_values.append(float(p808.run(None, {"input_1": p808_input})[0][0][0]))
        sig_values.append(float(np.poly1d(coefficients["sig"])(sig_raw)))
        bak_values.append(float(np.poly1d(coefficients["bak"])(bak_raw)))
        ovrl_values.append(float(np.poly1d(coefficients["ovrl"])(ovrl_raw)))

    if not p808_values:
        return DnsmosScores(segments=0, personalised=bool(personalised))
    return DnsmosScores(
        p808_mos=float(np.mean(p808_values)),
        sig=float(np.mean(sig_values)),
        bak=float(np.mean(bak_values)),
        ovrl=float(np.mean(ovrl_values)),
        segments=len(p808_values),
        personalised=bool(personalised),
    )


def score_audio_file(audio_file_path, personalised=PERSONALISED):
    import soundfile
    waveform, sample_rate = soundfile.read(str(audio_file_path))
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{audio_file_path} is {sample_rate} Hz; DNSMOS requires "
            f"{SAMPLE_RATE} Hz and resampling here would change the score")
    return score_waveform(waveform, personalised=personalised)


def mean_scores(score_list):
    """Average over trials. Trials too short to score are excluded."""
    scored = [s for s in score_list if s.p808_mos is not None]
    if not scored:
        return DnsmosScores(segments=0)
    return DnsmosScores(
        p808_mos=float(np.mean([s.p808_mos for s in scored])),
        sig=float(np.mean([s.sig for s in scored])),
        bak=float(np.mean([s.bak for s in scored])),
        ovrl=float(np.mean([s.ovrl for s in scored])),
        segments=len(scored),
        personalised=scored[0].personalised,
    )
