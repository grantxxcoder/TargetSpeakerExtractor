"""Streamed output must equal whole-clip output.

The live demo claims it runs the evaluated model. That is only true if
stream_push, fed audio in arbitrary chunks, reproduces extract() on the
whole clip -- asserted here on tiny random-weight models of both arms the demo
can load (the baseline and the extension, both from build_bsrnn_tfmap).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.bsrnn import build_bsrnn_tfmap, extract
from src.models.stft import stft_forward
from src.models.streaming import causal_stft_push, causal_stft_start, stream_flush, stream_push, stream_start

def _build(context, lookahead_frames=0):
    torch.manual_seed(0)
    embedding_dim = 8 if context else None
    model = build_bsrnn_tfmap(sample_rate=16000, n_fft=512, hop=128, feature_dim=16, hidden_dim=16, num_repeat=2, mlp_hidden=16, tfmap_scale=16.0, tfmap_parts=True, tfmap_part_scales=[2.0, 0.5, 1.5], embedding_dim=embedding_dim, lookahead_frames=lookahead_frames).eval()
    if context:
        # The context fusion is zero-initialised, i.e. a no-op; randomise it so the
        # test would catch the embedding being dropped.
        torch.nn.init.normal_(model.context.project.weight, std=0.3)
    return model


def _stream(model, mixture, enrollment, emb, sizes):
    stream = stream_start(model, enrollment[0], emb)
    out, pos, i = [], 0, 0
    while pos < mixture.shape[-1]:
        n = sizes[i % len(sizes)]
        out.append(stream_push(stream, mixture[0, pos:pos + n])["audio"])
        pos, i = pos + n, i + 1
    out.append(stream_flush(stream)["audio"])
    return torch.from_numpy(np.concatenate(out))


@pytest.mark.parametrize("lookahead_frames", [0, 3])
@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize("sizes", [[32], [320], [7, 100, 1, 555, 64]])
def test_streamed_equals_whole_clip(context, sizes, lookahead_frames):
    model = _build(context, lookahead_frames)
    g = torch.Generator().manual_seed(1)
    mixture = torch.randn(1, 8000 * 2 + 13, generator=g) * 0.1   # not a hop multiple
    enrollment = torch.randn(1, 8000, generator=g) * 0.1
    emb = torch.nn.functional.normalize(torch.randn(1, 8, generator=g), dim=1) if context else None
    with torch.no_grad():
        whole = extract(model, mixture, enrollment, emb)[0]
    streamed = _stream(model, mixture, enrollment, emb, sizes)
    assert streamed.shape == whole.shape
    assert (streamed - whole).abs().max() < 1e-5 * whole.abs().max().clamp_min(1e-3)


@pytest.mark.parametrize("lookahead_frames", [0, 3])
def test_lookahead_hears_exactly_that_far(lookahead_frames):
    """Changing the audio from one sample on must leave every output frame that cannot hear it untouched, and must reach the frame exactly lookahead frames earlier.

    Added by Claude Code on 2026-10-29 with the lookahead (decisions-pending.md 2026-10-29).
    """
    from src.models.bsrnn import extract_spectrogram
    model = _build(False, lookahead_frames)
    g = torch.Generator().manual_seed(3)
    mixture = torch.randn(1, 8000, generator=g) * 0.1
    enrollment = torch.randn(1, 8000, generator=g) * 0.1
    changed = mixture.clone()
    change_from = 4000
    changed[:, change_from:] += torch.randn(1, 8000 - change_from, generator=g) * 0.1
    with torch.no_grad():
        before = extract_spectrogram(model, mixture, enrollment, None)[0]
        after = extract_spectrogram(model, changed, enrollment, None)[0]
    stft = model.stft
    # Frame t covers samples t * hop - padding to t * hop - padding + n_fft, so the first frame that contains the change is this one.
    first_changed_frame = (change_from + stft.padding - stft.n_fft) // stft.hop_length + 1
    frame_moved = (after - before).abs().amax(dim=(0, 1)) > 1e-6
    first_moved = int(torch.nonzero(frame_moved)[0])
    assert first_moved == first_changed_frame - lookahead_frames


def test_causal_stft_matches_whole_clip_frames():
    """The demo draws its output spectrogram with this, so it must frame a
    stream exactly as stft_forward frames the finished clip."""
    stft = _build(False).stft
    x = torch.randn(8000 + 13, generator=torch.Generator().manual_seed(2))
    whole = stft_forward(stft, x[None])[0]
    n_frames = whole.shape[-1]
    c, parts, pos = causal_stft_start(stft), [], 0
    for n in [5, 300, 64, 1000] * 20:
        parts.append(causal_stft_push(c, x[pos:pos + n])[0]); pos += n
        if pos >= x.numel():
            break
    parts.append(causal_stft_push(c, torch.zeros(n_frames * stft.hop_length - x.numel()))[0])
    streamed = torch.cat(parts, dim=-1)
    assert streamed.shape == whole.shape
    assert (streamed - whole).abs().max() < 1e-4
