"""Streamed output must equal whole-clip output.

The live demo claims it runs the evaluated model. That is only true if
StreamingExtractor, fed audio in arbitrary chunks, reproduces forward() on the
whole clip -- asserted here on tiny random-weight models of both arms the demo
can load (1a: BSRNN_TFMAP, 1c: BSRNN_TFMAP_CONTEXT).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.bsrnn import BSRNN_TFMAP, BSRNN_TFMAP_CONTEXT  # noqa: E402
from src.models.streaming import CausalSTFT, StreamingExtractor  # noqa: E402

KW = dict(sample_rate=8000, n_fft=128, hop=32, num_repeat=2,
          band_segments={"uniform": 5}, feature_dim=16, hidden_dim=16,
          mlp_hidden=16, tfmap_parts=True, tfmap_part_scales=[2.0, 0.5, 1.5])


def _build(context):
    torch.manual_seed(0)
    model = (BSRNN_TFMAP_CONTEXT(embedding_dim=8, **KW) if context
             else BSRNN_TFMAP(**KW)).eval()
    if context:
        # ContextFusion is zero-initialised, i.e. a no-op; randomise it so the
        # test would catch the embedding being dropped.
        torch.nn.init.normal_(model.context.project.weight, std=0.3)
    return model


def _stream(model, mixture, enrollment, emb, sizes):
    stream = StreamingExtractor(model, enrollment[0], emb)
    out, pos, i = [], 0, 0
    while pos < mixture.shape[-1]:
        n = sizes[i % len(sizes)]
        out.append(stream.push(mixture[0, pos:pos + n])["audio"])
        pos, i = pos + n, i + 1
    out.append(stream.flush()["audio"])
    return torch.from_numpy(np.concatenate(out))


@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize("sizes", [[32], [320], [7, 100, 1, 555, 64]])
def test_streamed_equals_whole_clip(context, sizes):
    model = _build(context)
    g = torch.Generator().manual_seed(1)
    mixture = torch.randn(1, 8000 * 2 + 13, generator=g) * 0.1   # not a hop multiple
    enrollment = torch.randn(1, 8000, generator=g) * 0.1
    emb = torch.nn.functional.normalize(torch.randn(1, 8, generator=g), dim=1) if context else None
    kw = {"enrol_embedding": emb} if context else {}
    with torch.no_grad():
        whole = model(mixture, enrollment, **kw)[0]
    streamed = _stream(model, mixture, enrollment, emb, sizes)
    assert streamed.shape == whole.shape
    assert (streamed - whole).abs().max() < 1e-5 * whole.abs().max().clamp_min(1e-3)


def test_refuses_non_causal():
    model = BSRNN_TFMAP(**{**KW, "causal": False}).eval()
    with pytest.raises(ValueError):
        StreamingExtractor(model, torch.zeros(800))


def test_causal_stft_matches_whole_clip_frames():
    """The demo draws its output spectrogram with this, so it must frame a
    stream exactly as STFT.forward frames the finished clip."""
    stft = _build(False).stft
    x = torch.randn(8000 + 13, generator=torch.Generator().manual_seed(2))
    whole = stft(x[None])[0]
    n_frames = whole.shape[-1]
    c, parts, pos = CausalSTFT(stft), [], 0
    for n in [5, 300, 64, 1000] * 20:
        parts.append(c.push(x[pos:pos + n])[0]); pos += n
        if pos >= x.numel():
            break
    parts.append(c.push(torch.zeros(n_frames * stft.hop_length - x.numel()))[0])
    streamed = torch.cat(parts, dim=-1)
    assert streamed.shape == whole.shape
    assert (streamed - whole).abs().max() < 1e-4
