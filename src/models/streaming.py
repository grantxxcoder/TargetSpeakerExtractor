"""Stateful chunk-by-chunk inference for the causal BSRNN_TFMAP family.

`BSRNN_TFMAP.forward()` takes a whole clip. Every stage inside it works on one
STFT frame at a time EXCEPT the time-axis LSTM in each separator block, so a
stream has to carry exactly three things from one chunk to the next:

    1. the last n_fft - hop input samples   (the STFT's left context)
    2. each block's time-LSTM (h, c)        (the model's only memory)
    3. the overlap-add tail of the iSTFT    (samples not yet final)

With those carried, streamed output equals whole-clip output up to float
rounding (tests/test_streaming.py). That is the property a live demo needs: it
runs the model that was evaluated, not an approximation of it. Independent
chunks are NOT equivalent -- pass_a_test_case_through.py measured the seams.

The enrolment's spectrogram and speaker embedding are computed once, before
the first chunk, as a deployed system would.

Streaming STFT / weighted overlap-add: Allen & Rabiner, Proc. IEEE 1977.
The model itself: Luo & Yu, TASLP 2023; Yu et al., Interspeech 2023 (bsrnn.py).
"""

import torch


def _bsnet_step(block, x, state):
    """BSNet.forward with the time-LSTM's state passed in and returned.

    Mirrors modules.BSNet / ResRNN line for line; only `rnn(h)` becomes
    `rnn(h, state)`. The band LSTM runs within a frame, so it has no state.
    """
    B, K, N, T = x.shape
    rnn = block.time_rnn
    y = x.reshape(B * K, N, T)
    h, state = rnn.rnn(rnn.norm(y).transpose(1, 2), state)
    y = (y + rnn.proj(h).transpose(1, 2)).reshape(B, K, N, T)
    z = block.band_rnn(y.permute(0, 3, 2, 1).reshape(B * T, N, K))
    return z.reshape(B, T, N, K).permute(0, 3, 2, 1), state


class CausalSTFT:
    """STFT.forward, one chunk at a time: frames come out exactly as the
    whole-clip transform frames them (left pad n_fft - hop, center=False).

    Used for the extractor's input and, in the demo, to draw the spectrogram
    of the audio actually emitted -- not the model's pre-iSTFT estimate, which
    carries energy the overlap-add cancels.
    """

    def __init__(self, stft):
        self.n_fft, self.hop, self.pad = stft.n_fft, stft.hop_length, stft.padding
        self.window = stft.window
        self._left = torch.zeros(self.pad, device=self.window.device)
        self._pending = torch.zeros(0, device=self.window.device)

    @torch.no_grad()
    def push(self, samples):
        """1-D audio -> (1, F, m) complex, m = the frames now complete."""
        x = torch.as_tensor(samples, dtype=torch.float32,
                            device=self.window.device).reshape(-1)
        buf = torch.cat([self._pending, x])
        m = buf.numel() // self.hop
        self._pending = buf[m * self.hop:]
        if m == 0:
            return torch.zeros(1, self.n_fft // 2 + 1, 0, dtype=torch.complex64,
                               device=self.window.device)
        seg = torch.cat([self._left, buf[:m * self.hop]])   # (m-1)*hop + n_fft
        self._left = seg[-self.pad:]
        return torch.stft(seg[None], self.n_fft, self.hop, self.n_fft, self.window,
                          center=False, return_complex=True)


class StreamingExtractor:
    """Push audio in any chunk size; get back every output sample now final.

        stream = StreamingExtractor(model, enrollment, enrol_embedding)
        for chunk in mic:
            out = stream.push(chunk)      # dict: audio, mix_mag, est_mag
        tail = stream.flush()

    Output lags input by n_fft - hop samples (24 ms at 512/128) plus whatever
    part of a hop is still pending: sample s is final only once every frame
    overlapping it has been computed.
    """

    def __init__(self, model, enrollment, enrol_embedding=None):
        if model.training:
            raise ValueError("call model.eval() first")
        if model.lookahead_frames:
            raise NotImplementedError(
                "lookahead_frames > 0 needs a k-frame delay line; no checkpoint "
                "uses it (decisions-m1.md 2026-08-18)")
        if any(b.time_rnn.rnn.bidirectional for b in model.separator.blocks):
            raise ValueError("non-causal model: its time LSTM reads the future")
        self.model = model
        stft = model.stft
        self.n_fft, self.hop, self.pad = stft.n_fft, stft.hop_length, stft.padding
        self.window = stft.window
        self.device = stft.window.device
        enrollment = torch.as_tensor(enrollment, dtype=torch.float32,
                                     device=self.device).reshape(1, -1)
        with torch.no_grad():
            self.enrol_mag = stft(enrollment).abs()       # (1, F, Te), once
        self.context = enrol_embedding
        self.reset()

    def reset(self):
        z = lambda n: torch.zeros(n, device=self.device)  # noqa: E731
        self._analysis = CausalSTFT(self.model.stft)
        self._states = [None] * len(self.model.separator.blocks)
        self._ola = z(self.n_fft - self.hop)   # tail, padded coordinates
        self._env = z(self.n_fft - self.hop)
        self._skip = self.pad            # padded -> original offset still to drop
        self.n_in = 0
        self.n_out = 0

    @torch.no_grad()
    def push(self, samples):
        """1-D float audio in -> dict of numpy arrays out.

        audio    newly final output samples (may be empty)
        mix_mag  (F, m) |X| of the m frames this call computed
        est_mag  (F, m) |S|, the model's pre-iSTFT estimate for the same frames
        """
        x = torch.as_tensor(samples, dtype=torch.float32, device=self.device).reshape(-1)
        self.n_in += x.numel()
        X = self._analysis.push(x)                            # (1, F, m)
        m = X.shape[-1]
        if m == 0:
            empty = X[0].abs().cpu().numpy()
            return {"audio": empty[0, :0], "mix_mag": empty, "est_mag": empty}
        S = self._frames(X)
        audio = self._overlap_add(S, m)
        return {"audio": audio.cpu().numpy(),
                "mix_mag": X[0].abs().cpu().numpy(),
                "est_mag": S[0].abs().cpu().numpy()}

    @torch.no_grad()
    def flush(self):
        """Zero-pad to the frame count the offline STFT uses, emit the rest.

        After flush the total emitted equals the total pushed, sample for
        sample, which is what makes streamed and whole-clip output comparable.
        """
        n_real, before = self.n_in, self.n_out
        n_frames = -(-(n_real + self.pad) // self.hop)         # ceil, = STFT._n_frames
        out = self.push(torch.zeros(n_frames * self.hop - n_real, device=self.device))
        out["audio"] = out["audio"][:n_real - before]
        self.n_in = self.n_out = n_real
        return out

    def _frames(self, X):
        """forward() from the STFT to the estimate, for m frames at once."""
        m = self.model
        tf = m.tfmap(X.abs(), self.enrol_mag)
        feats_in = torch.cat([torch.stack([X.real, X.imag], dim=1), tf], dim=1)
        mix_bands, feat_bands = m.split(X), m.split(feats_in)
        cue = gates = None
        if m.tfmap_inject is not None:
            cue, gates = m.tfmap_inject(m.split(tf)), m.tfmap_inject.gates
        z = m._fuse_context(m.subband_norm(feat_bands), self.context)
        for i, block in enumerate(m.separator.blocks):
            if cue is not None:
                z = z + gates[i].view(1, -1, 1, 1) * cue
            z, self._states[i] = _bsnet_step(block, z, self._states[i])
        return m.estimator(z, mix_bands)

    def _overlap_add(self, S, m):
        """STFT.inverse, incrementally. Samples before the next frame's start
        will receive no further contributions, so they are final."""
        frames = torch.fft.irfft(S[0], n=self.n_fft, dim=0) * self.window[:, None]
        length = (m - 1) * self.hop + self.n_fft
        acc = torch.zeros(length, device=self.device)
        env = torch.zeros(length, device=self.device)
        acc[:self._ola.numel()] += self._ola
        env[:self._env.numel()] += self._env
        w2 = self.window.pow(2)
        for k in range(m):
            acc[k * self.hop:k * self.hop + self.n_fft] += frames[:, k]
            env[k * self.hop:k * self.hop + self.n_fft] += w2
        done = m * self.hop
        self._ola, self._env = acc[done:], env[done:]
        out = acc[:done] / env[:done].clamp_min(1e-11)
        drop = min(self._skip, out.numel())
        self._skip -= drop
        out = out[drop:]
        self.n_out += out.numel()
        return out
