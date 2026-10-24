"""Running the extractor chunk by chunk, for the live demo.

extract() takes a whole clip. Every stage inside it works one STFT frame at a time except the time-axis LSTMs, so a stream only has to carry three things from one chunk to the next:
the last window-minus-hop input samples, each separator block's LSTM state, and the tail of the overlap-add that is not final yet.
With those carried, the streamed output equals the whole-clip output up to float rounding, so the demo runs exactly the model that was evaluated.

The stream's state is a plain dictionary, made by stream_start and updated in place by stream_push and stream_flush.
Streaming STFT and weighted overlap-add follow Allen & Rabiner, Proc. IEEE 1977.

Claude Code used to generate this file (Hard).
"""

import torch

from src.models.bsrnn import fuse_context
from src.models.conditioning import tfmap
from src.models.modules import channel_norm_forward, estimator_forward, res_rnn_forward, split_bands, subband_norm_forward
from src.models.stft import stft_forward


# this runs one band-and-sequence block on new frames, carrying the time LSTM's state in and out
def bsnet_step(block, features, state):
    """The same as modules.bsnet_forward, except the time LSTM starts from state and returns its new state. The band LSTM works within a frame, so it needs no state."""
    batch_size, num_bands, feature_dim, num_frames = features.shape
    time_rnn = block.time_rnn
    time_input = features.reshape(batch_size * num_bands, feature_dim, num_frames)
    hidden, state = time_rnn.rnn(channel_norm_forward(time_rnn.norm, time_input).transpose(1, 2), state)
    time_output = (time_input + time_rnn.proj(hidden).transpose(1, 2)).reshape(batch_size, num_bands, feature_dim, num_frames)
    band_input = time_output.permute(0, 3, 2, 1).reshape(batch_size * num_frames, feature_dim, num_bands)
    band_output = res_rnn_forward(block.band_rnn, band_input)
    return band_output.reshape(batch_size, num_frames, feature_dim, num_bands).permute(0, 3, 2, 1), state


# this starts a chunk-by-chunk STFT that frames the audio exactly as stft_forward frames a whole clip
def causal_stft_start(stft):
    """The extractor uses it for its input, and the demo uses it to draw the spectrogram of the audio actually played out."""
    device = stft.window.device
    return {"n_fft": stft.n_fft, "hop": stft.hop_length, "padding": stft.padding, "window": stft.window, "left_context": torch.zeros(stft.padding, device=device), "pending_samples": torch.zeros(0, device=device)}


# this adds new audio and returns the frames that are now complete, shaped (1, frequency, new frames)
@torch.no_grad()
def causal_stft_push(analysis, samples):
    window = analysis["window"]
    hop = analysis["hop"]
    new_samples = torch.as_tensor(samples, dtype=torch.float32, device=window.device).reshape(-1)
    buffer = torch.cat([analysis["pending_samples"], new_samples])
    num_frames = buffer.numel() // hop
    analysis["pending_samples"] = buffer[num_frames * hop:]
    if num_frames == 0:
        return torch.zeros(1, analysis["n_fft"] // 2 + 1, 0, dtype=torch.complex64, device=window.device)
    # The left context from the previous chunk plus the complete hops makes exactly num_frames full frames.
    segment = torch.cat([analysis["left_context"], buffer[:num_frames * hop]])
    analysis["left_context"] = segment[-analysis["padding"]:]
    return torch.stft(segment[None], analysis["n_fft"], hop, analysis["n_fft"], window, center=False, return_complex=True)


# this starts a stream for one enrolled speaker. enrol_embedding is None for the baseline
def stream_start(model, enrollment, enrol_embedding):
    """Push audio in chunks of any size with stream_push, then call stream_flush for the rest.

    The output lags the input by one window minus one hop (24 ms at 512/128), plus whatever part of a hop is still waiting.
    The enrolment's spectrogram is computed once, here, as a deployed system would.
    """
    if model.training:
        raise ValueError("Call model.eval() before streaming.")
    stft = model.stft
    device = stft.window.device
    enrollment = torch.as_tensor(enrollment, dtype=torch.float32, device=device).reshape(1, -1)
    with torch.no_grad():
        enrollment_magnitude = stft_forward(stft, enrollment).abs()
    return {
        "model": model,
        "n_fft": stft.n_fft,
        "hop": stft.hop_length,
        "padding": stft.padding,
        "window": stft.window,
        "device": device,
        "enrollment_magnitude": enrollment_magnitude,
        "enrol_embedding": enrol_embedding,
        "analysis": causal_stft_start(stft),
        "lstm_states": [None] * len(model.separator.blocks),
        # The overlap-add tail and its window envelope, still waiting for later frames.
        "overlap_tail": torch.zeros(stft.n_fft - stft.hop_length, device=device),
        "envelope_tail": torch.zeros(stft.n_fft - stft.hop_length, device=device),
        # The left padding still to be dropped from the front of the output.
        "samples_to_drop": stft.padding,
        "samples_in": 0,
        "samples_out": 0,
    }


# this adds new audio and returns the output samples that are now final, plus the mixture spectrogram of the new frames for the demo to draw
@torch.no_grad()
def stream_push(stream, samples):
    new_samples = torch.as_tensor(samples, dtype=torch.float32, device=stream["device"]).reshape(-1)
    stream["samples_in"] += new_samples.numel()
    mixture_frames = causal_stft_push(stream["analysis"], new_samples)
    num_frames = mixture_frames.shape[-1]
    if num_frames == 0:
        empty = mixture_frames[0].abs().cpu().numpy()
        return {"audio": empty[0, :0], "mix_mag": empty}
    estimate_frames = stream_frames(stream, mixture_frames)
    audio = stream_overlap_add(stream, estimate_frames, num_frames)
    return {"audio": audio.cpu().numpy(), "mix_mag": mixture_frames[0].abs().cpu().numpy()}


# this pads the end with zeros to the frame count the whole-clip STFT uses and returns the last output, so the stream emits exactly as many samples as went in
@torch.no_grad()
def stream_flush(stream):
    samples_in = stream["samples_in"]
    samples_out_before = stream["samples_out"]
    # The same frame count as stft_frame_count, written as a ceiling division.
    num_frames = -(-(samples_in + stream["padding"]) // stream["hop"])
    output = stream_push(stream, torch.zeros(num_frames * stream["hop"] - samples_in, device=stream["device"]))
    output["audio"] = output["audio"][:samples_in - samples_out_before]
    stream["samples_in"] = samples_in
    stream["samples_out"] = samples_in
    return output


# this runs the model from the STFT frames to the estimated frames, the same steps as extract(), for the new frames only
def stream_frames(stream, mixture_frames):
    model = stream["model"]
    cue = tfmap(mixture_frames.abs(), stream["enrollment_magnitude"], model.tfmap_scale, model.tfmap_parts, model.tfmap_part_scales)
    network_input = torch.cat([torch.stack([mixture_frames.real, mixture_frames.imag], dim=1), cue], dim=1)
    mixture_bands = split_bands(mixture_frames, model.band_widths)
    input_bands = split_bands(network_input, model.band_widths)
    features = fuse_context(model, subband_norm_forward(model.subband_norm, input_bands), stream["enrol_embedding"])
    for block_index, block in enumerate(model.separator.blocks):
        features, stream["lstm_states"][block_index] = bsnet_step(block, features, stream["lstm_states"][block_index])
    return estimator_forward(model.estimator, features, mixture_bands)


# this does stft_inverse a little at a time: samples before the next frame starts get no more contributions, so they are final
def stream_overlap_add(stream, estimate_frames, num_frames):
    n_fft = stream["n_fft"]
    hop = stream["hop"]
    window = stream["window"]
    frames = torch.fft.irfft(estimate_frames[0], n=n_fft, dim=0) * window[:, None]
    length = (num_frames - 1) * hop + n_fft
    summed = torch.zeros(length, device=stream["device"])
    envelope = torch.zeros(length, device=stream["device"])
    summed[:stream["overlap_tail"].numel()] += stream["overlap_tail"]
    envelope[:stream["envelope_tail"].numel()] += stream["envelope_tail"]
    squared_window = window.pow(2)
    for frame_index in range(num_frames):
        start = frame_index * hop
        summed[start:start + n_fft] += frames[:, frame_index]
        envelope[start:start + n_fft] += squared_window
    finished = num_frames * hop
    stream["overlap_tail"] = summed[finished:]
    stream["envelope_tail"] = envelope[finished:]
    output = summed[:finished] / envelope[:finished].clamp_min(1e-11)
    drop = min(stream["samples_to_drop"], output.numel())
    stream["samples_to_drop"] -= drop
    output = output[drop:]
    stream["samples_out"] += output.numel()
    return output
