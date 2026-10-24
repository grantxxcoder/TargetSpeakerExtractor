"""The separator's building blocks, each written as a build function and a forward function.

A build function returns a plain nn.Module that only holds torch's own layers, so parameters(), state_dict(), .to() and .eval() all still work. The matching forward function runs it.
The attribute names on these holders are the checkpoint keys (for example subband_norm.blocks.0.0.norm.weight), so they must not be renamed. The trained models only load because the names match.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# this cuts a spectrogram into one piece per band, along the frequency axis
def split_bands(spectrogram, band_widths):
    """
    Slice and dice a spectrogram along frequency into per band tensors.

    I will be following the code from Luo & Yu, "Music Source Separation with Band-Split RNN", TASLP 2023.

    A complex (batch, frequency, frames) spectrogram gives a list of (batch, band width, frames) pieces.
    A real (batch, channels, frequency, frames) tensor gives a list of (batch, channels, band width, frames) pieces.
    """
    band_widths = list(band_widths) # remember that we have many bands, and each band has a different width in terms of the number of frequency bins.
    if spectrogram.dim() not in (3, 4):
        raise ValueError(f"Expected a 3D or 4D spectrogram, but got {spectrogram.dim()}D.")
    if spectrogram.shape[-2] != sum(band_widths):
        raise ValueError(f"The spectrogram has {spectrogram.shape[-2]} frequency bins, but the band plan covers {sum(band_widths)}.")
    bands = []
    start_bin = 0
    for band_width in band_widths:
        bands.append(spectrogram.narrow(-2, start_bin, band_width))
        start_bin += band_width
    return bands


# this builds a layer norm that normalises the channels of each frame on their own
def build_channel_norm(num_channels):
    """LayerNorm over the channel axis at each time step.

    Each frame is normalised using only its own channels, so nothing carries over between frames, and a 4 s training crop is normalised the same way as a 60 s live stream.
    WeSep calls this "cLN", but in Conv-TasNet (Luo & Mesgarani, 2019) cLN means cumulative layer norm, which is something else, so it is called channel-wise here.
    The LayerNorm sits one level down (block.norm) so the checkpoint key stays "...norm.norm.weight", the name the trained models were saved under.
    """
    block = nn.Module()
    block.norm = nn.LayerNorm(num_channels)
    return block


# this runs the channel norm on features shaped (batch, channels, frames)
def channel_norm_forward(block, features):
    return block.norm(features.transpose(1, -1)).transpose(1, -1)


# this builds one norm and one projection per band, so every band reaches the RNN at the same size
def build_subband_norm(band_widths, in_channels, feature_dim):
    """Per-band normalisation, then a projection to a common feature size.

    Yu et al. (Interspeech 2023) give each band its own fully connected layer, which turns a band of any width into a feature vector of the same size, 128 for their 16 kHz model.
    This matters because speech energy falls by about 6 dB per octave. Normalising all 257 bins together would leave the high bands almost invisible, so a quiet 6 kHz band gets its own norm and reaches the RNN as strongly as a loud 200 Hz one.
    """
    subband_norm = nn.Module()
    subband_norm.band_widths = list(band_widths)
    subband_norm.feature_dim = feature_dim
    subband_norm.blocks = nn.ModuleList()
    for band_width in subband_norm.band_widths:
        input_size = band_width * in_channels
        norm = build_channel_norm(input_size)
        subband_norm.blocks.append(nn.ModuleList([norm, nn.Conv1d(input_size, feature_dim, 1)]))
    return subband_norm


# this turns the list of band pieces into one tensor shaped (batch, bands, features, frames)
def subband_norm_forward(subband_norm, bands):
    band_features = []
    for (norm, projection), band in zip(subband_norm.blocks, bands):
        batch_size, channels, band_width, num_frames = band.shape
        # reshape rather than view, because split_bands uses narrow(), which leaves the band non-contiguous in memory
        flattened = band.reshape(batch_size, channels * band_width, num_frames)
        band_features.append(projection(channel_norm_forward(norm, flattened)))
    return torch.stack(band_features, dim=1)


# this builds one residual LSTM block: normalise, run the LSTM, project back, then add the input back on
def build_res_rnn(feature_dim, hidden_dim, bidirectional=False):
    """A residual LSTM block (Luo & Yu, TASLP 2023), with the residual connection of He et al. (2016).

    Yu et al. (Interspeech 2023, section 4.2) use a 192-unit LSTM in a stack of six.
    The same block runs along time or across bands, depending on how the caller reshapes its input.
    It must be one-directional along time, because a backward pass would read the future, and it should be bidirectional across bands, because frequency has no past or future.
    """
    block = nn.Module()
    block.norm = build_channel_norm(feature_dim)
    block.rnn = nn.LSTM(feature_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=bidirectional)
    # A bidirectional LSTM gives twice as many outputs, so the projection back to feature_dim has to take both directions.
    block.proj = nn.Linear(hidden_dim * (2 if bidirectional else 1), feature_dim)
    return block


# this runs one residual LSTM block on features shaped (batch, features, sequence length) and returns the same shape
def res_rnn_forward(block, features):
    # Normalising first keeps the residual path itself untouched.
    hidden = channel_norm_forward(block.norm, features)
    # The LSTM wants (batch, sequence length, features).
    hidden = hidden.transpose(1, 2)
    hidden, final_state = block.rnn(hidden)
    hidden = block.proj(hidden)
    # Add the input back on, the residual connection.
    return features + hidden.transpose(1, 2)


# this builds one band-and-sequence block: an LSTM along time, then an LSTM across the bands
def build_bsnet(feature_dim, hidden_dim):
    """One band-and-sequence block (Luo & Yu, TASLP 2023). Yu et al. (Interspeech 2023, section 4.2) stack six of them.

    WeSep calls the two parts band_rnn and band_comm. time_rnn and band_rnn say more clearly which axis each one runs along.
    """
    block = nn.Module()
    # Along time the LSTM runs forward only, because a backward pass would read the future.
    block.time_rnn = build_res_rnn(feature_dim, hidden_dim, bidirectional=False)
    # Across bands it is always bidirectional, even in the causal model, because all the bands of a frame arrive together.
    block.band_rnn = build_res_rnn(feature_dim, hidden_dim, bidirectional=True)
    return block


# this runs one band-and-sequence block on features shaped (batch, bands, features, frames) and returns the same shape
def bsnet_forward(block, features):
    batch_size, num_bands, feature_dim, num_frames = features.shape

    # Along time, every (clip, band) pair is its own sequence. The batch and band axes are already next to each other, so a plain reshape is enough.
    time_input = features.reshape(batch_size * num_bands, feature_dim, num_frames)
    time_output = res_rnn_forward(block.time_rnn, time_input).reshape(batch_size, num_bands, feature_dim, num_frames)

    # Across bands, every (clip, frame) pair is its own sequence. The frame axis has to be moved next to the batch axis first,
    # because reshaping without the permute still runs but mixes bands up with frames.
    band_input = time_output.permute(0, 3, 2, 1).reshape(batch_size * num_frames, feature_dim, num_bands)
    band_output = res_rnn_forward(block.band_rnn, band_input)
    return band_output.reshape(batch_size, num_frames, feature_dim, num_bands).permute(0, 3, 2, 1)


# this builds the stack of band-and-sequence blocks that does the separating (Yu et al., Interspeech 2023, section 4.2)
def build_band_sequence_model(feature_dim, hidden_dim, num_repeat=6):
    separator = nn.Module()
    separator.blocks = nn.ModuleList()
    for block_index in range(num_repeat):
        separator.blocks.append(build_bsnet(feature_dim, hidden_dim))
    return separator


# this runs the features through every block of the stack in turn
def band_sequence_model_forward(separator, features):
    for block in separator.blocks:
        features = bsnet_forward(block, features)
    return features


# this builds the per-band heads that predict a complex mask and a residual spectrogram
def build_estimator(band_widths, feature_dim, mlp_hidden):
    """Per-band complex mask plus a residual spectrogram (Yu et al., Interspeech 2023, section 2).

    Each band gets a small MLP that predicts a complex mask M. A second head predicts a residual spectrogram R, to repair the artefacts a mask alone leaves, so the estimate is S = M times X plus R (their equation 2).
    The hidden width of 384 is from the paper. Each trunk has one hidden layer, where the WeSep reference uses two (Wang et al., 2024; decisions-m1.md 2026-08-19). The mask uses a GLU (Dauphin et al., 2017).
    The WeSep reference leaves the residual head out. Keeping it here is deliberate.
    """
    estimator = nn.Module()
    estimator.band_widths = list(band_widths)
    estimator.trunks = nn.ModuleList()
    estimator.mask_heads = nn.ModuleList()
    estimator.res_heads = nn.ModuleList()  # the additional head that predicts the residual spectrogram R

    for band_width in estimator.band_widths:
        # Each trunk is the norm, then one convolution and a tanh.
        estimator.trunks.append(nn.ModuleList([build_channel_norm(feature_dim), nn.Conv1d(feature_dim, mlp_hidden, 1), nn.Tanh()]))
        # Four outputs per bin: a value and a gate, each with a real and an imaginary part.
        estimator.mask_heads.append(nn.Conv1d(mlp_hidden, 4 * band_width, 1))
        estimator.res_heads.append(nn.Conv1d(mlp_hidden, 2 * band_width, 1))
    return estimator


# use this method to actually move the features through the model and get the output complex spectrogram
def estimator_forward(estimator, features, mixture_bands):
    """Turn the separator's features, shaped (batch, bands, features, frames), into the estimated complex spectrogram, shaped (batch, frequency, frames).

    mixture_bands is the list of complex mixture pieces from split_bands, which the mask is applied to.
    """
    masks = []
    residuals = []
    for band_index, band_width in enumerate(estimator.band_widths):
        trunk = estimator.trunks[band_index]
        hidden = channel_norm_forward(trunk[0], features[:, band_index])
        for layer in trunk[1:]:
            hidden = layer(hidden)
        batch_size, hidden_size, num_frames = hidden.shape

        # need to do the GLU activation here to BOUND an unstable mask
        mask_values = estimator.mask_heads[band_index](hidden)
        masks.append(F.glu(mask_values, dim=1).reshape(batch_size, 2, band_width, num_frames))
        residuals.append(estimator.res_heads[band_index](hidden).reshape(batch_size, 2, band_width, num_frames))

    mask = torch.cat(masks, dim=2)
    mixture = torch.cat(mixture_bands, dim=-2)
    mask_real, mask_imag = mask[:, 0], mask[:, 1]
    mixture_real, mixture_imag = mixture.real, mixture.imag

    # Apply the mask by complex multiplication, S = M times X, then add the residual R.
    estimate_real = mixture_real * mask_real - mixture_imag * mask_imag
    estimate_imag = mixture_real * mask_imag + mixture_imag * mask_real
    residual = torch.cat(residuals, dim=2)
    estimate_real, estimate_imag = estimate_real + residual[:, 0], estimate_imag + residual[:, 1]
    return torch.complex(estimate_real, estimate_imag)
