"""How the model is told who the target speaker is.

tfmap builds the spectral cue from the enrolment, which both final models use. The context fusion multiplies a speaker embedding into the features, which only the extension uses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Keeps every division by a frame's length safe when the frame is silent.
EPSILON = 1e-8


# this builds the TF-Map cue: for every mixture frame, how much of it looks like the enrolled speaker
def tfmap(mix_mag, enroll_mag, scale, return_parts, part_scales):
    """Spectral-level speaker cue (Zhang et al., "Multi-Level Speaker Representation for Target Speaker Extraction", ICASSP 2025, section II-A, equations 1 and 2).

    Every enrolment frame is used directly as a basis vector, and each mixture frame weights them by cosine similarity. This is their Spectral Similarity variant, which needs no speaker encoder.
    It is causal, because each mixture frame only looks at the enrolment, which is complete before the stream starts.

    mix_mag is (batch, frequency, mixture frames) and enroll_mag is (batch, frequency, enrolment frames).
    The baseline gets one channel back, (batch, 1, frequency, mixture frames). With return_parts the extension gets three: the direction, the match fraction and the unmatched remainder, each multiplied by its scale in part_scales.
    """
    # Divide each frame by its own length, so only the shape of the spectrum is compared, not how loud it was.
    mix_shape = F.normalize(mix_mag, p=2, dim=1, eps=EPSILON)
    enrol_shape = F.normalize(enroll_mag, p=2, dim=1, eps=EPSILON)

    # Every mixture frame against every enrolment frame. Both sides are unit vectors, so each entry is a cosine.
    similarity = torch.matmul(mix_shape.transpose(1, 2), enrol_shape)

    # Cosines only span -1 to 1, so without the scale the softmax averages all the enrolment frames instead of picking the closest ones (decisions-m2.md 2026-08-25).
    enrol_weights = torch.softmax(similarity * scale, dim=-1)

    # Blend the enrolment frames with those weights, giving a fresh template for every mixture frame, then make it unit length again.
    template = torch.matmul(enrol_weights, enrol_shape.transpose(1, 2)).transpose(1, 2)
    direction = template / template.norm(dim=1, keepdim=True).clamp_min(EPSILON)

    # How far each mixture frame reaches along the template's direction. It uses the loud spectrum, so it mixes loudness with match.
    matched_level = (mix_mag * direction).sum(dim=1, keepdim=True)

    if not return_parts:
        # The baseline's cue: the match and the loudness fused into one channel.
        return (matched_level * direction).unsqueeze(1)

    # The extension hands over the parts instead, because the mask is a ratio and needs the match fraction on its own, not mixed with loudness (decisions-m2.md 2026-09-22).
    # The mixture frame splits exactly into the matched part plus an unmatched part at right angles to it, and the unmatched part shows where the other speaker sits.
    frame_loudness = mix_mag.norm(dim=1, keepdim=True).clamp_min(EPSILON)
    match_fraction = (matched_level / frame_loudness).expand(-1, mix_mag.shape[1], -1)
    unmatched = mix_mag - matched_level * direction
    parts = torch.stack([direction, match_fraction, unmatched], dim=1)

    # Each part is scaled so its size matches the real and imaginary channels it sits beside. The scales were derived on 2026-09-22 and are never trained.
    scale_values = []
    for part_scale in part_scales:
        scale_values.append(float(part_scale))
    scales = torch.tensor(scale_values)
    return parts * scales.to(device=parts.device, dtype=parts.dtype).view(1, -1, 1, 1)


# this builds the extension's fusion layer, which turns the speaker embedding into one gain per feature
def build_context_fusion(embedding_dim, feature_dim):
    """A speaker embedding scales the separator's features (Delcroix et al., "Improving speaker discrimination of target speech extraction with time-domain SpeakerBeam", ICASSP 2020).

    It is the same form as the gamma term of FiLM (Perez et al., AAAI 2018): gamma = W e + b, and the features become z times (1 + gamma), the same for every band and frame.
    Multiplying, rather than adding or concatenating, changes how strongly each feature is expressed for this speaker, and a later normalisation cannot remove it.
    W and b start at zero, so at the start the extension behaves exactly like its control, but it still learns because the gradient of W is not zero.
    """
    fusion = nn.Module()
    fusion.project = nn.Linear(embedding_dim, feature_dim)
    nn.init.zeros_(fusion.project.weight)
    nn.init.zeros_(fusion.project.bias)
    return fusion


# this scales features shaped (batch, bands, features, frames) by the gains the speaker embedding gives, and returns the same shape
def context_fusion_forward(fusion, features, enrol_embedding):
    if enrol_embedding.dim() != 2:
        raise ValueError(f"The speaker embedding must be shaped (batch, embedding), but it is {tuple(enrol_embedding.shape)}.")
    if enrol_embedding.shape[0] != features.shape[0]:
        raise ValueError(f"The features have a batch of {features.shape[0]}, but the embedding has a batch of {enrol_embedding.shape[0]}.")
    gamma = fusion.project(enrol_embedding)
    return features * (1.0 + gamma[:, None, :, None])
