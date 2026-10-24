"""The training objective both final models were trained with (decisions-m2.md 2026-08-20).

The argument order is (reference, output) throughout, the reverse of the usual (prediction, target), so keep it that way.
The reference is the clean target for the terms on crops where the target speaks, and the mixture for the silence term.
"""

import torch


# this gives the energy of each example: the sum of its squared samples, shaped (batch,)
def energy(signal):
    return signal.pow(2).sum(dim=-1)


# this is L_pres: the floored SI-SDR, used on crops where the target speaks. Lower is better, and -30 is perfect
def loss_target_present(s_target, s_output, present_floor):
    """CARTSE equation 1, changed so the floor scales with the projected target. A floor on the target's own energy let an amplified output score without limit.

    Only use it on crops where the target speaks. On a silent target the projection is 0/0 and the NaN would destroy every weight.
    """
    # alpha is the single volume knob that best explains the output as the target turned up or down.
    alpha = (s_output * s_target).sum(dim=-1, keepdim=True) / energy(s_target).unsqueeze(-1)
    s_projected = alpha * s_target
    numerator = energy(s_projected)
    error_energy = energy(s_output - s_projected)
    denominator = error_energy + present_floor * numerator
    return -10 * torch.log10((numerator + 1e-12) / (denominator + 1e-12))


# this is L_abs: how far the output was pushed below the mixture on crops where the target is silent. 0 means it did nothing, -30 means silent
def loss_target_absent(x_input, s_output, absent_floor):
    """CARTSE equation 2, divided by the mixture's energy so that 0 dB means "did nothing" on every crop. A positive value means the output got louder, which is a bug."""
    numerator = energy(s_output) + absent_floor * energy(x_input)
    denominator = energy(x_input)
    return 10 * torch.log10((numerator + 1e-12) / (denominator + 1e-12))


# this is L_gain: zero while the output's level is within gain_deadzone_db of the target's, and the excess in dB outside it
def loss_gain_match(s_target, s_output, gain_deadzone_db):
    """Only for crops where the target speaks. Nothing else in the objective stops the model muting itself, because L_pres ignores level and L_abs rewards silence (decisions-m2.md 2026-08-28).

    The small constant sits inside the square root rather than clamping the result, so a fully muted output still gets a gradient back up.
    """
    eps = 1e-12
    rms_output = (s_output.pow(2).mean(dim=-1) + eps).sqrt()
    rms_target = (s_target.pow(2).mean(dim=-1) + eps).sqrt()
    error_db = 20 * torch.log10(rms_output / rms_target)
    return (error_db.abs() - gain_deadzone_db).clamp_min(0.0)


# this is L_MR: the multi-resolution compressed-magnitude and complex L1 loss (Yu et al., Interspeech 2023, equation 3). 0 is perfect
def loss_multi_res_stft(s_target, s_output, window_lengths_ms, compression_exponent, sample_rate):
    """Only for crops where the target speaks. It prices spectral detail. The window lengths are in milliseconds."""
    summation = s_output.new_zeros(s_output.shape[0])
    for window_ms in window_lengths_ms:
        n_fft = int(round(window_ms * sample_rate / 1000))
        # A plain centred torch.stft, not the model's streaming STFT, so a latency change can never alter the loss.
        window = torch.hann_window(n_fft, device=s_output.device, dtype=s_output.dtype)
        S_target = torch.stft(s_target, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft, window=window, center=True, return_complex=True)
        S_output = torch.stft(s_output, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft, window=window, center=True, return_complex=True)

        # Not torch.abs(), whose gradient is infinite at zero, where silent bins sit.
        magnitude_target = (S_target.real.pow(2) + S_target.imag.pow(2) + 1e-8).sqrt()
        magnitude_output = (S_output.real.pow(2) + S_output.imag.pow(2) + 1e-8).sqrt()

        # Mean rather than sum, so this term does not swamp L_pres. Each example keeps its own value.
        compressed_magnitude = (magnitude_target.pow(compression_exponent) - magnitude_output.pow(compression_exponent)).abs().mean(dim=(-2, -1))
        complex_term = (S_target.real - S_output.real).abs().mean(dim=(-2, -1)) + (S_target.imag - S_output.imag).abs().mean(dim=(-2, -1))
        summation = summation + compressed_magnitude + complex_term
    return summation / len(window_lengths_ms)


# this is the full objective: (1 - w) times the speaking-crop terms plus w times the silence term. It returns the total and a dictionary of its parts for logging
def bsrnn_loss(s_target, s_output, x_input, crop_absent, loss_settings):
    """L = (1 - w) * mean over speaking crops of [L_pres + w_m * L_MR + w_g * L_gain] + w * mean over silent crops of L_abs.

    loss_settings is the dictionary from train.build_loss_settings. crop_absent must come from the loader, because about 6 % of crops from trials where the target speaks still land in their silence (decisions-m1.md 2026-08-18).
    """
    crop_absent = crop_absent.bool()
    present = ~crop_absent

    # The rows are selected before the loss is taken rather than masked afterwards, because NaN times 0 is still NaN in the backward pass.
    n_present = int(present.sum().item())
    n_absent = int(crop_absent.sum().item())

    nan = float("nan")
    total = s_output.new_zeros(())
    # The same keys every batch, so a batch with no silent crops leaves a gap in the curve rather than a missing column.
    parts = {"L_pres": nan, "L_MR": nan, "L_gain": nan, "L_abs": nan, "n_present": n_present, "n_absent": n_absent}

    absent_weight = loss_settings["absent_weight"]
    if n_present:
        target_present = s_target[present]
        output_present = s_output[present]
        if not bool((target_present.abs().amax(dim=-1) > 0).all()):
            raise ValueError("A crop flagged as speaking has a silent target. Use the loader's crop_absent, not the trial's label.")
        loss_present = loss_target_present(target_present, output_present, loss_settings["present_floor"]).mean()
        loss_mr = loss_multi_res_stft(target_present, output_present, loss_settings["window_lengths_ms"], loss_settings["compression_exponent"], loss_settings["sample_rate"]).mean()
        loss_gain = loss_gain_match(target_present, output_present, loss_settings["gain_deadzone_db"]).mean()
        total = total + (1 - absent_weight) * (loss_present + loss_settings["multi_resolution_weight"] * loss_mr + loss_settings["gain_weight"] * loss_gain)
        parts["L_pres"] = float(loss_present.detach())
        parts["L_MR"] = float(loss_mr.detach())
        parts["L_gain"] = float(loss_gain.detach())

    if n_absent:
        loss_absent = loss_target_absent(x_input[crop_absent], s_output[crop_absent], loss_settings["absent_floor"]).mean()
        total = total + absent_weight * loss_absent
        parts["L_abs"] = float(loss_absent.detach())

    parts["total"] = float(total.detach())
    return total, parts
