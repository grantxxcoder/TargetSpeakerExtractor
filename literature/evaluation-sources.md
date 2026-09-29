# Sources for lit review §"How extraction is evaluated"

Compiled 2026-09-29. Every row verified by fetching its abstract page (numbers
from full text where noted). `[pre]` = preprint/tech report. `[unv]` = lead only.
Already in `references.bib`: Vincent 2006, Le Roux 2019, Zmolikova 2023,
Reddy 2021/2022, ITU P.835/P.808, REAL-TSE (Wang 2026), Fela & Mowlaee 2026.

## 1. Signal metrics (SDR / SI-SDR) vs recognition and listeners

| Cite | Venue | ID | Finding |
|---|---|---|---|
| Iwamoto et al. 2022 | Interspeech, 5418–22 | 10.21437/Interspeech.2022-318 | Artefacts, not residual noise, cause the WER damage; adding back some mixture lowers SDR but cuts WER ~20 % rel. |
| Ochiai et al. 2024 | TASLP 32:3589–3602 | arXiv:2404.14860 | 2 talkers + noise: leftover interferer barely moves WER. Artefact-aware loss: SDR 9.9→8.5 dB, WER 19.1→14.9 % |
| Chen et al. 2020 (LibriCSS) | ICASSP | arXiv:2001.11482 | "signal-based metrics have very weak correlations with ASR accuracy" → ASR protocol instead |
| Zhang et al. 2021 | WASPAA | arXiv:2110.14139 | Best simulated SDR (18.0 dB) → real WER 41.5 % vs 19.5 % unprocessed |
| Sato et al. 2021 | Interspeech, 1149–53 | arXiv:2106.00949 | SpeakerBeam cuts CER ~89 % at SIR 0 dB, raises it 6–75 % at SIR 20 dB |
| Torcoli et al. 2021 | TASLP 29 | arXiv:2110.11438 | 7 listening tests: (SI-)SDR/SAR in bottom third of measures |
| Wisdom et al. 2020 (MixIT) | NeurIPS | arXiv:2006.12701 | Thresholded SNR; SI-SNRi "not meaningful" when a reference is silent |
| Jepsen et al. 2025 | ASRU | arXiv:2508.14623 | Noisy references cap SI-SDR |
| de Oliveira et al. 2023 | ITG Speech Comm. | arXiv:2306.03014 | Reference metrics favour predictive, non-reference favour generative models |

## 2. Perceptual / non-intrusive metrics and gaming

| Cite | Venue | ID | Finding |
|---|---|---|---|
| Rix et al. 2001 (PESQ) | ICASSP | 10.1109/ICASSP.2001.941023 | origin |
| Taal et al. 2011 (STOI) | TASLP 19(7) | 10.1109/TASL.2011.2114881 | origin |
| Fu et al. 2019 / 2021 (MetricGAN / +) | ICML / Interspeech | arXiv:1905.04874 / 2104.03538 | Canonical train-on-the-metric |
| de Oliveira et al. 2024 (PESQetarian) | Interspeech | arXiv:2406.03460 | PESQ 3.82, rated below the noisy input; SI-SDR −19.8 dB |
| Close et al. 2024 | EUSIPCO | arXiv:2403.11732 | Optimising a learned quality predictor yields a consistent hallucinated artefact |
| Huang & Toda 2026 | SLT | arXiv:2606.31105 | UTMOS attackable; degraded audio keeps a high score |
| Leglaive et al. 2024 (CHiME-7 UDASE) | CSL | 10.1016/j.csl.2024.101685 | Real audio: DNSMOS-SIG vs listeners r = 0.18; DNSMOS-best system worst in listening |
| Zhang et al. 2025 (URGENT 2024 lessons) | Interspeech | arXiv:2506.01611 | DNSMOS lags UTMOS/SCOREQ in rank agreement with MOS |
| Saijo et al. 2025 (URGENT 2025) | Interspeech | arXiv:2505.23212 | Generative T13: 1st on DNSMOS (3.10), 21st on char. accuracy (67.87 %); P.808 "difficult to penalize the correctness of the spoken content" (full text) |
| Lopez-Espejo et al. 2023 | Speech Comm. 150 | 10.1016/j.specom.2023.04.001 | STOI-trained nets raise STOI, no intelligibility gain in listening tests |

## 3. Recogniser-based evaluation (WER)

| Cite | Venue | ID | Finding |
|---|---|---|---|
| de Oliveira et al. 2026 `[pre]` | arXiv | 2605.12107 | Rank agreement with human WER 0.43 (weak ASR) to 1.00 (strong); normalisation alone reorders systems in ~17 % of resamples |
| Chondhekar et al. 2025 `[pre]` | tech report | arXiv:2512.17562 | MetricGAN+ denoising worsened Whisper, Parakeet, **Gemini Flash 2.0**, Parrotlet in 40/40 configs (+1.1 to +46.6 pts semWER). Noise only, medical |
| Huo et al. 2026 `[pre]` | arXiv | 2607.11157 | Best enhancement strength is recogniser-specific (wav2vec2 α=.85, Whisper α=.25) |
| Gong et al. 2023 | Interspeech | arXiv:2307.03183 | Whisper's representations encode noise type (not noise-invariant) |
| Barański et al. 2025 | ICASSP | arXiv:2501.11378 | large-v3 emits text on 40.3 % of non-speech clips. **Resolves the author TODO on `whisper_hallucination_2025`** |
| Koenecke et al. 2024 | FAccT, 1672–81 | 10.1145/3630106.3658996 | ~1 % of transcripts contain invented phrases; 38 % of those harmful |
| Atwany et al. 2025 | Findings ACL | 10.18653/v1/2025.findings-acl.1190 | Low WER can hide hallucination |
| Kim et al. 2021/22 (SemDist) | Interspeech | arXiv:2104.02138 | Meaning-based distance tracks users better than WER |
| von Neumann et al. 2023 (MeetEval) | CHiME-7 | arXiv:2307.11394 | cpWER / tcpWER: speaker-attributed WER |
| Polok et al. 2025 (DiCoW) | ICASSP | arXiv:2409.09543 | Target-speaker ASR beats separation cascade by 12.9 pts (NOTSOFAR-1) |
| Liu & Peng 2020 | Interspeech, 596–600 | 10.21437/Interspeech.2020-1338 | Blockwise bootstrap for correlated utterances |
| Bisani & Ney 2004 | ICASSP | 10.1109/ICASSP.2004.1326009 | Bootstrap CIs for WER |

## 4. TSE-specific: wrong speaker, absent target, generative output

| Cite | Venue | ID | Finding |
|---|---|---|---|
| Zhang et al. 2020 (X-TaSNet) | Interspeech, 1421–25 | 10.21437/Interspeech.2020-1706 | Wrong-speaker rate = % trials with SI-SDRi < 0; 9.2 % vs 9.5 % human-rated |
| Zhao et al. 2022 | Interspeech, 5333–37 | arXiv:2204.01355 | Target confusion analysed in embedding space |
| Delcroix et al. 2022 | Interspeech, 216–20 | 10.21437/Interspeech.2022-11252 | SNR losses "ill-defined" for silent reference; baseline always outputs sound when target absent; fail = SDRi < 1 dB |
| Borsdorf et al. 2021 | Interspeech, 1469–73 | 10.21437/Interspeech.2021-1939 | Silence-aware SI-SDR; present-only model ≈ −184 dB on absent trials |
| Zhang et al. 2023 | Interspeech, 3714–18 | 10.21437/Interspeech.2023-655 | Silence-aware SI-SDR can't tell absence detection from turning the volume down |
| Eskimez et al. 2022 | ICASSP | arXiv:2110.09625 | Target over-suppression measure + ASR deletion rate |
| Liu et al. 2026 (EvoTSE) `[pre]` | arXiv | 2604.06810 | Wrong-speaker rate + SI-SDRi on correct trials only |
| Zeng et al. 2026 (LauraTSE) `[pre]` | arXiv | 2601.06006 | Generative: best NISQA, but ~2× discriminative WER-type error |
| Yang et al. 2026 | Interspeech (acc.) | arXiv:2602.15519 | No TSE model beats the unprocessed mixture's WER in hard real scenes |
| Ma et al. 2025 | IEEE SPL | arXiv:2501.14477 | Generative TSE alters or loses content |

## Unverified leads `[unv]`
Pirklbauer et al. 2023 (ITG, phoneme-distance metric) · Emiya et al. 2011 (PEASS) · Wang et al. SLT 2024 "rectifying target confusion" · Yang/Pandey/Wang CSL 2026 Whisper claim (arXiv abstract doesn't mention Whisper)
