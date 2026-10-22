# Online Target Speaker Extraction — Ranked Reading List

Compiled 5 August 2026. Ranked by importance for online/streaming TSE, low latency, a small on-device model, handling interruptions and interpretable STT output, anchored on the REAL-TSE Online track.

> **Read with the 2026-08-07 re-scope in mind.** Written under the original brief: the summaries still hold, but the *rankings* no longer match the project.
>
> - **On-device / small-model work is out of scope.** TF-MLPNet (#8) and the "highest-value gap" at the bottom are now background and named future work, not a build target. See `docs/decisions/decisions-m0.md`.
> - **Replicating the challenge baselines is dropped.** Read "the baseline you must replicate" as "the architecture borrowed as a well-characterised instrument."
> - **PS4 and Ma et al. moved up.** They are the closest prior art to the contribution, though listed in Tier 2. Read them first.
> - **"Interpretable STT output" is now a measured reference condition, not a goal.** The model outputs audio; text is a benchmark row. The TS-ASR literature (especially Ma et al., and USEF-TP's joint framing) shows *what a serious text path would look like*, and our deliberately cheap extractor-to-ASR cascade should be cited against it.

---

## Before starting: three findings that should shape plan

**1. The REAL-TSE Challenge is finished.** Submissions closed 25 June 2026, system reports 1 July 2026, and final rankings and an official overview paper are public. Track 1 (Online) winner: **CARTSE** (CyberAgent), score 3.25. Runner-up: **DeepSound** (SA-Mamba). Then SonicAGI / WasedaM (tied 3rd), smellycat, WHU_IASP, insta360, ChuEst, AIVOX, SHNU-TSE. **Download all 12 valid Track 1 system reports** from the rankings table at https://real-tse.github.io/challenge/#rankings. They are short and are the state of the art on this task.

**2. The top challenge reports *are* the state of the art for online TSE on real conversational audio**, but not because of their architectures. The organisers found the top Track 1 entries were nearly all **BSRNN-style backbones close to the provided baseline**, with gains from data simulation, real-data adaptation, pseudo-label generation and filtering, multi-objective loss design and latency control. The overview says architecture and training recipe have not been co-optimised for real-world TSE, and stronger backbones (TF-GridNet-style) likely have headroom *if* paired with an equally strong data pipeline. **For the supervisors' note 4** ("reimplement first, new architecture after"): a new architecture without the data pipeline will lose to a conventional one with it.

**3. The two current papers.** Žmolíková et al. is a survey: essential for orientation, but not "the state of the art", and it predates everything that matters for the online track. TF-MLPNet *is* state of the art for its niche (real-time on a hearable-class NPU) and the best paper for the "small enough to fit on a small device" leg. The spec cites it as Interspeech 2025; it is the **6th Clarity Workshop (Clarity 2025)**. Fix this before the proposal is marked. Neither paper covers real conversational TSE, the biggest part of the problem.

**Caution:** several teams badly over-optimised DNSMOS-OVRL (one with adversarial waveform perturbations), so the organisers switched the official metric to DNSMOS-P808 post hoc. OVRL's correlation with human MOS on Track 1 was essentially zero (LCC +0.003). **Build gaming-resistance into the live-model metric from the start.** That alone is a defensible contribution and fits the supervisor's note 1 (the metric matters more than the score).

---

# 1. SLT 2026 REAL-TSE Challenge: Real-world Target Speaker Extraction from Conversational Recordings (2026)

**Citation:** S. Wang, Z. Qian, K. Zhang, J. Han, Z. Liu, X. Yu, H. Li, M. Delcroix, K. Yu, L. Xie, M. Li, H. Li, "SLT 2026 REAL-TSE Challenge: Real-world Target Speaker Extraction from Conversational Recordings," arXiv:2607.15198, Jul. 2026.
**Link:** https://arxiv.org/abs/2607.15198 · HTML: https://arxiv.org/html/2607.15198v1 · PDF: https://arxiv.org/pdf/2607.15198v1
**Also:** official challenge report PDF at https://real-tse.github.io/assets/pdf/Real_TSE_Challenge_Report.pdf

**One-sentence summary:** The official overview of the challenge anchoring the project: task, data, baselines, metrics, an analysis of all 24 submitted systems, and the organisers' lessons.

**Problem it addresses:** TSE benchmarks are mostly simulated (LibriMix, WSJ0-2mix) and miss what breaks real systems: real reverberation, uncontrolled loudness, ambient noise, turn-taking, reactive overlap, disfluencies, and enrollment/mixture device mismatch.

**Method (high level):** Real Mandarin + English conversational recordings. DEV (1,991 pairs, from REAL-T) and EVAL-1 (2,000 pairs, seen corpora) come from AISHELL-4, AliMeeting, AMI, DipCo and CHiME-6. EVAL-2 (3,000 pairs) is newly recorded in meeting rooms, cafés, homes and vehicles, simultaneously on two high-quality mics (H1/H2), a phone and headsets: a 3×5 mixture-device × enrollment-device matrix. Two tracks: Online (≤100 ms end-to-end algorithmic latency, checked by a perturbation-based response-delay test) and Offline. Four metrics, combined by averaging dense ranks: TER (Zipformer ASR, WER for EN / CER for ZH), SpkSim (WeSpeaker ResNet-34 cosine to *enrollment*), DNSMOS-P808 and target-activity F1 (FireRedVAD). No official training set; open data is allowed except a blacklist of the source corpora's dev/test splits.

**Key results:** Mean overlap ratio ~0.48–0.53, mean target-activity ratio ~0.73–0.75, mixtures ~17–18 s. Causal baselines score around TER 0.81 / SIM 0.37–0.39 on EVAL-2; the winning online system reached TER 0.699 / SIM 0.504. The **mixture** device matters far more than the enrollment device (far-field H2 is consistently worst). Enrollment–mixture channel mismatch mainly hurts SpkSim, plausibly because the metric uses enrollment as reference. Coarse scenario labels (café vs meeting vs car) do *not* reliably predict difficulty. Real-data adaptation consistently beat synthetic-only training; the top three teams all trained on real recordings.

**Limitations / what it doesn't solve:** An overview, so no method is reproducible from it; use the system reports. Single-channel only. No compute/parameter/energy budget, so "runs on-device" is *not* measured (only latency). All metrics are automatic and, as the paper documents, gameable. DEV/EVAL data needs team registration, now closed.

**Relevance to my project:** Problem statement, evaluation pipeline and literature map in one. Of the four project goals it measures latency, not model size, interruptions only indirectly (activity F1), and downstream live-model intelligibility not at all. The project's metric fills that last gap.

**Could I combine this with:** Everything downstream: its "lessons" section with PS4 and CARTSE for the training pipeline, and TF-MLPNet for the missing on-device axis.

**Reproducibility:** Strong. Official repos: https://github.com/REAL-TSE/REAL-TSE-Challenge (inference + scoring + latency script) and https://github.com/REAL-TSE/wesep-real-tse (baseline training). Checkpoints and DEV/EVAL data went by email to registered teams; registration is closed. **Email realtse.challenge@gmail.com early**, as a masters student. Fallback: reproduce the baselines from the code on Libri2Mix-100 and evaluate on self-built simulated + REAL-T-style data.

---

# 2. CARTSE Submission to the REAL-TSE Challenge, Track 1: Online Target Speaker Extraction (2026)

**Citation:** L. Li, S. Seki, "CARTSE Submission to the REAL-TSE Challenge, Track 1: Online Target Speaker Extraction," REAL-TSE Challenge system description, CyberAgent Inc., 2026.
**Link:** https://real-tse.github.io/assets/pdf/CARTSE-Track1.pdf

**One-sentence summary:** The online winner: a causal BSRNN with TF-Map + frame-level ECAPA conditioning, initialised from the official causal baseline and lifted almost entirely by a pseudo-label training pipeline. Verified to stream at ~22.9 ms.

**Problem it addresses:** Real far-field mixtures have no clean per-speaker reference, the target is often silent for long stretches, and enrollment and mixture devices often differ in frequency response.

**Method (high level):** STFT (512 win / 128 hop), TF-Map (enrollment-magnitude × mixture-magnitude attention) concatenated to the RI spectrogram, band-split into 32 non-uniform sub-bands, 6× (bidirectional band-RNN + **unidirectional** temporal RNN) with cumulative layer norm, complex band mask, iSTFT. Bilingual ECAPA-TDNN speaker encoder (VoxCeleb2 + CN-Celeb) fine-tuned on real recordings; non-causal, since enrollment is available in advance. Training: (a) pre-train on on-the-fly simulated mixtures with ~38% target-absent examples (distractor enrollment p=0.35, noise-only p=0.05) and a split loss (masked SI-SDR when the target is present, push-to-silence when absent); (b) two-stage fine-tune on 7,710 real clips (37.9 h) whose targets are **pseudo-labels from their own offline Track-2 teacher**, kept only if ΔOVRL>0, OVRL≥2.2, VAD precision≥0.80 and TER<0.6; (c) scenario-aware loss split over target-active vs target-silent frames, plus frozen auxiliary losses (WeSpeaker cosine, log-mel L1, direct DNSMOS maximisation, multi-layer Zipformer feature matching). Also "channel-gap" augmentation: random RMS-preserving EQ curves (spectral tilt, cubic-spline EQ, Butterworth band limits) on the enrollment, so the conditioning path learns device invariance.

**Key results:** Over its own initialisation: EVAL TER 0.805 to 0.699, SIM 0.456 to 0.504, DNSMOS-OVRL 1.67 to 3.44, F1 0.836 to 0.848. Algorithmic latency 24 ms (window − hop), buffering 8 ms, measured effective future dependency 22.2–23.7 ms (mean 22.9 ms). On DEV, learned mixture–enrollment similarity was the strongest single predictor of WER (Spearman ρ = −0.52).

**Limitations / what it doesn't solve:** No parameter count or MAC budget. Not on-device (its baseline initialisation is ~27 M params, ~39 GMAC/s). Trains directly against DNSMOS, the metric gaming the organisers later criticised. Needs an offline teacher, so two systems. Interruption/turn-taking handling is implicit (target-absent training + scenario-aware loss) and never evaluated directly.

**Relevance to my project:** Reference implementation and ablation ladder. Nearly every component is separable: reproduce the causal baseline, then add target-absent simulation, pseudo-labels, channel-gap EQ and each auxiliary loss in turn, attributing the gain to each. That is a publishable replication study and satisfies note 5 (reimplement and match reported results).

**Could I combine this with:** TF-MLPNet or SA-Mamba: same training recipe, an on-device backbone. The most obvious high-value experiment. Also PS4, which independently uses differentiable proxy losses on real data.

**Reproducibility:** Medium-high in principle; no code. The architecture *is* the public wesep-real-tse baseline, data sources are listed in Appendix B, all hyperparameters and loss weights are given, and Appendix A specifies the channel-gap augmentation well enough to reimplement. The hard dependency is their offline teacher; a public offline TSE model or guided source separation could stand in.

---

# 3. Multi-Level Speaker Representation for Target Speaker Extraction (2025)

**Citation:** K. Zhang, J. Li, S. Wang, Y. Wei, Y. Wang, Y. Wang, H. Li, "Multi-Level Speaker Representation for Target Speaker Extraction," in *Proc. ICASSP 2025*. doi:10.1109/ICASSP49660.2025.10889409 · arXiv:2410.16059
**Link:** https://arxiv.org/abs/2410.16059 · PDF: https://arxiv.org/pdf/2410.16059

**One-sentence summary:** Introduces the TF-Map (spectral-level) + contextual + neural-embedding multi-level conditioning used by the strongest REAL-TSE baseline and the challenge winner.

**Problem it addresses:** One pre-trained speaker embedding squeezes the enrollment into a single vector, is prone to speaker confusion and generalises poorly to unseen conditions.

**Method (high level):** Three levels of reference cue: a raw spectral-level representation from the enrollment magnitude spectrogram (the "TF-Map"), a frame-level contextual representation, and a conventional neural speaker embedding, each injected at a suitable point in a BSRNN extractor.

**Key results:** The raw spectral-level feature drives most of the generalisation gain. In the REAL-TSE baselines, TF-Map + Context beats plain ECAPA embeddings on nearly every metric. The *causal* TF-Map variant (TER 0.652 DEV / 0.808 EVAL-2) even beats the non-causal embedding baseline on TER, which is unusual and useful for an online project.

**Limitations / what it doesn't solve:** A second spectrogram-sized input stream plus cross-attention costs parameters and compute, a problem on a microcontroller-class device. The original paper is mostly simulated benchmarks; real far-field gains are smaller.

**Relevance to my project:** Needed to understand the baseline you must replicate. Also the likeliest place for efficiency wins: TF-Map is conceptually cheap (an outer-product-style attention map) but built as a full-resolution extra channel, so it is a plausible compression target.

**Could I combine this with:** USEF-TSE attacks the same "don't rely on a single embedding" problem from the other side (cross-attention retrieval vs raw spectral features); comparing them is a clean thesis chapter. SA-Mamba's progressive speaker distillation is a cost-aware way to decide *where* to spend expensive frame-level conditioning.

**Reproducibility:** Good. Open arXiv preprint, and an implementation ships in WeSep / wesep-real-tse, so the baseline code can be read directly.

---

# 4. Music Source Separation with Band-Split RNN (2023)

**Citation:** Y. Luo, J. Yu, "Music Source Separation with Band-Split RNN," *IEEE/ACM Trans. Audio, Speech, Lang. Process.*, vol. 31, pp. 1893–1901 (also cited as 1215–1225), 2023. doi:10.1109/TASLP.2023.3250264
**Link:** https://arxiv.org/abs/2209.15174 (search "Band-Split RNN music source separation arXiv"; the TASLP version is paywalled)
**Companion to read with it:** J. Yu, Y. Luo, H. Chen, R. Gu, C. Weng, "High Fidelity Speech Enhancement with Band-Split RNN," arXiv:2212.00406. The speech-domain adaptation, closer to what will be built.

**One-sentence summary:** The dual-path band-split architecture (non-uniform sub-bands, alternating an across-band module with an across-time RNN). The de facto backbone of every strong REAL-TSE online system.

**Problem it addresses:** Full-band T-F models waste capacity on 257 bins with very different statistics; time-domain models lose explicit frequency structure. Band-splitting gives per-band normalisation and a much shorter sequence along frequency.

**Method (high level):** STFT; split the bins into K hand-designed sub-bands (narrow at low frequency where F0 and low harmonics live, wide at high frequency); per-band normalisation + linear projection to a fixed dimension; a stack of blocks, each with a band-axis (frequency) RNN and a time-axis RNN; per-band complex mask; iSTFT. Causality comes from making only the *time-axis* RNN unidirectional and switching to cumulative layer norm. The band-axis RNN can stay bidirectional because frequency is not causal.

**Key results:** State-of-the-art music source separation at publication. The speech-enhancement variant became the backbone of several DNS Challenge and TEA-PSE-lineage systems. In REAL-TSE, all four official baselines and the top online systems are BSRNN derivatives.

**Limitations / what it doesn't solve:** Band boundaries are hand-designed, not learned (SA-Mamba uses 36 bands, CARTSE 32, neither justified). RNNs are sequential and awkward to quantise or run on parallel NPUs. The models are large: the REAL-TSE causal baselines are 25–27 M parameters at ~39 GMAC/s, orders of magnitude beyond a hearable-class budget.

**Relevance to my project:** Essential background. Causality only matters on the time axis, and per-band normalisation keeps low-latency streaming stable; knowing this lets you reason about latency budgets. Learned or perceptually-motivated band splitting is an obvious, tractable thesis novelty.

**Could I combine this with:** TF-MLPNet (a parallel MLP-Mixer instead of the sequential bidirectional RNN, for the on-device reasons BSRNN ignores) and SA-Mamba (a fixed-state SSM instead of the temporal RNN). Both are "BSRNN-shaped, with the expensive part swapped out" and could be unified.

**Reproducibility:** Good. No official code, but several faithful community reimplementations exist, and the TSE-conditioned causal version is in wesep-real-tse.

---

# 5. Neural Target Speech Extraction: An Overview (2023)

**Citation:** K. Žmolíková, M. Delcroix, T. Ochiai, K. Kinoshita, J. Černocký, D. Yu, "Neural Target Speech Extraction: An Overview," *IEEE Signal Processing Magazine*, vol. 40, no. 3, pp. 8–29, 2023. doi:10.1109/MSP.2023.3240008
**Link:** IEEE Xplore (institutional access); search for an author-hosted PDF — Delcroix and Žmolíková both post preprints.

**One-sentence summary:** The canonical neural TSE survey: how clues (audio enrollment, visual, spatial, text) are encoded and fused, losses, datasets, and open problems as of 2023.

**Problem it addresses:** Orientation. It unifies TSE's scattered vocabulary (speaker extraction / personalised speech enhancement / target speech extraction / informed separation) into one taxonomy.

**Method (high level):** Not a method paper. It organises the field by clue type, fusion mechanism (concatenation, multiplication, FiLM, attention, adaptation-layer), domain (time vs time-frequency), training objective and evaluation, the axes the literature review needs.

**Key results:** No new results. Lasting contributions: the taxonomy, speaker confusion and target-absent false alarms as the field's characteristic failure modes, and real-recording generalisation as the key open problem, which REAL-TSE then benchmarked.

**Limitations / what it doesn't solve:** Three years old. Predates TF-GridNet-era backbones, Mamba/SSM sequence models, generative and flow-matching TSE, embedding-free cross-attention conditioning, and real-conversational benchmarks. Barely treats latency or on-device deployment, two of the four goals. Do not cite it as the current state of the art.

**Relevance to my project:** Background chapter, terminology and notation. Marc Delcroix is also a REAL-TSE organiser, and the survey's framing explains the challenge's metric choices.

**Could I combine this with:** Connects everything else on this list. Its bibliography leads to the pre-2023 classics: SpeakerBeam, VoiceFilter, SpEx/SpEx+, td-SpeakerBeam.

**Reproducibility:** N/A (survey). No code.

---

# 6. REAL-T: Real Conversational Mixtures for Target Speaker Extraction (2025)

**Citation:** S. Li, S. Wang, J. Han, K. Zhang, W. Wang, H. Li, "REAL-T: Real Conversational Mixtures for Target Speaker Extraction," in *Proc. Interspeech 2025*, pp. 1923–1927. doi:10.21437/Interspeech.2025-2662
**Link:** https://www.isca-archive.org/interspeech_2025/li25da_interspeech.pdf (open access) · project page https://real-tse.github.io/

**One-sentence summary:** The automated pipeline and dataset that turns five diarisation corpora into real-conversational TSE trials; the source of the REAL-TSE DEV and EVAL-1 sets.

**Problem it addresses:** Real-conversational TSE needs mixture/enrollment/target triples, and real recordings have no clean targets. REAL-T builds the trials automatically.

**Method (high level):** From AISHELL-4, AliMeeting, AMI, DipCo and CHiME-6, diarisation annotations locate naturally overlapping segments (mixtures) and non-overlapping single-speaker segments of ≥5 s from the same speaker (enrollment). With no clean target, evaluation uses reference-free and downstream metrics (ASR, speaker similarity).

**Key results:** Models trained on simulated fully-overlapped mixtures degrade sharply on real conversational trials; these numbers motivated the challenge.

**Limitations / what it doesn't solve:** Inherits every error in the source diarisation labels (the supervisors' note 3 on training artefacts from unclean data). No clean target, so evaluation is indirect: *why* the challenge scores TER/SpkSim/DNSMOS/F1, not SI-SDR. Two languages, mostly meeting/dinner-party acoustics; voice-assistant and in-car conditions only appear in the separately recorded EVAL-2.

**Relevance to my project:** How to build the project's own training and validation data without the closed registration. The pipeline is fully described, the source corpora are public, and the official training splits are allowed. Probably the most useful infrastructure to build in the first month.

**Could I combine this with:** PS4, which builds a larger (71,771-sample) corpus in the same spirit, with transcripts and frame-level VAD labels. CARTSE's quality filters, a recipe for cleaning a REAL-T-style corpus.

**Reproducibility:** Medium. Open-access paper; the dataset was still "to be released soon" on the project page. The pipeline can be rebuilt from the description and the source corpora (AMI and AISHELL-4 are free to download; CHiME-6 and AliMeeting need registration).

---

# 7. Exploring Time-Frequency Domain Target Speaker Extraction for Causal and Non-Causal Processing (2023)

**Citation:** W. Zhang, L. Yang, Y. Qian, "Exploring Time-Frequency Domain Target Speaker Extraction for Causal and Non-Causal Processing," in *Proc. IEEE ASRU 2023*, pp. 1–6.
**Link:** https://ieeexplore.ieee.org/document/10389752/ · author page with paper/slides/poster: https://sites.google.com/view/wangyou-zhang/publications

**One-sentence summary:** Ports a TF-GridNet-class separator into TSE, compares speaker-conditioning fusions including a new speaker-token fusion, and shows it works causally with strong performance.

**Problem it addresses:** Strong TSE work was then mostly time-domain. It was unclear whether the dominant T-F separators could be speaker-conditioned and made causal without collapsing.

**Method (high level):** Take the top T-F separation backbone. Compare concatenation with a proposed **speaker-token** fusion (the speaker representation is a token in the sequence, not a concatenated feature). Then make it causal and measure the cost.

**Key results:** Beats widely-used time-domain TSE models by a large margin, causal and non-causal, on WSJ0-2mix and LibriMix. Quantifies the causal-vs-non-causal gap for a T-F architecture, the number online-track design depends on.

**Limitations / what it doesn't solve:** Simulated benchmarks only: no real conversational data, device mismatch or target-absent regions. No latency accounting in the ICASSP-DNS sense (algorithmic vs buffering) and no compute or parameter budget, so "causal" does not mean "deployable."

**Relevance to my project:** The REAL-TSE overview names TF-GridNet variants as the *other* online-track family besides BSRNN, with possible headroom given a strong data pipeline. This paper is the entry point. Speaker-token fusion is under-explored and could be revisited under a latency budget.

**Could I combine this with:** CARTSE's training pipeline (the overview's own suggestion: strong backbone + strong data recipe). USEF-TSE, whose USEF-TFGridNet is the same backbone with cross-attention instead of tokens.

**Reproducibility:** Medium. IEEE-paywalled; slides and poster are on the author's site, and ESPnet has open implementations of the backbone. Wangyou Zhang maintains ESPnet-SE, so ESPnet's TSE recipes are likely the fastest route to a working causal T-F TSE baseline.

---

# 8. TF-MLPNet: Tiny Real-Time Neural Speech Separation (2025)

**Citation:** M. Itani, T. Chen, S. Gollakota, "TF-MLPNet: Tiny Real-Time Neural Speech Separation," in *Proc. 6th Clarity Workshop on Improving Speech-in-Noise for Hearing Devices (Clarity 2025)*. arXiv:2508.03047
**Link:** https://arxiv.org/abs/2508.03047 · PDF: https://arxiv.org/pdf/2508.03047 · https://www.isca-archive.org/clarity_2025/itani25_clarity.pdf
**Note:** the specification cites this as Interspeech 2025; it is Clarity 2025. Correct it.

**One-sentence summary:** The first separation/extraction network to run in real time on a hearable-class low-power neural accelerator (GAP9). It swaps the sequential parts of a dual-path T-F model for parallel-friendly components and trains with mixed-precision quantisation awareness.

**Problem it addresses:** State-of-the-art separation networks cannot run in real time on tiny low-power accelerators. The bottleneck is *sequential* structure, not just FLOPs; the paper profiles a dual-path model to show where the runtime goes.

**Method (high level):** Time-frequency domain, causal 3×3 conv encoder / transposed-conv decoder. Two key components: a **conv-batched LSTM** that infers a batch of LSTM inputs in parallel using convolutional layers, and an **all-MLP-Mixer** module (fully-connected layers alternating along channel and frequency) that replaces the sequential bidirectional LSTM along frequency. Convolutions handle the time axis of each frequency bin independently. Optional frequency compression. Loss: PIT + negative SI-SDR for blind separation, SI-SDR + a differentiable PESQ term for TSE. Deployed with mixed-precision quantisation-aware training and GAPFlow kernel compilation.

**Key results:** Processes 6 ms chunks in real time on GAP9, a 3.5–4× runtime reduction over prior separation models, and beats existing streaming models on blind separation and TSE. The mixed-precision quantised model loses only ~0.6 dB against full floating point.

**Limitations / what it doesn't solve:** Simulated benchmarks only: no device mismatch, long target-absent stretches or meeting reverberation. Speaker conditioning is simple and not the contribution. A short workshop paper with compressed details. *Not* entered in REAL-TSE, so it has no comparable number there; producing one is an open, valuable experiment.

**Relevance to my project:** The on-device leg, and the strongest evidence that sequential-vs-parallel structure, not parameter count, governs deployability. No REAL-TSE entry addresses "small enough to fit on a small device", so **evaluating a TF-MLPNet-class model on the REAL-TSE online protocol is an open question with a clear novelty claim.**

**Could I combine this with:** CARTSE (their training recipe, this backbone), the highest-value single experiment here. SA-Mamba: a fixed-size SSM state and an MLP-Mixer are rival answers to "remove the sequential bottleneck", never compared under a shared latency + hardware budget.

**Reproducibility:** Medium. The Gollakota lab (UW Mobile Intelligence Lab) reliably releases code: check https://github.com/vb000, the lab's TF-MLPNet project pages, and their prior "Look Once to Hear" and "Target Conversation Extraction" releases, which share infrastructure. GAP9 deployment needs the GreenWaves toolchain and hardware, probably unavailable; the PyTorch model and quantisation-aware training reproduce without it.

---

# 9. USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction (2025)

**Citation:** B. Zeng, M. Li, "USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction," *IEEE/ACM Trans. Audio, Speech, Lang. Process.*, 2025. arXiv:2409.02615
**Link:** https://arxiv.org/abs/2409.02615 · PDF: https://arxiv.org/pdf/2409.02615 · IEEE: https://ieeexplore.ieee.org/document/11012711/
**Companion:** B. Zeng, M. Li, "Universal Speaker Embedding Free Target Speaker Extraction and Personal Voice Activity Detection" (USEF-TP), arXiv:2501.03612. Joint TSE + personal VAD with a scenario-aware differentiated loss, evaluated on SparseLibriMix and CALLHOME.

**One-sentence summary:** Drops the pre-trained speaker-recognition model and retrieves frame-level target-speaker features from the enrollment with multi-head cross-attention, in a wrapper for any time-domain or T-F separation backbone.

**Problem it addresses:** A verification-trained embedding optimises the wrong objective, discards the enrollment's phonetic and contextual detail, and adds a second model to ship. Choosing the speaker encoder is itself an unprincipled hyperparameter.

**Method (high level):** One shared encoder for mixture and enrollment. Mixture encodings are queries, enrollment encodings keys and values; cross-attention gives a frame-level speaker-conditioned representation for any separator. Shown as USEF-SepFormer and USEF-TFGridNet. (Predecessor SEF-Net reverses the query/key roles and fuses inside the separator.)

**Key results:** State-of-the-art SI-SDR on WSJ0-2mix, WHAM! and WHAMR!; USEF-TFGridNet reports around 23.3 dB SI-SDRi. Holds up on more diverse out-of-domain data (LibriMix, ICASSP 2023 DNS Challenge blind test set). USEF-TP shows the same features support personal VAD jointly with extraction.

**Limitations / what it doesn't solve:** Per-frame cross-attention cost grows with enrollment length, a real problem for streaming with 9–11 s enrollments (the REAL-TSE mean is ~9–11 s). Not causal or latency-aware. Keys and values *can* be cached at enrollment time (SA-Mamba and SonicAGI both do), but the paper does not analyse streaming cost.

**Relevance to my project:** Firstly, it is the main alternative to the baseline's conditioning, so TF-Map vs cross-attention under a fixed 100 ms latency budget is a well-posed, tractable thesis experiment. Secondly, **USEF-TP's joint TSE + personal VAD framing is the cleanest existing handle on the "handles interruptions" goal**: target-activity F1 measures *when* the target speaks, and doing this jointly rather than post hoc is a defensible contribution.

**Could I combine this with:** SA-Mamba's progressive speaker distillation (cheap global condition in shallow layers, cross-attention retrieval only in deep ones) answers USEF-TSE's cost problem. Multi-Level Speaker Representation is the head-to-head comparison.

**Reproducibility:** Good. Code at https://github.com/ZBang/USEF-TSE (Bang Zeng, Duke Kunshan / Ming Li's group, who also maintain SEF-PNet and related releases). The arXiv version is open, and at least one REAL-TSE team (SonicAGI) used it as an offline backbone, so it rebuilds.

---

# 10. Speaker-Aware State Space Modeling for Streaming Target Speaker Extraction (SA-Mamba) (2026)

**Citation:** DeepSound team (B. Liu), "Speaker-Aware State Space Modeling for Streaming Target Speaker Extraction," REAL-TSE Challenge Track 1 system description, 2026.
**Link:** https://real-tse.github.io/assets/pdf/DeepSound-Track1.pdf

**One-sentence summary:** Second in the online track and the most architecturally novel online entry. The speaker condition goes into a selective state-space model's parameters, so target selectivity happens *inside* the recurrence, with fewer parameters and MACs than the baseline.

**Problem it addresses:** Conditioning is almost always external fusion *after* temporal modelling (concatenate, multiply, FiLM, attention). Mamba's selective parameters depend only on the current acoustic input, so the state never knows which speaker to track. Streaming also needs a fixed-size state, ruling out growing attention caches.

**Method (high level):** Complex-STFT domain (512 win / 256 hop, 257 bins), 36 non-uniform bands (eight 2-bin bands below 0.5 kHz, coarsening to 14 wide bands above 3 kHz), shared band-split frontend for mixture and enrollment. **SA-Mamba:** the SSM's input matrix B, output matrix C and discretisation step Δt are generated from the current acoustic features *and* a speaker condition, so writing, reading and forgetting in the state are all speaker-dependent. 36 dual-path blocks: bidirectional Mamba across bands (frequency is not causal) + causal SA-Mamba along time per band. **Progressive speaker distillation:** the first 30 blocks use a cheap precomputed global speaker condition for coarse target localisation; the last 6 use multi-head cross-attention over cached enrollment keys/values for detail reconstruction. No separate speaker-recognition model. Gated complex ratio mask, iSTFT overlap-add.

**Key results:** 15.89 M params / 30.01 GMAC/s vs the baselines' 25–27 M / ~39 GMAC/s, with EVAL TER 0.713, F1 0.849, SIM 0.524: better on every metric at ~60% of the parameters. Measured future dependency 44.5–49.9 ms (mean 46.7 ms). The ablations are the most useful part: layer-wise cross-attention beats embedding multiplication (SI-SNRi 9.23 to 11.25 dB), and SA-Mamba beats causal GRU by 3.11 dB and causal TCN by 5.37 dB SI-SNRi under matched conditions.

**Limitations / what it doesn't solve:** The ablations use their own simulated CN-Celeb/VoxCeleb1 two-speaker setup, *not* REAL-TSE data, so they do not transfer cleanly to the leaderboard number. 36 blocks is deep, and 15.89 M params is far too big for a hearable. No real-data pseudo-label adaptation, which the overview names as the dominant factor: arguably a strong architecture with a weak data recipe, the untested combination the organisers flag. Their own future work lists offline-teacher distillation. Mamba kernels are also awkward to quantise and deploy on fixed-point NPUs.

**Relevance to my project:** Evidence that architectural novelty *can* pay off online, and the closest thing to "a fundamentally new TSE architecture" for streaming. SA-Mamba's architecture + CARTSE's data pipeline is an experiment nobody has run, and both papers give enough detail to try it.

**Could I combine this with:** CARTSE (architecture × data recipe). TF-MLPNet (SSM fixed state vs MLP-Mixer parallelism under one hardware budget). USEF-TSE (its progressive distillation *is* a cost-managed USEF-TSE).

**Reproducibility:** Medium-low; say so in the review. No code or checkpoint. The system description is unusually detailed (band widths, d_state=16, d_conv=4, expand=2, 36 blocks, 4 attention heads, loss configuration, optimiser schedule), but the ablation dataset is bespoke and unreleased: the *architecture* can be reproduced, their *numbers* not. Mamba has good open implementations (mamba-ssm, plus SPMamba and SepMamba as speech-separation precedent).

---

# Tier 2 — read these next, or when you hit the specific problem

Supporting papers, not written up in full. Download several now.

**For the live-model / downstream-intelligibility metric (the novel contribution):**

- **PS4: Proxy-Supervised Joint Training for Real Target Speaker Extraction** (Ning et al., arXiv:2607.08111, 2026; https://arxiv.org/abs/2607.08111). YiJiaHe's 2nd-place offline system. A 71,771-sample real-conversational corpus with transcripts *and* frame-level VAD labels, used to fine-tune a BSRNN with four differentiable proxy objectives: ASR cross-entropy, speaker similarity, frame-level VAD and perceptual quality. **The closest published prior art to "optimise TSE for what a downstream model actually needs", and the most useful paper here for the metric contribution.** Read it right after the top 3.
- **Enhancing Intelligibility for Generative Target Speech Extraction via Joint Optimization with Target Speaker ASR** (Ma et al., arXiv:2501.14477, 2025; https://arxiv.org/abs/2501.14477). Generative TSE on Whisper, jointly optimised with target-speaker ASR for intelligibility *and* perceptual quality. Addresses the observed effect that TSE output can be more transcribable yet worse for a downstream speech model.
- **Listen only to me! How well can target speech extraction handle false alarms?** (Delcroix et al., Interspeech 2022). The foundational treatment of target-absent behaviour and false alarms, by a REAL-TSE organiser. Essential for the interruption goal and for why activity F1 is scored.

**Architecture / backbone alternatives:**

- **TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation** (Wang et al., TASLP 31:3221–3236, 2023, arXiv:2209.03952; https://arxiv.org/abs/2209.03952). The other backbone family in the overview; open code in ESPnet. For the "stronger backbone" route.
- **The SonicAGI System for the REAL-TSE Challenge** (arXiv:2607.11083, 2026; https://arxiv.org/abs/2607.11083). Joint 3rd, online track. **SwiftNet-Lookahead**: one bounded-lookahead module before a strictly causal iterative separator, 96 ms total system latency. The only entry that deliberately *spends* the full 100 ms budget rather than staying at 25 ms: the best latency/quality case study. Also uses a frozen offline enhancer for denoised auxiliary supervision on real targets.
- **StarTSE / Towards Streaming TSE via Chunk-wise Interleaved Splicing of an Autoregressive Language Model** (Peng et al., arXiv:2604.19635, 2026; https://arxiv.org/abs/2604.19635). First streaming AR generative backbone, RTF 0.248 on a 4090. Interesting, but a datacentre GPU is the opposite of on-device.
- **High Fidelity Speech Enhancement with Band-Split RNN** (Yu et al., arXiv:2212.00406). Speech-domain BSRNN; read with #4.
- **TEA-PSE 2.0 / 3.0** (Ju et al., SLT 2023 / ICASSP 2023). Sub-band real-time personalised speech enhancement: the industrial lineage of low-latency TSE-adjacent systems and the source of much practical sub-band engineering.
- **DSINet: Towards Real-Time Target Speaker Extraction with Dynamic Speaker Information Fusion** (Hao et al., ICASSP 2024). Dynamic speaker-information fusion for real-time TSE; one of the two papers behind the baseline's "Context" module.

**Infrastructure to use:**

- **WeSep: A Scalable and Flexible Toolkit Towards Generalizable Target Speaker Extraction** (Wang et al., Interspeech 2024, pp. 4273–4277; https://github.com/wenet-e2e/wesep). The challenge baselines are WeSep recipes. Read the paper; work mostly in `wesep-real-tse`.
- **ECAPA-TDNN** (Desplanques et al., Interspeech 2020) and **WeSpeaker** (Wang et al., ICASSP 2023): the speaker encoder and toolkit for conditioning *and* the official SpkSim metric. SpkSim is computed against the *enrollment*, not a clean reference, which limits what it can tell you.
- **Look Once to Hear: Target Speech Hearing with Noisy Examples** (Veluri, Itani, Chen, Yoshioka, Gollakota, CHI 2024, doi:10.1145/3613904.3642057). Real-time on-device target speech hearing with noisy enrollment, from the TF-MLPNet group. Noisy enrollment is the microphone-mismatch problem from another angle.

---

# Suggested reading order (different from the importance ranking)

The supervisors asked for 5–6 papers reviewed in depth, then reimplementation, then novelty. Mapped onto this list:

1. **Orient (week 1):** #5 Žmolíková survey, then #1 REAL-TSE overview.
2. **Understand the baseline you must replicate (weeks 2–3):** #4 BSRNN, then #3 Multi-Level Speaker Representation, with the `wesep-real-tse` code alongside. Reproduce `BSRNN_TFMAP_CAUSAL` on Libri2Mix-100 and confirm it reaches the published DEV numbers (note 5).
3. **Understand why the winner won (week 4):** #2 CARTSE, then PS4, then #6 REAL-T. Build the data pipeline; this is where the marks are.
4. **Pick the novelty axis (weeks 5–6):** #8 TF-MLPNet for on-device/efficiency; #10 SA-Mamba or #7 causal T-F TSE for architecture; #9 USEF-TSE/USEF-TP for conditioning + interruption handling. Note 7 says pick one, not all.
5. **Metric contribution, in parallel throughout:** PS4 and the Whisper joint-optimisation paper are the prior art. Make the metric hard to game; the challenge's DNSMOS failure is the justification.

## The one experiment I'd flag as the highest-value gap

> **Superseded 2026-08-07.** Written under the original brief, this points at the on-device leg, now out of scope (spec note 10 assumes server-class compute). Kept as the strongest future-work candidate, since the reasoning is sound. The current highest-value experiment is the leg-2 divergence table: see `docs/decisions/research-plan.md` §4.

The REAL-TSE overview says the top online systems used baseline-like backbones lifted by data pipelines, and stronger backbones probably have headroom with an equally strong pipeline. **No REAL-TSE entry reported a parameter or MAC budget near hearable-class hardware**: the smallest online system was 15.89 M params / 30 GMAC/s, against TF-MLPNet running 6 ms chunks on a GAP9. So: *take an on-device backbone (TF-MLPNet-class or a compressed SA-Mamba), train it with CARTSE-class real-data pseudo-label adaptation, and evaluate it on the REAL-TSE online protocol with a reported compute budget.* Nobody has done this, all ingredients are public, and it targets the project's goal of a sufficiently accurate low-latency system on on-device hardware.

---

## Checklist form

- [X] SLT 2026 REAL-TSE Challenge overview — arXiv:2607.15198
- [X] CARTSE Track-1 system report — real-tse.github.io/assets/pdf/CARTSE-Track1.pdf
- [X] Multi-Level Speaker Representation for TSE — arXiv:2410.16059
- [X] Music Source Separation with Band-Split RNN — TASLP 2023 / arXiv:2209.15174
- [ ] Neural Target Speech Extraction: An Overview — doi:10.1109/MSP.2023.3240008
- [ ] REAL-T: Real Conversational Mixtures for TSE — Interspeech 2025
- [ ] Exploring T-F Domain TSE for Causal and Non-Causal Processing — ASRU 2023
- [ ] TF-MLPNet: Tiny Real-Time Neural Speech Separation — arXiv:2508.03047
- [ ] USEF-TSE — arXiv:2409.02615 (+ USEF-TP, arXiv:2501.03612)
- [X] SA-Mamba / DeepSound Track-1 report — real-tse.github.io/assets/pdf/DeepSound-Track1.pdf
- [X] PS4: Proxy-Supervised Joint Training — arXiv:2607.08111
- [ ] SonicAGI / SwiftNet-Lookahead — arXiv:2607.11083
