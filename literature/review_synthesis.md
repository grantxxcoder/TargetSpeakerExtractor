# Online Target Speaker Extraction — Ranked Reading List

Compiled 5 August 2026. Ranked most → least important online/streaming TSE, low latency, small on-device model, handles interruptions, interpretable STT output, anchored on the REAL-TSE Online track.

> **Read with the 2026-08-07 re-scope in mind.** This list was written under
> the original brief and its *rankings* no longer match the project. The
> summaries are all still accurate; what changed is which papers matter.
>
> - **On-device / small-model work is out of scope.** TF-MLPNet (#8) and the
>   "highest-value gap" at the bottom of this file are now background and
>   named future work, not a build target. See `docs/decisions.md`.
> - **Replicating the challenge baselines is dropped.** Where this file says
>   "the baseline you must replicate", read "the architecture you are
>   borrowing as a well-characterised instrument."
> - **PS4 and Ma et al. moved up.** They are the closest prior art to the
>   actual contribution and are currently only in Tier 2. Read them first.
> - **"Interpretable STT output" is now a measured reference condition, not a
>   goal.** The model outputs audio; text is a benchmark row. This makes the
>   TS-ASR literature — Ma et al. especially, and USEF-TP's joint framing —
>   relevant as *what a serious text path would look like*, which is the
>   comparison our deliberately cheap extractor→ASR cascade should be cited
>   against.

---

## Before starting: three findings that should shape plan

**1. The REAL-TSE Challenge is finished.** Submissions closed 25 June 2026, system reports 1 July 2026, and final rankings plus an official overview paper are now public. Track 1 (Online) winner: **CARTSE** (CyberAgent), score 3.25. Runner-up: **DeepSound** (SA-Mamba). Then SonicAGI / WasedaM (tied 3rd), smellycat, WHU_IASP, insta360, ChuEst, AIVOX, SHNU-TSE. All 12 valid Track 1 system reports are downloadable as PDFs from the rankings table at https://real-tse.github.io/challenge/#rankings — **download all of them, they are short and they are the actual state of the art on this exact task.**

**2. Yes, the top challenge reports *are* the state of the art for online TSE on real conversational audio** — but not in the way you might expect. The organisers' own conclusion is blunt: the top Track 1 entries were nearly all built on **BSRNN-style backbones that are architecturally close to the provided baseline**, and the large score gains came from data simulation, real-data adaptation, pseudo-label generation and filtering, multi-objective loss design, and latency control — not from new architectures. The overview explicitly says architecture and training recipe have not been co-optimised for real-world TSE, and that stronger backbones (TF-GridNet-style) likely still have headroom *if* paired with an equally strong data pipeline. **This is directly relevant to your supervisors' note 4** ("reimplement first, new architecture after"): the evidence says a new architecture without the data pipeline will lose to a boring architecture with one. Budget accordingly.

**3. Your two current papers.** Žmolíková et al. is a survey — essential orientation, but it is not "the state of the art" and it predates everything that matters for the online track. TF-MLPNet *is* state of the art for its niche (real-time on a hearable-class NPU), and is the single best paper for your "small enough to fit on a small device" leg — but note your spec cites it as Interspeech 2025 when it is actually the **6th Clarity Workshop (Clarity 2025)**; worth fixing before your proposal is marked. Neither paper is about real conversational TSE, so on their own they leave the biggest part of the problem uncovered.

**One caution you should design around now:** the organisers found that DNSMOS-OVRL was badly over-optimised by several teams (in one case with adversarial waveform perturbations), and swapped the official metric to DNSMOS-P808 post hoc. Human-MOS correlation for OVRL on Track 1 was essentially zero (LCC +0.003). If your project defines a new metric (the live-model measurement), **build gaming-resistance into the metric definition from the start** — that alone is a defensible contribution, and it fits your supervisor's note 1 (the metric matters more than the score).

---

# 1. SLT 2026 REAL-TSE Challenge: Real-world Target Speaker Extraction from Conversational Recordings (2026)

**Citation:** S. Wang, Z. Qian, K. Zhang, J. Han, Z. Liu, X. Yu, H. Li, M. Delcroix, K. Yu, L. Xie, M. Li, H. Li, "SLT 2026 REAL-TSE Challenge: Real-world Target Speaker Extraction from Conversational Recordings," arXiv:2607.15198, Jul. 2026.
**Link:** https://arxiv.org/abs/2607.15198 · HTML: https://arxiv.org/html/2607.15198v1 · PDF: https://arxiv.org/pdf/2607.15198v1
**Also:** official challenge report PDF at https://real-tse.github.io/assets/pdf/Real_TSE_Challenge_Report.pdf

**One-sentence summary:** The official overview of the challenge that anchors your project — task definition, data, baselines, metrics, a cross-cutting analysis of all 24 submitted systems, and the organisers' lessons.

**Problem it addresses:** TSE benchmarks are overwhelmingly simulated (LibriMix, WSJ0-2mix) and therefore do not measure the things that actually break real systems: real reverberation, uncontrolled loudness, ambient noise, turn-taking, reactive overlap, disfluencies, and enrollment/mixture device mismatch.

**Method (high level):** Real Mandarin + English conversational recordings. DEV (1,991 pairs, from REAL-T) and EVAL-1 (2,000 pairs, seen corpora) come from AISHELL-4, AliMeeting, AMI, DipCo, CHiME-6; EVAL-2 (3,000 pairs) was newly recorded across meeting rooms, cafés, homes and in-vehicle, synchronously captured on two high-quality mics (H1/H2), a phone and headsets, with a deliberate 3×5 mixture-device × enrollment-device matrix. Two tracks: Online (≤100 ms end-to-end algorithmic latency, verified by a perturbation-based response-delay test) and Offline. Four metrics — TER (Zipformer ASR, WER for EN / CER for ZH), SpkSim (WeSpeaker ResNet-34 cosine to *enrollment*), DNSMOS-P808, and target-activity F1 (FireRedVAD) — combined by averaging dense ranks. No official training set; open training data with a blacklist of the source corpora's dev/test splits.

**Key results:** Mean overlap ratio ~0.48–0.53, mean target-activity ratio ~0.73–0.75, mixtures ~17–18 s. Causal baselines land around TER 0.81 / SIM 0.37–0.39 on EVAL-2; the winning online system reached TER 0.699 / SIM 0.504. Condition-wise: the **mixture** device matters far more than the enrollment device (far-field H2 consistently worst); enrollment–mixture channel mismatch mainly hurts SpkSim, plausibly because the metric uses enrollment as reference; coarse scenario labels (café vs meeting vs car) do *not* reliably predict difficulty. Real-data adaptation beat synthetic-only training consistently, and the top three teams all trained on real recordings.

**Limitations / what it doesn't solve:** It is an overview, so no single method is described in reproducible depth — you must go to the individual system reports. Single-channel only. No compute/parameter/energy budget is enforced, so "runs on-device" is *not* measured by the challenge at all (only latency is). Metrics are all automatic and, as the paper itself documents, gameable. The DEV/EVAL data is behind team registration, which has closed.

**Relevance to my project:** This is your problem statement, your evaluation pipeline, and your literature map in one document. It also tells you exactly which of your project's four goals the challenge does and does not measure: latency yes, model size no, interruptions only indirectly (via activity F1), downstream live-model intelligibility not at all — which is precisely the gap your defined metric fills.

**Could I combine this with:** Everything downstream. Specifically, pair its "lessons" section with PS4 and the CARTSE report to design your training pipeline, and with TF-MLPNet to add the missing on-device axis.

**Reproducibility:** Strong. Official repos: https://github.com/REAL-TSE/REAL-TSE-Challenge (inference + scoring + latency script) and https://github.com/REAL-TSE/wesep-real-tse (baseline training). Baseline checkpoints and DEV/EVAL data were email-distributed to registered teams and registration is closed — **email realtse.challenge@gmail.com early** and explain you are a masters student; worst case you can reproduce the baselines yourself from the code on Libri2Mix-100 and evaluate on your own simulated + REAL-T-style data.

---

# 2. CARTSE Submission to the REAL-TSE Challenge, Track 1: Online Target Speaker Extraction (2026)

**Citation:** L. Li, S. Seki, "CARTSE Submission to the REAL-TSE Challenge, Track 1: Online Target Speaker Extraction," REAL-TSE Challenge system description, CyberAgent Inc., 2026.
**Link:** https://real-tse.github.io/assets/pdf/CARTSE-Track1.pdf

**One-sentence summary:** The winning online system — a causal BSRNN separator with TF-Map + frame-level ECAPA conditioning, initialised from the official causal baseline and lifted almost entirely by a pseudo-label training pipeline, verified to stream at ~22.9 ms.

**Problem it addresses:** Real far-field mixtures have no clean per-speaker reference, the target speaker is often silent for long stretches, and enrollment and mixture routinely come from different devices with different frequency responses.

**Method (high level):** STFT (512 win / 128 hop), TF-Map (enrollment-magnitude × mixture-magnitude attention) concatenated to the RI spectrogram, band-split into 32 non-uniform sub-bands, 6× (bidirectional band-RNN + **unidirectional** temporal RNN) with cumulative layer norm, complex band mask, iSTFT. Bilingual ECAPA-TDNN speaker encoder (VoxCeleb2 + CN-Celeb) fine-tuned on real recordings, non-causal because enrollment is available in advance. Training: (a) pre-train on on-the-fly simulated mixtures with ~38% target-absent examples (distractor enrollment p=0.35, noise-only p=0.05) and a split loss — masked SI-SDR when target present, push-to-silence when absent; (b) two-stage fine-tune on 7,710 real clips (37.9 h) whose targets are **pseudo-labels from their own offline Track-2 teacher**, kept only if ΔOVRL>0, OVRL≥2.2, VAD precision≥0.80 and TER<0.6; (c) scenario-aware loss split over target-active vs target-silent frames, plus frozen auxiliary losses (WeSpeaker cosine, log-mel L1, direct DNSMOS maximisation, and a multi-layer Zipformer feature-matching loss). Plus "channel-gap" augmentation: random RMS-preserving EQ curves (spectral tilt, cubic-spline EQ, Butterworth band limits) applied to the enrollment so the conditioning path learns device invariance.

**Key results:** EVAL TER 0.805 → 0.699, SIM 0.456 → 0.504, DNSMOS-OVRL 1.67 → 3.44, F1 0.836 → 0.848 over its own initialisation. Algorithmic latency 24 ms (window − hop), buffering 8 ms, measured effective future dependency 22.2–23.7 ms (mean 22.9 ms). Reports that on DEV, learned mixture–enrollment similarity was the strongest single predictor of WER (Spearman ρ = −0.52).

**Limitations / what it doesn't solve:** No parameter count or MAC budget reported — this is not a small model (its baseline initialisation is ~27 M params, ~39 GMAC/s), so it is *not* an on-device solution. It explicitly trains against DNSMOS, i.e. it is on the metric-gaming spectrum the organisers later criticised. Requires an offline teacher, so you must build two systems. Interruption/turn-taking handling is implicit (target-absent training + scenario-aware loss), never evaluated directly.

**Relevance to my project:** This is your reference implementation and your ablation ladder. Nearly every component is separable: you can reproduce the causal baseline, then add target-absent simulation, then pseudo-labels, then channel-gap EQ, then each auxiliary loss, and attribute the gain to each. That is a complete, publishable replication study on its own, and it satisfies note 5 (reimplement and match reported results).

**Could I combine this with:** TF-MLPNet or SA-Mamba (keep the training recipe, swap the backbone for something that fits on-device — this is the single most obvious high-value experiment available to you). Also PS4, which independently pursues differentiable proxy losses on real data.

**Reproducibility:** Medium-high in principle, no code released. But the architecture *is* the public wesep-real-tse baseline, the data sources are fully enumerated in its Appendix B, all hyperparameters and loss weights are given, and the Appendix A channel-gap augmentation is specified precisely enough to reimplement. The hard dependency is their offline teacher for pseudo-labels; you could substitute a public offline TSE model or guided source separation.

---

# 3. Multi-Level Speaker Representation for Target Speaker Extraction (2025)

**Citation:** K. Zhang, J. Li, S. Wang, Y. Wei, Y. Wang, Y. Wang, H. Li, "Multi-Level Speaker Representation for Target Speaker Extraction," in *Proc. ICASSP 2025*. doi:10.1109/ICASSP49660.2025.10889409 · arXiv:2410.16059
**Link:** https://arxiv.org/abs/2410.16059 · PDF: https://arxiv.org/pdf/2410.16059

**One-sentence summary:** Introduces the TF-Map (spectral-level) + contextual + neural-embedding multi-level conditioning that the strongest REAL-TSE baseline and the challenge winner both use.

**Problem it addresses:** A single pre-trained speaker embedding compresses the whole enrollment into one vector, is prone to speaker confusion, and generalises poorly to unseen conditions.

**Method (high level):** Build the reference cue at three levels of abstraction rather than one — a raw spectral-level representation derived from the enrollment magnitude spectrogram (the "TF-Map"), a frame-level contextual representation, and a conventional neural speaker embedding — and inject them at appropriate points in a BSRNN extractor.

**Key results:** The spectral-level raw feature is the main driver of improved generalisation. In the REAL-TSE baselines, TF-Map + Context beats plain ECAPA embeddings on nearly every metric, and notably the *causal* TF-Map variant (TER 0.652 DEV / 0.808 EVAL-2) outperforms even the non-causal embedding baseline on TER — an unusual and useful result for an online project.

**Limitations / what it doesn't solve:** Adds a second spectrogram-sized input stream and cross-attention, so it costs parameters and compute — a problem if your endpoint is a microcontroller-class device. Evaluated primarily on simulated benchmarks in the original paper. Gains on real far-field data are smaller than the paper's simulated numbers suggest.

**Relevance to my project:** You need to understand this to understand the baseline you are required to replicate. It is also the most likely place to find efficiency wins: TF-Map is conceptually cheap (an outer-product-style attention map) but implemented as a full-resolution extra channel, which is a plausible target for compression.

**Could I combine this with:** USEF-TSE (both attack the same "don't rely on a single embedding" problem from opposite directions — raw spectral features vs cross-attention retrieval; a comparison is a clean thesis chapter). Also SA-Mamba's progressive speaker distillation, which is a cost-aware way of deciding *where* to spend the expensive frame-level conditioning.

**Reproducibility:** Good. arXiv preprint is open, and an implementation ships inside WeSep / wesep-real-tse — you can read the actual baseline code rather than inferring from the paper.

---

# 4. Music Source Separation with Band-Split RNN (2023)

**Citation:** Y. Luo, J. Yu, "Music Source Separation with Band-Split RNN," *IEEE/ACM Trans. Audio, Speech, Lang. Process.*, vol. 31, pp. 1893–1901 (also cited as 1215–1225), 2023. doi:10.1109/TASLP.2023.3250264
**Link:** https://arxiv.org/abs/2209.15174 (search "Band-Split RNN music source separation arXiv"; the TASLP version is paywalled)
**Companion worth reading with it:** J. Yu, Y. Luo, H. Chen, R. Gu, C. Weng, "High Fidelity Speech Enhancement with Band-Split RNN," arXiv:2212.00406 — this is the speech-domain adaptation, and closer to what you'll build.

**One-sentence summary:** The dual-path band-split architecture — split the spectrum into non-uniform sub-bands, alternate an across-band module with an across-time RNN — that is the de facto backbone for every strong REAL-TSE online system.

**Problem it addresses:** Full-band time-frequency models waste capacity modelling 257 bins that have very different statistics, while time-domain models lose explicit frequency structure. Band-splitting gives per-band normalisation and a much shorter sequence to model along frequency.

**Method (high level):** STFT → split frequency bins into K hand-designed sub-bands (narrow at low frequency where F0 and low harmonics live, wide at high frequency) → per-band normalisation + linear projection to a fixed dimension → stack of blocks each containing a band-axis (frequency) RNN and a time-axis RNN → per-band complex mask → iSTFT. Causality is obtained by making only the *time-axis* RNN unidirectional and switching to cumulative layer norm; the band-axis RNN can stay bidirectional because frequency is not a causal dimension.

**Key results:** State-of-the-art music source separation at publication; the speech-enhancement variant became the backbone of multiple DNS Challenge and TEA-PSE-lineage systems. In REAL-TSE, all four official baselines and the top online systems are BSRNN derivatives.

**Limitations / what it doesn't solve:** Band boundaries are hand-designed hyperparameters, not learned (both SA-Mamba and CARTSE quietly use different splits — 36 vs 32 bands — with no principled justification). RNNs are sequential and awkward to quantise or run on parallel NPUs. The published models are large: the REAL-TSE causal baselines are 25–27 M parameters at ~39 GMAC/s, which is orders of magnitude beyond a hearable-class budget.

**Relevance to my project:** Non-negotiable background. Understanding *why* the time-axis is the only place causality bites, and that per-band normalisation is what makes low-latency streaming stable, is what lets you reason about latency budgets instead of guessing. Learned or perceptually-motivated band splitting is an obvious, tractable novelty for your thesis.

**Could I combine this with:** TF-MLPNet (which replaces the sequential bidirectional RNN with a parallel MLP-Mixer for exactly the on-device reasons BSRNN ignores) and SA-Mamba (which replaces the temporal RNN with a fixed-state SSM). Those two papers are both "BSRNN-shaped, but the expensive part swapped out" — you could unify them.

**Reproducibility:** Good. No official code from the authors, but multiple faithful community reimplementations exist, and the TSE-conditioned causal version is in wesep-real-tse, which is what actually matters to you.

---

# 5. Neural Target Speech Extraction: An Overview (2023)

**Citation:** K. Žmolíková, M. Delcroix, T. Ochiai, K. Kinoshita, J. Černocký, D. Yu, "Neural Target Speech Extraction: An Overview," *IEEE Signal Processing Magazine*, vol. 40, no. 3, pp. 8–29, 2023. doi:10.1109/MSP.2023.3240008
**Link:** IEEE Xplore (institutional access); search for an author-hosted PDF — Delcroix and Žmolíková both post preprints.

**One-sentence summary:** The canonical survey of neural TSE: how clues (audio enrollment, visual, spatial, text) are encoded, where and how they're fused, what the loss functions and datasets are, and what the open problems were as of 2023.

**Problem it addresses:** Orientation. TSE has a scattered vocabulary (speaker extraction / personalised speech enhancement / target speech extraction / informed separation) and this paper reconciles it into one taxonomy.

**Method (high level):** Not a method paper. It organises the field along the axes you will actually need for your literature review: clue type, fusion mechanism (concatenation, multiplication, FiLM, attention, adaptation-layer), domain (time vs time-frequency), training objective, and evaluation.

**Key results:** No new results. Its lasting contributions are the taxonomy, the framing of speaker confusion and target-absent false alarms as the field's characteristic failure modes, and its identification of real-recording generalisation as the key open problem — which the REAL-TSE Challenge then went and built a benchmark for.

**Limitations / what it doesn't solve:** Three years old and pre-dates essentially everything relevant to you: TF-GridNet-era backbones, Mamba/SSM sequence models, generative and flow-matching TSE, embedding-free cross-attention conditioning, and real-conversational benchmarks. It also barely treats latency or on-device deployment, which are two of your four goals. Do not cite it as evidence of the current state of the art.

**Relevance to my project:** Use it to write your background chapter and to get the terminology and notation right. Marc Delcroix is also a REAL-TSE organiser, so the survey's framing is visible in how the challenge is designed — reading it makes the challenge's metric choices legible.

**Could I combine this with:** It is the connective tissue for everything else on this list. Its bibliography is also your route to the pre-2023 classics you should be able to name: SpeakerBeam, VoiceFilter, SpEx/SpEx+, td-SpeakerBeam.

**Reproducibility:** N/A (survey). No code.

---

# 6. REAL-T: Real Conversational Mixtures for Target Speaker Extraction (2025)

**Citation:** S. Li, S. Wang, J. Han, K. Zhang, W. Wang, H. Li, "REAL-T: Real Conversational Mixtures for Target Speaker Extraction," in *Proc. Interspeech 2025*, pp. 1923–1927. doi:10.21437/Interspeech.2025-2662
**Link:** https://www.isca-archive.org/interspeech_2025/li25da_interspeech.pdf (open access) · project page https://real-tse.github.io/

**One-sentence summary:** The automated pipeline and dataset that turns five existing diarisation corpora into real-conversational TSE trials, and which the REAL-TSE DEV and EVAL-1 sets are built from.

**Problem it addresses:** You cannot train or evaluate real-conversational TSE without mixture/enrollment/target triples, and real recordings don't come with clean targets. REAL-T constructs the trials automatically instead.

**Method (high level):** From AISHELL-4, AliMeeting, AMI, DipCo and CHiME-6, use existing diarisation annotations to find naturally overlapping segments as mixtures, and non-overlapping single-speaker segments of ≥5 s from the same speaker as enrollment. Evaluate with reference-free and downstream metrics (ASR, speaker similarity) since no clean target exists.

**Key results:** Establishes that models trained on simulated fully-overlapped mixtures degrade sharply on real conversational trials, and provides the numbers that motivated the challenge.

**Limitations / what it doesn't solve:** Inherits every error in the source diarisation labels (your supervisors' note 3 about training artefacts from unclean data lands here). There is no clean target, so the evaluation is inherently indirect — this is exactly *why* the challenge scores TER/SpkSim/DNSMOS/F1 rather than SI-SDR. Only two languages, and the acoustic conditions are meeting/dinner-party-dominated, so voice-assistant and in-car conditions are only covered by the challenge's separately-recorded EVAL-2.

**Relevance to my project:** This is how you build your own training and validation data without waiting on the closed challenge registration. The pipeline is fully described, the source corpora are publicly available, and the official training splits are permitted under the challenge rules. Reproducing REAL-T-style trial construction is probably the single most useful piece of infrastructure you can build in your first month.

**Could I combine this with:** PS4, which builds a much larger (71,771-sample) corpus in the same spirit and adds transcripts and frame-level VAD labels; and CARTSE's quality-filtering criteria, which are essentially a recipe for cleaning a REAL-T-style corpus.

**Reproducibility:** Medium. Paper is open access; the dataset itself was still marked "to be released soon" on the project page. The pipeline is reimplementable from the description given the source corpora, which you can obtain independently (AMI and AISHELL-4 are freely downloadable; CHiME-6 and AliMeeting require registration).

---

# 7. Exploring Time-Frequency Domain Target Speaker Extraction for Causal and Non-Causal Processing (2023)

**Citation:** W. Zhang, L. Yang, Y. Qian, "Exploring Time-Frequency Domain Target Speaker Extraction for Causal and Non-Causal Processing," in *Proc. IEEE ASRU 2023*, pp. 1–6.
**Link:** https://ieeexplore.ieee.org/document/10389752/ · author page with paper/slides/poster: https://sites.google.com/view/wangyou-zhang/publications

**One-sentence summary:** Ports a TF-GridNet-class separator into the TSE framework, compares speaker-conditioning fusion mechanisms including a novel speaker-token fusion, and shows the model extends to causal processing with strong performance.

**Problem it addresses:** Most strong TSE work at the time was time-domain, and it was unclear whether the then-dominant time-frequency separation architectures could be conditioned on a speaker cue and made causal without collapsing.

**Method (high level):** Take the top-performing T-F domain separation backbone; test conditioning by concatenation vs a proposed **speaker-token** fusion (treat the speaker representation as a token in the sequence rather than a feature to concatenate); then convert the model to causal processing and measure the cost.

**Key results:** Beats widely-used time-domain TSE models by a large margin in both causal and non-causal settings on WSJ0-2mix and LibriMix. Quantifies the causal-vs-non-causal gap for a T-F architecture, which is the number your online-track design decisions hinge on.

**Limitations / what it doesn't solve:** Simulated benchmarks only — no real conversational data, no device mismatch, no target-absent regions. No latency accounting in the ICASSP-DNS sense (algorithmic vs buffering), and no compute or parameter budget, so "causal" here does not imply "deployable."

**Relevance to my project:** The REAL-TSE overview names TF-GridNet variants as the *other* online-track family besides BSRNN, and says explicitly that TF-GridNet-style extractors may still have headroom if paired with a strong data pipeline. This paper is the entry point to that route, and speaker-token fusion is an under-explored conditioning mechanism you could revisit under a latency budget.

**Could I combine this with:** CARTSE's training pipeline (the overview's own suggested experiment: strong backbone + strong data recipe). Also USEF-TSE, whose USEF-TFGridNet is the same backbone with cross-attention conditioning instead of tokens.

**Reproducibility:** Medium. IEEE-paywalled but slides and poster are on the author's site, and the underlying backbone has open implementations in ESPnet (Wangyou Zhang is an ESPnet-SE maintainer, so check ESPnet's TSE recipes first — that is likely the fastest path to a working causal T-F TSE baseline).

---

# 8. TF-MLPNet: Tiny Real-Time Neural Speech Separation (2025)

**Citation:** M. Itani, T. Chen, S. Gollakota, "TF-MLPNet: Tiny Real-Time Neural Speech Separation," in *Proc. 6th Clarity Workshop on Improving Speech-in-Noise for Hearing Devices (Clarity 2025)*. arXiv:2508.03047
**Link:** https://arxiv.org/abs/2508.03047 · PDF: https://arxiv.org/pdf/2508.03047 · https://www.isca-archive.org/clarity_2025/itani25_clarity.pdf
**Note:** your specification cites this as Interspeech 2025 — it is Clarity 2025. Worth correcting.

**One-sentence summary:** The first separation/extraction network that actually runs in real time on a hearable-class low-power neural accelerator (GAP9), by replacing the sequential parts of a dual-path T-F model with parallel-friendly components and training with mixed-precision quantisation awareness.

**Problem it addresses:** State-of-the-art separation networks cannot run in real time on tiny low-power accelerators. The bottleneck is not FLOPs alone but *sequential* structure — the paper profiles an existing dual-path model and shows where the runtime actually goes.

**Method (high level):** Time-frequency domain, causal 3×3 conv encoder / transposed-conv decoder. Two key components: a **conv-batched LSTM** that lets a batch of LSTM inputs be inferred in parallel using convolutional layers, and an **all-MLP-Mixer** module (stacks of fully-connected layers alternating along channel and frequency) that replaces the sequential bidirectional LSTM along frequency; the time axis at each frequency bin is handled independently by convolutions. Optional frequency compression. Trained with PIT + negative SI-SDR for blind separation, and SI-SDR + a differentiable PESQ term for TSE. Deployed via mixed-precision quantisation-aware training and GAPFlow kernel compilation.

**Key results:** Processes 6 ms audio chunks in real time on GAP9, a 3.5–4× runtime reduction versus prior separation models, while outperforming existing streaming models on both blind separation and target speech extraction. The mixed-precision quantised model loses only ~0.6 dB versus full floating point.

**Limitations / what it doesn't solve:** Evaluated on simulated benchmarks, not real conversational audio — no device mismatch, no long target-absent stretches, no meeting reverberation. Conditioning on the target speaker is not the paper's contribution and is comparatively simple. Very short (workshop paper), so many details are compressed. And critically: it was *not* entered in REAL-TSE, so there is no comparable number for it on the data you care about — producing one is an open, valuable experiment.

**Relevance to my project:** This is your on-device leg and the strongest evidence in the literature that the sequential-vs-parallel distinction, not parameter count, is what governs deployability. Your project's "small enough to fit on a small device" goal is essentially unaddressed by every REAL-TSE entry, so **evaluating a TF-MLPNet-class model on the REAL-TSE online protocol is a genuinely open question with a clear novelty claim.**

**Could I combine this with:** CARTSE (their training recipe, this backbone) — the highest-value single experiment on this whole list. Also with SA-Mamba, since a fixed-size SSM state and an MLP-Mixer are competing answers to the same "get rid of the sequential bottleneck" question, and nobody has compared them under a shared latency + hardware budget.

**Reproducibility:** Medium. The Gollakota lab (UW Mobile Intelligence Lab) has a consistent track record of releasing code — check https://github.com/vb000 and the lab's project pages for TF-MLPNet, and look at their prior "Look Once to Hear" and "Target Conversation Extraction" releases, which share infrastructure. GAP9 deployment requires the GreenWaves toolchain and hardware you probably don't have; the PyTorch model and the quantisation-aware training are reproducible without it.

---

# 9. USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction (2025)

**Citation:** B. Zeng, M. Li, "USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction," *IEEE/ACM Trans. Audio, Speech, Lang. Process.*, 2025. arXiv:2409.02615
**Link:** https://arxiv.org/abs/2409.02615 · PDF: https://arxiv.org/pdf/2409.02615 · IEEE: https://ieeexplore.ieee.org/document/11012711/
**Companion:** B. Zeng, M. Li, "Universal Speaker Embedding Free Target Speaker Extraction and Personal Voice Activity Detection" (USEF-TP), arXiv:2501.03612 — joint TSE + personal VAD with a scenario-aware differentiated loss, evaluated on SparseLibriMix and CALLHOME.

**One-sentence summary:** Drops the pre-trained speaker-recognition model entirely and instead retrieves frame-level target-speaker features from the enrollment via multi-head cross-attention, in a wrapper that plugs into any time-domain or T-F separation backbone.

**Problem it addresses:** A speaker embedding trained for verification is optimised for the wrong objective, throws away the enrollment's phonetic and contextual detail, and forces you to pick and ship a second model. Choosing which speaker encoder to use is itself an unprincipled hyperparameter.

**Method (high level):** One shared encoder processes both mixture and enrollment. The mixture encoding forms queries, the enrollment encoding forms keys and values, cross-attention produces a frame-level speaker-conditioned representation, and that feeds an arbitrary separator. Demonstrated as USEF-SepFormer and USEF-TFGridNet. (Contrast with its predecessor SEF-Net, where the query/key roles are reversed and fusion happens inside the separator.)

**Key results:** State-of-the-art SI-SDR on WSJ0-2mix, WHAM! and WHAMR!; USEF-TFGridNet reported around 23.3 dB SI-SDRi. Also validated on LibriMix and the ICASSP 2023 DNS Challenge blind test set, i.e. it holds up on more diverse out-of-domain data. The companion USEF-TP shows the same frame-level features support personal VAD jointly with extraction.

**Limitations / what it doesn't solve:** Per-frame cross-attention cost grows with enrollment length, which is a real problem for streaming with 9–11 s enrollments (REAL-TSE mean enrollment is ~9–11 s). The paper is not causal or latency-aware. Keys and values *can* be precomputed and cached at enrollment time — SA-Mamba and SonicAGI both exploit this — but USEF-TSE itself doesn't analyse the streaming cost.

**Relevance to my project:** Two things. First, it is the main alternative to your baseline's conditioning scheme, so a TF-Map vs cross-attention comparison under a fixed 100 ms latency budget is a well-posed, tractable thesis experiment. Second, **USEF-TP's joint TSE + personal VAD framing is the cleanest existing handle on your "handles interruptions" goal** — knowing *when* the target is speaking is what target-activity F1 measures, and doing it jointly rather than post hoc is a defensible contribution.

**Could I combine this with:** SA-Mamba's progressive speaker distillation (use the cheap global condition in shallow layers, expensive cross-attention retrieval only in deep layers — this directly answers USEF-TSE's cost problem). Also with Multi-Level Speaker Representation as the head-to-head comparison.

**Reproducibility:** Good. Code at https://github.com/ZBang/USEF-TSE (Bang Zeng, Duke Kunshan / Ming Li's group — they also maintain SEF-PNet and related releases). arXiv version is open, and it was used as the offline backbone by at least one REAL-TSE team (SonicAGI), which is independent evidence it rebuilds.

---

# 10. Speaker-Aware State Space Modeling for Streaming Target Speaker Extraction (SA-Mamba) (2026)

**Citation:** DeepSound team (B. Liu), "Speaker-Aware State Space Modeling for Streaming Target Speaker Extraction," REAL-TSE Challenge Track 1 system description, 2026.
**Link:** https://real-tse.github.io/assets/pdf/DeepSound-Track1.pdf

**One-sentence summary:** Second place in the online track, and the most architecturally novel online entry — injects the speaker condition directly into a selective state-space model's parameters so that target selectivity happens *inside* the recurrence, with fewer parameters and MACs than the baseline.

**Problem it addresses:** Conditioning is almost always external fusion *after* temporal modelling (concatenate, multiply, FiLM, attention). Mamba's selective parameters are generated from the current acoustic input only, so the state dynamics never know which speaker they're supposed to be tracking. Separately: streaming needs a fixed-size state, which rules out growing attention caches.

**Method (high level):** Complex-STFT domain (512 win / 256 hop, 257 bins), 36 non-uniform bands (fine at low frequency: eight 2-bin bands below 0.5 kHz, coarsening to 14 wide bands above 3 kHz), shared band-split frontend for mixture and enrollment. **SA-Mamba:** generate the SSM's input matrix B, output matrix C and discretisation step Δt from the concatenation of the current acoustic features *and* a speaker condition, so writing, reading and forgetting in the recurrent state all become speaker-dependent. 36 dual-path blocks: bidirectional Mamba across bands (frequency is not causal) + causal SA-Mamba along time per band. **Progressive speaker distillation:** the first 30 blocks use a cheap precomputed global speaker condition for coarse target localisation; the last 6 use multi-head cross-attention retrieval over cached enrollment keys/values for detail reconstruction. No separate speaker-recognition model at all. Gated complex ratio mask, iSTFT overlap-add.

**Key results:** 15.89 M params / 30.01 GMAC/s vs the baselines' 25–27 M / ~39 GMAC/s, with EVAL TER 0.713, F1 0.849, SIM 0.524, i.e. better on every metric at ~60% of the parameters. Measured future dependency 44.5–49.9 ms (mean 46.7 ms). Their ablations are the most useful part: layer-wise cross-attention beats embedding multiplication (SI-SNRi 9.23 → 11.25 dB), and SA-Mamba beats causal GRU by 3.11 dB and causal TCN by 5.37 dB SI-SNRi under matched conditions.

**Limitations / what it doesn't solve:** The ablations are run on their own simulated CN-Celeb/VoxCeleb1 two-speaker setup, *not* on REAL-TSE data, so the component attributions don't transfer cleanly to the leaderboard number. 36 blocks is deep, and 15.89 M params is still far too big for a hearable. No real-data pseudo-label adaptation — which the overview paper identifies as the dominant factor — so this is arguably a strong architecture with a weak data recipe, i.e. the exact untested combination the organisers flag. Their own future work lists offline-teacher distillation, confirming the gap. Mamba kernels are also awkward to quantise and deploy on fixed-point NPUs.

**Relevance to my project:** This is your evidence that architectural novelty *can* pay off in the online track, and it is the closest thing in the literature to "a fundamentally new TSE architecture" for streaming. It also hands you a ready-made experiment: SA-Mamba's architecture + CARTSE's data pipeline is a combination nobody has run, and both papers describe themselves well enough to attempt it.

**Could I combine this with:** CARTSE (architecture × data recipe). TF-MLPNet (SSM fixed state vs MLP-Mixer parallelism, compared under one hardware budget). USEF-TSE (its progressive distillation *is* a cost-managed USEF-TSE).

**Reproducibility:** Medium-low, and be honest about this in your review. No code, no checkpoint. The system description is unusually detailed for a challenge report — band widths, d_state=16, d_conv=4, expand=2, 36 blocks, 4 attention heads, loss configuration, optimiser schedule are all given — but the ablation dataset is bespoke and unreleased, so you can reproduce the *architecture* but not verify their *numbers*. Mamba itself has good open implementations (mamba-ssm, plus SPMamba and SepMamba for the speech-separation precedent).

---

# Tier 2 — read these next, or when you hit the specific problem

Not written up in full because they're supporting rather than core, but several are important enough that you should download them now.

**For your live-model / downstream-intelligibility metric (your novel contribution):**

- **PS4: Proxy-Supervised Joint Training for Real Target Speaker Extraction** — Ning et al., arXiv:2607.08111 (2026). https://arxiv.org/abs/2607.08111 — YiJiaHe's 2nd-place offline system. Builds a 71,771-sample real-conversational corpus with transcripts *and* frame-level VAD labels, then fine-tunes a BSRNN with four differentiable proxy objectives: ASR cross-entropy, speaker similarity, frame-level VAD, and perceptual quality. **This is the closest published prior art to "optimise TSE for what a downstream model actually needs," and the most directly useful paper on this list for your metric contribution.** Read it right after the top 3.
- **Enhancing Intelligibility for Generative Target Speech Extraction via Joint Optimization with Target Speaker ASR** — Ma et al., arXiv:2501.14477 (2025). https://arxiv.org/abs/2501.14477 — generative TSE built on Whisper, jointly optimised with target-speaker ASR to get intelligibility *and* perceptual quality. Directly addresses your observed phenomenon that TSE output can be more transcribable yet worse for a downstream speech model.
- **Listen only to me! How well can target speech extraction handle false alarms?** — Delcroix et al., Interspeech 2022. The foundational treatment of target-absent behaviour and false alarms, from a REAL-TSE organiser. Essential for both the interruption goal and for understanding why activity F1 is scored.

**Architecture / backbone alternatives:**

- **TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation** — Wang et al., TASLP 31:3221–3236, 2023. arXiv:2209.03952. https://arxiv.org/abs/2209.03952 — the other backbone family named in the overview; open code in ESPnet. Read if you go the "stronger backbone" route.
- **The SonicAGI System for the REAL-TSE Challenge** — arXiv:2607.11083 (2026). https://arxiv.org/abs/2607.11083 — joint 3rd, online track. Introduces **SwiftNet-Lookahead**: one bounded-lookahead module in front of a strictly causal iterative separator, total system latency 96 ms. This is the only entry that deliberately *spends* the full 100 ms budget rather than hiding at 25 ms, which makes it the best case study in the latency/quality trade-off. Also uses a frozen offline enhancer to provide denoised auxiliary supervision on real targets.
- **StarTSE / Towards Streaming TSE via Chunk-wise Interleaved Splicing of an Autoregressive Language Model** — Peng et al., arXiv:2604.19635 (2026). https://arxiv.org/abs/2604.19635 — first AR generative backbone made to stream, RTF 0.248 on a 4090. Interesting but note the honesty problem for you: RTF 0.248 on a datacentre GPU is the opposite of on-device.
- **High Fidelity Speech Enhancement with Band-Split RNN** — Yu et al., arXiv:2212.00406 — the speech-domain BSRNN; read alongside #4.
- **TEA-PSE 2.0 / 3.0** — Ju et al., SLT 2023 / ICASSP 2023 — sub-band real-time personalised speech enhancement; the industrial lineage of low-latency TSE-adjacent systems, and where a lot of the practical sub-band engineering comes from.
- **DSINet: Towards Real-Time Target Speaker Extraction with Dynamic Speaker Information Fusion** — Hao et al., ICASSP 2024 — dynamic speaker-information fusion for real-time TSE; one of the two papers behind the baseline's "Context" module.

**Infrastructure you will actually use:**

- **WeSep: A Scalable and Flexible Toolkit Towards Generalizable Target Speaker Extraction** — Wang et al., Interspeech 2024, pp. 4273–4277. https://github.com/wenet-e2e/wesep — the challenge baselines are WeSep recipes. Read the paper, but spend most of your time in `wesep-real-tse`.
- **ECAPA-TDNN** (Desplanques et al., Interspeech 2020) and **WeSpeaker** (Wang et al., ICASSP 2023) — the speaker encoder and the toolkit used both for conditioning *and* for computing the official SpkSim metric. You need to know the metric is computed against the *enrollment*, not a clean reference, because that shapes what SpkSim can and cannot tell you.
- **Look Once to Hear: Target Speech Hearing with Noisy Examples** — Veluri, Itani, Chen, Yoshioka, Gollakota, CHI 2024. doi:10.1145/3613904.3642057 — real-time on-device target speech hearing with noisy enrollment, from the TF-MLPNet group. The enrollment-from-noisy-audio problem is exactly your microphone-mismatch problem, from a different angle.

---

# Suggested reading order (different from the importance ranking)

Your supervisors asked for 5–6 papers reviewed deeply, then reimplementation, then novelty. Mapping that onto the above:

1. **Orient (week 1):** #5 Žmolíková survey → #1 REAL-TSE overview. You now know the field and the exact problem.
2. **Understand the baseline you must replicate (weeks 2–3):** #4 BSRNN → #3 Multi-Level Speaker Representation → read the `wesep-real-tse` code alongside them. Reproduce `BSRNN_TFMAP_CAUSAL` on Libri2Mix-100 and confirm you can hit the published DEV numbers. This is note 5 satisfied.
3. **Understand why the winner won (week 4):** #2 CARTSE → PS4 → #6 REAL-T. Build the data pipeline. This is where your marks are.
4. **Pick your novelty axis (weeks 5–6):** #8 TF-MLPNet if you go on-device/efficiency; #10 SA-Mamba or #7 causal T-F TSE if you go architecture; #9 USEF-TSE/USEF-TP if you go conditioning + interruption handling. Note 7 says you don't have to solve all of these — pick one.
5. **Your metric contribution, in parallel throughout:** PS4 and the Whisper joint-optimisation paper are your prior art. Design the metric to be hard to game; the challenge's own DNSMOS failure is your justification for caring.

## The one experiment I'd flag as the highest-value gap

> **Superseded 2026-08-07.** This recommendation was made under the original
> brief and points at the on-device leg, which is now explicitly out of scope
> (spec note 10 assumes server-class compute). Retained because the reasoning
> is sound and it is the strongest candidate for the future-work section. The
> current highest-value experiment is the leg-2 divergence table — see
> `docs/research-plan.md` §4.

The REAL-TSE overview says the top online systems all used baseline-like backbones lifted by data pipelines, and that stronger backbones probably still have headroom with an equally strong pipeline. Separately, **not a single REAL-TSE entry reported a parameter or MAC budget anywhere near hearable-class hardware** — the smallest online system was 15.89 M params / 30 GMAC/s, versus TF-MLPNet running 6 ms chunks on a GAP9. So: *take a genuinely on-device backbone (TF-MLPNet-class or a compressed SA-Mamba), train it with CARTSE-class real-data pseudo-label adaptation, and evaluate it on the REAL-TSE online protocol with a reported compute budget.* Nobody has done this, the ingredients are all publicly described, and it lands squarely on your project's stated goal of a sufficiently accurate low-latency system that runs on on-device hardware.

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