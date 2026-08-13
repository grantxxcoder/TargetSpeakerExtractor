# Online Target Speaker Extraction — Glossary & Reference

A newcomer's reference for the terminology that papers in this field assume you already know. ML basics are skipped. Terms marked **★** are the ones you will hit constantly in *your* project specifically.

Both spellings **enrollment** (US) and **enrolment** (UK) appear in the literature; they mean the same thing.

---

## 0. Start here: the 15 terms you cannot read a paper without

| Term | Plain meaning |
|---|---|
| **TSE** | Target Speaker Extraction — pull one chosen person's voice out of a noisy mixture. Your task. |
| **Mixture** | The messy input audio containing everyone and everything. |
| **Target** | The person you want. **Interference** = other people. **Noise** = non-speech sounds. |
| **Enrollment** | A separate short clip of the target speaking alone, used to tell the model *which* person to extract. |
| **Causal** | The model only uses audio it has already heard — never the future. Required for live use. |
| **Streaming** | Processing audio continuously as it arrives, rather than waiting for the whole file. |
| **Latency** | The delay between sound arriving and the processed output coming out. |
| **STFT** | Short-Time Fourier Transform — converts audio into a picture of frequency over time. |
| **Mask** | The model's output: a per-pixel "keep this / discard this" multiplier applied to that picture. |
| **SI-SDR** | The standard signal-quality score in dB. Higher = better. Needs a clean reference. |
| **WER / CER / TER** | Word/Character/Term error rate when transcribing the output. **Lower = better.** |
| **DNSMOS** | An AI that predicts how good audio sounds to a human. No reference needed. However this can be gamified by tricks to trick the AI into thinking the audio sounds better than what it is which is what some teams did. |
| **Speaker embedding** | A fixed-length vector summarising a voice's identity. |
| **Simulated vs real** | Artificially added-together clips vs genuine recordings of real conversations. |
| **Pseudo-label** | A fake "correct answer" produced by a bigger model, used to train on real data that has no answer key. The idea behind this it to train one massive model on the combined audio streams and to separate the different targets noisily so that there is more realistic input. This is an offline method fed into an online model potentially.|

---

## 1. Task names and neighbouring problems

| Acronym | Full name | What it is |
|---|---|---|
| **TSE** ★ | Target Speaker Extraction | Extract one enrolled speaker from a mixture. Your task. |
| **TSS / TSES** | Target Speech Extraction / Separation | Same thing, different naming convention. Papers use these interchangeably with TSE. |
| **BSS** | Blind Source Separation | Separate *all* speakers at once, without being told who to look for. |
| **SS** | Speech Separation | Usually synonymous with BSS in this literature. |
| **SE** | Speech Enhancement | Remove *noise* (not other speakers) from one voice. |
| **PSE** ★ | Personalized Speech Enhancement | Industry term for TSE. Common in Microsoft/Tencent papers and the DNS Challenge. |
| **CSS** | Continuous Speech Separation | Separating a long, continuous meeting recording into speaker streams. |
| **SD** | Speaker Diarization | "Who spoke when" — segment a recording by speaker, without extracting audio. |
| **VAD** ★ | Voice Activity Detection | Detect when *anyone* is speaking vs silence. |
| **PVAD** ★ | Personal VAD | Detect when the *target* speaker specifically is speaking. Relevant to your interruption goal. |
| **TS-VAD** | Target Speaker VAD | Same idea, common in diarization papers. |
| **ASR** ★ | Automatic Speech Recognition | Speech-to-text. |
| **TS-ASR** | Target Speaker ASR | Transcribe only the target speaker directly, skipping extraction. |
| **SV** | Speaker Verification | Decide whether two clips are the same person. Where speaker embeddings come from. |
| **KWS** | Keyword Spotting | Detecting a wake word. |
| **Cocktail party problem** | — | The classic framing: humans can follow one voice in a crowded room; machines struggle. |
| **Permutation problem / permutation ambiguity** | — | In BSS, if you output two voices you don't know which output is which person. **TSE avoids this entirely** — a key selling point you'll see stated in every intro. |
| **Informed separation** | — | Separation guided by a cue. TSE is a form of informed separation. |

---

## 2. Audio and signal basics

| Term | Meaning |
|---|---|
| **Sample rate** ★ | Samples per second. **16 kHz** is standard in this field. 8 kHz = narrowband/telephone, 44.1/48 kHz = fullband/music. Always check — mismatched sample rates silently ruin experiments. |
| **Waveform** | The raw audio signal over time. Also called the **time domain**. |
| **T-F domain** ★ | Time-Frequency domain. The audio represented as frequency content changing over time. |
| **STFT** ★ | Short-Time Fourier Transform. Chop audio into short overlapping chunks, take the frequency content of each. Output is a **spectrogram**. |
| **iSTFT** ★ | Inverse STFT. Converts back to audio. |
| **Window (length)** ★ | How long each STFT chunk is. E.g. 512 samples = 32 ms at 16 kHz. |
| **Hop (length) / frame shift / stride** ★ | How far you slide along between chunks. E.g. 128 samples = 8 ms. Window minus hop determines your latency. |
| **Frame** | One STFT time-slice. Models process frame by frame when streaming. |
| **Frequency bin** ★ | One horizontal row of the spectrogram. A 512-point STFT gives **257** bins (you'll see this number constantly). |
| **Magnitude / phase** | A spectrogram value is a complex number: magnitude = how loud, phase = timing alignment. Early models predicted magnitude only and reused the noisy phase. |
| **Complex spectrogram / RI** ★ | Real and Imaginary parts, stacked as 2 channels. Modern models use this so they can fix phase too. |
| **Overlap-add (OLA)** | Standard way to reassemble audio from overlapping processed frames. |
| **Overlap-save** | Alternative that (per the challenge rules) adds **no** extra algorithmic latency. |
| **Mel / filterbank / log-mel** | A perceptually-spaced compressed version of the spectrogram, common as an ASR input and in loss functions. |
| **CMVN** | Cepstral Mean and Variance Normalisation. Standard per-utterance feature normalisation. |
| **dB (decibel)** | Logarithmic loudness/ratio scale. All the quality scores are in dB. |
| **Monaural / single-channel** ★ | One microphone. Your setting. |
| **Multichannel / array** | Multiple mics, which lets you use direction. Often 8-channel in meeting corpora. |
| **Beamforming** | Using a mic array to spatially focus on one direction. Classical (non-neural) technique. |
| **MVDR / GEV** | Specific beamformer types you'll see named in multichannel papers. |

### Room acoustics

| Term | Meaning |
|---|---|
| **Reverberation** ★ | Echo from room surfaces. Smears speech and badly degrades separation. |
| **RIR** ★ | Room Impulse Response. A recording of a room's echo signature; convolving clean speech with an RIR simulates that room. |
| **RT60 / T60** ★ | Reverberation time — how long for sound to decay by 60 dB. ~0.05–0.3 s = dry, ~1 s = very echoey. |
| **Anechoic** | No echo at all. Lab conditions. |
| **Dereverberation** | Removing echo. |
| **Near-field / close-talk** ★ | Mic near the mouth (headset, phone at your face). Clean. |
| **Far-field / distant** ★ | Mic across the room. Echoey and noisy. **This is where models fail** — the challenge found the farthest mic was consistently worst. |
| **Channel / device mismatch** ★ | Enrollment recorded on one device, mixture on another. Different frequency responses confuse the model. Central to your project's microphone-quality concern. |
| **EQ (equalisation) / band-limiting** | Boosting/cutting certain frequencies, or cutting off high/low ends. Used to *simulate* cheap microphones during training. |

### Ratios

| Acronym | Meaning |
|---|---|
| **SNR** ★ | Signal-to-Noise Ratio. Target loudness vs noise, in dB. Higher = easier. |
| **SIR** | Signal-to-Interference Ratio. Target vs *other speakers*. |
| **SAR** | Signal-to-Artifacts Ratio. How much damage the model itself introduced. |
| **SDR** | Signal-to-Distortion Ratio. Overall quality; combines the above. |
| **Overlap ratio** ★ | Fraction of the mixture where more than one person talks. REAL-TSE averages ~0.5. |
| **Target ratio** ★ | Fraction of the mixture where the target is speaking at all. REAL-TSE ~0.75. Strongly predicts difficulty. |
| **Fully overlapped** | Both speakers talk the entire time. LibriMix is like this — unrealistic. |
| **Sparsely overlapped** | Realistic conversation with pauses and turn-taking. |

---

## 3. Latency and real-time (★ core to your project)

| Term | Meaning |
|---|---|
| **Causal** ★ | Uses only past and present audio. Required for streaming. |
| **Non-causal / offline** ★ | Allowed to see the whole recording, including the future. Performs better; can't run live. |
| **Lookahead / future context** ★ | Peeking a small number of frames into the future. Improves quality, costs latency. |
| **Algorithmic latency** ★ | Delay inherent to the maths: STFT window − hop, plus any lookahead. Example: 32 ms window with 8 ms hop = **24 ms**. |
| **Buffering latency** ★ | Delay from having to collect a block of samples before processing. Equals the hop size. |
| **End-to-end / effective latency** ★ | The total real-world delay. The challenge caps this at **100 ms** and verifies it by perturbing the input and measuring when the output changes. |
| **Chunk-wise / frame-wise** | Processing in small blocks vs one frame at a time. |
| **RTF** ★ | Real-Time Factor. Processing time ÷ audio duration. **Must be < 1** to keep up; 0.25 means 4× faster than real time. |
| **Unidirectional vs bidirectional** ★ | A one-directional RNN only looks backwards (causal). Bidirectional looks both ways (not causal). **Key trick:** in band-split models the *frequency* direction can stay bidirectional because frequency isn't time — only the time axis must be causal. |
| **cLN / gLN** ★ | Cumulative vs Global Layer Normalisation. Global normalisation uses statistics from the whole utterance, which secretly breaks causality. Causal models must use cumulative (running) statistics. A classic hidden bug. |
| **State / cache** | What a streaming model carries between frames. Fixed-size is good; growing is bad. |
| **MACs / GMAC/s** ★ | Multiply-Accumulate operations, i.e. compute cost. Reported per second of audio. Proxy for whether it fits on-device. |
| **Params** ★ | Parameter count, i.e. model size. REAL-TSE online systems were 15–27 M; a hearable chip wants far less. |
| **QAT** | Quantization-Aware Training. Training the model to survive being squashed to low-precision integers for cheap hardware. |
| **Knowledge distillation** ★ | Training a small model to imitate a big one. |
| **Endpointing** ★ | Deciding that the speaker has finished, so the recognised text can be sent onward. A text output path cannot emit anything until this fires, which is why it is slower than just streaming audio. |
| **Cascade** ★ | Chaining separate models (extractor → ASR → downstream model) instead of one end-to-end system. Each stage adds its own latency, and errors made early cannot be undone later. Your text reference condition is a cascade. |
| **On-device / edge** | Running locally rather than in the cloud. |
| **NPU / DSP / MCU** | Neural Processing Unit / Digital Signal Processor / Microcontroller — the tiny chips you'd deploy on. |
| **GAP9** | A specific ultra-low-power chip used in hearing-device research (TF-MLPNet targets it). **GAPFlow** is its compiler. |

---

## 4. Enrollment and conditioning (how the model knows who to extract)

| Term | Meaning |
|---|---|
| **Enrollment / reference / cue / clue** ★ | The clip of the target speaking alone. All four words mean this. |
| **Conditioning** ★ | Feeding the speaker information into the separation network. *How* and *where* you do this is the main design question in TSE. |
| **Speaker embedding** ★ | Fixed-length vector (often 192 or 256 dims) summarising a voice. |
| **i-vector / d-vector / x-vector** | Successive generations of speaker embedding: statistical factor analysis → early neural → TDNN-based. Mostly historical context now. |
| **ECAPA-TDNN** ★ | *Emphasized Channel Attention, Propagation and Aggregation in Time-Delay Neural Network.* The current default speaker embedding model. Used in the REAL-TSE baselines. |
| **ResNet34 (speaker)** | Alternative embedding architecture. The challenge computes its official SpkSim metric with a WeSpeaker ResNet-34. |
| **Global vs frame-level conditioning** ★ | One vector for the whole enrollment (cheap, loses detail) vs a value per time-frame (expensive, keeps detail). |
| **TF-Map** ★ | Time-Frequency Map. A spectrogram-shaped conditioning signal built from enrollment × mixture magnitudes. Used by the best REAL-TSE baseline and the winner. |
| **Multi-level speaker representation** ★ | Using raw spectral features *and* contextual features *and* a neural embedding together. |
| **Fusion** ★ | The mechanism combining speaker info with audio features. Common types: **concatenation**, **element-wise multiplication**, **FiLM**, **cross-attention**, **speaker token**. |
| **FiLM** | Feature-wise Linear Modulation. Conditioning by learned per-channel scale and shift. |
| **Cross-attention** ★ | Let the mixture "query" the enrollment and retrieve the most relevant parts. The basis of embedding-free methods. |
| **Embedding-free / speaker-encoder-free** ★ | Skipping the pre-trained speaker model entirely; use cross-attention on the raw enrollment instead. (USEF-TSE, SEF-Net, SEF-PNet.) |
| **Speaker token / prefix token** | Injecting speaker info as an extra token in the sequence. |
| **Attractor** | A learned vector that "pulls out" one source. Diarization/separation term. |
| **Adaptation layer** | Older SpeakerBeam term for the layer where the speaker embedding is injected. |

### The characteristic failure modes ★

| Term | Meaning |
|---|---|
| **Speaker confusion** ★ | The model extracts the *wrong* person. The signature TSE failure. |
| **Target-absent** ★ | Stretches where the target isn't speaking at all. The model should output **silence**. |
| **False alarm** ★ | Outputting speech during target-absent regions — usually leaking an interfering speaker. Directly relevant to your "handles interruptions" goal. |
| **Over-suppression** | Being so aggressive that you cut the target's own speech. |
| **Leakage / bleed** | Residual interfering speaker audible in the output. |
| **Musical noise / artefacts** | Unnatural warbling introduced by the processing itself. Often what tanks perceptual quality even when SI-SDR looks fine. |

---

## 5. Architectures and backbones

| Name | What it is |
|---|---|
| **TasNet / Conv-TasNet** | Early, influential time-domain separation network using a learned encoder + temporal convolutions. The historical baseline. |
| **TCN** | Temporal Convolutional Network. Stack of dilated causal convolutions. Fixed receptive field. |
| **DPRNN** | Dual-Path RNN. Introduced the "split the sequence and model along two axes alternately" idea. |
| **Dual-path** ★ | The general pattern: alternate a module along one axis (e.g. frequency) with a module along another (e.g. time). Nearly every modern model is dual-path. |
| **SepFormer** | Transformer-based separation. |
| **TF-GridNet** ★ | Strong time-frequency separation model combining full-band and sub-band modelling. One of the two backbone families in the challenge. |
| **BSRNN** ★ | Band-Split RNN. Splits the spectrum into non-uniform frequency bands, then alternates a band-axis RNN with a time-axis RNN. **The backbone of every top REAL-TSE online system.** |
| **Band-split / sub-band** ★ | Grouping frequency bins into bands — narrow at low frequencies (where voice identity lives), wide at high frequencies (to save compute). |
| **Full-band vs sub-band** | Modelling all frequencies jointly vs in groups. |
| **U-Net** | Encoder–decoder with skip connections. Common in enhancement. |
| **SSM / S4 / Mamba** ★ | State Space Model. A sequence model with a **fixed-size** running state — attractive for streaming because memory doesn't grow. **Mamba** is the popular "selective" variant. |
| **MLP-Mixer** | All-fully-connected architecture that replaces sequential layers with parallel ones. Used by TF-MLPNet for hardware friendliness. |
| **LSTM / GRU / BLSTM** | Recurrent layers. B = bidirectional (not causal). |
| **SSL** ★ | Self-Supervised Learning. Large models pre-trained on unlabelled audio: **WavLM**, **HuBERT**, **wav2vec 2.0**. Used as feature extractors. |
| **Generative TSE** | Instead of masking the mixture, *generate* clean speech. Sub-approaches: **diffusion**, **flow matching**, **autoregressive (AR) language models** over audio tokens, **vocoder** reconstruction. Higher perceptual quality, risk of inventing words that weren't said. |
| **Neural codec / discrete tokens** | Representing audio as a sequence of discrete symbols so language-model techniques apply. |
| **Vocoder** | Converts a compressed/spectral representation back to a waveform. |

### Masking

| Term | Meaning |
|---|---|
| **Mask** ★ | Per-bin multiplier applied to the mixture spectrogram to keep the target. |
| **Ratio mask (IRM) / cIRM** | Ideal Ratio Mask / complex Ideal Ratio Mask. The theoretical best mask; used as an upper bound. |
| **Complex mask** ★ | A mask with real and imaginary parts, so it can correct phase. |
| **Oracle** ★ | A hypothetical system given perfect information (e.g. the ideal mask). Used as a performance ceiling. |

---

## 6. Training objectives and losses

| Term | Meaning |
|---|---|
| **SI-SDR / SI-SNR** ★ | Scale-Invariant Signal-to-Distortion (or -Noise) Ratio. **The default loss and metric.** "Scale-invariant" = doesn't punish you for output volume. Used interchangeably in practice. |
| **SI-SDRi / SI-SNRi** ★ | The **improvement** over the unprocessed mixture. The honest number to report. |
| **PIT / uPIT** | (Utterance-level) Permutation Invariant Training. Solves the "which output is which speaker" problem in BSS. **Not needed in TSE** — a stated advantage. |
| **MRSTFT** ★ | Multi-Resolution STFT loss. Compare magnitude spectrograms at several window sizes. Standard companion to SI-SDR. |
| **Perceptual loss** | Any loss that targets how it *sounds* rather than sample-exact accuracy. |
| **PESQ loss** | Differentiable approximation of the PESQ metric, used directly as a loss (e.g. `torch-pesq`). |
| **Speaker consistency loss** ★ | Penalise the output's speaker embedding for drifting from the enrollment's. |
| **DNSMOS loss** ★ | Directly maximise a frozen DNSMOS predictor. Effective, and exactly how metric-gaming starts. |
| **ASR / CE loss** ★ | Feed output into a frozen ASR and penalise transcription error. Makes output optimised for machine readability — very relevant to your live-model metric. |
| **Feature-matching loss** | Match intermediate activations of a frozen network (e.g. an ASR encoder) rather than the waveform. |
| **VAD / activity loss** ★ | Penalise speaking when the target is silent. Targets the false-alarm failure mode. |
| **Scenario-aware loss** ★ | Apply *different* losses to target-active vs target-silent regions (maximise quality when present, force silence when absent). |
| **Multi-objective / multi-task** ★ | Optimising several of the above at once. The clear trend in the challenge. |
| **Curriculum learning** | Train on easy examples first, then hard. |
| **Teacher–student** ★ | A big "teacher" model supervises a small "student". |
| **Self-training** | Use your own model's outputs as training labels, iteratively. |
| **Fine-tuning / domain adaptation** ★ | Continue training a pre-trained model on data closer to the real target conditions. |
| **On-the-fly mixing** ★ | Generating training mixtures during training rather than pre-writing files. Gives effectively infinite variety. |
| **Model soup / weight averaging** | Averaging the weights of several fine-tuned models. |

---

## 7. Metrics (how systems are scored)

### Intrusive (need a clean reference recording)

| Metric | Range / direction | What it measures |
|---|---|---|
| **SI-SDR / SI-SDRi** ★ | dB, higher better | Signal accuracy. The field's default. |
| **SDR / SIR / SAR** | dB, higher better | From the older BSS_EVAL toolkit. Distortion / interference / artefacts separately. |
| **PESQ** ★ | ~−0.5 to 4.5, higher better | Predicted perceptual quality (ITU-T P.862). Wideband and narrowband variants. |
| **STOI / ESTOI** ★ | 0–1, higher better | Short-Time Objective Intelligibility — predicted understandability. E = Extended. |

### Non-intrusive / reference-free (no clean reference needed) ★

**These matter most to you, because real recordings have no clean reference.**

| Metric | Direction | What it measures |
|---|---|---|
| **DNSMOS** ★ | higher better | Neural predictor of human opinion score. |
| **SIG / BAK / OVRL** ★ | higher better | DNSMOS **P.835** sub-scores: **SIG**nal quality, **BAK**ground noise, **OVR**a**L**. |
| **P808** ★ | higher better | DNSMOS variant following ITU-T P.808 crowdsourcing. **The challenge's official metric**, because OVRL got gamed. |
| **DNSMOS Pro** | higher better | A smaller probabilistic variant. |
| **MOS** ★ | 1–5, higher better | Mean Opinion Score — the *actual human* rating that all the above try to predict. |
| **Subjective listening test** | — | Real humans rating clips. The ground truth for quality. Expensive. |

### Downstream / task-based ★

| Metric | Direction | What it measures |
|---|---|---|
| **WER** ★ | **lower better** | Word Error Rate from an ASR system. Word-level; used for English. |
| **CER** ★ | **lower better** | Character Error Rate. Used for Mandarin (no word boundaries). |
| **TER** ★ | **lower better** | Token Error Rate. The challenge's umbrella term: WER for English, CER for Mandarin. |
| **SpkSim / SIM** ★ | 0–1, higher better | Cosine similarity between speaker embeddings of output and **enrollment**. Note: measured against enrollment, not a clean target — so it's sensitive to device mismatch. |
| **EER** | lower better | Equal Error Rate. Standard speaker-verification metric. |
| **Timing precision / recall / F1** ★ | higher better | Did you emit speech when — and *only* when — the target was actually talking? Your false-alarm/interruption measure. |

### Meta-metrics (for evaluating metrics — relevant to your contribution)

| Term | Meaning |
|---|---|
| **LCC** ★ | Linear Correlation Coefficient (Pearson). How well a metric's values track human MOS. |
| **SRCC** ★ | Spearman's Rank Correlation Coefficient. How well a metric's *ranking* matches humans'. |
| **Metric over-optimisation / gaming** ★ | Raising the score without improving the real quality it stands for. Happened with DNSMOS-OVRL in this challenge. |

### This project's own vocabulary ★

Defined in full in `docs/data/metric-definitions.md`. These are terms you will use
constantly and will not find in any paper, because they are yours.

| Term | Meaning |
|---|---|
| **Judge** ★ | The live speech-to-speech model that scores you by reporting what it understood. Held out from training entirely — never a loss, never a proxy, never a data filter. |
| **LCF** ★ | Live-model Content Fidelity. The metric family: how much of what the target actually said the judge recovered. |
| **LCF-WER** ★ | The headline score. Word error rate of the judge's report against the target's true words. **Lower better.** |
| **ICR** ★ | Interferer Content Rate. How often the judge reports words the *other* speaker said. Stops you scoring well by passing everything through. **Lower better.** |
| **NRR** ★ | Non-Response Rate. How often the judge declines, hears nothing, or returns silence. Stops you scoring well by outputting silence. **Lower better.** |
| **Floor / ceiling** ★ | Mandatory reference rows: the unprocessed mixture (doing nothing) and the clean target (the best extraction could ever do on this judge). A system that doesn't beat the floor is worthless; a score above the ceiling means something is wrong with your harness. |
| **Proxy** ★ | A differentiable stand-in for the judge, used in training because you cannot backprop through an API. Must be a **different model family** from the judge. |
| **Output modality** ★ | Whether your system hands the judge **audio** or **text**. Audio is what we build; text is measured as a reference condition. Must be recorded on every result. |
| **Text reference condition** ★ | Extractor → off-the-shelf streaming ASR → text → judge. A benchmark row, not a build target. Not an upper bound either — an ASR error is permanent, whereas audio leaves the judge acoustic evidence. |
| **Text floor / text ceiling** ★ | The text path's own anchors: ASR run on the raw mixture, and the ground-truth transcript handed over directly. |
| **Front-end ASR vs response ASR** ★ | Easy to confuse and important not to. The **front-end** ASR is part of your system, inside the latency budget, only present in the text condition. The **response** ASR is part of the measuring instrument, outside the budget, used to transcribe the judge's spoken reply in every condition. |
| **Paralinguistics** ★ | Everything in speech that isn't the words: tone, emphasis, hesitation, emotion, identity. Destroyed by the text path, invisible to LCF because LCF is purely lexical. A known blind spot, stated rather than fixed. |

### Reading a results table

- **↓** = lower is better; **↑** = higher is better.
- **Dense ranking** ★ = ties share a rank and the next rank isn't skipped (1, 2, 2, 3). The challenge averaged dense ranks across four metrics to get one score.
- **Ablation study** ★ = remove one component at a time to prove each contributes. Expected in your thesis.

---

## 8. Data concepts

| Term | Meaning |
|---|---|
| **Simulated / synthetic** ★ | Clean recordings digitally added together. You know the exact clean answer. Unrealistic. |
| **Real / in-the-wild** ★ | Genuine recordings of real conversations. Realistic, but **no clean answer exists**. |
| **Re-recorded / replayed** | Clean audio played through speakers and re-recorded in a real room (LibriCSS, REAL-M). A halfway house. |
| **Mixture / target / interference / noise** ★ | Input / wanted / other speakers / non-speech sounds. |
| **Pseudo-label** ★ | A fake "correct answer" generated by a model, used where no real answer exists. |
| **Quality filtering** ★ | Discarding pseudo-labels that fail quality checks so you don't train on garbage. |
| **GSS** | Guided Source Separation. A classical CHiME technique for producing clean-ish targets from real multichannel meeting audio. Another pseudo-label source. |
| **Augmentation** ★ | Artificially varying training data: add noise, add reverb, change speed, apply EQ. |
| **Speed perturbation** | Playing audio at 0.9×/1.1× to create "new" speakers/variation. |
| **min / max mode** ★ | LibriMix options: **min** truncates both utterances to the shorter one; **max** pads to the longer. Papers must state which — the numbers differ. |
| **Train / dev / test split** ★ | Learn / tune / final evaluation. In speech, splits must be **speaker-disjoint**, not just file-disjoint. |
| **Seen vs unseen** ★ | Test conditions that resemble training vs deliberately novel ones. The challenge's EVAL-1 vs EVAL-2. |
| **Blind test set** | Evaluation data whose answers you never see. |
| **Domain mismatch / generalisation gap** ★ | Performing well in training conditions and badly in real ones. The whole motivation for REAL-TSE. |

---

## 9. Datasets and corpora

### Simulated benchmarks

| Name | Notes |
|---|---|
| **WSJ0-2mix** | The historical standard 2-speaker mixture benchmark (from Wall Street Journal read speech). |
| **WHAM!** | WSJ0-2mix plus real recorded noise. |
| **WHAMR!** | WHAM! plus reverberation. |
| **LibriSpeech** ★ | ~1000 h of read audiobooks. Open, CC BY 4.0. The base for most simulated data. |
| **LibriMix / Libri2Mix** ★ | Mixtures generated from LibriSpeech by an open script. **train-100** (58 h, 251 speakers) and **train-360** (212 h, 921 speakers) are the standard subsets. 100% overlap. |
| **SparseLibriMix** ★ | LibriMix with realistic sparse overlap and silence. Much better for testing target-absent behaviour. |
| **Libri2Talker** | LibriMix with target/interferer roles swapped, doubling the data. |
| **LibriCSS** | LibriSpeech replayed through loudspeakers into a real room. |
| **REAL-M** | Real simultaneous read-aloud recordings. |
| **Libri2Vox** | Newer synthetic TSE set with more diverse speaker conditions. |

### Real conversational corpora ★

| Name | Language | Notes |
|---|---|---|
| **AMI** ★ | English | Real meetings. **CC BY 4.0, direct download.** Has synchronous **IHM** (Individual Headset Mic), **lapel**, **SDM** (Single Distant Mic) and **MDM** (Multiple Distant Mic) channels of the same speech — which is how you get real device mismatch for free. Ships human transcripts and speaker annotations. |
| **AISHELL-4** | Mandarin | Real meetings, 8-channel array. Open via OpenSLR. |
| **AliMeeting** | Mandarin | Meetings, near-field headset + far-field array. From the **M2MeT** challenge. Research agreement. |
| **CHiME-5 / CHiME-6** | English | Real dinner parties. Binaural in-ear mics + distant Kinect arrays. Notoriously hard. Licence agreement required. |
| **DipCo** | English | Dinner Party Corpus (Amazon). CC BY 4.0. No official train split. |
| **Fisher** | English | Large telephone conversation corpus. |
| **REAL-T** ★ | EN + ZH | The TSE benchmark built automatically from the five corpora above. Defines **BASE** (easier) and **PRIMARY** (harder) test subsets. |
| **CHiME-9 ECHI** | English | Newer real conversational data aimed at hearing assistance. |

### Speaker recognition corpora (for the conditioning model)

| Name | Notes |
|---|---|
| **VoxCeleb1 / VoxCeleb2** ★ | Large speaker sets from YouTube interviews. VoxCeleb2-dev is the standard training set. Access now requires a request. |
| **CN-Celeb1 / CN-Celeb2** ★ | Mandarin equivalent. Open via OpenSLR. |
| **AISHELL-1** | Mandarin read speech. |
| **VCTK** | Multi-speaker English, studio quality. |
| **EARS** | Anechoic fullband speech, benchmarked for enhancement. |
| **CALLHOME** | Telephone conversations; common diarization test set. |

### Noise and RIR collections

| Name | Contents |
|---|---|
| **MUSAN** ★ | Music, speech and noise. |
| **WHAM! noise** | Real recorded café/restaurant/bar ambience. |
| **DEMAND** ★ | Multi-channel environmental noise: vehicle, home, café, office. Matches the scenarios you care about. |
| **FSD50K** | 50k human-labelled sound events. |
| **FMA** | Free Music Archive — music as interference. |
| **RIRS_NOISES / SLR28** ★ | Standard room impulse response collection. |
| **DNS Challenge data (DNS4/DNS5)** ★ | Large open speech + noise sets from Microsoft's Deep Noise Suppression challenges. |

---

## 10. Toolkits, models and software

| Name | What it is |
|---|---|
| **WeSep** ★ | Target speaker extraction toolkit. **The REAL-TSE baselines are WeSep recipes.** |
| **WeSpeaker** ★ | Speaker embedding toolkit. Provides the ResNet-34 used for the official SpkSim metric. |
| **WeNet** | Speech recognition toolkit from the same family. |
| **ESPnet / ESPnet-SE** ★ | Large end-to-end speech toolkit; has separation and TSE recipes including TF-GridNet. |
| **Asteroid** | PyTorch source-separation toolkit; where LibriMix generation lives. |
| **SpeechBrain** | General-purpose PyTorch speech toolkit. |
| **Kaldi** | The classic pre-deep-learning-era speech toolkit. You'll see its data formats everywhere (`wav.scp`, `utt2spk`, "Kaldi-style"). |
| **k2 / icefall / sherpa-onnx** ★ | Next-generation ASR ecosystem. **icefall** = recipes, **sherpa-onnx** = deployment runtime. Where the challenge's Zipformer models come from. |
| **Zipformer** ★ | The ASR model used for the challenge's official TER metric, chosen because it hallucinates less than Whisper on long real audio. |
| **Whisper** ★ | OpenAI's ASR model. Widely used, but prone to **hallucinated looped repetitions** on long conversational audio — a real problem the challenge repo explicitly warns about. |
| **FireRedASR / FireRedASR2S** | Industrial-grade Mandarin ASR. **FireRedVAD** (inside FireRedASR2S) provides the challenge's activity timing. |
| **Silero VAD** | Lightweight open voice activity detector. |
| **pyannote** | Popular speaker diarization toolkit. |
| **DNSMOS** ★ | Microsoft's released MOS-prediction models (from the DNS Challenge repo). |
| **torch-pesq** | Differentiable PESQ for use as a loss. |
| **ONNX / ONNX Runtime** ★ | Portable model format and runtime — how you'd actually deploy a trained model. |
| **TorchAudio / soundfile / librosa** | Standard Python audio I/O and processing libraries. |

---

## 11. Named methods you'll see cited constantly

Recognising these names is enough; you don't need to read them all.

| Name | One-line significance |
|---|---|
| **SpeakerBeam / td-SpeakerBeam** | The original neural TSE method. Cited in every paper's introduction. |
| **VoiceFilter / VoiceFilter-Lite** | Google's TSE; Lite is the on-device version. |
| **SpEx / SpEx+ / MC-SpEx** | Influential time-domain TSE family. |
| **SEF-Net / SEF-PNet** | Early speaker-embedding-free methods. |
| **USEF-TSE / USEF-TP** ★ | Current embedding-free approach via cross-attention; TP adds personal VAD. |
| **X-TF-GridNet / X-CrossNet / X-SepFormer** | TF-GridNet/SepFormer variants with different speaker fusion. |
| **TEA-PSE (2.0 / 3.0)** | Tencent's industrial low-latency personalised enhancement systems. |
| **E3Net** | Fast real-time personalised enhancement with distillation. |
| **DSINet** | Real-time TSE with dynamic speaker information fusion. |
| **3S-TSE** | Ultra-compact three-stage TSE (0.19 M params). |
| **TF-MLPNet** ★ | Tiny real-time separation for hearable-class chips. |
| **Look Once to Hear** | On-device target speech hearing using a noisy enrollment. |
| **SpeakerBeam-SS** | Real-time TSE using state-space modelling in Conv-TasNet. |
| **SPMamba / SepMamba** | Mamba-based separation. |
| **LauraTSE / TSELM / FlowTSE / MeanFlow-TSE / AlphaFlowTSE** | Generative TSE approaches (language-model and flow-matching based). |
| **CARTSE** ★ | Winner, REAL-TSE online track. |
| **SA-Mamba** ★ | Runner-up; speaker-conditioned state-space model. |
| **PS4** ★ | Proxy-supervised training on real data with differentiable ASR/speaker/VAD/quality losses. |

---

## 12. Venues and challenges

| Name | What it is |
|---|---|
| **ICASSP** | IEEE's flagship signal-processing conference. Annual, ~May. |
| **Interspeech** | The main speech-science conference. Annual, ~September. Run by **ISCA**. |
| **SLT** | IEEE Spoken Language Technology Workshop. **REAL-TSE is an SLT 2026 satellite challenge.** |
| **ASRU** | IEEE Automatic Speech Recognition and Understanding Workshop. |
| **WASPAA** | Workshop on Applications of Signal Processing to Audio and Acoustics. |
| **TASLP** | *IEEE/ACM Transactions on Audio, Speech and Language Processing.* The main journal — journal papers are longer and more detailed than conference ones. |
| **SPM** | *IEEE Signal Processing Magazine.* Where survey/overview papers appear (e.g. the Žmolíková TSE overview). |
| **REAL-TSE** ★ | Your anchor challenge. Real conversational TSE, online and offline tracks. |
| **DNS Challenge** ★ | Microsoft's Deep Noise Suppression challenge. **Source of the standard algorithmic/buffering latency definitions the REAL-TSE online track adopts.** |
| **CHiME** | Long-running series on speech in real environments. |
| **Clarity** | Workshop series on speech-in-noise for hearing devices (where TF-MLPNet appeared). |
| **M2MeT** | ICASSP 2022 multi-party meeting transcription challenge (produced AliMeeting). |
| **MISP** | Multimodal Information-Based Speech Processing challenge. |
| **OpenSLR** ★ | openslr.org — the main open repository for speech datasets. Where many corpora live. |
| **System description paper** ★ | A short paper each challenge team writes describing exactly what they built. Often more practically useful than a polished conference paper. |
| **AoE** | "Anywhere on Earth" — a deadline timezone convention meaning you get until the deadline in the last timezone on Earth. |

---

## 13. Common mathematical notation in these papers

| Symbol | Meaning |
|---|---|
| **x** | the mixture (input) |
| **s** | the target (clean, wanted) |
| **ŝ** ("s-hat") | the estimate — what your model produced |
| **e** | the enrollment |
| **X, S** (capitals) | the same signals in the STFT/frequency domain |
| **F** | number of frequency bins (usually 257) |
| **T** | number of time frames |
| **K** | number of frequency bands after band-splitting (32, 36…) |
| **D / C** | feature or channel dimension (often 128) |
| **M** | the estimated mask |
| **⊙** | element-wise multiplication |
| **θ** | model parameters |
| **ℒ** | a loss function |

---

## 14. Five things that trip up newcomers

1. **Lower is better for error rates** (WER, CER, TER) and higher is better for everything else. Tables mix both, so always read the arrows.
2. **SI-SNR and SI-SDR are used interchangeably** in practice despite technically differing.
3. **"Causal" is easy to break accidentally.** Global normalisation, bidirectional layers on the time axis, and whole-utterance statistics all secretly use the future. Your model can look causal and not be.
4. **The frequency axis is not the time axis.** You can look "both ways" across frequency within a single frame without breaking causality. This confuses almost everyone at first, and it's why band-split models can use bidirectional layers and still stream.
5. **Simulated numbers do not transfer.** A model at 18 dB SI-SDR on LibriMix can be near-useless on real meeting audio. Never compare a score across datasets.