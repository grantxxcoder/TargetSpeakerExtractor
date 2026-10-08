# Reference audit — 2026-10-08

Every `\cite` in `report/` (51 keys, REAL-TSE challenge excluded) checked against the source.
Each verdict comes from text fetched on 2026-10-08. **S** supported · **P** partly supported (says
less, or something different) · **C** contradicted. ★ = re-checked by hand against the source.

**Headline: all 51 references exist. None is hallucinated.** Of 102 claims checked, 79 are
supported, 21 partly supported and 2 contradicted. 16 bib entries need a metadata fix.

## Fix first

| # | Where | Problem | Fix |
|---|---|---|---|
| 1 | `setup.tex:50` | ★ `gemini-3.7-flash` is **not a Live API model** ("Live API: Not supported"), but is cited to the Live API page. Package name fixed to `google-genai` 2026-10-08. | Cite the [model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash); call it a stand-in for the Live model (todo 0c). |
| 2 | `data.tex:50` | ★ WHAM! clips are not 5–20 s. Website: 3.5–47.7 s. Our copy (1,000-clip sample): 4.6–30.2 s, median 10.3 s, 99.2 % inside 5–20 s. | "mostly 5–20 s (median about 10 s)". |
| 3 | `litreview.tex:30` | ★ Speaker identity is **not** near chance (61.5–70 % vs 50 %); pronunciation and speaking rate are. Noise test was GPT-4o-Audio only. | "speaking rate or pronunciation … close to chance"; "GPT-4o-Audio was robust to added noise". |
| 4 | `litreview.tex:34` | ★ Fela & Mowlaee's LLM never hears audio: it classifies intent from Whisper/wav2vec2 transcripts. | Do not present it as a "model as a listener" study. |
| 5 | `baseline_architecture.tex:68` | ★ Luo *does* give a band-specific reason ("different instruments … frequency range and timbres"). | Delete "rather than by an argument specific to bands". |
| 6 | `data.tex:37` | ★ CARTSE uses EQ only in its speaker-encoder fine-tune, prob 0.7. | EQ idea from CARTSE; enrolment-side use at p = 0.5 is this work's. |
| 7 | `baseline_architecture.tex:161` | CARTSE's L_abs means target-**absent**, not "absolute level"; its τ = 10⁻³ floors at −30 dB, ours at −20 dB. | Rename; state the different floor. |
| 8 | `metrics.tex:72` | ★ Le Roux did not introduce SI-SDR: "The scale-invariant SDR (SI-SDR) measure was used in [6, 7, 11, 20–23]". | "formalised by" / "advocated by". |
| 9 | `baseline_architecture.tex:144,158` | ★ Yu's loss windows are 10/20/30/40 ms, no hop given. | Say 8/16/32/64 ms and hop N/4 are this work's choice. |
| 10 | `baseline_architecture.tex:42,61,68` | ★ Yu uses layer norm only offline; "six-layer module" may mean 3 blocks (Luo counts 2 layers per block). | Cite Luo for per-band norm; say "six blocks are used here". |

## Bib entries to fix

| Key | Fix |
|---|---|
| `reddy2021dnsmos` | add doi 10.1109/ICASSP39728.2021.9414878 |
| `reddy2022dnsmos835` | add doi 10.1109/ICASSP43922.2022.9746108 |
| `desplanques2020ecapa` | add pages 3830--3834, doi 10.21437/Interspeech.2020-2650 |
| `wichern2019wham` | add doi 10.21437/Interspeech.2019-2821 |
| `carletta2005ami` | add doi 10.1007/11677482_3 |
| `scheibler2018pyroomacoustics` | add doi 10.1109/ICASSP.2018.8461310 |
| `leroux2019sisdr`, `vincent2006bsseval` | optional dois 10.1109/ICASSP.2019.8683855, 10.1109/TSA.2005.858005 |
| `faster_whisper` | year 2025 (v1.2.1 released 2025-10-31) |
| `jiwer` | year 2025 (v4.0.0 released 2025-06-19) |
| `silero2021vad` | optional: repo's own citation says 2024 |
| `whisper_normalizer` | author Kurian Benoy |
| `itu2003p835` | note "not revised since" is false: new P.835 (07/26) approved 2026-07-29; 2003 edition superseded |
| `radford2023whisper` | note: normaliser is Appendix **D** in the ICML version (C only on arXiv) |
| `whisper_hallucination_2025` | note: paper does not test "confidence thresholding" |
| `li2026cartse` | add url https://real-tse.github.io/assets/pdf/CARTSE-Track1.pdf |
| `zhang2025multi` | optional pages 1--5 |

---

## 1. Architecture: Yu 2023, Luo 2023

**yu2023high** — exists — [ISCA](https://www.isca-archive.org/interspeech_2023/yu23b_interspeech.html) · arXiv 2212.00406 — bib OK
- `figures/baseline_architecture.tex:90` **S** — [§2 Eq. 2](https://arxiv.org/html/2212.00406#S2.E2) — "an MLP is additionally used to directly predict the target speech's complex-valued residual spectrogram"
- `litreview.tex:14` **S** — [§4.2](https://arxiv.org/html/2212.00406#S4.SS2) — "on an Intel i5 2.50GHz CPU, the online PSE model achieves 0.42 real-time factor"
- `baseline_architecture.tex:7` **S** — [Abstract](https://arxiv.org/html/2212.00406#abstract1) — "an additional speaker enrollment module is added to BSRNN … for suppressing the interfering speech" — paper calls it PSE; "interfering speech", not "any interference"
- `baseline_architecture.tex:17` **S** — [§4.2](https://arxiv.org/html/2212.00406#S4.SS2) — "Hanning analysis window … 32 ms and 8 ms for the 16 kHz model" — causal front-padding is ours
- `baseline_architecture.tex:42` **P** ★ — [§4.2](https://arxiv.org/html/2212.00406#S4.SS2) — "layer normalization [31] for offline configuration and batch normalization [32] for online configuration" — per-band norm+FC is Luo §II-A
- `baseline_architecture.tex:44` **S** — [§4.2](https://arxiv.org/html/2212.00406#S4.SS2) — "batch normalization [32] for online configuration"
- `baseline_architecture.tex:56` **S** — [§4.2](https://arxiv.org/html/2212.00406#S4.SS2) — "six-layer band and sequence modeling module with 192 dimensional LSTM" — system-wide, not "their 16 kHz model"
- `baseline_architecture.tex:68` **P** ★ — [§4.2](https://arxiv.org/html/2212.00406#S4.SS2) — same quote — "six-layer" ≠ clearly six blocks
- `baseline_architecture.tex:73` **S** — [§2](https://arxiv.org/html/2212.00406#S2) — "band-specific MLPs to predict the complex-value T-F mask M" — "Yu et al. realised …" reasoning is not in Yu
- `baseline_architecture.tex:75` **S** — [§2](https://arxiv.org/html/2212.00406#S2) — "Regarding the artifacts brought by the complex masks, an MLP is additionally used …" — "It is useful to not bound this mask …" is in neither paper
- `baseline_architecture.tex:82` **S** — [§4.2](https://arxiv.org/html/2212.00406#S4.SS2) — "384-dimensional MLP with Tanh activation function and a gated linear unit (GLU) output layer"
- `baseline_architecture.tex:84` **S** — [§4.2](https://arxiv.org/html/2212.00406#S4.SS2) — "a gated linear unit (GLU) [30] output layer"
- `baseline_architecture.tex:144` **P** ★ — [§3](https://arxiv.org/html/2212.00406#S3) — "STFT window size from [10, 20, 30, 40] ms. We use p = 0.3" — different windows, no hop given

**luo2023music** — exists — [DOI](https://doi.org/10.1109/TASLP.2023.3271145) · arXiv 2209.15174 — bib OK
- `figures/baseline_architecture.tex:89` **S** — [Abstract](https://arxiv.org/html/2209.15174#abstract1) — "splits the spectrogram of the mixture into subbands and perform interleaved band-level and sequence-level modeling"
- `litreview.tex:14` **S** — [§II-B](https://arxiv.org/html/2209.15174#S2.SS2) — BLSTM over time ⇒ non-causal (paper never says "offline")
- `baseline_architecture.tex:7` **S** — [§I](https://arxiv.org/html/2209.15174#S1) — "specially designed for high sample rate signals" — "modelled independently" overstates: band RNN models dependencies between bands
- `baseline_architecture.tex:54` **S** — [§II-B](https://arxiv.org/html/2209.15174#S2.SS2) — "group normalization … BLSTM layer followed by an FC layer … Residual connection"
- `baseline_architecture.tex:68` **P** ★ — [§II-D](https://arxiv.org/html/2209.15174#S2.SS4) — "different instruments may have significantly different frequency range and timbres"
- `baseline_architecture.tex:84` **S** — [§IV-A4](https://arxiv.org/html/2209.15174#S4.SS1.SSS4) — "We use a gated linear unit (GLU) [74] for the output layer of the MLP."

## 2. TF-Map, CARTSE, REAL-T, WeSep

**zhang2025multi** — exists — [IEEE](https://ieeexplore.ieee.org/document/10889409/) · arXiv 2410.16059 — bib OK
- `setup.tex:22` **S** — [§II](https://arxiv.org/html/2410.16059v2#S2) — "The Band-Split RNN (BSRNN) [28] model is selected as the backbone"
- `figures/baseline_architecture.tex:91` **S** — [§II-A](https://arxiv.org/html/2410.16059v2#S2.SS1) — "concatenated with the spectrogram of the mixture"
- `baseline_architecture.tex:7` **P** — [§II-A](https://arxiv.org/html/2410.16059v2#S2.SS1) — paper never cites Yu; the PSE→TSE step is in the WeSep paper §3.3
- `baseline_architecture.tex:27` **S** — [§III-B](https://arxiv.org/html/2410.16059v2#S3.SS2) — "below 1.5 kHz by a 100 Hz bandwidth … This split results in 32 subbands" — bin counts match WeSep code
- `baseline_architecture.tex:88` **S** — [§II-A](https://arxiv.org/html/2410.16059v2#S2.SS1) — "referred to as the TF Map feature"
- `baseline_architecture.tex:105` **S** — [§II-A2](https://arxiv.org/html/2410.16059v2#S2.SS1.SSS2) — "All vectors are length-normalized … cosine similarity is used"; Eq. 2 has no scale (κ is ours) · "energy … is recovered by projecting the amplitude spectrogram of the mixed signal"
- `extension_architecture.tex:6` **S** — same passage
- `extension_architecture.tex:49` **S** — [§II-B](https://arxiv.org/html/2410.16059v2#S2.SS2) — "cross-attention mechanism to generate a time-varying speaker feature" — next sentence ("recomputes the embedding for every frame") is loose: encoder runs once on enrolment; per-frame cost is the cross-attention

**li2026cartse** — exists — [report PDF](https://real-tse.github.io/assets/pdf/CARTSE-Track1.pdf) (1st, Track 1) — bib OK, add URL
- `nomenclature.tex:12` **S** — [p. 1](https://real-tse.github.io/assets/pdf/CARTSE-Track1.pdf#page=1) — "submitted to the online track (Track 1)"
- `introduction.tex:16` **S** — [§IV-A p. 3](https://real-tse.github.io/assets/pdf/CARTSE-Track1.pdf#page=3) — "zero-target masked SI-SDR loss, split by target presence with floor τ=10⁻³"
- `baseline_architecture.tex:125` **S** — p. 3 Eq. 1 — "s_t = (⟨ŝ, s⟩/∥s∥²) s the scaled target projection"
- `baseline_architecture.tex:141` **S** — p. 3 — "floor τ=10⁻³" — CARTSE floors with ∥s∥², we use ∥s_proj∥²; −30 dB rationale is ours
- `baseline_architecture.tex:161` **P** — p. 3 Eq. 2 — "L_abs = η · 10 log10(∥ŝ∥² + τ∥x∥²), η = 2.0" — "abs" = absent; floor differs
- `baseline_architecture.tex:197` **S** — p. 3 Eq. 2 — η = 2.0 multiplies the absent branch
- `data.tex:37` **P** ★ — [p. 2](https://real-tse.github.io/assets/pdf/CARTSE-Track1.pdf#page=2) — "a smooth, RMS-preserving EQ filter (Appendix A) is used only in the speaker-encoder fine-tune"; App. A preset "prob 0.7"

**li2025realt** — exists — [ISCA](https://www.isca-archive.org/interspeech_2025/li25da_interspeech.html) — bib OK
- `data.tex:53` **P** — [§2.3.1 p. 2](https://www.isca-archive.org/interspeech_2025/li25da_interspeech.pdf#page=2) — "overlapping segments with a cumulative overlap duration of at least 5 seconds" — 10–20 s is ours (REAL-T PRIMARY caps at 30 s); REAL-T enrols from the same recording

**wang2024wesep** — exists — [ISCA](https://www.isca-archive.org/interspeech_2024/wang24fa_interspeech.html) · arXiv 2409.15799 — bib OK
- `setup.tex:22` **S** — [§3.3](https://arxiv.org/html/2409.15799v1#S3.SS3) — "WeSep, a toolkit designed for research and practical applications in TSE" — the 27.2 M checkpoint is not described in the paper
- `extension_architecture.tex:49` **P** — [§3.4](https://arxiv.org/html/2409.15799v1#S3.SS4) — utterance-level embeddings only; contextual/TF-Map support is in the [repo](https://github.com/wenet-e2e/wesep) (`wesep/models/tse_bsrnn_spk.py`), not the paper

## 3. LLM-as-judge literature

**manakul2026audiojudge** — exists — [ACL Anthology](https://aclanthology.org/2026.eacl-long.168/) · arXiv 2507.12705 — bib OK
- `litreview.tex:30` rank like humans **S** — [§5.2](https://arxiv.org/html/2507.12705#S5.SS2) — "current LAMs are able to rank speech-in speech-out systems at a reliable level"
- `litreview.tex:30` near chance **P** ★ — [§4.2](https://arxiv.org/html/2507.12705#S4.SS2) — "accuracy on pronunciation (46.0%) and speaking rate (46.9%) approximates random guessing" — speaker ID 61.5→70.0 %
- `litreview.tex:30` robust to noise **S** — [§6.1](https://arxiv.org/html/2507.12705#S6.SS1) — "GPT-4o-Audio maintains robust performance against noise perturbations" — one model only
- `litreview.tex:30` lexical vs non-lexical **S** — [§7](https://arxiv.org/html/2507.12705#S7) — "still struggles with non-lexical judgments, often performing near random chance"
- `litreview.tex:34` rater **S** — [§3.1](https://arxiv.org/html/2507.12705#S3.SS1) — "the LAM directly compares two audio responses to determine which is better"
- `litreview.tex:34` position bias **S** — [§6.3](https://arxiv.org/html/2507.12705#S6.SS3) — "GPT-4o-Audio, Gemini-1.5-Flash, and Gemini-2.5-Flash all favor the first position"
- `litreview.tex:34` white noise only **S** ★ — [§6.1](https://arxiv.org/html/2507.12705#S6.SS1) — "incrementally adding white Gaussian noise"

**zheng2023judging** — exists — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html) · arXiv 2306.05685 — bib OK
- `litreview.tex:32` >80 % **S** — [§1](https://arxiv.org/html/2306.05685#S1) — "agreement rate exceeding 80%, achieving the same level of human-human agreement"
- `litreview.tex:32` biases **P** — [§3.3](https://arxiv.org/html/2306.05685#S3.SS3) — "our study cannot determine whether the models exhibit a self-enhancement bias" — position and verbosity supported

**yang2024airbench** — exists — [ACL Anthology](https://aclanthology.org/2024.acl-long.109/) — bib OK
- `litreview.tex:32` **S (reword)** — [§4.4](https://arxiv.org/html/2402.07729#S4.SS4) — "scoring twice by interchanging the positions of the hypothesis and reference and calculating the average" — hypothesis vs reference, averaged, text-only GPT-4 judge, "mitigate" not "prevent"

**foo2026glitters** — exists — [arXiv 2604.24401](https://arxiv.org/abs/2604.24401) — bib OK (preprint)
- `litreview.tex:32` **S** — [§1](https://arxiv.org/html/2604.24401#S1) — "even without audio input, models retain 60–72% of their full-audio accuracy" — multiple-choice audio-QA benchmarks

**koenecke2024careless** — exists — [DOI](https://doi.org/10.1145/3630106.3658996) — bib OK
- `litreview.tex:32` **P** — [§4.1](https://arxiv.org/html/2402.08021#S4.SS1) — "perhaps implying that Whisper's over-reliance on … language modeling … leads to hallucinations" — hypothesis; measured factor is long pauses
- `metrics.tex:49` **P** — [§2.4](https://arxiv.org/html/2402.08021#S2.SS4) — "entire hallucinated phrases or sentences which did not exist in any form in the underlying audio" — phrase level; never says "plausible"

**chondhekar2025denoising** — exists — [arXiv 2512.17562](https://arxiv.org/abs/2512.17562) — bib OK
- `litreview.tex:32` **S** — [§4.4](https://arxiv.org/html/2512.17562#S4.SS4) — "Original noisy audio achieves lower semWER than enhanced audio in all 40 tested configurations (4 models x 10 conditions)" — one enhancer (MetricGAN+)

**fela2026perceptually** — exists — [arXiv 2608.30348](https://arxiv.org/abs/2608.30348) (EMNLP 2026) — bib OK
- `litreview.tex:34` **S (reframe)** ★ — [§4.2](https://arxiv.org/html/2608.30348#S4.SS2) — "no metric provides meaningful signal about which clips will diverge" — LLM reads ASR transcripts, not audio

## 4. TSE and evaluation literature

**zmolikova2023overview** — exists — [DOI](https://doi.org/10.1109/MSP.2023.3240008) · arXiv 2301.13341 — bib OK
- `litreview.tex:8` **S** — [p. 2](https://arxiv.org/pdf/2301.13341#page=2) — "the key difference between TSE and BSS and noise reduction"
- `litreview.tex:10` silence **S** — [p. 17](https://arxiv.org/pdf/2301.13341#page=17) — "output no signal when the target speaker is inactive, which may not actually be the case with most current systems"
- `litreview.tex:10` SDR **P** — [p. 17](https://arxiv.org/pdf/2301.13341#page=17) — "may not always be correlated with human perception and intelligibility" — wrong speaker *does* lower SDR; the issue is SDR can't separate identification from extraction errors
- `litreview.tex:10` streaming (uncited sentence) **S** — [p. 17](https://arxiv.org/pdf/2301.13341#page=17) — "Research on lightweight and low-latency TSE systems is gaining momentum" — add the `\cite`
- `litreview.tex:18` **P** — [p. 9](https://arxiv.org/pdf/2301.13341#page=9) — "must be able to see enough context in the mixture" — not "entire clip"

**cherry1953cocktail** — exists — [DOI](https://doi.org/10.1121/1.1907229) — bib OK
- `litreview.tex:8` **S** — [p. 976](https://jontalle.web.engr.illinois.edu/Public/CherrySpeech2Ears53.pdf#page=2) — "how do we recognize what one person is saying when others are speaking at the same time (the "cocktail party problem")?"

**zhang2021closing** — exists — [DOI](https://doi.org/10.1109/WASPAA52581.2021.9632720) · arXiv 2110.14139 — bib OK
- `litreview.tex:22` **S** ★ — [Table 1 p. 4](https://arxiv.org/pdf/2110.14139#page=4) — Test (Real): noisy CH5 19.5 %, MC-Conv-TasNet* 41.5 % — name the CHiME-4 real test set (Dev Real: 10.9 → 18.5 %)

**iwamoto2022bad** — exists — [ISCA PDF](https://www.isca-archive.org/interspeech_2022/iwamoto22_interspeech.pdf) — bib OK
- `litreview.tex:26` **S (scope)** — [§5 p. 4](https://www.isca-archive.org/interspeech_2022/iwamoto22_interspeech.pdf#page=4) — "the impact of noise errors is relatively limited, while the impact of artifact errors is particularly detrimental to ASR" — speech enhancement, not TSE

**luo2019convtasnet** — exists — [DOI](https://doi.org/10.1109/TASLP.2019.2915167) — bib OK
- `litreview.tex:14` **S** — [p. 1](https://arxiv.org/pdf/1809.07454v3#page=1) — "a long temporal window for the calculation of STFT. This requirement increases the minimum latency"

**hershey2016deep** — exists — [DOI](https://doi.org/10.1109/ICASSP.2016.7471631) — bib OK
- `litreview.tex:22` **S** — [§3.1 p. 5](https://arxiv.org/pdf/1508.04306#page=5) — "a new dataset of speech mixtures based on the Wall Street Journal (WSJ0) corpus"

**cosentino2020librimix** — exists — [arXiv 2005.11262](https://arxiv.org/abs/2005.11262) — bib OK (@misc correct)
- `setup.tex:22` **S** — [§2.2 p. 2](https://arxiv.org/pdf/2005.11262#page=2) — "Libri2Mix and Libri3Mix, consist of clean and noisy, two- and three-speaker mixtures" — "WeSep trained on clean Libri2Mix" rests on the WeSep citation
- `litreview.tex:22` **S** — [Abstract](https://arxiv.org/pdf/2005.11262#page=1) — "wsj0-2mix has become the reference dataset for single-channel speech separation"

**reddy2021dnsmos** — exists — [DOI](https://doi.org/10.1109/ICASSP39728.2021.9414878) — add DOI
- `litreview.tex:24` **S** — [Abstract](https://arxiv.org/pdf/2010.15258#page=1) — "a high correlation to human ratings"
- `metrics.tex:58` **S** — [§1](https://arxiv.org/pdf/2010.15258#page=1) — "trained using the ground truth human ratings obtained using ITU-T P.808"

**reddy2022dnsmos835** — exists — [DOI](https://doi.org/10.1109/ICASSP43922.2022.9746108) — add DOI
- `metrics.tex:58` **S** — [Abstract](https://arxiv.org/pdf/2110.01763#page=1) — "output 3 scores: i) speech quality (SIG), ii) background noise quality (BAK), and iii) the overall quality (OVRL)"

## 5. ML and statistics classics

**dauphin2017language** — exists — [PMLR](https://proceedings.mlr.press/v70/dauphin17a.html) — bib OK
- `baseline_architecture.tex:84` **S** — [§2 Eq. 1](https://proceedings.mlr.press/v70/dauphin17a/dauphin17a.pdf#page=2) — "We dub this gating mechanism Gated Linear Units (GLU)."

**he2016deep** — exists — [DOI](https://doi.org/10.1109/CVPR.2016.90) — bib OK
- `baseline_architecture.tex:54` **S** — [§3.2 p. 772](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf#page=3) — "The operation F + x is performed by a shortcut connection and element-wise addition."

**desplanques2020ecapa** — exists — [ISCA](https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.html) — add pages + DOI
- `extension_architecture.tex:47` **S** — [§4.1 p. 3](https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.pdf#page=3) — "The number of nodes in the final fully-connected layer is 192." — checkpoint ([`lin_neurons: 192`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/blob/main/hyperparams.yaml)) trained on VoxCeleb1+2

**loshchilov2019adamw** — exists — [arXiv 1711.05101](https://arxiv.org/abs/1711.05101) (ICLR 2019) — bib OK
- `setup.tex:11` **S** — [§2 p. 3](https://arxiv.org/pdf/1711.05101v3#page=3) — "our variant of Adam with decoupled weight decay (AdamW)"

**leroux2019sisdr** — exists — [DOI](https://doi.org/10.1109/ICASSP.2019.8683855) — bib OK
- `metrics.tex:72` **P** ★ — [§1 p. 2](https://arxiv.org/pdf/1811.02508#page=2) — "The scale-invariant SDR (SI-SDR) measure was used in [6, 7, 11, 20–23]." — not "introduced by"

**vincent2006bsseval** — exists — [DOI](https://doi.org/10.1109/TSA.2005.858005) — bib OK
- `metrics.tex:62` **P** — [§II-A](https://inria.hal.science/inria-00544230/document#page=4) — "ŝj = s_target + e_interf + e_noise + e_artif" — four terms; we fold noise into interference — say it is adapted

**efron1979bootstrap** — exists — [DOI](https://doi.org/10.1214/aos/1176344552) — bib OK
- `metrics.tex:80` **P** — [§2 p. 3](https://sites.stat.washington.edu/courses/stat527/s13/readings/ann_stat1979.pdf#page=4) — "selected with replacement from the set {x1, x2, …, xn}" — introduces the bootstrap, not the percentile interval. Cite Efron for the bootstrap and Bisani & Ney (Eq. 7) for the interval.

**bisani2004bootstrap** — exists — [DOI](https://doi.org/10.1109/ICASSP.2004.1326009) — bib OK
- `metrics.tex:80` **S** — [§3 Eq. 4 p. 2](https://www-i6.informatik.rwth-aachen.de/PostScript/InterneArbeiten/Bisani_BootstrapEstimatesForConfidenceIntervalsInASRPerformanceEvaluation_ICASSP_2004.pdf#page=2) — "repeated B times (typically B = 10³ … 10⁴)"; Eq. 4 = total errors / total reference words

**sabine1922** — exists — [archive.org](https://archive.org/details/collectedpaperso00sabiuoft) — bib OK
- `data.tex:43` **S** — [p. 43](https://archive.org/details/collectedpaperso00sabiuoft/page/43/mode/1up) — "a = absorbing power of the room … s = area of wall … V = volume … aT = k = KV"

## 6. Datasets and ASR

**panayotov2015librispeech** — exists — [DOI](https://doi.org/10.1109/ICASSP.2015.7178964) — bib OK
- `data.tex:19` **S** — [Abstract](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf#page=1) — "a new corpus of read English speech … derived from audiobooks"
- `metrics.tex:43` **P** — [§2.3–2.4 p. 2](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf#page=2) — "a good likelihood of having accurate transcripts" — not "exact"

**wichern2019wham** — exists — [ISCA](https://www.isca-archive.org/interspeech_2019/wichern19_interspeech.html) — add DOI
- `data.tex:19` **P** — [§2 p. 2](https://www.isca-archive.org/interspeech_2019/wichern19_interspeech.pdf#page=2) — "coffee shops, restaurants, bars, office buildings, parks" — no "streets"
- `data.tex:50` **C** ★ — [wham.whisper.ai](http://wham.whisper.ai/) — "shortest clip being 3.5 seconds and the longest 47.7 seconds"

**carletta2005ami** — exists — [DOI](https://doi.org/10.1007/11677482_3) — add DOI
- `appendices.tex:180` **S** — [§3 p. 3](https://publications.idiap.ch/attachments/reports/2005/carletta-2005-mlmi.pdf#page=3) — "the participants play the roles of employees"
- `data.tex:48`, `data.tex:53` **S** — [Abstract](https://publications.idiap.ch/attachments/reports/2005/carletta-2005-mlmi.pdf#page=1) — "a multi-modal data set consisting of 100 hours of meeting recordings"

**scheibler2018pyroomacoustics** — exists — [DOI](https://doi.org/10.1109/ICASSP.2018.8461310) — add DOI
- `appendices.tex:51` **S** — [§3 p. 3](https://arxiv.org/pdf/1710.04196#page=3) — "The RIR generator is based on the ISM … shoe box, i.e. rectangular"
- `data.tex:33` **S** — [Abstract](https://arxiv.org/pdf/1710.04196#page=1) — "a fast C implementation of the image source model"

**radford2023whisper** — exists — [PMLR](https://proceedings.mlr.press/v202/radford23a.html) — fix note (App. D)
- `appendices.tex:27`, `metrics.tex:7` **S** — [App. C (arXiv)](https://arxiv.org/pdf/2212.04356#page=21) — "normalize English texts in different styles into a standardized form"
- `appendices.tex:28`, `setup.tex:45` **S** — [Table 1](https://arxiv.org/pdf/2212.04356#page=5) — "Large 32 1280 20 1550M"

**whisper_turbo** — exists — [HF model card](https://huggingface.co/openai/whisper-large-v3-turbo) — bib OK
- `appendices.tex:28`, `setup.tex:45` **S** — "a finetuned version of a pruned Whisper large-v3 … decoding layers have reduced from 32 to 4"; 809 M vs 1550 M

**anastassiou2024seedtts** — exists — [arXiv 2406.02430](https://arxiv.org/abs/2406.02430) — bib OK
- `setup.tex:45` **S** — [§3.1](https://arxiv.org/html/2406.02430v1#S3.SS1) — "we employ Whisper-large-v3 … as the automatic speech recognition (ASR) engines"

**wang2025solospeech** — exists — [arXiv 2505.19314](https://arxiv.org/abs/2505.19314) — bib OK
- `setup.tex:45` **S** — [§V-B](https://arxiv.org/html/2505.19314v3#S5.SS2) — "we performed ASR … using the Whisper large-v3-turbo model"

**whisper_hallucination_2025** — exists — [DOI](https://doi.org/10.1109/ICASSP49660.2025.10890105) — fix note
- `setup.tex:56` **P** — [§V-B](https://arxiv.org/html/2501.11378v1#S5.SS2) — "an effective VAD, such as SileroVAD, yields a significant reduction in WER … as well as the incidence of hallucinations" — Whisper only; VAD concatenates segments rather than gating; "none of these approaches can be considered a complete solution" vs our "covers all possible scenarios"

## 7. Standards, software, products

| Key | Exists | Claim | Link |
|---|---|---|---|
| `itu2003p835` | yes; note out of date | **S** SIG/BAK/OVRL scales | [ITU](https://www.itu.int/rec/T-REC-P.835-200311-S/en) |
| `itu2018p808` | yes | **S** crowdsourced ACR | [summary](https://www.itu.int/dms_pubrec/itu-t/rec/p/T-REC-P.808-201806-S!!SUM-HTM-E.htm) |
| `itu2015bs1770` | yes (superseded by -5, 2023) | **S** gated loudness (LKFS = LUFS) | [ITU](https://www.itu.int/rec/R-REC-BS.1770-4-201510-S/en) |
| `bird2009nltk` | yes | **S** English stopword corpus | [ch. 2 §4.1](https://www.nltk.org/book/ch02.html) |
| `faster_whisper` | v1.2.1 yes; year 2025 | **S** large-v3-turbo, int8 | [release](https://github.com/SYSTRAN/faster-whisper/releases/tag/v1.2.1) |
| `jiwer` | v4.0.0 yes; year 2025 | **S** | [PyPI](https://pypi.org/project/jiwer/4.0.0/) |
| `whisper_normalizer` | 0.1.15 yes; author K. Benoy | **S** EnglishTextNormalizer | [source](https://github.com/kurianbenoy/whisper_normalizer/blob/0.1.15/whisper_normalizer/english.py) |
| `silero2021vad` | v6.2.1 yes | **S** | [release](https://github.com/snakers4/silero-vad/releases/tag/v6.2.1) |
| `ravanelli2021speechbrain` | yes | **S** VoxCeleb ECAPA, 192-dim | [model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| `google2026geminilive` | yes | `intro:3` **S**; `setup:50` **C** ★ | [Live API](https://ai.google.dev/gemini-api/docs/live) · [gemini-3.7-flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash) |
| `anthropic2026claude` | yes | **S** product exists | [models](https://platform.claude.com/docs/en/about-claude/models/overview) |

Not checked: `wang2026realtse` (excluded by request). 17 bib entries are never cited and were not checked.
