# Novelty review — variational presence gate for streaming TSE

Searched 2026-10-14. Two parallel searches: neural target-speaker / personalised systems, and the probabilistic / variational side. Subject: the gate in `docs/extra/variational-presence-gate.md`. A per-frame target-presence probability $\pi_t$ multiplies e21's output. It is trained with an ELBO-style objective: the loss on the gated output, plus β·KL to a two-state Markov prior built from run lengths, plus an optional BCE on frame labels. A binary Concrete sample is used in training.

**Verdict: novel only as a combination and an application. Every component has prior art.** The gate itself is published (USEF-TP, 2025). Gating an output by a speech-presence probability is 1980s–2001 enhancement. The Markov prior on activity goes back to Sohn 1999. The closed-form KL follows from Ghahramani & Jordan 1997. No paper was found that trains a target-extraction gate as a variational posterior under a Markov activity prior. Confidence that this combination is unpublished: about 65–70 %.

**Claim it as:** *a causal TSE presence gate trained through a relaxed ELBO-style objective with a two-state Markov prior on target activity.* **Do not claim:** a new variational method, a first Markov prior on activity, or a first presence gate.

**AI-use.** Both searches and the novelty judgement were done by Claude (2026-10-14).

---

## 1. Closest existing work

| Paper | Venue | What it shares with the gate | What it lacks |
|---|---|---|---|
| Zeng & Li, "Universal Speaker Embedding Free Target Speaker Extraction and Personal Voice Activity Detection" (USEF-TP) | Computer Speech & Language 2025 *(volume and pages unconfirmed)*; arXiv:2501.03612 | per-frame target-active head (BCE), upsampled and **multiplied onto the extractor output** before the inverse STFT; joint training | offline (BLSTM); no prior; no variational objective; no Concrete |
| Zhang, Wu & Li, "Dialogue-Based Streaming Audio-Visual Target Speaker Extraction with Predictive Dialogue Information" | arXiv:2609.30774 (Sep 2026, submitted to ICASSP 2027) | **streaming** extractor with a per-frame target-activity head (BCE), joint training | hard threshold (0.4, attenuate to 0.05); audio-visual; no prior |
| Bando, Nakamura & Watanabe, "Neural blind source separation and diarization for distant speech recognition" | Interspeech 2024; arXiv:2406.08396 † | **per-frame binary activity latent** multiplying each speaker's spectrum; factorised Bernoulli posterior; ELBO + BCE | multichannel, non-causal blind separation; authors state no prior is defined for the activity |
| Mao, Fontaine, Larcher & Badeau, "...with Beta speaker activity prior" | Interspeech 2026; arXiv:2608.28661 † | activity prior + ELBO with a closed-form KL | frame-independent Beta prior; **lists a conversational-dynamics activity prior as future work**, which is the step the Markov prior takes |
| Zhang, Borsdorf, Pan, Li, Wei & Wang, "Speaker Extraction with Detection of Presence and Absence of Target Speakers" (TSEJoint) | Interspeech 2023, 3714–3718; doi:10.21437/Interspeech.2023-655 | presence branch sharing modules with SpEx+; joint training | one decision per clip; hard zeroing |

† = full text read.

**Consequence for the design.** Arm B (β = 0, λ > 0) is essentially a causal USEF-TP-style gate, so treat it as a literature baseline, not only an ablation. The thesis question becomes arm A against arm B.

## 2. Prior art by component (must be cited)

| Component | Prior art | Status |
|---|---|---|
| Enrolment-conditioned per-frame target-presence head | Ding et al., "Personal VAD: Speaker-Conditioned Voice Activity Detection", Odyssey 2020, arXiv:1908.04284 (pages unconfirmed). Ding et al., "Personal VAD 2.0", Interspeech 2022, arXiv:2204.03793. Medennikov et al., "Target-Speaker Voice Activity Detection" (TS-VAD), Interspeech 2020, 274–278, arXiv:2005.07272 | confirmed |
| Head trained jointly with the extractor and multiplied onto its output | USEF-TP; arXiv:2609.30774; TSEJoint (clip-level). Boeddeker et al., "TS-SEP", IEEE/ACM TASLP 32:1185–1197, 2024, doi:10.1109/TASLP.2024.3350887. Zhao, Chen, Yu & Wang, "Continuous Target Speech Extraction", arXiv:2401.15993 (venue unconfirmed; models trained separately) | confirmed except as marked |
| Output gain multiplied by a speech-presence probability (classical) | McAulay & Malpass, "Speech enhancement using a soft-decision noise suppression filter", IEEE TASSP 28(2):137–145, 1980 (via Cohen 2001's references; no URL found). Malah, Cox & Accardi, "Tracking speech-presence uncertainty...", ICASSP-99, 789–792. Cohen & Berdugo, "Speech enhancement for non-stationary noise environments", Signal Processing 81:2403–2418, 2001 † | confirmed (McAulay & Malpass via citation only) |
| Speech-presence probability estimation | Cohen, "Noise spectrum estimation in adverse environments: IMCRA", IEEE TSAP 11(5):466–475, 2003. Gerkmann & Hendriks, "Unbiased MMSE-based noise power estimation...", IEEE TASLP 2012, doi:10.1109/TASL.2011.2180896 (volume and pages unconfirmed) | confirmed except as marked |
| First-order Markov model of speech activity | Sohn, Kim & Sung, "A statistical model-based voice activity detection", IEEE SPL 6(1):1–3, 1999 (hang-over as a first-order Markov process). Landini et al., "Bayesian HMM clustering of x-vector sequences (VBx)", CSL 71, 2022, doi:10.1016/j.csl.2021.101254 | confirmed |
| Geometric run lengths, mean $1/(1-a_{ii})$ (eqs. 6–8) | Rabiner, "A tutorial on hidden Markov models...", Proc. IEEE 77(2):257–286, 1989, eq. (5) † | confirmed |
| Closed-form KL, factorised posterior vs Markov prior (eq. 16) | Ghahramani & Jordan, "Factorial hidden Markov models", Machine Learning 29:245–273, 1997, doi:10.1023/A:1007425814087, §3.4 (completely factorised variational inference; the expected log-transition cross-term) † | confirmed. **Present (16) as a derivation following §3.4, not a contribution.** |
| Discrete latent with a Markov prior, variational, in enhancement | Sadeghi & Alameda-Pineda, "Switching variational auto-encoders for noise-agnostic audio-visual speech enhancement", ICASSP 2021, arXiv:2102.04144 † (keeps the chain structure in the posterior) | confirmed |
| VAE / dynamical-VAE speech enhancement (background) | Bando et al., ICASSP 2018, arXiv:1710.11439 †. Leglaive, Girin & Horaud, MLSP 2018, arXiv:1902.01605. Leglaive et al., ICASSP 2020, arXiv:1910.10942. Girin et al., "Dynamical VAEs", FnT ML 15(1–2), 2021, doi:10.1561/2200000089 | confirmed |
| Binary Concrete / hard-concrete gate | Maddison, Mnih & Teh, ICLR 2017, arXiv:1611.00712. Jang, Gu & Poole, ICLR 2017, arXiv:1611.01144. Louizos, Welling & Kingma, ICLR 2018, arXiv:1712.01312 | confirmed (already in the note's bib, except Louizos) |
| Loss as a Gibbs pseudo-likelihood (eq. 11) | Bissiri, Holmes & Walker, "A general framework for updating belief distributions", JRSS-B 78(5):1103–1130, 2016, doi:10.1111/rssb.12158. Alquier, Ridgway & Chopin, JMLR 17(236), 2016, arXiv:1506.04091. Knoblauch, Jewson & Damoulas, JMLR 23(132), 2022 | confirmed |
| Over-suppression vs leakage trade-off (the AMI fine-tune finding) | Taherian, Eskimez & Yoshioka, "Breaking the trade-off in personalized speech enhancement with cross-task knowledge distillation", arXiv:2211.02944 (ICASSP 2023 unconfirmed). Eskimez et al. 2022, arXiv:2110.09625 (already in `evaluation-sources.md`) | confirmed except as marked |

## 3. Corrections the search forces on the derivation note

1. **Call it a "relaxed ELBO-style objective", not an ELBO.** It uses a relaxed Concrete sample in the extraction term but the analytic discrete KL (16). Maddison et al. (2017, App. C.2) show this is not guaranteed to be a lower bound. The pseudo-likelihood (11) already makes (10) non-strict for SI-SDR.
2. **β and κ are not separately identifiable.** Only their product scales the KL. The note already folds κ into β; say explicitly that β is the single temperature.
3. **The posterior is mean-field.** A factorised posterior cannot represent temporal correlation, which Ghahramani & Jordan (1997) argue the structured approximation captures better. Sadeghi & Alameda-Pineda (2021) keep the chain in the posterior. State this as a limitation.
4. **(16) is not Claude-originated maths.** It follows from Ghahramani & Jordan (1997) §3.4. The note's AI-use line should credit Claude with applying it here, not with deriving a new result.

## 4. Already in the repo

- **`report/references.bib`:** `delcroix2022listen`, `zmolikova2023overview`, `sato2021separate`, `zhang2020xtasnet`, `wang2026realtse`, `li2026cartse`, `liu2026samamba`.
- **`literature/review_synthesis.md` notes USEF-TP and PS4 (arXiv:2607.08111), but neither is in the bib.**
- **`literature/evaluation-sources.md`:** Borsdorf 2021, Eskimez 2022. Not in the bib.
- **Not cited anywhere yet:** Personal VAD, TS-VAD, TSEJoint, arXiv:2609.30774, Bando 2024, Mao 2026, Taherian, TS-SEP, Sohn, Rabiner, Ghahramani & Jordan, Bissiri.

## 5. Not checked

- Full text of Zhang, Liu, Yang & Yang, "Gated Cross-Attention for Universal Speaker Extraction", Electronics 13(11):2046, 2024 (HTTP 403), so whether its gate is per frame or per clip is unknown.
- Exact pages and volumes as marked above. Ephraim 1992 (HMM enhancement) volume (40 or 41). MacKay 1997 "Ensemble learning for HMMs" venue.
- HMM/Viterbi smoothing of TS-VAD outputs in diarisation systems. These very likely exist and are why "first Markov prior on activity" must not be claimed.
- Sekiguchi et al.; non-English venues; patents; full ICASSP and Interspeech 2025–2026 coverage; REAL-TSE system reports beyond MERL's.
- arXiv:2609.12154 (Sep 2026), seen only in search results.
