# Report proofread, 2026-10-29

Grammar and typo pass over the working tree (uncommitted edits included). Spell check (aspell, en_GB): clean. `(opt)` = judgement call. Nothing in `report/` was edited.

## Fix first: the meaning is wrong
- [x] abstract.tex l.2 "Perceptual metrics do follow the ranking" → "do **not** follow" (results.tex l.5 says they do not)
- [x] results.tex l.189 "degrades the target speech at the cost of removing the background interfering speech" → reversed: "removes the interfering speech at the cost of degrading the target speech"
- [x] results.tex l.173 "Lower is better in every column." → "in every row" (metrics are rows)
- [x] results.tex l.116 "neither fine-tune improves" → "none of the fine-tunes improves" (there are three)
- [x] data.tex l.40 "compared to the validation set of" → as written, the −48.6 dB reads as the constructed set's level; say which set each number belongs to

## frontmatter/abstract.tex
- [ ] l.2 "holding the same conclusions as the simulated data" → "with the same conclusions as on the simulated data"
- [x] l.2 "at the cost of increased rate" → "at the cost of an increased rate"
- [ ] l.2 "talk, however the different data strategy" → "talk; however, the different data strategy"
- [ ] l.2 "by dropping the word error rate from 62.94" → run-on list: "…: the word error rate drops from …, the rate … drops …, and the share … drops …"
- [ ] (opt) l.2 "and states the relationship" → "and to state the relationship"
- [ ] (opt) l.2 "standard band separation architecture" → "band-split", as earlier in the abstract
- [ ] (opt) l.2 "two or more words fabricated" → "invented", as earlier in the abstract

## frontmatter/acknowledgements.tex
- [ ] l.5 "Guidelines for using GenAI tools in Learning and Assessment" → LaTeX quotes ``…''
- [ ] l.12 "not given access to writing directly to the report" → "not given write access to the report"
- [ ] l.12 "were sourced by the original authors" → "sourced from"
- [ ] l.13 "specifically I prompted" → "For \autoref{fig:pipeline} specifically, I prompted"
- [ ] l.15 "the REAL TSE paper" → "REAL-TSE"
- [ ] l.17 "Equations and Formulations:" → "Equations and formulations:"
- [x] l.20 "two state target present Markov model" → "two-state target-presence Markov model"
- [ ] l.20 "cited this identity to the paper by" → "attributed this identity to"
- [ ] l.22 "Created Metrics:" → "Created metrics:"
- [ ] (opt) l.5 "code was needed" → "were needed"; "and then cases where" → "and in cases where"
- [ ] (opt) l.12 "was not clear or inaccurate" → "was unclear or inaccurate"
- [ ] (opt) l.15 "the baseline and extension model" → "models"
- [ ] (opt) l.16 "but rather for setup" → "but rather serve setup"
- [ ] (opt) l.19 "I used Claude to verify" → "Claude Code", as elsewhere
- [ ] (opt) l.20 "until the KL term myself" → "up to the KL term"; "mean field assumption" → "mean-field"
- [ ] (opt) l.22 "the interferer rate" → "interferer content rate"

## introduction/introduction.tex
- [x] l.12 "adapted by CARTSE" → "adapted from CARTSE"
- [ ] (opt) l.5 "though every measurement" → ", though"; l.12 "Finally to test" → "Finally, to test"

## litreview/litreview.tex
- [x] l.1 "Literature Review" → "Literature review"
- [x] l.8 "can understand" → no subject: "…: using visual, spatial and acoustic cues, a listener can understand…"
- [x] l.10 "absent target trials" → "target-absent trials"
- [x] l.18 "whether given a causal extractor" → "whether giving"
- [ ] l.26 "and secondly another word" → run-on: "Suppose one word comes from the interfering speaker and another was said by neither speaker; both can appear…"
- [ ] l.30 "as opposed to non-lexical content" → "than on non-lexical content"
- [ ] l.32 "agreed with human preferences" → "agrees"; comma before "over 80 %"
- [ ] l.32 "Gemini Flash 2.0" → "Gemini 2.0 Flash"
- [ ] (opt) l.3 "The section presents" → "This section"
- [ ] (opt) l.10 "target speaker always present" → "always being present"; "signal level metrics" → "signal-level"; "low latency streaming" → "low-latency"
- [x] (opt) l.22 "signal based metrics" → "signal-based"; SDR and WER expanded twice
- [x] (opt) l.26 "ground truth transcription" → "ground-truth"
- [ ] (opt) l.32 "positional bias which is an idea" → "AIR-Bench, which scores … positional bias, an idea"
- [ ] (opt) l.32 "audio language models keep" → "LAMs" (defined l.30); "too as found by" → "as found by" (found twice)
- [ ] (opt) l.34 "presented to the LLM" → "LAM"

## methodology/data.tex
- [x] l.12 "which is a dataset" → comma before: "WHAM!~\cite{…}, which is a dataset"
- [x] l.21 "and hard trials (40" → "speech, and hard trials"; also "difficult case" vs "hard trials": pick one
- [ ] l.21 "\textit{Case 1}, both speakers present, \textit{Case 2}" → separate the cases with semicolons
- [ ] l.21 "Finally the noise clips" → "Finally, the noise clips"
- [ ] l.21 "the other speaker alone" → l.33 says "interferer only"; one label per case
- [ ] l.25 "first attempted to be drawn" → "is drawn first from a different book; failing that, a different chapter; failing that, a different utterance"
- [ ] l.25 "by applying a random EQ filter to the clip" → "applying" twice, clips/clip: "This is done by applying a random EQ filter to each enrolment clip…"
- [ ] l.31 "which covers a carpeted office to a large room" → "which ranges from"
- [ ] l.31 "rather than sampled and a room" → "rather than sampled, and a room"
- [x] l.35 "trials, consisting of" → "trials drawn from 1172 unique speakers"
- [x] l.40 "background noise, thus the enrolment" → "background noise, so the enrolment"
- [x] l.40 "rather than a perfect one" → "rather than a clean reference"
- [ ] (opt) l.6 "every trial has its unprocessed mixture" → "must have", to match (i)–(iii)
- [ ] (opt) l.8 "enrolment quality and injecting" → "enrolment quality, and injecting"
- [ ] (opt) l.12 "considering environments such as" → "covering"; "is towards a base case" → "is on a base case"
- [ ] (opt) l.17 "drawn from WHAM!." → drop the "."
- [ ] (opt) l.25 "manner} that the target speaker speaks" → "manner} in which"; "Again to avoid bias" → "Again, to avoid bias"
- [ ] (opt) l.33 "49.5\% both" → "49.5\,\%"
- [ ] (opt) l.40 "As a final test, the extractor is tested on real recordings" → repeats l.38; delete one

## methodology/baseline_architecture.tex
- [x] l.11 "rather than a waveform" → "rather than on the waveform"
- [x] l.15 "and used by the challenge baseline" → no verb: "…, which is also used by the challenge baseline"
- [x] l.20 "Single frame normalisation" → "Single-frame"
- [x] l.35 "Yu et al. realised" → "Yu et al.\ realised"
- [x] l.35 "the drowned out target" → "drowned-out"
- [x] l.35 "It is useful to not bound this mask and is what" → "Leaving this mask unbounded is useful, and is what…"
- [x] l.45 "remains and does not keep the loudness" → "remains, not the loudness"
- [x] l.62 "lie between $[0, 1]$" → "lie in $[0, 1]$"
- [x] l.62 "the similarities which sharpens" → "the similarities, which sharpens"
- [x] l.77 "when it is present, otherwise, output silence" → "when it is present, and output silence otherwise"
- [ ] l.112 "Finally the term is evaluated" → "Finally, the term…"; "window lengths namely:" → "window lengths, namely" (no colon)
- [x] l.124 "the target present case" → "target-present"
- [ ] l.147 "(occurs in" → "(this occurs in"
- [x] l.159 "(Multi-Resolution)" → "(multi-resolution)"
- [ ] (opt) l.4 "Band-split Recurrent Neural Network" → sentence case
- [x] (opt) l.6 "high sample rate audio" → "high-sample-rate"
- [ ] (opt) l.15 "The split is 257 frequency bins into 32 bands" → "The 257 frequency bins are split into 32 bands"
- [ ] (opt) l.71 "The extra channel is the only addition as it widens" → "as" reads as "because"
- [ ] (opt) l.97 "It views" → "First, it treats…", to pair with "Secondly"
- [ ] (opt) l.100 "it does not constrain" → "the loss does not constrain"; "low and high frequency bands" → "low- and high-frequency"
- [ ] (opt) l.115 "The model should penalise" → "The loss should penalise"
- [ ] (opt) l.147 "can draw a trial" → "can come from a trial"; "rewarded for silence where the loss" → "…, and that reward should not be diluted"
- [ ] (opt) l.151 "on the silent branch" → "absent branch", as named elsewhere
- [ ] (opt) l.158 "The reason for this term" → "this weight"
- [ ] (opt) l.159 "\num{-5.91}" vs l.161 "5.9088": one precision

## methodology/extension_architecture.tex
- [x] l.20 "once multiplying the loudness and the similarity into a scalar" → "once the loudness and the similarity are multiplied into a scalar"
- [x] l.43 "the single channel cue" → "single-channel"
- [x] l.49 "is to consider utterance-level embedding" → "is to use an utterance-level embedding, and the second is to use the contextual embedding of…"
- [x] l.54 "is present by $z_t$" → two "by"s: "gated by the presence variable $z_t$:"
- [ ] l.66 "the average of the band states, $\hat{h}_t$" → comma splice: "states and $\hat{h}_t$"
- [ ] l.66 "and let $\psi" → "with weights … and bias …. Let $\psi = (\omega, \omega_0)$."
- [x] l.71 "mean field assumption" → "mean-field" (also l.194)
- [x] l.72 "Markov Prior." → "Markov prior."; "first order Markov" → "first-order"
- [x] l.88 "is a geometric distribution" → "$D$ follows a geometric distribution"
- [x] l.93 "sample the mean speech run length" → no subject; "the sample mean speech-run length … is the maximum likelihood estimate …; treating silence runs the same way gives" *(hyphen done; rest open)*
- [ ] l.161 "Closed Form KL." → "Closed-form KL."; "posterior has a closed form" → "posterior it has a closed form" *(hyphen done; rest open)*
- [x] l.180 "Recalling that being in state" → "Recall that … Bernoulli distribution. Therefore, for the first frame only the state is considered…"
- [ ] l.212 "binary Concrete distribution" → end the sentence with ":" (a display follows)
- [x] l.218 "where $\tau$ is annealed, linearly per epoch" → "draw. $\tau$ is annealed linearly per epoch…"
- [ ] (opt) l.212 "is an unbiased estimate of the expectation" → ", which gives an unbiased estimate"

## methodology/metrics.tex
- [ ] l.7 "and the interferer speaker" → "and of the interfering speaker" (drop the comma before it)
- [x] l.7 "contractions expanded and numbers written consistently" → "expanding contractions and writing numbers consistently"
- [x] l.22 "Content Words." → "Content words."
- [x] l.33 "\frac{\ell_i}{a_i}," → "." (next line starts a sentence)
- [x] l.40 "neither present in the target nor the interferer" → "present in neither the target nor the interferer"
- [x] l.49 "chosen in this work." → "," (continues with "where")
- [x] l.56 "overall quality (OVRL) and ITU-T P.808" → "(OVRL), and"
- [ ] l.58 "Signal to Distortion/Interference/Artefact Ratios" → "Signal-to-distortion, -interference and -artefact ratios" *(hyphen done; rest open)*
- [x] l.71 "Reference Points" → "Reference points"
- [x] l.73 "worst case scenario" → "unprocessed, to a listener as the worst-case scenario" *(hyphen done; rest open)*
- [x] l.73 "passing a clean target audio to the listener as best case" → "the clean target audio … as the best case"
- [ ] (opt) l.7 "When a system produces output, either the offline" → "When a system's output is passed to either the offline recogniser or the live model…"
- [ ] (opt) l.7 "The two anchors get transcribed as well" → "The two anchors, the untouched mixture and the clean target, are transcribed as well"
- [ ] (opt) l.9 "Listener Content Fidelity Word Error Rate" and l.18, l.38, l.54 headings: Title Case vs sentence case
- [ ] (opt) l.13 WER display → end with ","
- [ ] (opt) l.22 "filler words" / "Function words" / "stopword": one name for one concept
- [ ] (opt) l.40 "hallucinated a word" → "invented"; "and is only computable" → ". It is only computable"
- [ ] (opt) l.41 "the mean invented count per trial of the number of words" → "the mean number of words per trial in the model output that are present in neither…"
- [ ] (opt) l.52 "as defined by \cite{koenecke2024careless}" → "Koenecke et al.~\cite{…}"
- [ ] (opt) l.73 "bounds of the results section" → "used in"; floor/ceiling vs anchors vs reference points: pick one

## experiments/experiments.tex
- [x] l.3 "the voice specific additions" → "voice-specific"

## experiments/setup.tex
- [x] l.17 "a shortlist of best candidate checkpoints" → "of the best"
- [x] l.21 "for both baseline and the extension" → "both the baseline"
- [x] l.21 "which is slight decrease" → "a slight decrease"
- [x] l.21 "its epoch four score" → "epoch 4" (paragraph writes "Epoch 11")
- [x] l.28 "with TF-Map cue of" → "with the TF-Map cue of"
- [x] l.45 "which was connected to via" → "which was queried via"
- [ ] l.49 "a trial will cut short" → "will be cut short"
- [x] l.49 "Gemini 3.7 Flash however can receive" → "Gemini 3.7 Flash, however, can receive"
- [x] l.49 "listens, thus Live only provides" → "listens, so Live only provides"
- [x] l.53 "without the need of calling the listener" → "without calling the listener"
- [ ] (opt) l.11 "stops improving" → "stopped improving" (tense)
- [ ] (opt) l.17 "chosen by the model with the lowest validation loss" → "chosen as the checkpoint with"; "where the lowest WER is chosen as the final model" → "and the checkpoint with the lowest LCF-WER is chosen"
- [ ] (opt) l.21 "validation WER" → "LCF-WER" (this is Gemini, not Whisper)
- [ ] (opt) l.23 "for the context of" → "in the context of"
- [ ] (opt) l.28 "): baseline BSRNN" → "the baseline BSRNN"
- [ ] (opt) l.45 "hallucinates" → "invents words"
- [ ] (opt) l.53 "reduces inventing words on silence" → "reduces invented words on silence"; "a target said but were suppressed" → "the target said but that were suppressed"; "secondly shows correct output of silence" → "shows a correct silent output"

## results/results.tex
- [x] l.5 "Crucially however, despite" → "Crucially, however, despite"
- [x] l.5 "the speaker specific information provides the extractor meaningful knowledge" → "speaker-specific … provides the extractor with" *(hyphen done; rest open)*
- [x] l.50 "but rather due to the simplistic nature" → "but rather to the simplistic nature"
- [ ] l.52 "however, when considering" → comma splice: "; however,"; also "Table~\ref" → "\autoref"
- [ ] l.52 "The metric in itself is subjective to the actual perceptual system" → "The metrics themselves are specific to human perception"
- [x] l.52 "does not provide indication" → "does not provide any indication"
- [x] l.72 "including less interfering speaker's content" → "less of the interfering speaker's content"
- [x] l.72 "subsequently has fabricated/invented more words" → "consequently invents more words"
- [x] l.72 "both baseline and the extension" → "both the baseline and the extension"
- [x] l.116 "any artefact related fine-tune" → "artefact-related"
- [ ] l.125 "more often than WeSep getting 18 of the 42 clips silenced as compared against" → "more often than WeSep, silencing 18 of the 42 clips compared with WeSep's 7"
- [x] l.132 "Figure \ref{fig:voice_picking}" → "\autoref{fig:voice_picking}"
- [x] l.132 "In the same gender case" → "same-gender"
- [x] l.141 "the worst case delay" → "worst-case"; "196 milliseconds" → "196\,ms", as in the same paragraph
- [ ] l.189 "can be seen by the target words missed metric" → "can be seen in the target-words-missed metric"
- [x] l.189 "Finally there is an increase of artefacts" → "Finally, there is an increase in artefacts"
- [ ] (opt) l.20 "the target from the background noise (distortion)" → non-parallel list
- [ ] (opt) l.28 "The signal decomposition and DNSMOS predicted opinion scores" → caption fragment
- [ ] (opt) l.50, l.72 "the floor" → "no processing", the label the figures use
- [ ] (opt) l.52 "with a clear lowest LCF-WER" → "with clearly the lowest LCF-WER"
- [ ] (opt) l.72 "a trade-off with the number of hallucinated words" → "at the cost of more invented words"; "has both reduced the share" → "has reduced both the share"
- [ ] (opt) l.72 "which indicates how the model introduces" → "which suggests the models introduce"
- [ ] (opt) l.116 "By weighting by a different error" → dangling: "Weighting a different error changes the masking behaviour"
- [ ] (opt) l.118 "masking architectures" → "architecture"
- [ ] (opt) l.125 "where the extension shines" → "where the extension does best"
- [ ] (opt) l.132 "which is indicative that" → "which indicates that"
- [ ] (opt) l.137 "two-voice validation pairs" → "two-speaker"
- [ ] (opt) l.183 "invented /trial" → "invented per trial"
- [ ] (opt) l.196 "baseline of the REAL-TSE model" → "the REAL-TSE baseline" (REAL-TSE is a challenge, not a model)

## conclusion/conclusion.tex
- [x] l.6 "if more target will be recovered" → "if more of the target will be recovered"
- [x] l.6 "the signal metric cannot give" → "metrics"
- [x] l.6 "and does not necessarily have" → "and do not necessarily have"
- [x] l.6 "The cue was split into parts and adding" → "Splitting the cue into parts and adding the speaker embedding lowered"
- [ ] l.9 "can confound the extractor to expect" → "may lead the extractor to expect"
- [x] l.12 "rather be trained on the two speaker condition rather than" → drop the first "rather"; "two-speaker" *(hyphen done; rest open)*
- [x] l.12 "general target present condition" → "target-present"
- [x] l.12 "a gate that when a frame is shared" → "a gate that, when a frame is shared …, estimates" (drop "it")
- [ ] (opt) l.6 "include how perceptual measures" → "include that"
- [ ] (opt) l.12 "I believe" → impersonal; "state space layers" → "state-space"

## appendices/appendices.tex
- [ ] l.105 "The target's headset and it also picks up" → "The target's headset, which also picks up"
- [ ] l.130 "The transition matrix of the Markov prior" → no main verb: "… set by \eqref{…} is:"
- [ ] (opt) l.111 "same-sex" → "same-gender", as in data.tex

## figures/
- [x] training_curves.tex l.10 "target specific components" → "target-specific"
- [ ] (opt) training_curves.tex l.10 "\autoref{eq:total}" → "\eqref{eq:total}"
- [ ] (opt) pipeline.tex l.60 "(causal, real time)" → "real-time"; l.90 "The output passes a speech gate" → "passes through"
