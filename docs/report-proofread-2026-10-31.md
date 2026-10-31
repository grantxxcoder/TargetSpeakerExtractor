# Proofread, 2026-10-31 (Claude Code)

Spelling: clean (aspell en_GB found only labels, acronyms and technical terms; spelling is consistently British).
Format: where → now → suggested. Nothing here is applied to `report/`.

## A. Factual accuracy (fix first)
1. AI declaration AI2: "The tool was not given access to writing directly to the report" is no longer true: Claude Code edited report/*.tex at the author's instruction (2026-10-30/31). Reword, e.g. "…and, at my instruction, applied agreed corrections directly to the report files."
2. AI declaration AI3: figures 5.1–5.6 were drawn by plotting scripts Claude Code wrote from the author's specification, not only "reformatted". Also the test-set stems rebuild and per-case scoring scripts (2026-10-30/31). See docs/ai-declaration-todo.md.
3. setup.tex augmentation: "for each trial per epoch" and "a single random 4 s crop … for each epoch". Persistent workers fixed crops, enrollment variant and remix per Kaggle session, not per epoch.
4. setup.tex: "L_abs … is off for the first 4 epochs". It was 0 for 4 epochs, then ramped in over 3.
5. setup.tex Live judge: "This allows for a clear distinction for whether the model hallucinates as the status field…". The status-field check was contradicted (0/6). Drop or soften.
6. setup.tex curves: "L_MR consistently decreases … showing how the extractor is able to reconstruct the audio with fewer artefacts". e21 validation L_MR 0.281 is above the mixture's 0.270; say it improves over training, not that artefacts are fewer.
7. setup.tex gate: "with the oscillation in the validation loss" vs "validation stays flat between 12.5 and 18.5". Pick one ("stays flat").
8. metrics.tex intro: "…the offline recogniser's word error rate are the conventional measures compared against them". No Whisper result appears in §5. Remove or add the comparison.
9. setup.tex latency: never defines the 80 ms chunk + 40 ms lookahead that Fig 5.5 uses.
10. results.tex §5.1: "The three kinds of separation … the target from the background noise (distortion)". SDR is total error, and SIR covers the other speaker and the noise (metrics.tex). Reword to match §3.4.6.

## B. Consistency
11. Results text says "the floor" (e.g. "14.67 points lower than the floor", "the floor of 1.48", "lower than the floor for … SIG") while tables say "No processing". Use "no processing".
12. "live judge" / "live model" for Gemini 3.7 Flash, which is not Live. Pick one name ("Gemini" or "the judge").
13. Abstract: "two or more words fabricated" vs "invented" everywhere else.
14. Validation labels still missing: §5.3 voice-picking sentence; §3.3 "Measured on the baseline architecture…" (71.8 / 52.6 %).
15. AI declaration AI8: "the interferer rate" → "the interferer content rate". AI5: "REAL TSE" → "REAL-TSE".
16. P2 decimals still one place: data.tex shares 49.5/25.5/19.6/5.3, 5.8 %, −48.6/−29.2 dB, 71.8/52.6, WeSep 27.2 M; w_m text −5.91/0.184 vs equation 5.9088/0.1842.

## C. Grammar and wording
### Abstract
17. "This work builds … This work analyses …": two openers in a row. Merge or vary.
18. "An evaluation suite is thus proposed to consider both errors and states the relationship between them." → "An evaluation suite is proposed that measures both errors and the relationship between them."
19. "The extractor built is a causal band-split network" → "The extractor is a causal band-split network".
20. "The model is seen to perform in real time with a 175 ms latency" → "The extractor runs in real time with a mean delay of 175 ms".
21. "evaluated on real recordings, holding the same conclusions as the simulated data" → "On real meeting recordings (AMI), the same conclusions hold."
22. "beats just passing the mixture … by dropping … , the rate … drops …, the share … drops …" mixes "by dropping" and "drops". Make parallel.
### Introduction
23. "though every measurement uses Gemini 3.7 Flash as its stand-in and is used to select checkpointed models" → "…as its stand-in, which is also used to select checkpoints."
24. "An evaluation suite in which the listener's errors are categorised into leaked versus invented words is developed." → "This work develops an evaluation suite that splits the listener's errors into leaked and invented words."
25. "Secondly, a causal implementation …" has no "Firstly". "Finally to test" → "Finally, to test".
### Literature review
26. l.3 "The section presents and reviews … and is structured to present the gaps of the research questions." → "This section reviews … and is structured around the gaps behind the research questions."
27. "benefit from the ability for more robust speech recognition" → "benefit from more robust speech recognition". "The survey done by" → "The survey by".
28. "The confound of the target speaker always present during training" → "Training with the target always present".
29. "either a raw waveform or spectrogram" → "or a spectrogram". "set a benchmark to say that a system must run with" → "requires a system to run with".
30. Zhang et al.: "…more than doubled its word error rate (WER) compared with the unprocessed input on real recordings from 19.5 % to 41.5 %" → "…raised the word error rate on real recordings from 19.5 % (unprocessed) to 41.5 %".
31. Double space in "such as  DNSMOS".
32. "Suppose a word is heard from the interfering speaker, and secondly another word that neither speaker said and was invented, both can appear…" → "A word leaked from the interfering speaker and a word that neither speaker said can both appear…".
33. AIR-Bench sentence is one run-on and links position bias to role swapping loosely (27 Oct list: delete). "Gemini Flash 2.0" → "Gemini 2.0 Flash".
34. "This was explored by Foo et al. where audio language models keep…" → "Foo et al. found that audio language models keep…". "The prior was found to influence hallucination too as found by" (found twice).
35. "In Manakul et al. the study uses the model as a rater." → "Manakul et al. use the model as a rater."
### Data
36. "These considerations limit the available choices of publicly available datasets" → "limit the choice of public datasets".
37. "Speech was chosen for target and interferer from the LibriSpeech dataset" → "Target and interferer speech were drawn from LibriSpeech". "noise recordings considering environments such as" → "recorded in environments such as". "The primary focus is towards a base case" → "The primary focus is a base case".
38. Case list: use semicolons ("Case 1, both speakers present; Case 2, …"). "Finally the noise clips" → "Finally, noise clips".
39. Enrollment: "The enrollment clip is first attempted to be drawn from a different book reading, otherwise…" → "The enrollment clip is drawn from a different book where possible, otherwise from …". "Again to avoid bias" → "Again, to avoid bias". "This is done by applying EQ … by applying a random EQ filter" (applying twice).
40. "which covers a carpeted office to a large room" → "covering rooms from a carpeted office to …".
41. Trial cases: "The split is as follows for these trials" → say which set ("in the training set").
### Baseline architecture
42. "The baseline architecture section describes the architecture that was used as a backbone in…" → "This section describes the architecture used as the backbone of…".
43. Band plan: "no reason is given in the original paper … and no explanation is given in previous papers" (repeats).
44. "a deliberate design choice that opposes Yu et al." → "departs from".
45. "The band and sequence model is the implementation of the residual LSTM block … in a stack of six blocks again following" → "stacks six residual LSTM blocks (§), again following".
46. Estimator: "Yu et al. realised it would be more efficient to mask" → "Yu et al. mask the mixture rather than…". "as the multiplication operation would not be able to scale the drowned-out target" repeats the previous clause. "Leaving this mask unbounded is useful, and is what the original implementation encouraged" is vague.
47. TF-Map: "but they use softmax directly on the inner product rather than scaling it by a factor κ = 16" → "but without the scaling factor κ = 16 used here". "After testing it was found that" → "In testing,".
48. "which will be calculated as a projection" → "computed as a projection". "which is a suggestion from CARTSE" → "as suggested by CARTSE".
49. "Finally the term is evaluated at several window lengths namely: 8, 16, 32 and 64 ms." → "Finally, the term is evaluated at window lengths of 8, 16, 32 and 64 ms." "the places where most errors would occur can be expected to exist at frame boundaries" → "errors can be expected at frame boundaries".
50. "The model should penalise any output that is not silence." → "The loss should penalise…".
51. "(occurs in 5.8 % of cases). The model should therefore be rewarded for silence where the loss should not be diluted…" → "(5.8 % of crops). Such a crop should reward silence, without that reward being diluted by…".
52. w_m: "The reason for this term is that" → "This weight is needed because". "in units with respect to a ratio in decibels" → "a ratio in decibels". "which will simulate the scenario for the average losses for an untrained model" → "which approximates the losses of an untrained model".
### Extension architecture
53. "Even split into parts, the cue" → "Even when split into parts, the cue".
54. "The idea for this extension is to incorporate further target speaker identity. The second extension is to add a speaker embedding" → "The second extension therefore adds a speaker embedding for each target speaker."
55. "ECAPA itself is large at 20.77 million parameters and hence would cause storage concerns" → the concern is compute, not storage: "ECAPA is large (20.77 M parameters), but it runs once per enrollment clip…".
56. "The first (which is the one used in this work) is to use an utterance-level embedding" → "The first, used in this work, is an utterance-level embedding".
57. "no explicit, underlying mechanism" → "no explicit mechanism". "The gate represents a two-state Markov chain" → "The gate models target presence as a two-state Markov chain". "Here h_t is …, ĥ_t is …" → "…, and ĥ_t is …".
58. "This idea was suggested by Kingma et al. in an image classification context" → "This follows Kingma et al., who used it for image classification".
59. "The KL term of (…) is intractable, but under the mean-field posterior has a closed form" contradicts itself → "The KL term of (…) has a closed form under the mean-field posterior." "term 1" / "Term 2" are never defined.
60. "Instead leverage Monte Carlo sampling" has no subject → "Instead, Monte Carlo sampling is used". "will potentially not change when π_t is changed slightly" → "does not change for a small change in π_t".
### Metrics
61. "When a system produces output, either the offline or the live model with the prompt of Appendix D.1, it gives a transcript. The two anchors get transcribed as well…" → "Each output is transcribed by the offline recogniser or by Gemini (prompt in Appendix D.1), and so are the two reference points, the mixture and the clean target."
62. "the target speaker, and the interferer speaker" → "the target speaker and of the interfering speaker".
63. ICR: "the first metric used to highlight one particular nuance of the LCF-WER metric" → "the first of two measures that split the LCF-WER's extra words by source". "These content words will be used to make" → "are used to build".
64. FR: "The FR metric is defined as follows: Over the set … the statistic is the mean invented count per trial of the number of words…" is garbled → "Over the set T of trials with a non-empty response, FR_count is the mean number of words per trial that appear in neither transcript".
65. DNSMOS: "predicts what a collection of humans would have rated" → "predicts the score a panel of human listeners would give".
66. Reference points: "The following terms are used to describe the lower and upper bounds of the results section." → "Two reference points bound the results." Uncertainty: "unique speakers are chosen rather than individual trials" → "whole speakers are redrawn rather than individual trials".
### Experiments
67. "The training setup is described below and considers how…" → "This section covers how…". Same for the evaluation-setup opener.
68. "If a clip is only used once for a single speaker, the answer remains the same which can be memorised." → "With one direction only, each mixture always has the same answer, which can be memorised."
69. Model selection: "The final model to base the evaluation on cannot simply be chosen by the model with the lowest validation loss as a result of the proxy training objective." → "The final model cannot simply be the one with the lowest validation loss, because the loss is only a proxy." "where the approach in the proxy is to reward better separation" (redundant). "Finally a consideration is that Gemini is also too slow" → "Finally, Gemini is too slow". "Whisper … during training will create a shortlist" → "During training, Whisper shortlists…". "where the lowest WER is chosen" → "and the lowest LCF-WER is chosen".
70. Curves: "A positive note to be seen is how" → "Encouragingly,". "as by the validation live judge results it can be seen that on the final epoch the validation WER has returned to its epoch 4 score" → "because by its final epoch its validation WER had returned to its epoch-4 level". "although sporadic" → "although noisy". Caption: "\autoref{eq:total}" → "\eqref".
71. Systems: "Secondly, WeSep itself has a causal separator" follows no "Firstly".
72. Flash vs Live: "Gemini Live will reply then to a fragment" → "would then reply to a fragment". "so Live only provides additional difficulty for few justifiable reasons" → "so using Live would add difficulty for little benefit".
73. Speech gate: "This method is supported by [x] where the authors show that this method reduces inventing words on silence" → "This follows [x], who show that a VAD reduces words invented on silence". "The gated output also shows missed words that a target said but were suppressed" → "The gate also exposes target words the extractor suppressed".
74. Latency: "time to consume and time to process the audio" → "time to receive the audio and time to process it". "This forces a delay of the processing of the window as per §" → define the 80 ms chunk + 40 ms lookahead (item 9).
### Results
75. §5 intro: "This section covers the three research questions with results to answer them individually." → "This section answers the three research questions in turn."
76. §5.1: "The baseline has the highest model SAR metric" → "the highest SAR of the models". "Therefore by the signal metrics alone, WeSep cannot be established as a better separation model, and it does not explain…" ("it" has no subject).
77. §5.2: "This came at a trade-off with the number of invented words. The baseline system consequently invents more…" → "This came at the cost of more invented words: the baseline invents…".
78. Fine-tunes: "Four fine-tunes were run on the extension where 6 additional epochs were trained." → "Four fine-tunes of the extension were each trained for 6 further epochs." "Adding the artefact weight cut invented words only on top of the leak weight but against the control, the change is within the noise" needs a semicolon and the interval: "…on top of the leak weight [−1.28, −0.25]; against the control, the change is within the noise […]". "Crucially, it is a hypothesis that these errors are not trade-offs of each other, but a by-product of the trade-off of the masking architecture. By weighting by a different error" → "The hypothesis is that the two errors do not trade off directly, but both follow from how strongly the mask suppresses. Weighting a different error changes the masking behaviour."
79. Heading "The trade-off of the masking architectures." → singular. "a single frame only contains raw audio of unknown mixture quantities of different sources" → "a frame is a mixture of sources in unknown proportions".
80. §5.3: "The extension system is a clear improvement over the baseline, which is indicative that … but it must now be considered whether…" (run-on; split). "This is demonstrated by Fig 5.4" → "Fig 5.4 shows".
81. §5.4: "A demonstration of how the model distorts the speech can be seen by the target words missed metric, where Gemini is able without any processing to only miss 3.46 %" → "The target words missed show how the extractors distort the speech: with no processing Gemini misses only 3.46 % of the target's words, but even with WeSep it misses 27.71 %." "which is an expected result from the simulated dataset" → "as on the simulated data".
82. Closing: "The extension has been shown to beat the causal re-implementation in this work of the baseline of the REAL-TSE model, in which additional speaker information can be added…" → "The extension beats this work's causal re-implementation of the REAL-TSE baseline, showing that additional speaker information can be added…".
### Conclusion
83. "The general findings include how perceptual and signal domain measures do not align" → "The first finding is that perceptual and signal-domain measures do not align". "do not necessarily have a direct correlation" → "do not necessarily reflect".
84. "The word error rate alone does not discern how insertions or substitutions are obtained" → "does not show where its insertions and substitutions come from". "This involved considering the enrollment and the conditioning…" is vague.
85. Limitations: "which can confound the extractor to expect the target to be present if two speakers are present" → "which may teach the extractor to expect the target whenever two voices are present". "Another limitation is that Gemini can change as it is a closed model both between runs and versions." → "Gemini is a closed model that can change between runs and versions."
86. Future work: "The first suggestion is considering" → "The first suggestion concerns". "and secondly it did not have a training recipe built around one" → "and has no training recipe built around one". "The next suggestion would be to consider" → "The second suggestion concerns". "Right now the main issue is the discernment of how much to mask the interferer." → "The main open problem is deciding how much to mask the interferer." "a gate that, when a frame is shared by both speakers, rather than just a target being present, estimates…" → "a gate that, when both speakers share a frame, estimates how much of it belongs to each".
### AI declaration
87. "redundant tasks" (twice) → "routine tasks" ("redundant" means unnecessary).
88. "sourced by the original authors" → "sourced from". "\autoref{fig:pipeline} specifically I prompted" → "For Figure 3.1 specifically, I prompted". "in which the tool would make experiments" → "and the tool ran experiments".
89. AI6b: "I was able to derive the expressions by hand until the KL term myself" → "I derived the expressions by hand up to the KL term". "it cited this identity to the paper by" → "attributed this identity to". Double space after the citation.
