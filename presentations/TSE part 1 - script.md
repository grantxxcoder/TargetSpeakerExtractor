# TSE part 1 — 5-minute script

~720 words ≈ 4:50. Every beat maps to a card on the slide, in slide order.
Bracketed labels are the card you should be looking at.

---

## Slide 1 — Title (0:00–0:15)

Real-time target speech extraction.

In one sentence: I take a noisy recording with two people talking, pull out
the one voice I care about, live — and I score the result by whether a live AI
model understands what that person said.

---

## Slide 2 — Context of the problem (0:15–1:15)

**[01 / Innate Human Ability]**
You already do this. In a crowded restaurant you lock onto one voice and
everything else becomes background. It costs you no effort at all, and it is
still not solved in machines.

**[02 / Edge AI Integration]**
The reason it matters now is that listening is moving into everyday devices —
smart glasses, hearables — which have to make sense of whatever acoustic
environment their wearer walks into.

**[03 / Real-Time AI Transcription]**
And increasingly the thing consuming that audio is not a person. It is a live
AI model — Gemini Live and its equivalents. Those work beautifully in a quiet
room and fall apart the moment a second person is in it. So my setting is
robust voice understanding, in chaotic environments, for an AI listener.

---

## Slide 3 — Problem formulation (1:15–2:55)

**[the equation]**
Formally: what the model hears is a mixture — my target speaker, one
interfering speaker, and background noise. It also gets a few seconds of the
target talking alone, which is the only thing telling it which voice to keep.

**[spectrogram]**
The coloured stripes are the four situations inside one clip: target alone,
both speakers overlapping, interferer alone, noise only. The top panel is what
the model hears. The second panel is what it has to produce.

**[Technical Challenges]**
The technical difficulty is conditioning. The model has to transform its
internal speech representation using very little information about the target
— a few seconds of enrolment audio, and nothing else. It has to hold onto who
it is listening for across the whole recording.

**[Hardware Constraints]**
The constraint is that this runs streaming. No looking ahead, no waiting for
the sentence to finish — roughly a two hundred millisecond budget, and a model
small enough to be worth deploying.

**[Primary Objective]**
And this card is the actual research contribution. The objective is to
maximise downstream live AI transcription accuracy. Not signal quality, not
how nice it sounds — what the AI listener recovers.

Those are not the same thing, and that is the whole point. Cleaning up audio
leaves holes and artefacts. An offline transcriber shrugs those off; a live
model listens through its own learned encoder and appears to be more upset by
the artefacts you added than by the voice you removed. So a system can improve
on every conventional score and get worse at the only job I care about.

The two smaller goals under it: stay silent when the target isn't speaking,
and suppress the noise.

---

## Slide 4 — Current project state (2:55–4:35)

Three pillars, and all three now exist.

**[01 / Dataset Construction]**
The data is built, not borrowed. Twenty-one thousand mixtures, a hundred and
five hours — two-speaker scenarios, controlled noise levels, real recorded
noise and room reverb. Built rather than borrowed because training needs a
clean target signal and the exact words both speakers said, and real meeting
recordings give you neither.

**[02 / Baseline Signal Extraction]**
The baseline is a time-domain masking model. It streams at about a hundred and
sixty milliseconds on a laptop CPU, inside budget. The performance curve runs
between two anchors: do nothing to the mixture and the AI listener gets 63% of
the words wrong; hand it the clean target and that falls to 1%. My baseline
sits at 57% — so it closes about a tenth of the available gap. That is an
unflattering result and I report it as one.

**[03 / Holistic Metric Suite]**
This is the main contribution, and these four rows are why.

Hallucination tracking: feed the AI listener pure silence and it confidently
reports twenty or thirty words that were never said. Every extractor I have
tested makes it invent *more* than doing nothing — about half again as much.

Interference leakage: separately, how much of the *wrong* speaker got through.

Latency: measured per system, against that streaming budget.

And perceptual quality — where the headline finding is. My baseline made the
audio measurably worse to listen to while the AI listener understood more of
it. A quality-based metric would have rejected a system that helps.

---

## Close (4:35–4:50)

Where I am now: letting the live model supervise training directly. First
measuring how noisy that judge is — one clip moved sixteen points across five
identical calls — then tuning the extractor against its preferences, with an
untouched offline transcriber kept as the control.
