# TSE part 1 — 5-minute script

~720 words ≈ 4:50. Every beat maps to a card on the slide, in slide order. Bracketed labels are the card you should be looking at.

---

## Slide 1 — Title (0:00–0:15)

Real-time target speech extraction.

In one sentence: I take a noisy recording with two people talking and pull out the one voice I care about, live. I then score the result by whether a live AI model understands what that person said.

---

## Slide 2 — Context of the problem (0:15–1:15)

**[01 / Innate Human Ability]**
You already do this. In a crowded restaurant you can focus on one voice, and everything else becomes background. It takes you no effort at all, but it is still not solved in machines.

**[02 / Edge AI Integration]**
The reason this matters now is that listening is moving into everyday devices, such as smart glasses and hearables, which have to make sense of whatever acoustic environment their wearer walks into.

**[03 / Real-Time AI Transcription]**
Increasingly, the thing consuming that audio is not a person. It is a live AI model, such as Gemini Live and its equivalents. These work very well in a quiet room, but they fall apart as soon as a second person is in it. So my setting is robust voice understanding in chaotic environments, for an AI listener.

---

## Slide 3 — Problem formulation (1:15–2:55)

**[the equation]**
Formally, what the model hears is a mixture of my target speaker, one interfering speaker and background noise. It also gets a few seconds of the target talking alone, which is the only thing telling it which voice to keep.

**[spectrogram]**
The coloured stripes show the four situations inside one clip: target alone, both speakers overlapping, interferer alone, and noise only. The top panel is what the model hears, and the second panel is what it has to produce.

**[Technical Challenges]**
The main technical difficulty is conditioning. The model has to transform its internal speech representation using very little information about the target: a few seconds of enrolment audio, and nothing else. It then has to hold onto who it is listening for across the entire recording.

**[Hardware Constraints]**
The constraint is that this has to run streaming. There is no looking ahead and no waiting for the sentence to finish. There is roughly a two hundred millisecond budget, and the model must be small enough to be worth deploying.

**[Primary Objective]**
This card is the research contribution. The objective is to maximise downstream live AI transcription accuracy. In other words, not signal quality or how nice it sounds, but what the AI listener recovers.

These are not the same thing. Cleaning up audio leaves holes and artefacts. An offline transcriber mostly ignores those, but a live model listens through its own learned encoder, and it appears to be more affected by the artefacts you added than by the voice you removed. This means that a system can improve on every conventional score and still get worse at the one job I care about.

The two smaller goals under it are to stay silent when the target isn't speaking, and to suppress the noise.

---

## Slide 4 — Current project state (2:55–4:35)

There are three pillars, and all three now exist.

**[01 / Dataset Construction]**
The data is built, not borrowed. Twenty-one thousand mixtures, a hundred and five hours of two-speaker scenarios, with controlled noise levels, real recorded noise and room reverb. The reason it is built rather than borrowed is that training needs a clean target signal and the exact words both speakers said, and real meeting recordings give you neither.

**[02 / Baseline Signal Extraction]**
The baseline is a time-domain masking model. It streams at about a hundred and sixty milliseconds on a laptop CPU, which is inside the budget. The performance runs between two anchors. If you do nothing to the mixture, the AI listener gets 63% of the words wrong, and if you hand it the clean target, that falls to 1%. My baseline sits at 57%, so it closes about a tenth of the available gap. This is not a flattering result, and I report it as such.

**[03 / Holistic Metric Suite]**
This is the main contribution, and these four rows show why.

Hallucination tracking: if you feed the AI listener pure silence, it confidently reports twenty or thirty words that were never said. Every extractor I have tested makes it invent *more* than doing nothing, by about half again as much.

Interference leakage: separately, how much of the *wrong* speaker got through.

Latency: measured per system, against the streaming budget.

Finally, perceptual quality, which is where the headline finding is. My baseline made the audio measurably worse to listen to, while the AI listener understood more of it. Therefore, a quality-based metric would have rejected a system that helps.

---

## Close (4:35–4:50)

Where I am now is letting the live model supervise training directly. First, I am measuring how noisy that judge is: one clip moved sixteen points across five identical calls. Next, I am tuning the extractor against its preferences, with an untouched offline transcriber kept as the control.
