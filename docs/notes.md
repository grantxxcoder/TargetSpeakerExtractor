Plain-language notes to myself. Formal versions of all of this live in
docs/decisions.md.

**LibriMix** takes clean audiobook recordings of people reading, and digitally
adds two of them together to fake a conversation. It's not a real conversation
— nobody interrupted anyone, there's no real room echo, no background clatter.
It's clean and artificial (this is what "simulated" means). This is what I use
for **training**, and — updated after the 2026-08-07 re-scope — also for my
**primary eval**. The reason is that my training losses need a clean target
signal and exact ground-truth text, and real recordings give me neither. My
primary eval has to be constructed too, because the metric needs to know
exactly what the target said, and needs a true "clean target" ceiling to
compare against.

**AMI** is recordings of real meetings — real people talking over each other,
real rooms, real microphones. This is what the voice-assistant problem actually
looks like. It is my **secondary** eval: the real-audio transfer check. Not
primary, because AMI has no clean target (the headset channel still picks up
other people bleeding through) and its transcripts aren't word-for-word. So
its "ceiling" is only approximate, and I have to say so every time I quote an
AMI number.

**Audio out, not text out.** Gemini Live will take either audio or text, so I
could have my model transcribe the target speaker and just send the words. I'm
not doing that, for three reasons. It's slower — you have to finish decoding
the speech and decide the sentence has ended before you can send anything.
It deletes the actual research question — my whole premise is that live models
choke on audio processing artefacts, and there are no audio artefacts if
there's no audio. And text throws away tone, emphasis, hesitation, emotion,
which is a lot of what a speech-to-speech model is actually using.

But I still *measure* the text route, because it's nearly free: run my
extractor, run an off-the-shelf ASR on its output, send the text. One extra row
in the results table. It tells me how much of the content is recoverable at
all. It is **not** an upper bound though — once the ASR gets a word wrong,
that word is gone forever, whereas sending audio at least leaves the live model
something to work with.
