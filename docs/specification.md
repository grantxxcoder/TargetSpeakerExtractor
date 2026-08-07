Real-time target speaker extraction
Supervisors: DJ Swanevelder and David Botha (past SU ML/AI students now working at Smart
Operator, Cape Town)
Real-world audio reaching a voice assistant is a single mixed source containing the target
speaker’s voice (signal), non-target voices and sounds (interference), and irregular background
sounds (noise). Understanding what the user specifically wants across noisy environments
therefore requires Target Speaker Extraction: isolating the target speaker’s speech from the
mixture using an enrollment clue [1].
Accuracy has improved steadily in recent years, but no existing approach is both accurate
enough and low-latency enough for real-time, on-device use [2], and our own experiments
confirmed this gap. The model must also be robust to the wide differences in microphone
quality between input devices, such as earpieces versus mobile phones. The REAL-TSE Challenge
Online track [3], which evaluates streaming TSE on real conversational recordings from diverse
recording conditions under a 100 ms latency budget, captures precisely this problem and
anchors the project.
Voice assistants also increasingly send audio directly to live speech-to-speech models such as
Gemini Live. We have found that traditional TSE can improve transcription accuracy while
making the audio harder for these live models to understand. The challenge does not measure
this, so the project should define a way to measure it and test the top models, and the model
built in this project, against it.
The student will review the state of the art, replicate the challenge’s online-track baselines and
evaluation pipeline (synthesizing training data and reproducing its metrics), define the
live-model measurement above, and then explore a fundamentally new TSE architecture, aiming
to move closer to a suﬀiciently accurate, low-latency system that runs on on-device hardware.
[1] K. Žmolíková et al., Neural Target Speech Extraction: An Overview, IEEE Signal Processing
Magazine, 40(3), 2023. doi:10.1109/MSP.2023.3240008
[2] M. Itani, T. Chen, S. Gollakota, TF-MLPNet: Tiny Real-Time Neural Speech Separation,
Interspeech 2025. arXiv:2508.03047
[3] REAL-TSE Challenge, satellite of IEEE SLT 2026. https://real-tse.github.io/challenge/

Meeting notes concerning the goals and outcomes for this research project:
1. The actual score values from my defined metric do not matter - the metric itself matters more.
2. Focus on the online track for the TSE challenge over the offline track.
3. Can expect the neural networks themselves to have artefacts from the training process due to unclean training data.
4. A suggested approach is to begin with the literature review of 5-6 papers. Choose a subset of them and do one of the following: Either think of ways to combine multiple approaches together that seem good, better the implementation of some that seem like they have not been implemented optimally, or, make a new architecture. The new architecture though is after reimplementing some of the methods first.
5. The literature reviews are of utmost importance. It is very important to be able to reimplement the results that the respective papers got.
6. The idea for this project is to use the state of the art methods to build a model that leverages low latency methods for live target speaker extraction, with a small enough model to fit on a small device, that can handle interruptions, and can produce interpretable speech to text output. 
7. The project is open ended however to take any one of these approaches to improve upon and does not need to solve all of these at the same time.
8. The scientific idea behind this project is to first choose/build a metric that considers the goal in (10), then to implement some architecture and measure it against the metric (this should be some state of the art method but baseline can also be sufficient), and then, time permitting, to compare it against another method or my own architecture. I do NOT need to reimplement anything from the 2026 challenge but I can borrow ideas from it. 
9. I am permitted to make my own training/val/eval sets as a mixture of whatever was used in the challenge.
10. The actual goal can be refined as follows: I need to build a model that focuses on improving what a LIVE AI model interprets when listening to combined audio for the target speaker. It is not to optimise for the QUALITY of the separated output, although this might help the transcription, but rather, to optimise the intelligibility of what a live ai model like Gemini Live understands of our audio. So the general idea: Have combined audio with the target speaker present, feed it into my model, model produces separated audio to be fed into Gemini Live, measure how well Gemini Live understands the audio through intelligibillity metrics. Constraints: the model needs to be live, therefore latency becomes a secondary metric to optimise for. We can assume we are also working with something as big as a server, and not necessarily optimise for a small device like a wearable.

