"""The live target speaker extraction demo, for presenting.

    ../tse_venv/bin/python scripts/demo_live.py        # then open the printed URL

1. Upload or record a target voice, an enrolment, and optionally another speaker.
2. Build a mixture the way the trials were built (src/demo/mixer.py).
3. Stream it through the extractor 80 ms at a time, paced like a microphone, with both spectrograms drawn live (src/models/streaming.py).
4. Send the mixture, the extracted audio and the clean target to Gemini with the benchmark's judge prompt, and play back what it heard.

Everything is set in experiments/configs/demo_live.yaml. Every run is saved to experiments/results/demo-live/<date>-<time>/: the audio, and record.json with the checkpoint, seed, mixture draws, streaming timings, and the exact Gemini model ID, prompt, modality and date.
GEMINI_API_KEY is read from the environment, or from .env if it is not set there.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.demo.server import serve

CONFIG = REPO / "experiments/configs/demo_live.yaml"


# this is where the demo starts
def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    os.chdir(REPO)
    # A key already exported in the shell wins over the file.
    load_dotenv(REPO / ".env", override=False)
    if not os.environ.get("GEMINI_API_KEY"):
        print("note: GEMINI_API_KEY not set -- step 4 (Gemini) will fail until it is", flush=True)
    serve(CONFIG)


if __name__ == "__main__":
    main()
