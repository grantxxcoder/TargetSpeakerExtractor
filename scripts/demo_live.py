"""Live target speaker extraction demo, for presenting.

    ../tse_venv/bin/python scripts/demo_live.py        # then open the printed URL

1. upload or record a target voice, an enrolment, optionally another speaker
2. build a mixture the way the trials were built (src/demo/mixer.py)
3. stream it through the extractor 80 ms at a time, paced like a microphone,
   with both spectrograms drawn live (src/models/streaming.py)
4. send the mixture, the extracted audio and the clean target to Gemini with
   the benchmark's judge prompt, and play back what it heard

Everything is set in experiments/configs/demo_live.yaml. Every run is saved to
experiments/results/demo-live/<date>-<time>/: the audio files, and record.json
with checkpoint, commit, seed, mixture draws, streaming timings, and the exact
Gemini model ID, prompt, modality and date.

GEMINI_API_KEY is read from the environment, or from .env if unset there.
"""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/configs/demo_live.yaml")
    ap.add_argument("--port", type=int, default=None, help="override server.port")
    args = ap.parse_args()

    os.chdir(REPO)
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env", override=False)   # an exported key wins
    if not os.environ.get("GEMINI_API_KEY"):
        print("note: GEMINI_API_KEY not set -- step 4 (Gemini) will fail until it is",
              flush=True)

    from src.demo.server import serve
    serve(REPO / args.config, port=args.port)


if __name__ == "__main__":
    main()
