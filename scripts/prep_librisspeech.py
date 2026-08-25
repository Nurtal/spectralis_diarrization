"""Prepare a LibriSpeech subset as a flat corpus for the mixture generator.

Converts <spk>/<chapter>/<utt>.flac trees into <speaker>/<utterance>.wav,
optionally subsampling speakers and utterances and resampling.

Usage:
    python scripts/prep_librisspeech.py \
        --librispeech data/downloads/LibriSpeech/test-clean \
        --out data/clips_testclean --sample-rate 16000 \
        --max-speakers 20 --max-utterances-per-speaker 3
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from benchmark.audio import load_audio, save_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-speakers", type=int, default=0, help="0 = all")
    parser.add_argument("--max-utterances-per-speaker", type=int, default=0, help="0 = all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.librispeech)
    by_speaker = defaultdict(list)
    for flac in sorted(root.rglob("*.flac")):
        speaker = flac.relative_to(root).parts[0]
        by_speaker[speaker].append(flac)

    rng = random.Random(args.seed)
    speakers = sorted(by_speaker)
    if args.max_speakers:
        speakers = rng.sample(speakers, min(args.max_speakers, len(speakers)))

    out_root = Path(args.out)
    n_clips = 0
    for speaker in sorted(speakers):
        clips = by_speaker[speaker]
        if args.max_utterances_per_speaker:
            clips = rng.sample(clips, min(args.max_utterances_per_speaker, len(clips)))
        dest_dir = out_root / speaker
        dest_dir.mkdir(parents=True, exist_ok=True)
        for flac in clips:
            audio, sr = load_audio(flac, sample_rate=args.sample_rate)
            save_audio(dest_dir / (flac.stem + ".wav"), audio, args.sample_rate)
            n_clips += 1
        print(f"{speaker}: {len(clips)} clips")

    print(f"total: {n_clips} clips for {len(speakers)} speakers -> {out_root}")


if __name__ == "__main__":
    main()
