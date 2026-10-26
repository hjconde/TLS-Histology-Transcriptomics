"""OPTIONAL: train Gene2Vec from independently authorized co-expression pairs.

The vector training stage is optional when a local 200-dimensional
Gene2Vec text file is already available.
The input corpus and generated vectors are stored in the local data/ directory.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random

from gensim.models import Word2Vec

from gene2vec.generateMatrix import outputTxt

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def load_source(input_dir: Path, suffix: str, *, seed: int = 2024):
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Private Gene2Vec training corpus missing: {input_dir}")
    rng = random.Random(seed)
    corpus = []
    files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.name.endswith(suffix))
    rng.shuffle(files)
    for file in files:
        with file.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                row = line.strip().split()
                if len(row) >= 2:
                    corpus.append(row)
    if not corpus:
        raise ValueError("No gene pairs found; expected whitespace-separated genes per line")
    rng.shuffle(corpus)
    return corpus


def gene_vectorization(corpus, output_dir: Path, *, dimension: int,
                       workers: int, window: int, iterations: int,
                       sg: int, seed: int, write_text: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    model = None
    latest = None
    for epoch in range(1, iterations + 1):
        rng.shuffle(corpus)
        if epoch == 1:
            model = Word2Vec(corpus, vector_size=dimension, window=window,
                             min_count=1, workers=workers, epochs=1,
                             sg=sg, seed=seed)
        else:
            model.train(corpus, total_examples=len(corpus), epochs=1)
        latest = output_dir / f"gene2vec_dim_{dimension}_iter_{epoch}"
        model.save(str(latest))
        if write_text:
            outputTxt(latest)
    return latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/gene2vec_pairs"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/gene2vec_models"))
    parser.add_argument("--file-suffix", default="txt")
    parser.add_argument("--dimension", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--sg", type=int, default=1, choices=(0, 1))
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--write-txt", action="store_true", help="also write gene vectors to private .txt")
    args = parser.parse_args()
    if args.dimension < 1 or args.max_iter < 1 or args.num_workers < 1:
        parser.error("dimension, iterations, and number of workers must be positive")
    corpus = load_source(args.data_dir, args.file_suffix, seed=args.seed)
    result = gene_vectorization(corpus, args.output_dir, dimension=args.dimension,
                                workers=args.num_workers, window=args.window_size,
                                iterations=args.max_iter, sg=args.sg,
                                seed=args.seed, write_text=args.write_txt)
    print(f"Private Gene2Vec model saved: {result}")
    print("Generated vector files are saved under the selected local output directory.")


if __name__ == "__main__":
    main()
