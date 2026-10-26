"""Convert a locally trained gensim Word2Vec/KeyedVectors model to text.

This project helper has been updated to work with gensim 4.x.  Gene-vector
training data and output files must remain under the private data/ tree.
"""
from pathlib import Path
import numpy as np
from gensim.models import Word2Vec, KeyedVectors


def load_embeddings(file_name):
    """Return (word vectors, gene symbols) from a gensim model."""
    try:
        original = Word2Vec.load(str(file_name))
        keyed = original.wv
    except (AttributeError, KeyError, TypeError, ValueError):
        keyed = KeyedVectors.load(str(file_name))
    names = list(keyed.index_to_key)
    values = np.asarray(keyed.vectors, dtype=np.float32)
    if not names or values.shape[0] != len(names):
        raise ValueError("Gene2Vec model has no aligned gene vectors")
    return values, names


def outputTxt(embeddings_file):
    vectors, names = load_embeddings(embeddings_file)
    destination = Path(str(embeddings_file) + ".txt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as out:
        for name, vec in zip(names, vectors):
            out.write(f"{name}\t" + " ".join(f"{float(x):.9g}" for x in vec) + "\n")
    return destination
