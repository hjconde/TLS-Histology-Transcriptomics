# Histology–Transcriptomics Learning: Core Implementation

This repository contains the core implementation for **gene representation learning**, **histopathology image representation learning**, and **cross-modal fusion for patient-level tertiary lymphoid structure (TLS) classification**.

The workflow is organized into three modules:

| Module | Main implementation | Output |
|---|---|---|
| **Gene representation — Gene2Vec / gaBERT** | `gene2vec/gene2vec.py`, `gene2vec/generateMatrix.py`, `prepare_gene_embeddings.py`, `gene2vec/gene2vec_positional_embedding.py`, `performer_pytorch/performer.py` | Gene2Vec vectors and gene-token representations |
| **Image representation — CAE** | `cae_train/main_train.py`, `cae_train/networks.py`, `cae_train/trainer_exchange.py`, `extract_cae_embeddings.py` | 256-dimensional histology embeddings |
| **Multimodal fusion — TLS classifier** | `tls_multimodal/model.py`, `tls_multimodal/engine.py`, `tls_multimodal/data.py`, `train_5fold.py`, `evaluate_5fold.py` | Class probabilities, model checkpoints, and evaluation summaries |

## Architecture overview

```text
Gene expression ── Gene2Vec positional vectors ── gaBERT / PerformerLM ──┐
                                                                         ├── Cross-modal attention
H&E images ─────── CAE image encoder ──────── 256-D image embeddings ─────┘          │
                                                                               Contrastive loss
                                                                                    │
                                                                          Attention pooling
                                                                                    │
                                                                            TLS classification
```

## Repository structure

```text
.
├── README.md
├── DATA_REQUIREMENTS.md           # Input paths and formats
├── requirements.txt               # Main dependencies
├── requirements-gene2vec-optional.txt
├── data/                          # Local input directory
│   └── README.md
├── gene2vec/                      # Gene2Vec training and positional embedding
├── performer_pytorch/             # Gene-level Performer / gaBERT encoder
├── prepare_gene_embeddings.py
├── cae_train/                      # CAE model and training code
├── extract_cae_embeddings.py      # CAE feature-export command
├── tls_multimodal/                # Multimodal model and training utilities
├── train_5fold.py
├── evaluate_5fold.py
├── tests/
└── scripts/audit_public_release.py
```

## Installation

Python 3.9+ and a compatible PyTorch environment are recommended. GPU acceleration is useful for long gene sequences and image-model training.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The project-specific `performer_pytorch/` and `gene2vec/` modules are included as source code. Prepare input files in the local `data/` directory following [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md).

## 1. Gene representation

**Gene2Vec training.** For a prepared gene co-expression pair corpus, generate 200-dimensional vectors using `gene2vec/gene2vec.py` and its text-output helper `gene2vec/generateMatrix.py`:

```bash
python -m pip install -r requirements-gene2vec-optional.txt
python -m gene2vec.gene2vec \
  --data-dir data/gene2vec_pairs \
  --output-dir data/gene2vec_models \
  --dimension 200 --write-txt
```

**Gene2Vec preparation.** With a gene-vector text file at `data/Gene2vec_0111.txt` and an expression matrix at `data/TLS_gene_fold5/hj_scaler_mat.csv`, build a gene-order-aligned positional embedding matrix:

```bash
python prepare_gene_embeddings.py --data-root data
```

This creates `data/gene_embeddings.npy` and `data/gene_embeddings.genes.txt`. The `Gene2VecPositionalEmbedding` layer is implemented in `gene2vec/gene2vec_positional_embedding.py`; the gene encoder (`PerformerLM`) is implemented in `performer_pytorch/performer.py`.

## 2. CAE image representation

**CAE training.** `cae_train/main_train.py` trains the CAE generator and discriminator, defined in `cae_train/networks.py` and `cae_train/trainer_exchange.py`. For example:

```bash
python cae_train/main_train.py \
  --cuda True --style_dim 256 \
  --trainA_img_path data/cae/fold1/images/train/A \
  --trainB_img_path data/cae/fold1/images/train/B \
  --testA_img_path data/cae/fold1/images/test/A \
  --testB_img_path data/cae/fold1/images/test/B \
  --train_label_file data/cae/fold1/trainAB_img-name_label.txt \
  --test_label_file data/cae/fold1/testAB_img-name_label.txt \
  --model_save_path data/cae/fold1/models \
  --image_save_path data/cae/fold1/display
```

**Image embedding export.** `extract_cae_embeddings.py` loads a trained CAE generator, processes RGB histology images, and writes an embedding file for each requested fold:

```bash
python extract_cae_embeddings.py \
  --fold 1 \
  --checkpoint data/cae/fold1/models/gen_00004000.pt \
  --train-image-dir data/cae/fold1/images/train \
  --test-image-dir data/cae/fold1/images/test \
  --split-dir data/TLS_gene_fold5 \
  --output-root data/hjImage \
  --style-dim 256 --method content_gap
```

`content_gap` applies spatial average pooling to the CAE content map, producing one 256-dimensional image vector per sample. The exporter also supports `--method style` to use the 256-dimensional individual-specific code. Repeat the export for each fold.

## 3. Multimodal fusion and classification

The fusion model in `tls_multimodal/model.py` combines gene-token and image representations with bidirectional cross-modal attention and patient-paired contrastive learning, then applies attention pooling and a binary classification head.

**Five-fold training:**

```bash
python train_5fold.py --data-root data --output-dir outputs
```

**Evaluation with saved fold checkpoints:**

```bash
python evaluate_5fold.py \
  --data-root data \
  --checkpoints outputs/validation_selected_RUN_DIRECTORY \
  --output-dir outputs/evaluation
```

`outputs/` contains the local training outputs and prediction summaries. See [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) for exact input names, required formats and fold layout.

## Development and attribution

Run the included tests and the pre-publication source scan as needed:

```bash
python -m pytest -q
python scripts/audit_public_release.py
```

CAE and GSEABert/gaBERT components were developed by the authors' research group. The optional Gene2Vec vector-training utility builds on the Gene2vec project (Du *et al.*, *BMC Genomics*, 2019; [DOI: 10.1186/s12864-018-5370-x](https://doi.org/10.1186/s12864-018-5370-x)). Its MIT license notice is retained in `third_party_licenses/Gene2Vec-MIT.txt`.
