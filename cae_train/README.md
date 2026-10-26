# CAE image encoder

The `cae_train/` package contains the source for image representation learning and 256-dimensional feature export.

| File | Function |
|---|---|
| `data.py` | Read locally available labeled RGB images |
| `networks.py` | CAE generator, content/style encoders, decoder and discriminator |
| `trainer_exchange.py` | Image reconstruction, class transfer, training losses and checkpoints |
| `main_train.py` | CAE training command-line entry |
| `extract_embeddings.py` | Extract and save 256-dimensional image vectors |

**Model outputs:** The content encoder produces a 256-channel spatial feature map. The configurable style encoder defaults to a 256-dimensional code. The exporter supports either `content_gap` (spatial mean over content channels) or `style` (individual-specific code).

From the repository root, use `python extract_cae_embeddings.py --help` for the feature export options; full examples are in the root `README.md`. Image files, class labels and checkpoints are read from the configured local `data/` paths.
