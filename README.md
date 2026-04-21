# word2vec

Skip-gram and CBOW Word2Vec with Negative Sampling, written in idiomatic Rust.

## Features

- **Two architectures**: Skip-gram (better rare words) and CBOW (faster)
- **Negative Sampling**: unigram noise table (`freq^0.75`), configurable k
- **Subsampling**: Mikolov frequent-word discard formula
- **Dynamic window**: per-sentence random window width
- **Xavier initialisation** for input weights
- **Linear LR decay**: `lr_start → lr_min` over all epochs
- **Progress bars**: per-epoch via `indicatif`
- **Structured logging**: via `log` + `env_logger`
- **Plots**: loss curve PNG + 2D PCA word projection PNG via `plotters`
- **Save/load**: JSON (full model) + word2vec text format
- **35 unit tests** + **10 integration tests**, all passing

## Project Structure

```
src/
├── lib.rs          — public API re-exports
├── error.rs        — Word2VecError (thiserror)
├── config.rs       — Config, ModelType
├── vocab.rs        — Vocabulary, subsampling, noise table
├── model.rs        — Weight matrices, SGD updates
├── trainer.rs      — Training loop, history, checkpointing
├── embeddings.rs   — most_similar, similarity, analogy, save/load
├── plot.rs         — Loss curve + PCA scatter plot
└── bin/
    ├── train.rs    — CLI trainer
    └── evaluate.rs — Interactive REPL
tests/
└── integration.rs  — End-to-end tests
examples/
└── basic_training.rs
```

## Usage

```bash
# Run all tests
cargo test

# Train on built-in sample corpus, generates all artifacts
cargo run --release --example basic_training

# Train on your own corpus (one sentence per line)
cargo run --release --bin train -- \
  --input corpus.txt \
  --output embeddings.json \
  --dim 100 \
  --window 5 \
  --epochs 10 \
  --model skipgram \
  --negative 5

# Interactive evaluation REPL
cargo run --release --bin evaluate -- --model embeddings.json
# > sim king queen
# > near paris 10
# > analogy king man woman
```

## Library API

```rust
use word2vec::{Config, ModelType, Trainer};

let config = Config {
    embedding_dim: 100,
    window_size: 5,
    negative_samples: 5,
    epochs: 10,
    model: ModelType::SkipGram,
    ..Config::default()
};

let mut trainer = Trainer::new(config);
let embeddings = trainer.train(&corpus)?;

// Nearest neighbours
let similar = embeddings.most_similar("king", 5);

// Cosine similarity
let sim = embeddings.similarity("paris", "berlin")?;

// Analogy: king - man + woman ≈ ?
let results = embeddings.analogy("king", "man", "woman", 5)?;

// Save / load
embeddings.save("embeddings.json")?;
let loaded = Embeddings::load("embeddings.json")?;

// Export word2vec text format (gensim compatible)
embeddings.save_text_format("vectors.txt")?;
```

## Output Artifacts

After training:
- `embeddings.json` — full model (loadable)
- `embeddings.txt` — word2vec text format
- `training_history.json` — per-epoch loss/lr/pairs/time
- `loss_curve.png` — training loss chart
- `word_pca.png` — 2D PCA word projection

## Environment Variables

```bash
RUST_LOG=debug cargo run --bin train  # verbose logging
RUST_LOG=info  cargo run --bin train  # default
```
