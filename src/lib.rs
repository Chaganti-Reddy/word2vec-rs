//! # word2vec
//!
//! A Word2Vec implementation in Rust supporting both
//! Skip-gram and CBOW architectures with Negative Sampling.
//!
//! ## Architecture
//!
//! - [`vocab`] — Vocabulary construction, subsampling, and unigram noise table
//! - [`model`] — Skip-gram and CBOW forward/backward pass
//! - [`trainer`] — Training loop with monitoring and checkpointing
//! - [`embeddings`] — Post-training embedding access, similarity, analogy
//! - [`config`] — Hyperparameter configuration
//! - [`error`] — Unified error type
//! - [`plot`] — Loss curves and 2D PCA projection plots
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use word2vec::{Config, ModelType, Trainer};
//!
//! let config = Config {
//!     embedding_dim: 100,
//!     window_size: 5,
//!     negative_samples: 5,
//!     epochs: 5,
//!     model: ModelType::SkipGram,
//!     ..Config::default()
//! };
//!
//! let corpus = vec![
//!     "the quick brown fox jumps over the lazy dog".to_string(),
//! ];
//!
//! let mut trainer = Trainer::new(config);
//! let embeddings = trainer.train(&corpus).unwrap();
//!
//! let similar = embeddings.most_similar("fox", 5);
//! println!("{:?}", similar);
//! ```

pub mod config;
pub mod embeddings;
pub mod error;
pub mod model;
pub mod plot;
pub mod trainer;
pub mod vocab;

pub use config::{Config, ModelType};
pub use embeddings::Embeddings;
pub use error::Word2VecError;
pub use trainer::Trainer;
