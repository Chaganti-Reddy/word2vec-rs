//! Unified error type for the word2vec crate.

use thiserror::Error;

/// All errors that can occur during vocabulary building, training, or inference.
#[derive(Debug, Error)]
pub enum Word2VecError {
    #[error("Vocabulary is empty — provide a non-empty corpus")]
    EmptyVocabulary,

    #[error("Word not found in vocabulary: `{0}`")]
    UnknownWord(String),

    #[error("Embedding dimension must be > 0, got {0}")]
    InvalidDimension(usize),

    #[error("Window size must be > 0, got {0}")]
    InvalidWindowSize(usize),

    #[error("Negative samples must be > 0, got {0}")]
    InvalidNegativeSamples(usize),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Plot error: {0}")]
    Plot(String),

    #[error("Corpus too small: need at least 2 unique tokens, got {0}")]
    CorpusTooSmall(usize),
}

pub type Result<T> = std::result::Result<T, Word2VecError>;
