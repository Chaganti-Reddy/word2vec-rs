//! Hyperparameter configuration for Word2Vec training.
//!
//! # Example
//!
//! ```rust
//! use word2vec::{Config, ModelType};
//!
//! let cfg = Config {
//!     embedding_dim: 128,
//!     window_size: 5,
//!     negative_samples: 10,
//!     epochs: 10,
//!     learning_rate: 0.025,
//!     min_learning_rate: 0.0001,
//!     min_count: 5,
//!     subsample_threshold: 1e-3,
//!     model: ModelType::SkipGram,
//!     num_threads: 4,
//!     seed: 42,
//! };
//!
//! assert_eq!(cfg.embedding_dim, 128);
//! ```

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// Predict context words from the center word. Better for rare words.
    SkipGram,
    /// Predict center word from context words. Faster; better for frequent words.
    Cbow,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::SkipGram => write!(f, "Skip-gram"),
            ModelType::Cbow => write!(f, "CBOW"),
        }
    }
}

/// Full training configuration.
///
/// All fields have sensible defaults via [`Config::default()`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Dimensionality of word embeddings (typical: 50–300).
    pub embedding_dim: usize,
    /// Half-width of the context window (words on each side of target).
    pub window_size: usize,
    /// Number of negative samples per positive pair (typical: 5–20).
    pub negative_samples: usize,
    /// Full passes over the corpus.
    pub epochs: usize,
    /// Initial learning rate (decays linearly to `min_learning_rate`).
    pub learning_rate: f32,
    /// Floor for decayed learning rate.
    pub min_learning_rate: f32,
    /// Discard words appearing fewer than this many times.
    pub min_count: usize,
    /// Frequent-word subsampling threshold (Mikolov et al. suggest 1e-3 – 1e-5).
    pub subsample_threshold: f64,
    /// Architecture choice.
    pub model: ModelType,
    /// Rayon thread count (0 = use all logical cores).
    pub num_threads: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for Config {
    /// Sensible defaults matching the original word2vec paper.
    fn default() -> Self {
        Self {
            embedding_dim: 100,
            window_size: 5,
            negative_samples: 5,
            epochs: 5,
            learning_rate: 0.025,
            min_learning_rate: 0.0001,
            min_count: 1,
            subsample_threshold: 1e-3,
            model: ModelType::SkipGram,
            num_threads: 0,
            seed: 42,
        }
    }
}

impl Config {
    /// Validate configuration, returning an error message if invalid.
    ///
    /// ```rust
    /// use word2vec::Config;
    /// let cfg = Config { embedding_dim: 0, ..Config::default() };
    /// assert!(cfg.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        if self.embedding_dim == 0 {
            return Err(format!("embedding_dim must be > 0, got {}", self.embedding_dim));
        }
        if self.window_size == 0 {
            return Err(format!("window_size must be > 0, got {}", self.window_size));
        }
        if self.negative_samples == 0 {
            return Err(format!("negative_samples must be > 0, got {}", self.negative_samples));
        }
        if self.epochs == 0 {
            return Err("epochs must be > 0".to_string());
        }
        if self.learning_rate <= 0.0 {
            return Err(format!("learning_rate must be > 0, got {}", self.learning_rate));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        assert!(Config::default().validate().is_ok());
    }

    #[test]
    fn zero_dim_is_invalid() {
        let cfg = Config { embedding_dim: 0, ..Config::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn zero_window_is_invalid() {
        let cfg = Config { window_size: 0, ..Config::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn model_type_display() {
        assert_eq!(ModelType::SkipGram.to_string(), "Skip-gram");
        assert_eq!(ModelType::Cbow.to_string(), "CBOW");
    }
}
