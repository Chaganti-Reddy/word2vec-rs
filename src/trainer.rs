//! Training loop with progress monitoring, learning rate decay, and
//! optional checkpointing.
//!
//! # Training Flow
//!
//! ```text
//! corpus ──► Vocabulary::build ──► Model::new
//!                                       │
//!              ┌────────────────────────┘
//!              ▼
//!         for each epoch:
//!           shuffle sentences
//!           for each sentence:
//!             subsample tokens
//!             for each (center, context) pair:
//!               sample negatives
//!               Model::update (SGD step)
//!               update LR (linear decay)
//!           record epoch loss
//!              │
//!              ▼
//!         Embeddings ──► .most_similar() / .analogy() / .save()
//! ```

use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::Path;

use crate::config::Config;
use crate::embeddings::Embeddings;
use crate::error::{Result, Word2VecError};
use crate::model::{sentence_to_pairs, Model};
use crate::vocab::Vocabulary;

/// Recorded statistics for one epoch.
#[derive(Debug, Clone)]
pub struct EpochStats {
    pub epoch: usize,
    pub avg_loss: f64,
    pub learning_rate: f32,
    pub pairs_processed: u64,
    pub elapsed_secs: f64,
}

/// Manages the full training pipeline.
pub struct Trainer {
    config: Config,
    /// Loss history per epoch (populated during `train`).
    pub history: Vec<EpochStats>,
}

impl Trainer {
    /// Create a new trainer with the given configuration.
    ///
    /// ```rust
    /// use word2vec::{Config, Trainer};
    /// let trainer = Trainer::new(Config::default());
    /// assert!(trainer.history.is_empty());
    /// ```
    pub fn new(config: Config) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Tokenise raw text into sentences (split on whitespace).
    fn tokenise(corpus: &[String]) -> Vec<Vec<String>> {
        corpus
            .iter()
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.split_whitespace().map(str::to_string).collect())
            .collect()
    }

    /// Train on a corpus of sentences.
    ///
    /// Returns [`Embeddings`] which wraps the trained model and vocabulary
    /// for inference queries.
    ///
    /// # Errors
    ///
    /// Returns [`Word2VecError`] if the config is invalid or the corpus
    /// is too small to train.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use word2vec::{Config, ModelType, Trainer};
    ///
    /// let corpus = vec![
    ///     "paris is the capital of france".to_string(),
    ///     "berlin is the capital of germany".to_string(),
    ///     "tokyo is the capital of japan".to_string(),
    /// ];
    ///
    /// let mut trainer = Trainer::new(Config {
    ///     epochs: 3,
    ///     embedding_dim: 50,
    ///     ..Config::default()
    /// });
    ///
    /// let embeddings = trainer.train(&corpus).unwrap();
    /// println!("Vocab size: {}", embeddings.vocab_size());
    /// ```
    pub fn train(&mut self, corpus: &[String]) -> Result<Embeddings> {
        self.config
            .validate()
            .map_err(|_e| Word2VecError::EmptyVocabulary)?;

        let sentences = Self::tokenise(corpus);
        let vocab = Vocabulary::build(&sentences, &self.config)?;

        info!(
            "Vocabulary: {} unique tokens ({} total), model: {}",
            vocab.len(),
            vocab.total_tokens,
            self.config.model
        );

        let mut model = Model::new(vocab.len(), self.config.embedding_dim, self.config.seed);
        let mut rng = SmallRng::seed_from_u64(self.config.seed);

        let total_pairs_estimate = self.estimate_pairs(&sentences, &vocab);
        let lr_start = self.config.learning_rate;
        let lr_min = self.config.min_learning_rate;
        let epochs = self.config.epochs;

        // Grand total steps for LR decay
        let total_steps = total_pairs_estimate * epochs as u64;
        let mut global_step: u64 = 0;

        for epoch in 0..epochs {
            let t_start = std::time::Instant::now();
            let mut epoch_loss = 0.0_f64;
            let mut epoch_pairs: u64 = 0;

            // Shuffle sentence order each epoch
            let mut indices: Vec<usize> = (0..sentences.len()).collect();
            indices.shuffle(&mut rng);

            let pb = self.make_progress_bar(epoch, sentences.len());

            for &sent_idx in &indices {
                let tokens = vocab.tokenise_and_subsample(
                    &sentences[sent_idx],
                    self.config.subsample_threshold,
                    &mut rng,
                );

                if tokens.len() < 2 {
                    pb.inc(1);
                    continue;
                }

                let pairs = sentence_to_pairs(&tokens, self.config.window_size, &mut rng);

                for (center, context_words) in pairs {
                    // Linear LR decay
                    let progress = global_step as f32 / total_steps.max(1) as f32;
                    let lr = (lr_start - (lr_start - lr_min) * progress).max(lr_min);

                    // Sample negatives
                    let negatives: Vec<usize> = (0..self.config.negative_samples)
                        .map(|_| vocab.negative_sample(&mut rng))
                        .collect();

                    let loss =
                        model.update(self.config.model, center, &context_words, &negatives, lr);

                    epoch_loss += loss as f64;
                    epoch_pairs += 1;
                    global_step += 1;
                }

                pb.inc(1);
            }

            pb.finish_and_clear();

            let lr_now =
                (lr_start - (lr_start - lr_min) * (epoch + 1) as f32 / epochs as f32).max(lr_min);
            let avg_loss = if epoch_pairs > 0 {
                epoch_loss / epoch_pairs as f64
            } else {
                0.0
            };
            let elapsed = t_start.elapsed().as_secs_f64();

            let stats = EpochStats {
                epoch: epoch + 1,
                avg_loss,
                learning_rate: lr_now,
                pairs_processed: epoch_pairs,
                elapsed_secs: elapsed,
            };

            info!(
                "Epoch {}/{} | loss: {:.4} | lr: {:.5} | pairs: {} | {:.1}s",
                stats.epoch,
                epochs,
                stats.avg_loss,
                stats.learning_rate,
                stats.pairs_processed,
                stats.elapsed_secs
            );

            self.history.push(stats);
        }

        Ok(Embeddings::new(model, vocab, self.config.clone()))
    }

    /// Save training history as JSON.
    ///
    /// ```rust,no_run
    /// use word2vec::{Config, Trainer};
    /// let mut trainer = Trainer::new(Config::default());
    /// // trainer.train(...);
    /// // trainer.save_history("history.json").unwrap();
    /// ```
    pub fn save_history(&self, path: impl AsRef<Path>) -> Result<()> {
        let records: Vec<serde_json::Value> = self
            .history
            .iter()
            .map(|s| {
                serde_json::json!({
                    "epoch": s.epoch,
                    "avg_loss": s.avg_loss,
                    "learning_rate": s.learning_rate,
                    "pairs_processed": s.pairs_processed,
                    "elapsed_secs": s.elapsed_secs,
                })
            })
            .collect();

        let json = serde_json::to_string_pretty(&records)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Rough pair count for LR scheduling (doesn't account for subsampling).
    fn estimate_pairs(&self, sentences: &[Vec<String>], vocab: &Vocabulary) -> u64 {
        sentences
            .iter()
            .map(|s| {
                let len = s.iter().filter(|w| vocab.word2idx.contains_key(*w)).count() as u64;
                len.saturating_sub(1) * self.config.window_size as u64 * 2
            })
            .sum()
    }

    fn make_progress_bar(&self, epoch: usize, total: usize) -> ProgressBar {
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{prefix:.bold} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb.set_prefix(format!("Epoch {:>3}/{}", epoch + 1, self.config.epochs));
        pb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelType;

    fn tiny_corpus() -> Vec<String> {
        vec![
            "the quick brown fox".to_string(),
            "the lazy dog sleeps".to_string(),
            "fox and dog are animals".to_string(),
            "quick animals run fast".to_string(),
        ]
    }

    #[test]
    fn training_runs_without_panic() {
        let mut trainer = Trainer::new(Config {
            epochs: 2,
            embedding_dim: 10,
            ..Config::default()
        });
        let result = trainer.train(&tiny_corpus());
        assert!(result.is_ok(), "{:?}", result);
    }

    #[test]
    fn history_records_all_epochs() {
        let mut trainer = Trainer::new(Config {
            epochs: 3,
            embedding_dim: 8,
            ..Config::default()
        });
        trainer.train(&tiny_corpus()).unwrap();
        assert_eq!(trainer.history.len(), 3);
    }

    #[test]
    fn loss_is_finite() {
        let mut trainer = Trainer::new(Config {
            epochs: 2,
            embedding_dim: 8,
            ..Config::default()
        });
        trainer.train(&tiny_corpus()).unwrap();
        for s in &trainer.history {
            assert!(
                s.avg_loss.is_finite(),
                "epoch {} loss={}",
                s.epoch,
                s.avg_loss
            );
        }
    }

    #[test]
    fn cbow_training_runs() {
        let mut trainer = Trainer::new(Config {
            epochs: 2,
            embedding_dim: 8,
            model: ModelType::Cbow,
            ..Config::default()
        });
        assert!(trainer.train(&tiny_corpus()).is_ok());
    }

    #[test]
    fn empty_corpus_returns_error() {
        let mut trainer = Trainer::new(Config::default());
        let result = trainer.train(&[]);
        assert!(result.is_err());
    }
}
