//! Post-training embedding access: similarity, analogy, save/load.
//!
//! # Usage
//!
//! ```rust,no_run
//! use word2vec::{Config, Trainer};
//!
//! let corpus = vec!["the cat sat on the mat".repeat(50)];
//! let mut trainer = Trainer::new(Config { epochs: 3, ..Config::default() });
//! let emb = trainer.train(&corpus).unwrap();
//!
//! let similar = emb.most_similar("cat", 3);
//! // [("mat", 0.92), ("sat", 0.87), ("on", 0.81)] (values illustrative)
//!
//! // king - man + woman ≈ queen
//! // let queen = emb.analogy("king", "man", "woman", 3);
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::config::Config;
use crate::error::{Result, Word2VecError};
use crate::model::Model;
use crate::vocab::Vocabulary;

/// Trained embeddings with vocabulary — the primary inference interface.
#[derive(Debug, Serialize, Deserialize)]
pub struct Embeddings {
    model: Model,
    vocab: Vocabulary,
    config: Config,
}

impl Embeddings {
    /// Wrap a trained model and vocabulary.
    pub(crate) fn new(model: Model, vocab: Vocabulary, config: Config) -> Self {
        Self {
            model,
            vocab,
            config,
        }
    }

    /// Number of words in the vocabulary.
    ///
    /// ```rust,no_run
    /// # use word2vec::{Config, Trainer};
    /// # let corpus = vec!["hello world hello".to_string()];
    /// # let mut t = Trainer::new(Config { epochs: 1, ..Config::default() });
    /// # let emb = t.train(&corpus).unwrap();
    /// assert!(emb.vocab_size() >= 2);
    /// ```
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Get the raw embedding vector for a word.
    ///
    /// Returns `None` if the word is not in the vocabulary.
    ///
    /// ```rust,no_run
    /// # use word2vec::{Config, Trainer};
    /// # let corpus = vec!["hello world".to_string()];
    /// # let mut t = Trainer::new(Config { epochs: 1, ..Config::default() });
    /// # let emb = t.train(&corpus).unwrap();
    /// assert!(emb.get_vector("hello").is_some());
    /// assert!(emb.get_vector("nonexistent").is_none());
    /// ```
    pub fn get_vector(&self, word: &str) -> Option<&[f32]> {
        self.vocab
            .word2idx
            .get(word)
            .map(|&i| self.model.input_vec(i))
    }

    /// Cosine similarity between two words.
    ///
    /// Returns a value in `[-1.0, 1.0]`, or an error if either word
    /// is not in the vocabulary.
    ///
    /// ```rust,no_run
    /// # use word2vec::{Config, Trainer};
    /// # let corpus = vec!["cat mat cat mat bat".to_string()];
    /// # let mut t = Trainer::new(Config { epochs: 1, ..Config::default() });
    /// # let emb = t.train(&corpus).unwrap();
    /// let sim = emb.similarity("cat", "mat").unwrap();
    /// assert!(sim >= -1.0 && sim <= 1.0);
    /// ```
    pub fn similarity(&self, word_a: &str, word_b: &str) -> Result<f32> {
        let va = self
            .get_vector(word_a)
            .ok_or_else(|| Word2VecError::UnknownWord(word_a.to_string()))?;
        let vb = self
            .get_vector(word_b)
            .ok_or_else(|| Word2VecError::UnknownWord(word_b.to_string()))?;
        Ok(cosine_similarity(va, vb))
    }

    /// Find the `top_k` most similar words to `query`.
    ///
    /// Returns `(word, cosine_similarity)` pairs sorted descending.
    /// The query word itself is excluded from results.
    ///
    /// ```rust,no_run
    /// # use word2vec::{Config, Trainer};
    /// # let corpus = (0..100).map(|_| "cat mat bat rat sat".to_string()).collect::<Vec<_>>();
    /// # let mut t = Trainer::new(Config { epochs: 3, ..Config::default() });
    /// # let emb = t.train(&corpus).unwrap();
    /// let nearest = emb.most_similar("cat", 3);
    /// assert!(nearest.len() <= 3);
    /// ```
    pub fn most_similar(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        let query_vec = match self.get_vector(query) {
            Some(v) => v,
            None => return vec![],
        };

        let mut scores: Vec<(usize, f32)> = (0..self.vocab.len())
            .filter(|&i| self.vocab.idx2word[i] != query)
            .map(|i| (i, cosine_similarity(query_vec, self.model.input_vec(i))))
            .collect();

        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        scores
            .into_iter()
            .map(|(i, sim)| (self.vocab.idx2word[i].clone(), sim))
            .collect()
    }

    /// Solve an analogy: `pos_a - neg_a + pos_b ≈ result`.
    ///
    /// Classic example: `king - man + woman ≈ queen`.
    ///
    /// Excludes `pos_a`, `neg_a`, and `pos_b` from the candidate list.
    ///
    /// Returns `(word, score)` pairs sorted by cosine similarity to
    /// the query vector.
    pub fn analogy(
        &self,
        pos_a: &str,
        neg_a: &str,
        pos_b: &str,
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let va = self
            .get_vector(pos_a)
            .ok_or_else(|| Word2VecError::UnknownWord(pos_a.to_string()))?;
        let vna = self
            .get_vector(neg_a)
            .ok_or_else(|| Word2VecError::UnknownWord(neg_a.to_string()))?;
        let vb = self
            .get_vector(pos_b)
            .ok_or_else(|| Word2VecError::UnknownWord(pos_b.to_string()))?;

        let dim = self.embedding_dim();
        let query: Vec<f32> = (0..dim).map(|i| va[i] - vna[i] + vb[i]).collect();
        let query_norm = normalize_vec(&query);

        let exclude = [pos_a, neg_a, pos_b];
        let mut scores: Vec<(usize, f32)> = (0..self.vocab.len())
            .filter(|&i| !exclude.contains(&self.vocab.idx2word[i].as_str()))
            .map(|i| (i, cosine_similarity(&query_norm, self.model.input_vec(i))))
            .collect();

        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        Ok(scores
            .into_iter()
            .map(|(i, s)| (self.vocab.idx2word[i].clone(), s))
            .collect())
    }

    /// Return all words in vocabulary sorted alphabetically.
    pub fn words(&self) -> Vec<&str> {
        let mut words: Vec<&str> = self.vocab.idx2word.iter().map(|s| s.as_str()).collect();
        words.sort_unstable();
        words
    }

    /// Save embeddings to JSON.
    ///
    /// The file contains both the model weights and vocabulary so it can
    /// be loaded independently.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load embeddings from a JSON file produced by [`save`](Self::save).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let emb: Self = serde_json::from_str(&json)?;
        Ok(emb)
    }

    /// Export just the word vectors as a plain-text file (word2vec format).
    ///
    /// Format: first line is `<vocab_size> <dim>`, then one word per line
    /// followed by space-separated floats.
    pub fn save_text_format(&self, path: impl AsRef<Path>) -> Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "{} {}", self.vocab.len(), self.embedding_dim())?;
        for (i, word) in self.vocab.idx2word.iter().enumerate() {
            let vec = self.model.input_vec(i);
            let vec_str: Vec<String> = vec.iter().map(|v| format!("{:.6}", v)).collect();
            writeln!(f, "{} {}", word, vec_str.join(" "))?;
        }
        Ok(())
    }

    /// Get a reference to the vocabulary.
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }
}

/// Cosine similarity between two vectors (handles zero-norm gracefully).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Return a L2-normalised copy of a vector.
pub fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn normalize_unit_vector() {
        let v = vec![3.0, 4.0];
        let n = normalize_vec(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    fn make_embeddings() -> Embeddings {
        use crate::{Config, Trainer};
        let corpus: Vec<String> = (0..30)
            .map(|i| format!("word{} word{} word{}", i % 5, (i + 1) % 5, (i + 2) % 5))
            .collect();
        let mut trainer = Trainer::new(Config {
            epochs: 2,
            embedding_dim: 10,
            ..Config::default()
        });
        trainer.train(&corpus).unwrap()
    }

    #[test]
    fn get_vector_known_word() {
        let emb = make_embeddings();
        assert!(emb.get_vector("word0").is_some());
    }

    #[test]
    fn get_vector_unknown_word() {
        let emb = make_embeddings();
        assert!(emb.get_vector("unknown_xyz").is_none());
    }

    #[test]
    fn most_similar_returns_sorted_results() {
        let emb = make_embeddings();
        let results = emb.most_similar("word0", 3);
        for window in results.windows(2) {
            assert!(window[0].1 >= window[1].1, "results not sorted");
        }
    }

    #[test]
    fn similarity_self_is_one() {
        let emb = make_embeddings();
        let sim = emb.similarity("word0", "word0").unwrap();
        assert!((sim - 1.0).abs() < 1e-4, "self-similarity={sim}");
    }

    #[test]
    fn save_load_roundtrip() {
        let emb = make_embeddings();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("emb.json");
        emb.save(&path).unwrap();
        let loaded = Embeddings::load(&path).unwrap();
        assert_eq!(loaded.vocab_size(), emb.vocab_size());
    }
}
