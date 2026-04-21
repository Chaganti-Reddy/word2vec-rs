//! Neural network weights and forward/backward pass for Skip-gram and CBOW
//! with Negative Sampling.
//!
//! # Weight Matrices
//!
//! - `input_weights` (`W_in`): shape `[vocab_size × embedding_dim]` — the
//!   "input" or center-word embedding matrix.
//! - `output_weights` (`W_out`): shape `[vocab_size × embedding_dim]` — the
//!   context/output embedding matrix used in the dot-product scoring.
//!
//! # Negative Sampling Loss
//!
//! For a positive pair (center `c`, context `o`) and `k` negatives `n_i`:
//!
//! `L = log σ(v_o · v_c) + Σ log σ(-v_{n_i} · v_c)`
//!
//! Gradients are applied in-place via SGD.

use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

use crate::config::ModelType;

/// Sigmoid activation.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Xavier uniform initialisation range for a given dimension.
#[inline]
fn xavier_range(dim: usize) -> f32 {
    (6.0_f32 / dim as f32).sqrt()
}

/// Core weight matrices for Word2Vec.
///
/// Both matrices are row-major flat `Vec<f32>` for cache efficiency.
/// Row `i` starts at byte offset `i * embedding_dim * 4`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub input_weights: Vec<f32>,
    pub output_weights: Vec<f32>,
}

impl Model {
    /// Initialise with Xavier uniform weights.
    ///
    /// ```rust
    /// use word2vec::model::Model;
    /// let m = Model::new(100, 50, 42);
    /// assert_eq!(m.input_weights.len(), 100 * 50);
    /// ```
    pub fn new(vocab_size: usize, embedding_dim: usize, seed: u64) -> Self {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(seed);
        let r = xavier_range(embedding_dim);
        let n = vocab_size * embedding_dim;
        let input_weights: Vec<f32> = (0..n).map(|_| rng.gen_range(-r..r)).collect();
        let output_weights = vec![0.0_f32; n]; // output weights start at zero

        Self {
            vocab_size,
            embedding_dim,
            input_weights,
            output_weights,
        }
    }

    /// Get the embedding vector for word at `idx` (slice into input_weights).
    #[inline]
    pub fn input_vec(&self, idx: usize) -> &[f32] {
        let start = idx * self.embedding_dim;
        &self.input_weights[start..start + self.embedding_dim]
    }

    /// Mutable access to input embedding.
    #[inline]
    pub fn input_vec_mut(&mut self, idx: usize) -> &mut [f32] {
        let start = idx * self.embedding_dim;
        &mut self.input_weights[start..start + self.embedding_dim]
    }

    /// Mutable access to output embedding.
    #[inline]
    pub fn output_vec_mut(&mut self, idx: usize) -> &mut [f32] {
        let start = idx * self.embedding_dim;
        &mut self.output_weights[start..start + self.embedding_dim]
    }

    /// Dot product between two embedding rows.
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Skip-gram update: given center word, update for one context word
    /// and `n_neg` negative samples.
    ///
    /// Returns the binary cross-entropy loss contribution.
    pub fn skipgram_update(
        &mut self,
        center: usize,
        context: usize,
        negatives: &[usize],
        lr: f32,
    ) -> f32 {
        let dim = self.embedding_dim;
        let mut grad_input = vec![0.0f32; dim];
        let mut loss = 0.0f32;

        // Positive pair
        {
            let score = Self::dot(self.input_vec(center), self.output_vec(context));
            let sig = sigmoid(score);
            let err = sig - 1.0;
            loss -= sig.ln().max(-30.0);

            let center_vec: Vec<f32> = self.input_vec(center).to_vec();
            let out_vec: Vec<f32> = self.output_vec(context).to_vec();

            for i in 0..dim {
                grad_input[i] += err * out_vec[i];
                self.output_weights[context * dim + i] -= lr * err * center_vec[i];
            }
        }

        // Negative pairs
        for &neg in negatives {
            if neg == context {
                continue;
            }
            let score = Self::dot(self.input_vec(center), self.output_vec(neg));
            let sig = sigmoid(score);
            loss -= (1.0 - sig).ln().max(-30.0);

            let center_vec: Vec<f32> = self.input_vec(center).to_vec();
            let neg_out: Vec<f32> = self.output_vec(neg).to_vec();

            for i in 0..dim {
                grad_input[i] += sig * neg_out[i];
                self.output_weights[neg * dim + i] -= lr * sig * center_vec[i];
            }
        }

        // Apply gradient to center word
        for (i, &grad) in grad_input.iter().enumerate() {
            self.input_weights[center * dim + i] -= lr * grad;
        }

        loss
    }

    /// CBOW update: average context embeddings, predict center word.
    pub fn cbow_update(
        &mut self,
        center: usize,
        context_words: &[usize],
        negatives: &[usize],
        lr: f32,
    ) -> f32 {
        if context_words.is_empty() {
            return 0.0;
        }
        let dim = self.embedding_dim;
        let scale = 1.0 / context_words.len() as f32;

        let mut ctx_avg = vec![0.0f32; dim];
        for &cidx in context_words {
            let v = self.input_vec(cidx);
            for i in 0..dim {
                ctx_avg[i] += v[i] * scale;
            }
        }

        let mut grad_ctx = vec![0.0f32; dim];
        let mut loss = 0.0f32;

        {
            let score: f32 = ctx_avg
                .iter()
                .zip(self.output_vec(center))
                .map(|(a, b)| a * b)
                .sum();
            let sig = sigmoid(score);
            let err = sig - 1.0;
            loss -= sig.ln().max(-30.0);
            let out_center: Vec<f32> = self.output_vec(center).to_vec();
            for i in 0..dim {
                grad_ctx[i] += err * out_center[i];
                self.output_weights[center * dim + i] -= lr * err * ctx_avg[i];
            }
        }

        for &neg in negatives {
            if neg == center {
                continue;
            }
            let score: f32 = ctx_avg
                .iter()
                .zip(self.output_vec(neg))
                .map(|(a, b)| a * b)
                .sum();
            let sig = sigmoid(score);
            loss -= (1.0 - sig).ln().max(-30.0);
            let out_neg: Vec<f32> = self.output_vec(neg).to_vec();
            for i in 0..dim {
                grad_ctx[i] += sig * out_neg[i];
                self.output_weights[neg * dim + i] -= lr * sig * ctx_avg[i];
            }
        }

        for &cidx in context_words {
            for (i, &grad) in grad_ctx.iter().enumerate() {
                self.input_weights[cidx * dim + i] -= lr * grad * scale;
            }
        }

        loss
    }

    /// Convenience: output vector slice.
    fn output_vec(&self, idx: usize) -> &[f32] {
        let start = idx * self.embedding_dim;
        &self.output_weights[start..start + self.embedding_dim]
    }

    /// Run a full Skip-gram or CBOW update, dispatching on model type.
    pub fn update(
        &mut self,
        model_type: ModelType,
        center: usize,
        context_window: &[usize],
        negatives: &[usize],
        lr: f32,
    ) -> f32 {
        match model_type {
            ModelType::SkipGram => {
                let mut total_loss = 0.0;
                for &ctx in context_window {
                    total_loss += self.skipgram_update(center, ctx, negatives, lr);
                }
                total_loss
            }
            ModelType::Cbow => self.cbow_update(center, context_window, negatives, lr),
        }
    }
}

/// Generate training examples from a tokenised sentence.
///
/// Returns `(center_idx, context_indices)` pairs using a dynamic window
/// (window size drawn uniformly from `[1, window_size]`).
pub fn sentence_to_pairs(
    tokens: &[usize],
    window_size: usize,
    rng: &mut SmallRng,
) -> Vec<(usize, Vec<usize>)> {
    use rand::Rng;
    let mut pairs = Vec::new();

    for (pos, &center) in tokens.iter().enumerate() {
        // Dynamic window (Mikolov et al. 2013 trick)
        let win: usize = rng.gen_range(1..=window_size);
        let start = pos.saturating_sub(win);
        let end = (pos + win + 1).min(tokens.len());

        let context: Vec<usize> = (start..end)
            .filter(|&i| i != pos)
            .map(|i| tokens[i])
            .collect();

        if !context.is_empty() {
            pairs.push((center, context));
        }
    }

    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn model_weight_dimensions() {
        let m = Model::new(50, 10, 0);
        assert_eq!(m.input_weights.len(), 50 * 10);
        assert_eq!(m.output_weights.len(), 50 * 10);
    }

    #[test]
    fn skipgram_update_returns_finite_loss() {
        let mut m = Model::new(10, 8, 0);
        let loss = m.skipgram_update(0, 1, &[2, 3, 4], 0.025);
        assert!(loss.is_finite(), "loss={loss}");
        assert!(loss >= 0.0, "loss should be non-negative");
    }

    #[test]
    fn cbow_update_returns_finite_loss() {
        let mut m = Model::new(10, 8, 0);
        let loss = m.cbow_update(0, &[1, 2, 3], &[4, 5], 0.025);
        assert!(loss.is_finite());
    }

    #[test]
    fn loss_decreases_after_repeated_updates() {
        let mut m = Model::new(10, 16, 99);
        let first = m.skipgram_update(0, 1, &[2, 3], 0.1);
        let mut last = first;
        for _ in 0..200 {
            last = m.skipgram_update(0, 1, &[2, 3], 0.01);
        }
        assert!(
            last < first,
            "loss should decrease with repetition: {first} -> {last}"
        );
    }

    #[test]
    fn sentence_to_pairs_respects_window() {
        let mut rng = SmallRng::seed_from_u64(0);
        let tokens = vec![0, 1, 2, 3, 4];
        let pairs = sentence_to_pairs(&tokens, 2, &mut rng);
        assert!(!pairs.is_empty());
        for (_, ctx) in &pairs {
            assert!(!ctx.is_empty());
        }
    }

    #[test]
    fn sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.9999);
        assert!(sigmoid(-100.0) < 0.0001);
    }
}
