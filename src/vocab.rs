//! Vocabulary construction with frequency counting, subsampling, and
//! the unigram noise distribution table for negative sampling.
//!
//! # Subsampling
//!
//! Frequent words are stochastically discarded using Mikolov's formula:
//!
//! `P(discard) = 1 - sqrt(t / f)`
//!
//! where `t` is [`Config::subsample_threshold`] and `f` is the word's
//! relative corpus frequency.
//!
//! # Negative Sampling Table
//!
//! A flat array of `TABLE_SIZE` word indices drawn from `freq^0.75`,
//! which downweights very frequent words as negatives.

use std::collections::HashMap;
use rand::Rng;
#[allow(unused_imports)]
use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::{Result, Word2VecError};

/// Size of the unigram noise table.
const TABLE_SIZE: usize = 1_000_000;

/// Maps tokens ↔ integer indices and stores frequency statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    /// word → index
    pub word2idx: HashMap<String, usize>,
    /// index → word
    pub idx2word: Vec<String>,
    /// Raw corpus frequency per index
    pub counts: Vec<u64>,
    /// Flat noise table for O(1) negative sampling
    pub noise_table: Vec<u32>,
    /// Total token count (after min_count filter)
    pub total_tokens: u64,
}

impl Vocabulary {
    /// Build vocabulary from a tokenised corpus.
    ///
    /// Steps:
    /// 1. Count every token
    /// 2. Drop tokens below `config.min_count`
    /// 3. Sort by descending frequency (stable index order)
    /// 4. Build unigram noise table
    ///
    /// ```rust
    /// use word2vec::{Config, vocab::Vocabulary};
    ///
    /// let corpus = vec!["the cat sat on the mat".to_string()];
    /// let tokens: Vec<Vec<String>> = corpus.iter()
    ///     .map(|s| s.split_whitespace().map(str::to_string).collect())
    ///     .collect();
    ///
    /// let vocab = Vocabulary::build(&tokens, &Config::default()).unwrap();
    /// assert!(vocab.word2idx.contains_key("the"));
    /// assert_eq!(vocab.count("the"), 2);
    /// ```
    pub fn build(sentences: &[Vec<String>], config: &Config) -> Result<Self> {
        let mut raw_counts: HashMap<String, u64> = HashMap::new();
        for sentence in sentences {
            for token in sentence {
                *raw_counts.entry(token.clone()).or_insert(0) += 1;
            }
        }

        // Apply min_count filter
        let mut filtered: Vec<(String, u64)> = raw_counts
            .into_iter()
            .filter(|(_, c)| *c >= config.min_count as u64)
            .collect();

        if filtered.is_empty() {
            return Err(Word2VecError::EmptyVocabulary);
        }
        if filtered.len() < 2 {
            return Err(Word2VecError::CorpusTooSmall(filtered.len()));
        }

        // Stable sort: descending frequency, then alphabetical for tie-breaking
        filtered.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let total_tokens: u64 = filtered.iter().map(|(_, c)| c).sum();
        let mut word2idx = HashMap::with_capacity(filtered.len());
        let mut idx2word = Vec::with_capacity(filtered.len());
        let mut counts = Vec::with_capacity(filtered.len());

        for (idx, (word, count)) in filtered.into_iter().enumerate() {
            word2idx.insert(word.clone(), idx);
            idx2word.push(word);
            counts.push(count);
        }

        let noise_table = Self::build_noise_table(&counts);

        Ok(Self { word2idx, idx2word, counts, noise_table, total_tokens })
    }

    /// Number of unique tokens in vocabulary.
    #[inline]
    pub fn len(&self) -> usize {
        self.idx2word.len()
    }

    /// Returns `true` if the vocabulary contains no words.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.idx2word.is_empty()
    }

    /// Frequency of a word (0 if not in vocab).
    ///
    /// ```rust
    /// use word2vec::{Config, vocab::Vocabulary};
    /// let corpus = vec!["a a b".to_string()];
    /// let tokens: Vec<Vec<String>> = corpus.iter()
    ///     .map(|s| s.split_whitespace().map(str::to_string).collect())
    ///     .collect();
    /// let vocab = Vocabulary::build(&tokens, &Config::default()).unwrap();
    /// assert_eq!(vocab.count("a"), 2);
    /// assert_eq!(vocab.count("z"), 0);
    /// ```
    pub fn count(&self, word: &str) -> u64 {
        self.word2idx.get(word).map(|&i| self.counts[i]).unwrap_or(0)
    }

    /// Returns `true` if this word should be subsampled (discarded) given
    /// a uniformly random `dice` in [0, 1).
    ///
    /// Uses Mikolov's formula: `P(keep) = min(1, sqrt(t/f) + t/f)`.
    pub fn should_subsample(&self, idx: usize, threshold: f64, dice: f64) -> bool {
        let freq = self.counts[idx] as f64 / self.total_tokens as f64;
        let keep_prob = ((threshold / freq).sqrt() + threshold / freq).min(1.0);
        dice >= keep_prob
    }

    /// Draw a negative sample index from the noise distribution.
    ///
    /// Uses the precomputed unigram table for O(1) lookup.
    pub fn negative_sample(&self, rng: &mut SmallRng) -> usize {
        let idx = rng.gen_range(0..TABLE_SIZE);
        self.noise_table[idx] as usize
    }

    /// Build flat unigram noise table from `freq^0.75`.
    fn build_noise_table(counts: &[u64]) -> Vec<u32> {
        let powered: Vec<f64> = counts.iter().map(|&c| (c as f64).powf(0.75)).collect();
        let total: f64 = powered.iter().sum();

        let mut table = Vec::with_capacity(TABLE_SIZE);
        let mut cumulative = 0.0_f64;
        let mut word_idx = 0usize;

        for i in 0..TABLE_SIZE {
            let threshold = (i as f64 + 1.0) / TABLE_SIZE as f64;
            while cumulative / total < threshold && word_idx < powered.len() - 1 {
                cumulative += powered[word_idx];
                word_idx += 1;
            }
            table.push(word_idx as u32);
        }

        table
    }

    /// Tokenise and subsample a sentence, returning word indices.
    pub fn tokenise_and_subsample(
        &self,
        sentence: &[String],
        threshold: f64,
        rng: &mut SmallRng,
    ) -> Vec<usize> {
        sentence
            .iter()
            .filter_map(|w| self.word2idx.get(w).copied())
            .filter(|&idx| !self.should_subsample(idx, threshold, rng.gen()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vocab(text: &str) -> Vocabulary {
        let tokens = vec![text.split_whitespace().map(str::to_string).collect()];
        Vocabulary::build(&tokens, &Config::default()).unwrap()
    }

    #[test]
    fn vocab_word_counts() {
        let vocab = make_vocab("a a a b b c");
        assert_eq!(vocab.count("a"), 3);
        assert_eq!(vocab.count("b"), 2);
        assert_eq!(vocab.count("c"), 1);
        assert_eq!(vocab.count("z"), 0);
    }

    #[test]
    fn vocab_sorted_by_frequency() {
        let vocab = make_vocab("a a a b b c");
        assert_eq!(vocab.idx2word[0], "a");
        assert_eq!(vocab.idx2word[1], "b");
    }

    #[test]
    fn vocab_len() {
        let vocab = make_vocab("hello world hello");
        assert_eq!(vocab.len(), 2);
    }

    #[test]
    fn min_count_filters() {
        let tokens = vec!["a a a b b c c".split_whitespace().map(str::to_string).collect()];
        let cfg = Config { min_count: 2, ..Config::default() };
        let _vocab = Vocabulary::build(&tokens, &cfg).unwrap();
        let tokens2 = vec!["a a a b".split_whitespace().map(str::to_string).collect()];
        let result = Vocabulary::build(&tokens2, &cfg);
        assert!(result.is_err());
        let tokens3 = vec!["a a b b".split_whitespace().map(str::to_string).collect()];
        let vocab3 = Vocabulary::build(&tokens3, &cfg).unwrap();
        assert!(!vocab3.word2idx.contains_key("x"));
        assert!(vocab3.word2idx.contains_key("a"));
        assert!(vocab3.word2idx.contains_key("b"));
    }

    #[test]
    fn empty_corpus_errors() {
        let tokens: Vec<Vec<String>> = vec![vec![]];
        let result = Vocabulary::build(&tokens, &Config::default());
        assert!(result.is_err());
    }

    #[test]
    fn noise_table_has_correct_size() {
        let vocab = make_vocab("a a b c d e");
        assert_eq!(vocab.noise_table.len(), TABLE_SIZE);
    }

    #[test]
    fn negative_sample_in_range() {
        let vocab = make_vocab("the cat sat on the mat");
        let mut rng = SmallRng::seed_from_u64(0);
        for _ in 0..1000 {
            let idx = vocab.negative_sample(&mut rng);
            assert!(idx < vocab.len());
        }
    }
}
