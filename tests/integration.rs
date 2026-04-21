//! Integration tests: end-to-end training and inference validation.

use word2vec::{Config, ModelType, Trainer};

/// Repeated sentences so the model sees enough signal.
fn make_corpus(n: usize) -> Vec<String> {
    let templates = [
        "the quick brown fox jumps over the lazy dog",
        "paris is the capital of france",
        "berlin is the capital of germany",
        "tokyo is the capital of japan",
        "king and queen are royalty",
        "man and woman are people",
        "cat and dog are animals",
        "deep learning neural networks artificial intelligence machine learning",
    ];
    (0..n)
        .map(|i| templates[i % templates.len()].to_string())
        .collect()
}

#[test]
fn skipgram_trains_and_produces_embeddings() {
    let corpus = make_corpus(40);
    let mut trainer = Trainer::new(Config {
        epochs: 5,
        embedding_dim: 30,
        model: ModelType::SkipGram,
        min_count: 1,
        ..Config::default()
    });
    let emb = trainer.train(&corpus).unwrap();
    assert!(emb.vocab_size() >= 10);
    assert_eq!(emb.embedding_dim(), 30);
}

#[test]
fn cbow_trains_and_produces_embeddings() {
    let corpus = make_corpus(40);
    let mut trainer = Trainer::new(Config {
        epochs: 5,
        embedding_dim: 30,
        model: ModelType::Cbow,
        ..Config::default()
    });
    let emb = trainer.train(&corpus).unwrap();
    assert!(emb.vocab_size() >= 5);
}

#[test]
fn loss_decreases_over_epochs() {
    let corpus = make_corpus(80);
    let mut trainer = Trainer::new(Config {
        epochs: 6,
        embedding_dim: 20,
        ..Config::default()
    });
    trainer.train(&corpus).unwrap();

    let losses: Vec<f64> = trainer.history.iter().map(|s| s.avg_loss).collect();
    let first_half: f64 =
        losses[..losses.len() / 2].iter().sum::<f64>() / (losses.len() / 2) as f64;
    let second_half: f64 =
        losses[losses.len() / 2..].iter().sum::<f64>() / (losses.len() - losses.len() / 2) as f64;
    assert!(
        second_half < first_half,
        "Loss should decrease: first_half={first_half:.4}, second_half={second_half:.4}"
    );
}

#[test]
fn similar_words_are_contextually_close() {
    let corpus = make_corpus(200);
    let mut trainer = Trainer::new(Config {
        epochs: 10,
        embedding_dim: 50,
        negative_samples: 5,
        min_count: 1,
        ..Config::default()
    });
    let emb = trainer.train(&corpus).unwrap();

    let paris_berlin = emb.similarity("paris", "berlin").unwrap();
    let paris_tokyo = emb.similarity("paris", "tokyo").unwrap();
    let paris_dog = emb.similarity("paris", "dog").unwrap();

    assert!(
        paris_berlin > paris_dog || paris_tokyo > paris_dog,
        "capital pair should beat capital-animal: paris-berlin={paris_berlin:.3}, paris-tokyo={paris_tokyo:.3}, paris-dog={paris_dog:.3}"
    );
}

#[test]
fn most_similar_excludes_query_word() {
    let corpus = make_corpus(40);
    let mut trainer = Trainer::new(Config {
        epochs: 3,
        embedding_dim: 20,
        ..Config::default()
    });
    let emb = trainer.train(&corpus).unwrap();
    let results = emb.most_similar("king", 10);
    assert!(
        !results.iter().any(|(w, _)| w == "king"),
        "query word should not appear in results"
    );
}

#[test]
fn most_similar_returns_at_most_k_results() {
    let corpus = make_corpus(20);
    let mut trainer = Trainer::new(Config {
        epochs: 2,
        embedding_dim: 10,
        ..Config::default()
    });
    let emb = trainer.train(&corpus).unwrap();
    let results = emb.most_similar("the", 3);
    assert!(results.len() <= 3);
}

#[test]
fn similarity_is_symmetric() {
    let corpus = make_corpus(30);
    let mut trainer = Trainer::new(Config {
        epochs: 2,
        embedding_dim: 10,
        ..Config::default()
    });
    let emb = trainer.train(&corpus).unwrap();
    let ab = emb.similarity("king", "queen").unwrap();
    let ba = emb.similarity("queen", "king").unwrap();
    assert!((ab - ba).abs() < 1e-5, "similarity should be symmetric");
}

#[test]
fn save_and_load_preserves_similarity() {
    let corpus = make_corpus(30);
    let mut trainer = Trainer::new(Config {
        epochs: 2,
        embedding_dim: 10,
        ..Config::default()
    });
    let emb = trainer.train(&corpus).unwrap();
    let original_sim = emb.similarity("king", "queen").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.json");
    emb.save(&path).unwrap();

    let loaded = word2vec::Embeddings::load(&path).unwrap();
    let loaded_sim = loaded.similarity("king", "queen").unwrap();

    assert!(
        (original_sim - loaded_sim).abs() < 1e-5,
        "similarity changed after save/load: {original_sim} vs {loaded_sim}"
    );
}

#[test]
fn unknown_word_returns_error() {
    let corpus = make_corpus(10);
    let mut trainer = Trainer::new(Config {
        epochs: 1,
        embedding_dim: 8,
        ..Config::default()
    });
    let emb = trainer.train(&corpus).unwrap();
    assert!(emb.similarity("zzz_notaword", "king").is_err());
}

#[test]
fn text_format_export_creates_valid_file() {
    let corpus = make_corpus(20);
    let mut trainer = Trainer::new(Config {
        epochs: 1,
        embedding_dim: 8,
        ..Config::default()
    });
    let emb = trainer.train(&corpus).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vectors.txt");
    emb.save_text_format(&path).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    let mut lines = content.lines();
    let header = lines.next().unwrap();
    let parts: Vec<&str> = header.split_whitespace().collect();
    assert_eq!(parts.len(), 2);
    let vocab_size: usize = parts[0].parse().unwrap();
    let dim: usize = parts[1].parse().unwrap();
    assert_eq!(vocab_size, emb.vocab_size());
    assert_eq!(dim, emb.embedding_dim());
}
