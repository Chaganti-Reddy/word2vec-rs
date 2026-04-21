//! Basic training example demonstrating the primary API.
//!
//! Run with: `cargo run --example basic_training`

use word2vec::{Config, ModelType, Trainer};
use word2vec::plot::{plot_loss_curve, plot_word_vectors_pca};

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    let corpus: Vec<String> = vec![
        "paris is the capital of france and a major european city",
        "berlin is the capital of germany in central europe",
        "tokyo is the capital of japan and the largest city in asia",
        "london is the capital of england and the united kingdom",
        "madrid is the capital of spain in southern europe",
        "rome is the capital of italy and a historic city",
        "the king rules over the kingdom with great power",
        "the queen stands beside the king in royal ceremonies",
        "man and woman are both important members of society",
        "a dog is a loyal pet that loves to play outside",
        "a cat is an independent pet that enjoys sleeping",
        "machine learning algorithms learn from large datasets",
        "deep learning is a branch of machine learning using neural networks",
        "natural language processing enables text understanding by computers",
        "word embeddings are dense vector representations of words",
        "the quick brown fox jumps over the lazy dog near the river",
        "artificial intelligence is transforming how we interact with computers",
        "data science combines statistics programming and domain knowledge",
        "the sun rises every morning in the east bringing warm light",
        "rivers flow from mountains to the sea through valleys",
    ].into_iter().map(str::to_string).collect::<Vec<_>>()
     .into_iter().cycle().take(200).collect();

    let config = Config {
        embedding_dim: 50,
        window_size: 5,
        negative_samples: 5,
        epochs: 15,
        learning_rate: 0.025,
        min_count: 1,
        model: ModelType::SkipGram,
        seed: 42,
        ..Config::default()
    };

    println!("Training Word2Vec ({})...", config.model);
    let mut trainer = Trainer::new(config);
    let embeddings = trainer.train(&corpus).expect("training failed");

    println!("\nVocabulary size: {}", embeddings.vocab_size());
    println!("Embedding dim:   {}", embeddings.embedding_dim());

    println!("\n--- Nearest Neighbours ---");
    for word in ["paris", "king", "machine", "dog"] {
        let similar = embeddings.most_similar(word, 5);
        let formatted: Vec<String> = similar.iter()
            .map(|(w, s)| format!("{w}({s:.2})"))
            .collect();
        println!("  {word:<12} → {}", formatted.join(", "));
    }

    println!("\n--- Similarities ---");
    let pairs = [("paris", "berlin"), ("king", "queen"), ("dog", "cat"), ("paris", "dog")];
    for (a, b) in pairs {
        if let Ok(sim) = embeddings.similarity(a, b) {
            println!("  sim({a}, {b}) = {sim:.4}");
        }
    }

    println!("\n--- Analogy: king - man + woman ≈ ---");
    if let Ok(results) = embeddings.analogy("king", "man", "woman", 5) {
        for (w, s) in &results {
            println!("  {w:<15} ({s:.4})");
        }
    }

    embeddings.save("embeddings.json").expect("save failed");
    embeddings.save_text_format("embeddings.txt").expect("text save failed");
    trainer.save_history("training_history.json").expect("history save failed");

    plot_loss_curve(&trainer.history, "loss_curve.png").expect("plot failed");
    plot_word_vectors_pca(&embeddings, 40, "word_pca.png").expect("pca plot failed");

    println!("\nArtifacts saved:");
    println!("  embeddings.json / embeddings.txt");
    println!("  training_history.json");
    println!("  loss_curve.png");
    println!("  word_pca.png");
}
