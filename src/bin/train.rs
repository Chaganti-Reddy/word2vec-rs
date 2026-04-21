//! CLI training entrypoint.
//!
//! Usage:
//! ```text
//! cargo run --release --bin train -- \
//!   --input corpus.txt \
//!   --output embeddings.json \
//!   --dim 100 \
//!   --window 5 \
//!   --epochs 10 \
//!   --model skipgram \
//!   --negative 5
//! ```

use std::env;
use std::process;
use word2vec::{Config, ModelType, Trainer};
use word2vec::plot::{plot_loss_curve, plot_word_vectors_pca};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = env::args().collect();
    let config = parse_args(&args);

    let corpus = if let Some(path) = args.iter().position(|a| a == "--input")
        .and_then(|i| args.get(i + 1))
    {
        match std::fs::read_to_string(path) {
            Ok(text) => text.lines().map(str::to_string).collect::<Vec<_>>(),
            Err(e) => {
                eprintln!("Error reading corpus: {e}");
                process::exit(1);
            }
        }
    } else {
        eprintln!("No --input provided; using built-in sample corpus.");
        sample_corpus()
    };

    println!("=== Word2Vec Training ===");
    println!("Model:      {}", config.model);
    println!("Dim:        {}", config.embedding_dim);
    println!("Window:     {}", config.window_size);
    println!("Negatives:  {}", config.negative_samples);
    println!("Epochs:     {}", config.epochs);
    println!("Corpus:     {} sentences", corpus.len());
    println!("========================\n");

    let mut trainer = Trainer::new(config);

    let embeddings = match trainer.train(&corpus) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Training failed: {e}");
            process::exit(1);
        }
    };

    println!("\n=== Training Summary ===");
    for stat in &trainer.history {
        println!(
            "Epoch {:>3} | loss: {:.4} | lr: {:.5} | pairs: {:>8} | {:.2}s",
            stat.epoch, stat.avg_loss, stat.learning_rate,
            stat.pairs_processed, stat.elapsed_secs
        );
    }

    let out_path = args.iter().position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("embeddings.json");

    embeddings.save(out_path).expect("Failed to save embeddings");
    println!("\nEmbeddings saved to: {out_path}");

    let txt_path = out_path.replace(".json", ".txt");
    embeddings.save_text_format(&txt_path).ok();

    trainer.save_history("training_history.json").ok();

    plot_loss_curve(&trainer.history, "loss_curve.png").ok();
    println!("Loss curve saved to: loss_curve.png");

    plot_word_vectors_pca(&embeddings, 50, "word_pca.png").ok();
    println!("PCA plot saved to: word_pca.png");

    println!("\n=== Nearest Neighbours (sample) ===");
    for query in embeddings.words().into_iter().take(5) {
        let similar = embeddings.most_similar(query, 5);
        let pairs: Vec<String> = similar.iter().map(|(w, s)| format!("{w}({s:.2})")).collect();
        println!("  {query:<15} → {}", pairs.join(", "));
    }
}

fn sample_corpus() -> Vec<String> {
    vec![
        "the quick brown fox jumps over the lazy dog".to_string(),
        "the dog chased the fox through the forest".to_string(),
        "machine learning is a subset of artificial intelligence".to_string(),
        "deep learning uses neural networks with many layers".to_string(),
        "natural language processing enables computers to understand text".to_string(),
        "word embeddings capture semantic relationships between words".to_string(),
        "paris is the capital of france and a beautiful city".to_string(),
        "berlin is the capital of germany in central europe".to_string(),
        "tokyo is the capital of japan and the largest city in asia".to_string(),
        "king and queen are royalty words in english language".to_string(),
        "man and woman are common words used in everyday speech".to_string(),
        "the cat sat on the mat near the door".to_string(),
        "a dog is a loyal animal that loves to play".to_string(),
        "the sun rises in the east and sets in the west".to_string(),
        "learning new skills requires practice and dedication every day".to_string(),
    ]
}

fn parse_args(args: &[String]) -> Config {
    let mut config = Config::default();

    let get_arg = |flag: &str| -> Option<&str> {
        args.iter().position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str())
    };

    if let Some(v) = get_arg("--dim").and_then(|s| s.parse().ok()) {
        config.embedding_dim = v;
    }
    if let Some(v) = get_arg("--window").and_then(|s| s.parse().ok()) {
        config.window_size = v;
    }
    if let Some(v) = get_arg("--epochs").and_then(|s| s.parse().ok()) {
        config.epochs = v;
    }
    if let Some(v) = get_arg("--negative").and_then(|s| s.parse().ok()) {
        config.negative_samples = v;
    }
    if let Some(v) = get_arg("--lr").and_then(|s| s.parse().ok()) {
        config.learning_rate = v;
    }
    if let Some(v) = get_arg("--min-count").and_then(|s| s.parse().ok()) {
        config.min_count = v;
    }
    if let Some(v) = get_arg("--seed").and_then(|s| s.parse().ok()) {
        config.seed = v;
    }
    if let Some(m) = get_arg("--model") {
        config.model = match m.to_lowercase().as_str() {
            "cbow" => ModelType::Cbow,
            _ => ModelType::SkipGram,
        };
    }

    config
}
