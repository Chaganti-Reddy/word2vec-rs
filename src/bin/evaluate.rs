//! Evaluation binary: load saved embeddings, run similarity & analogy queries.
//!
//! Usage:
//! ```text
//! cargo run --release --bin evaluate -- --model embeddings.json
//! ```

use std::env;
use word2vec::Embeddings;

fn main() {
    let args: Vec<String> = env::args().collect();

    let model_path = args.iter().position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("embeddings.json");

    println!("Loading embeddings from: {model_path}");
    let emb = match Embeddings::load(model_path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to load embeddings: {e}");
            std::process::exit(1);
        }
    };

    println!("Vocab size:  {}", emb.vocab_size());
    println!("Embed dim:   {}", emb.embedding_dim());
    println!();

    let stdin = std::io::stdin();
    let mut line = String::new();

    println!("Commands:");
    println!("  sim <word_a> <word_b>        — cosine similarity");
    println!("  near <word> <k>              — k nearest neighbours");
    println!("  analogy <pos> <neg> <pos2>   — solve analogy");
    println!("  quit");
    println!();

    loop {
        print!("> ");
        use std::io::Write;
        std::io::stdout().flush().ok();
        line.clear();
        stdin.read_line(&mut line).ok();
        let parts: Vec<&str> = line.split_whitespace().collect();

        match parts.as_slice() {
            ["quit"] | ["exit"] | ["q"] => break,

            ["sim", a, b] => {
                match emb.similarity(a, b) {
                    Ok(s) => println!("similarity({a}, {b}) = {s:.4}"),
                    Err(e) => println!("Error: {e}"),
                }
            }

            ["near", word, k] => {
                let k: usize = k.parse().unwrap_or(10);
                let results = emb.most_similar(word, k);
                if results.is_empty() {
                    println!("Word not found or no neighbours.");
                } else {
                    for (w, s) in &results {
                        println!("  {w:<20} {s:.4}");
                    }
                }
            }

            ["analogy", pos_a, neg_a, pos_b] => {
                match emb.analogy(pos_a, neg_a, pos_b, 5) {
                    Ok(results) => {
                        println!("{pos_a} - {neg_a} + {pos_b} ≈");
                        for (w, s) in &results {
                            println!("  {w:<20} {s:.4}");
                        }
                    }
                    Err(e) => println!("Error: {e}"),
                }
            }

            _ if line.trim().is_empty() => {}
            _ => println!("Unknown command. Try: sim, near, analogy, quit"),
        }
    }
}
