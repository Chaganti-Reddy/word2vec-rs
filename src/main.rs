use std::collections::HashMap;
use rand::{RngExt};

fn main() {
    let text = "the cat sat on the mat";
    let tokens: Vec<&str> = text.split_whitespace().collect();

    let mut vocab: HashMap<&str, usize> = HashMap::new();
    let mut id = 0;

    for token in &tokens {
        if !vocab.contains_key(token) {
            vocab.insert(token, id);
            id += 1;
        }
    }

    let mut context: HashMap<&str, Vec<&str>> = HashMap::new();

    for i in 0..tokens.len() {
        for j in i.saturating_sub(2)..=(i + 2).min(tokens.len() - 1) {
            if i == j {
                continue;
            }
            context.entry(tokens[i]).or_insert(vec![]).push(tokens[j]);
        }
    }

    let mut rng = rand::rng();
    let embedding_dim = 3;

    let mut embedding_matrix: Vec<Vec<f32>> = vec![vec![0.0; embedding_dim]; vocab.len()];
    for id in vocab.values() {
        embedding_matrix[*id] = (0..embedding_dim)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
    }

    println!("{:?}", embedding_matrix);
}
