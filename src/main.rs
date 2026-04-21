use std::collections::HashMap;
use rand::{RngExt, seq::IteratorRandom};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn dot_product(vec1: &[f32], vec2: &[f32]) -> f32 {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let mag_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (mag_a * mag_b)
}

fn update_embedding(embedding_matrix: &mut [Vec<f32>], input_id: usize, output_id: usize, error: f32, learning_rate: f32) {
    let input_vec = embedding_matrix[input_id].clone();
    let output_vec = embedding_matrix[output_id].clone();
    let embedding_dim = input_vec.len();

    for i in 0..embedding_dim {
        embedding_matrix[input_id][i] -= learning_rate * error * output_vec[i];
        embedding_matrix[output_id][i] -= learning_rate * error * input_vec[i];
    }
}

fn train(vocab: &HashMap<&str, usize>, context: &HashMap<&str, Vec<&str>>, embedding_matrix: &mut [Vec<f32>], learning_rate: f32) {
    for key in context.keys() {
        // real context word 
        for context_word in &context[key] {
            let input_id = vocab[key];
            let output_id = vocab[context_word];
            let input_vec = &embedding_matrix[input_id];
            let output_vec = &embedding_matrix[output_id];
            let score = dot_product(input_vec, output_vec);
            let error = sigmoid(score) - 1.0;
            update_embedding(embedding_matrix, input_id, output_id, error, learning_rate);
        }

        // negative sampling
        let mut rng = rand::rng();
        let negative_words = vocab.keys().filter(|w| !context[key].contains(w) && **w != *key).choose(&mut rng);

        if let Some(word) = negative_words {
            let input_id = vocab[key];
            let output_id = vocab[word];
            let input_vec = &embedding_matrix[input_id];
            let output_vec = &embedding_matrix[output_id];
            let score = dot_product(input_vec, output_vec);
            let error = sigmoid(score);
            update_embedding(embedding_matrix, input_id, output_id, error, learning_rate);
        }
    }
}

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

    let learning_rate = 0.01;
    for _ in 0..1000 {
        train(&vocab, &context, &mut embedding_matrix, learning_rate);
    }
}
