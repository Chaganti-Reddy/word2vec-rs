//! Training visualisation: loss curves and 2D PCA word projection plots.
//!
//! Uses [`plotters`] to render PNG files.
//!
//! # Example
//!
//! ```rust,no_run
//! use word2vec::plot::{plot_loss_curve, plot_word_vectors_pca};
//! use word2vec::trainer::EpochStats;
//!
//! // After training:
//! // plot_loss_curve(&trainer.history, "loss.png").unwrap();
//! // plot_word_vectors_pca(&embeddings, 50, "pca.png").unwrap();
//! ```

use plotters::prelude::*;

use crate::embeddings::Embeddings;
use crate::error::{Result, Word2VecError};
use crate::trainer::EpochStats;

/// Render a loss-vs-epoch line chart to a PNG file.
///
/// # Arguments
///
/// * `history` — Slice of per-epoch statistics from [`Trainer::history`].
/// * `output_path` — Destination PNG path (created or overwritten).
pub fn plot_loss_curve(history: &[EpochStats], output_path: &str) -> Result<()> {
    if history.is_empty() {
        return Err(Word2VecError::Plot("history is empty".to_string()));
    }

    let root = BitMapBackend::new(output_path, (900, 500)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| Word2VecError::Plot(e.to_string()))?;

    let losses: Vec<f64> = history.iter().map(|s| s.avg_loss).collect();
    let max_loss = losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_loss = losses.iter().cloned().fold(f64::INFINITY, f64::min);
    let padding = (max_loss - min_loss).max(0.1) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Word2Vec Training Loss", ("sans-serif", 28).into_font())
        .margin(30)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(
            1usize..history.len(),
            (min_loss - padding)..(max_loss + padding),
        )
        .map_err(|e| Word2VecError::Plot(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Average Loss")
        .axis_desc_style(("sans-serif", 16))
        .draw()
        .map_err(|e| Word2VecError::Plot(e.to_string()))?;

    // Line
    chart
        .draw_series(LineSeries::new(
            history.iter().enumerate().map(|(i, s)| (i + 1, s.avg_loss)),
            &BLUE,
        ))
        .map_err(|e| Word2VecError::Plot(e.to_string()))?
        .label("avg loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    // Dots at each epoch
    chart
        .draw_series(
            history.iter().enumerate().map(|(i, s)| {
                Circle::new((i + 1, s.avg_loss), 4, BLUE.filled())
            })
        )
        .map_err(|e| Word2VecError::Plot(e.to_string()))?;

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .map_err(|e| Word2VecError::Plot(e.to_string()))?;

    root.present().map_err(|e| Word2VecError::Plot(e.to_string()))?;
    Ok(())
}

/// Project the top-`n_words` most frequent words into 2D using PCA
/// (covariance-free: uses power iteration on two principal components)
/// and render a scatter plot with word labels.
pub fn plot_word_vectors_pca(emb: &Embeddings, n_words: usize, output_path: &str) -> Result<()> {
    let n = n_words.min(emb.vocab_size());
    if n < 2 {
        return Err(Word2VecError::Plot("need at least 2 words to plot".to_string()));
    }

    // Collect vectors (top-n by vocab order = most frequent due to sorted vocab)
    let words: Vec<&str> = emb.vocab().idx2word.iter().take(n).map(|s| s.as_str()).collect();
    let vectors: Vec<&[f32]> = words.iter()
        .filter_map(|w| emb.get_vector(w))
        .collect();

    let dim = vectors[0].len();
    let count = vectors.len();

    // Center the data
    let mean: Vec<f64> = (0..dim)
        .map(|d| vectors.iter().map(|v| v[d] as f64).sum::<f64>() / count as f64)
        .collect();

    let centered: Vec<Vec<f64>> = vectors.iter()
        .map(|v| (0..dim).map(|d| v[d] as f64 - mean[d]).collect())
        .collect();

    // Power iteration for top-2 PCs
    let pc1 = power_iteration(&centered, dim, 30, 0);
    let pc2 = power_iteration_deflated(&centered, dim, 30, &pc1);

    // Project
    let projected: Vec<(f64, f64)> = centered.iter()
        .map(|v| (dot_f64(v, &pc1), dot_f64(v, &pc2)))
        .collect();

    let x_min = projected.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = projected.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_min = projected.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_max = projected.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let xpad = (x_max - x_min).max(0.1) * 0.15;
    let ypad = (y_max - y_min).max(0.1) * 0.15;

    let root = BitMapBackend::new(output_path, (1100, 700)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| Word2VecError::Plot(e.to_string()))?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Word Vectors — PCA Projection", ("sans-serif", 24).into_font())
        .margin(40)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            (x_min - xpad)..(x_max + xpad),
            (y_min - ypad)..(y_max + ypad),
        )
        .map_err(|e| Word2VecError::Plot(e.to_string()))?;

    chart.configure_mesh()
        .x_desc("PC1")
        .y_desc("PC2")
        .draw()
        .map_err(|e| Word2VecError::Plot(e.to_string()))?;

    for (i, (&word, &(x, y))) in words.iter().zip(projected.iter()).enumerate() {
        let color = Palette99::pick(i % 99);

        chart.draw_series(std::iter::once(Circle::new((x, y), 5, color.filled())))
            .map_err(|e| Word2VecError::Plot(e.to_string()))?;

        chart.draw_series(std::iter::once(Text::new(
            word.to_string(),
            (x + xpad * 0.05, y),
            ("sans-serif", 12).into_font(),
        )))
        .map_err(|e| Word2VecError::Plot(e.to_string()))?;
    }

    root.present().map_err(|e| Word2VecError::Plot(e.to_string()))?;
    Ok(())
}

fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm_f64(v: &[f64]) -> f64 {
    dot_f64(v, v).sqrt()
}

fn normalize_f64(v: &mut [f64]) {
    let n = norm_f64(v);
    if n > 1e-10 {
        for x in v.iter_mut() { *x /= n; }
    }
}

/// Power iteration to find the dominant eigenvector of X^T X.
fn power_iteration(data: &[Vec<f64>], dim: usize, iters: usize, seed: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..dim).map(|d| data[seed % data.len()][d]).collect();
    normalize_f64(&mut v);

    for _ in 0..iters {
        let xv: Vec<f64> = data.iter().map(|row| dot_f64(row, &v)).collect();
        let mut w = vec![0.0f64; dim];
        for (row, &proj) in data.iter().zip(xv.iter()) {
            for (wd, &rd) in w.iter_mut().zip(row.iter()) {
                *wd += proj * rd;
            }
        }
        normalize_f64(&mut w);
        v = w;
    }
    v
}

/// Find second PC by deflating the first.
fn power_iteration_deflated(data: &[Vec<f64>], dim: usize, iters: usize, pc1: &[f64]) -> Vec<f64> {
    let deflated: Vec<Vec<f64>> = data.iter().map(|row| {
        let proj = dot_f64(row, pc1);
        (0..dim).map(|d| row[d] - proj * pc1[d]).collect()
    }).collect();

    power_iteration(&deflated, dim, iters, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plot_loss_curve_empty_errors() {
        let result = plot_loss_curve(&[], "/tmp/test_empty.png");
        assert!(result.is_err());
    }

    #[test]
    fn plot_loss_curve_creates_file() {
        let history = vec![
            EpochStats { epoch: 1, avg_loss: 2.5, learning_rate: 0.025, pairs_processed: 100, elapsed_secs: 0.5 },
            EpochStats { epoch: 2, avg_loss: 1.8, learning_rate: 0.020, pairs_processed: 100, elapsed_secs: 0.5 },
            EpochStats { epoch: 3, avg_loss: 1.2, learning_rate: 0.015, pairs_processed: 100, elapsed_secs: 0.5 },
        ];
        let path = "/tmp/word2vec_test_loss.png";
        plot_loss_curve(&history, path).unwrap();
        assert!(std::path::Path::new(path).exists());
    }

    #[test]
    fn pca_plot_creates_file() {
        use crate::{Config, Trainer};
        let corpus: Vec<String> = (0..50)
            .map(|i| format!("w{} w{} w{} w{}", i % 8, (i+1) % 8, (i+2) % 8, (i+3) % 8))
            .collect();
        let mut trainer = Trainer::new(Config { epochs: 2, embedding_dim: 20, ..Config::default() });
        let emb = trainer.train(&corpus).unwrap();
        let path = "/tmp/word2vec_test_pca.png";
        plot_word_vectors_pca(&emb, 8, path).unwrap();
        assert!(std::path::Path::new(path).exists());
    }
}
