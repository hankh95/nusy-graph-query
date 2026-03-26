//! fastembed-rs local embedding provider — ONNX-based, no network required.
//!
//! EX-3384: Replaces 50ms/chunk Ollama HTTP with ~2ms/chunk local ONNX inference.
//! Feature-gated behind `fastembed`.
//!
//! ```toml
//! [dependencies]
//! nusy-graph-query = { version = "0.14", features = ["fastembed"] }
//! ```

use crate::embedding::{EmbeddingError, EmbeddingProvider, Result};

/// Default fastembed model — BAAI/bge-small-en-v1.5 (384-dim, fast, good quality).
const DEFAULT_MODEL: &str = "BAAI/bge-small-en-v1.5";
const DEFAULT_DIM: usize = 384;

/// Local embedding provider using fastembed-rs (ONNX Runtime).
///
/// No network dependencies — downloads the model on first use, then runs
/// locally with ONNX Runtime. Batch-native for maximum throughput.
///
/// # Example
/// ```rust,ignore
/// use nusy_graph_query::fastembed_provider::FastembedProvider;
/// use nusy_graph_query::embedding::EmbeddingProvider;
///
/// let provider = FastembedProvider::new().unwrap();
/// let embeddings = provider.embed_batch(&["Hello".into(), "World".into()]).unwrap();
/// assert_eq!(embeddings.len(), 2);
/// assert_eq!(embeddings[0].len(), 384);
/// ```
pub struct FastembedProvider {
    model: std::sync::Mutex<fastembed::TextEmbedding>,
    dim: usize,
}

impl FastembedProvider {
    /// Create a provider with the default model (BAAI/bge-small-en-v1.5, 384-dim).
    pub fn new() -> std::result::Result<Self, EmbeddingError> {
        Self::with_model(DEFAULT_MODEL, DEFAULT_DIM)
    }

    /// Create a provider with a specific model.
    ///
    /// Supported models (see fastembed docs for full list):
    /// - `BAAI/bge-small-en-v1.5` (384-dim, fast, default)
    /// - `BAAI/bge-base-en-v1.5` (768-dim, balanced)
    /// - `BAAI/bge-large-en-v1.5` (1024-dim, highest quality)
    /// - `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
    pub fn with_model(model_name: &str, dim: usize) -> std::result::Result<Self, EmbeddingError> {
        let model_info = fastembed::TextEmbedding::list_supported_models()
            .into_iter()
            .find(|m| m.model_code == model_name)
            .ok_or_else(|| {
                EmbeddingError::Provider(format!(
                    "Model '{}' not supported by fastembed. Use TextEmbedding::list_supported_models() to see available models.",
                    model_name
                ))
            })?;

        let options = fastembed::InitOptions::new(model_info.model)
            .with_show_download_progress(true);

        let model = fastembed::TextEmbedding::try_new(options).map_err(|e| {
            EmbeddingError::Provider(format!("Failed to initialize fastembed model: {e}"))
        })?;

        Ok(Self { model: std::sync::Mutex::new(model), dim })
    }

    /// Create a provider with a custom cache directory for model files.
    pub fn with_cache_dir(
        model_name: &str,
        dim: usize,
        cache_dir: impl Into<std::path::PathBuf>,
    ) -> std::result::Result<Self, EmbeddingError> {
        let model_info = fastembed::TextEmbedding::list_supported_models()
            .into_iter()
            .find(|m| m.model_code == model_name)
            .ok_or_else(|| {
                EmbeddingError::Provider(format!("Model '{}' not supported", model_name))
            })?;

        let options = fastembed::InitOptions::new(model_info.model)
            .with_show_download_progress(true)
            .with_cache_dir(cache_dir.into());

        let model = fastembed::TextEmbedding::try_new(options).map_err(|e| {
            EmbeddingError::Provider(format!("Failed to initialize fastembed model: {e}"))
        })?;

        Ok(Self { model: std::sync::Mutex::new(model), dim })
    }
}

impl EmbeddingProvider for FastembedProvider {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut model = self.model.lock().map_err(|e| {
            EmbeddingError::Provider(format!("fastembed lock poisoned: {e}"))
        })?;
        let results = model
            .embed(texts.to_vec(), None)
            .map_err(|e| EmbeddingError::Provider(format!("fastembed embed failed: {e}")))?;

        // Validate dimensions.
        for (i, vec) in results.iter().enumerate() {
            if vec.len() != self.dim {
                return Err(EmbeddingError::DimensionMismatch {
                    expected: self.dim,
                    actual: vec.len(),
                });
            }
            let _ = i; // suppress unused warning
        }

        Ok(results)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model download on first run (~30MB)
    fn test_fastembed_single() {
        let provider = FastembedProvider::new().expect("init should succeed");
        let vec = provider.embed("Hello world").expect("embed should succeed");
        assert_eq!(vec.len(), DEFAULT_DIM);

        // Should be a unit-ish vector (fastembed normalizes by default).
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.1,
            "expected ~unit vector, got norm={norm}"
        );
    }

    #[test]
    #[ignore] // Requires model download
    fn test_fastembed_batch() {
        let provider = FastembedProvider::new().expect("init should succeed");
        let texts = vec![
            "The cat sat on the mat".to_string(),
            "Dogs are loyal animals".to_string(),
            "Mathematics is beautiful".to_string(),
        ];
        let results = provider
            .embed_batch(&texts)
            .expect("batch embed should succeed");
        assert_eq!(results.len(), 3);
        for vec in &results {
            assert_eq!(vec.len(), DEFAULT_DIM);
        }
    }

    #[test]
    #[ignore] // Requires model download
    fn test_fastembed_similarity() {
        use crate::embedding::cosine_similarity;

        let provider = FastembedProvider::new().expect("init should succeed");
        let cat = provider.embed("cat").expect("embed");
        let dog = provider.embed("dog").expect("embed");
        let math = provider.embed("integral calculus").expect("embed");

        let cat_dog = cosine_similarity(&cat, &dog);
        let cat_math = cosine_similarity(&cat, &math);

        // cat-dog should be more similar than cat-math.
        assert!(
            cat_dog > cat_math,
            "cat-dog ({cat_dog}) should be more similar than cat-math ({cat_math})"
        );
    }

    #[test]
    #[ignore] // Requires model download
    fn test_fastembed_empty_batch() {
        let provider = FastembedProvider::new().expect("init should succeed");
        let results = provider.embed_batch(&[]).expect("empty batch should succeed");
        assert!(results.is_empty());
    }
}
