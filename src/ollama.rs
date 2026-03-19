//! Ollama embedding provider — calls the Ollama `/api/embed` endpoint.
//!
//! Feature-gated behind `ollama`. Requires a running Ollama instance.
//!
//! ```toml
//! [dependencies]
//! nusy-graph-query = { version = "0.14", features = ["ollama"] }
//! ```

use crate::embedding::{EmbeddingError, EmbeddingProvider, Result};

/// Embedding provider that calls a local Ollama instance.
///
/// Uses the `/api/embed` endpoint (batch-native since Ollama 0.4.0).
/// Default model: `nomic-embed-text` (768-dim, fast, good quality).
pub struct OllamaEmbeddingProvider {
    /// Ollama API base URL (default: `http://localhost:11434`).
    base_url: String,
    /// Model name (default: `nomic-embed-text`).
    model: String,
    /// Embedding dimension (determined on first call).
    dim: usize,
}

impl OllamaEmbeddingProvider {
    /// Create a new Ollama provider with default settings.
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "nomic-embed-text".to_string(),
            dim: 768,
        }
    }

    /// Set the Ollama API base URL.
    pub fn with_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }

    /// Set the embedding model.
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    /// Set the expected embedding dimension.
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }
}

impl Default for OllamaEmbeddingProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingProvider for OllamaEmbeddingProvider {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/api/embed", self.base_url);

        let response: serde_json::Value = ureq::post(&url)
            .send_json(serde_json::json!({
                "model": self.model,
                "input": texts,
            }))
            .map_err(|e| EmbeddingError::Provider(format!("Ollama request failed: {e}")))?
            .into_json()
            .map_err(|e| EmbeddingError::Provider(format!("Ollama response parse failed: {e}")))?;

        // Parse embeddings array from response
        let embeddings: &Vec<serde_json::Value> = response
            .get("embeddings")
            .and_then(|v: &serde_json::Value| v.as_array())
            .ok_or_else(|| {
                EmbeddingError::Provider(format!(
                    "Ollama response missing 'embeddings' field: {}",
                    serde_json::to_string_pretty(&response).unwrap_or_default()
                ))
            })?;

        let mut result = Vec::with_capacity(texts.len());
        for (i, emb) in embeddings.iter().enumerate() {
            let vec: Vec<f32> = emb
                .as_array()
                .ok_or_else(|| {
                    EmbeddingError::Provider(format!("Embedding {i} is not an array"))
                })?
                .iter()
                .map(|v: &serde_json::Value| {
                    v.as_f64()
                        .ok_or_else(|| {
                            EmbeddingError::Provider(format!("Non-numeric value in embedding {i}"))
                        })
                        .map(|f| f as f32)
                })
                .collect::<Result<Vec<f32>>>()?;

            if vec.len() != self.dim {
                // Auto-detect dimension from first response
                if i == 0 && result.is_empty() {
                    // Accept whatever dimension Ollama returns
                    result.push(vec);
                    continue;
                }
                return Err(EmbeddingError::DimensionMismatch {
                    expected: self.dim,
                    actual: vec.len(),
                });
            }
            result.push(vec);
        }

        Ok(result)
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::Provider("empty result from Ollama".to_string()))
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_provider_default() {
        let provider = OllamaEmbeddingProvider::new();
        assert_eq!(provider.dim(), 768);
        assert_eq!(provider.base_url, "http://localhost:11434");
        assert_eq!(provider.model, "nomic-embed-text");
    }

    #[test]
    fn test_ollama_provider_builder() {
        let provider = OllamaEmbeddingProvider::new()
            .with_url("http://dgx:11434")
            .with_model("all-minilm")
            .with_dim(384);

        assert_eq!(provider.base_url, "http://dgx:11434");
        assert_eq!(provider.model, "all-minilm");
        assert_eq!(provider.dim(), 384);
    }

    #[test]
    fn test_ollama_provider_empty_batch() {
        let provider = OllamaEmbeddingProvider::new();
        let result = provider.embed_batch(&[]).unwrap();
        assert!(result.is_empty());
    }

    // Integration test — only runs if Ollama is available
    // #[test]
    // fn test_ollama_real_embedding() {
    //     let provider = OllamaEmbeddingProvider::new();
    //     let result = provider.embed("hello world");
    //     if let Ok(vec) = result {
    //         assert!(!vec.is_empty());
    //         assert!(vec.len() >= 384);
    //     }
    //     // OK to fail if Ollama is not running
    // }
}
