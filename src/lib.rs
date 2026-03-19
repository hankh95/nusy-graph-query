//! nusy-graph-query — Graph-native semantic search for Arrow RecordBatches.
//!
//! Provides embedding, graph traversal, hybrid ranking, and embedding caching
//! for Arrow-backed knowledge graphs. Useful for any system that needs to
//! search and traverse structured knowledge stored in Arrow RecordBatches.
//!
//! # Modules
//!
//! - [`embedding`] — `EmbeddingProvider` trait, hash provider, cosine similarity
//! - [`traversal`] — Generic BFS/DFS over Arrow edge RecordBatches
//! - [`hybrid_rank()`] — Combine structural + semantic scores
//! - [`cache`] — Content-hash embedding cache with Parquet persistence
//! - [`ollama`] — Ollama embedding provider (feature: `ollama`)
//! - [`subprocess`] — Python sentence-transformers provider (feature: `subprocess`)

pub mod cache;
pub mod embedding;
pub mod hybrid_rank;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "subprocess")]
pub mod subprocess;
pub mod traversal;

// Re-export key types at crate root for convenience.
pub use cache::EmbeddingCache;
pub use embedding::{
    EmbeddedItem, EmbeddingError, EmbeddingProvider, HashEmbeddingProvider, SearchResult,
    cosine_similarity, hash_to_vector, semantic_search,
};
pub use hybrid_rank::{HybridConfig, RankCandidate, RankedResult, hybrid_rank};
#[cfg(feature = "ollama")]
pub use ollama::OllamaEmbeddingProvider;
#[cfg(feature = "subprocess")]
pub use subprocess::SubprocessEmbeddingProvider;
pub use traversal::{
    Direction, EdgeSchema, TraversalNode, bfs, bfs_with_adjacency, build_adjacency,
    build_adjacency_from_list,
};
