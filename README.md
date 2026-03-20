# nusy-graph-query

[![Crates.io](https://img.shields.io/crates/v/nusy-graph-query.svg)](https://crates.io/crates/nusy-graph-query)
[![Documentation](https://docs.rs/nusy-graph-query/badge.svg)](https://docs.rs/nusy-graph-query)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

Graph-native semantic search for Arrow RecordBatches — embeddings, traversal,
hybrid ranking, and caching.

Add to your `Cargo.toml`:

```toml
[dependencies]
nusy-graph-query = "0.14"

# With Ollama embeddings
nusy-graph-query = { version = "0.14", features = ["ollama"] }

# With sentence-transformers (Python subprocess)
nusy-graph-query = { version = "0.14", features = ["subprocess"] }
```

## Embedding Providers

Three providers, all implementing the `EmbeddingProvider` trait:

| Provider | Feature | Backend | Default dim |
|----------|---------|---------|-------------|
| `HashEmbeddingProvider` | (always on) | Deterministic SHA-256 | 64 |
| `OllamaEmbeddingProvider` | `ollama` | Ollama REST API | 768 |
| `SubprocessEmbeddingProvider` | `subprocess` | Python sentence-transformers | 384 |

```rust
use nusy_graph_query::EmbeddingProvider;

// Hash provider — zero dependencies, deterministic, for testing
use nusy_graph_query::embedding::HashEmbeddingProvider;
let provider = HashEmbeddingProvider;
let vec = provider.embed("hello world")?;
```

### Ollama provider

```rust
use nusy_graph_query::ollama::OllamaEmbeddingProvider;

let provider = OllamaEmbeddingProvider::new()
    .with_url("http://localhost:11434")
    .with_model("nomic-embed-text")
    .with_dim(768);

let embeddings = provider.embed_batch(&[
    "knowledge graph".into(),
    "semantic search".into(),
])?;
```

### Subprocess provider

```rust
use nusy_graph_query::subprocess::SubprocessEmbeddingProvider;

let provider = SubprocessEmbeddingProvider::new()
    .with_python("python3")
    .with_model("all-MiniLM-L6-v2")
    .with_dim(384);
```

## Graph Traversal (BFS)

BFS over Arrow edge tables with configurable schema and predicate filtering:

```rust
use nusy_graph_query::traversal::{bfs, EdgeSchema, Direction};

// Configure which columns hold source/target/predicate
let schema = EdgeSchema::new("source_id", "target_id", Some("predicate"));

// BFS from a start node, max depth 3
let nodes = bfs("Alice", &edge_batches, &schema, Direction::Forward, None, 3)?;
for node in &nodes {
    println!("  {} (depth {})", node.id, node.depth);
}

// With predicate filter — only traverse "knows" edges
let nodes = bfs("Alice", &edge_batches, &schema, Direction::Forward, Some("knows"), 5)?;
```

### Pre-computed adjacency lists

```rust
use nusy_graph_query::traversal::{build_adjacency, bfs_with_adjacency};

let adj = build_adjacency(&edge_batches, &schema, Direction::Forward, None);
let nodes = bfs_with_adjacency("Alice", &adj, 3);
```

## Hybrid Ranking

Combine structural scores (text match, graph distance) with semantic similarity:

```rust
use nusy_graph_query::hybrid_rank::{hybrid_rank, HybridConfig, RankCandidate};

let candidates = vec![
    RankCandidate { id: "doc-1".into(), structural_score: 0.8 },
    RankCandidate { id: "doc-2".into(), structural_score: 0.3 },
    RankCandidate { id: "doc-3".into(), structural_score: 0.6 },
];

let config = HybridConfig {
    structural_weight: 0.6,  // 60% text/graph score
    semantic_weight: 0.4,    // 40% embedding similarity
};

let results = hybrid_rank(
    &candidates, &embeddings, "search query", &provider, &config, 10,
)?;

for r in &results {
    println!("{}: combined={:.3} (structural={:.3}, semantic={:.3})",
        r.id, r.combined_score, r.structural_score, r.semantic_score);
}
```

## Query Caching

Content-hash embedding cache — avoids recomputing embeddings when content hasn't changed.
Persists to Parquet:

```rust
use nusy_graph_query::cache::EmbeddingCache;

let mut cache = EmbeddingCache::new();

// embed_cached only computes embeddings for cache misses
let items = vec![
    ("doc-1", "hash-abc", "some text"),
    ("doc-2", "hash-def", "other text"),
];
let vectors = cache.embed_cached(&items, &provider)?;

println!("Hits: {}, Misses: {}", cache.hits(), cache.misses());

// Persist to Parquet
cache.save("embeddings.parquet")?;

// Restore
let mut cache2 = EmbeddingCache::new();
cache2.load("embeddings.parquet")?;
```

## Semantic Search

Direct semantic search over a set of embeddings:

```rust
use nusy_graph_query::embedding::{semantic_search, cosine_similarity};

let results = semantic_search(&embeddings, "find similar documents", &provider, 5)?;
for r in &results {
    println!("{}: {:.3}", r.id, r.score);
}

// Low-level similarity
let sim = cosine_similarity(&vec_a, &vec_b);
```

## Feature Flags

| Flag | Dependencies | Description |
|------|-------------|-------------|
| `ollama` | ureq, serde, serde_json | Ollama REST API embedding provider |
| `subprocess` | serde_json | Python sentence-transformers subprocess provider |

Default: no features (hash provider only — zero external dependencies).

## Requirements

- Rust 1.85+ (edition 2024)
- For `ollama`: Ollama server running locally (0.4.0+ recommended)
- For `subprocess`: Python 3 with `sentence-transformers` installed

## License

MIT — Copyright (c) Hank Head / Congruent Systems PBC
