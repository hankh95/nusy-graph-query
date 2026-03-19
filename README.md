# nusy-graph-query

Graph-native semantic search for Arrow RecordBatches.

Provides embeddings, BFS/DFS traversal, hybrid ranking, and query caching over
Arrow-backed knowledge graphs. Designed for AI systems that need to search and
traverse structured knowledge at scale.

## Features

- **Embedding providers** — Hash-based (zero-dep), Ollama, subprocess
- **Graph traversal** — BFS with adjacency lists built from Arrow ListArray columns
- **Hybrid ranking** — Combine text match, semantic similarity, and graph distance
- **Query caching** — Parquet-backed embedding cache with TTL eviction

## Quick Start

```rust
use nusy_graph_query::embedding::HashEmbeddingProvider;
use nusy_graph_query::EmbeddingProvider;

let provider = HashEmbeddingProvider;
let embedding = provider.embed("hello world").unwrap();
assert_eq!(embedding.len(), 64); // 64-dim hash embedding
```

## Feature Flags

| Flag | Dependencies | Description |
|------|-------------|-------------|
| `ollama` | serde, serde_json, ureq | Ollama API embedding provider |
| `subprocess` | serde_json | External process embedding provider |

## License

MIT
