# Vector Database and Search w/ C++

A mini vector index built from scratch in C++ 20. Supports exact and approximate nearest-neighbor search over float vectors.

## Indexes

- **FlatIndex** — brute-force linear scan, exact results, O(N·d). Supports L2 and inner product (cosine via pre-normalization). SIMD-accelerated
  distance computation.
- **IVFFlatIndex** — inverted file index. Clusters vectors with k-means at build time; searches only the `nprobe` nearest clusters at query time.
- **IVFPQIndex** — IVF with product quantization. Compresses vectors into short byte codes (8 subspaces × 256 centroids), then scores candidates
  via asymmetric distance computation (ADC). Lower memory, faster scan.

Note: All core algorithms (k-means, PQ, IVF, distance functions, etc.) written by hand.

## Measurement

- **Eval** (`eval_runner`) — Recall@K and QPS table across nprobe values, compared against FlatIndex ground truth.
- **Benchmarks** (`benchmarks`) — Google Benchmark micros for search latency and build time.

## Build & Run

```bash
# Tests
cd build && make unit_tests && ./unit_tests

# Eval (Recall@K + QPS)
cd build && make eval_runner && ./eval_runner

# Benchmarks
cd build_release && make benchmarks && ./benchmarks
```
