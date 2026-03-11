# Project: Mini Vector Index v1

## Overview
This project is a C++ implementation of a **Mini Vector Index**, designed to serve as a **ground-truth, in-memory baseline** for vector database experiments. 

**Goal:** Provide exact nearest neighbor search (100% recall) to validate future approximate nearest neighbor (ANN) algorithms.

## Architecture & Design (Planned)
Based on `plan.md`, the v1 implementation focuses on simplicity and correctness:

*   **Type:** Flat (brute-force) index.
*   **Storage:** In-memory, contiguous float buffer (`std::vector<float>` or similar).
*   **Search Algorithm:** Linear scan over all vectors O(N * dim).
*   **Result Management:** Max-heap of size K to maintain top results.
*   **Supported Metrics:**
    *   Squared L2 (Euclidean) distance.
    *   Inner Product (raw dot product). For cosine semantics, callers pre-normalize vectors before inserting.

## Scope & Constraints
*   **Included:** Exact results, fixed dimensionality, single thread (initially), simple insert/search API.
*   **Excluded:** Disk persistence, approximate search (HNSW, IVF), deletions, metadata filtering, network layers.

## Roadmap / Implementation Steps
1.  **Setup Build System:** Initialize `CMakeLists.txt`.
2.  **Core Data Structure:** Implement the class holding the flat vector buffer.
3.  **Distance Functions:** Implement optimized L2 and Dot Product functions.
4.  **Search Logic:** Implement the linear scan with a priority queue (max-heap).
5.  **Testing:** Add unit tests for correctness (identical vectors, orthogonal vectors).
6.  **Benchmarking:** Create a harness to measure latency and throughput.

## Development Conventions
*   **Language:** C++20.
*   **Build System:** CMake. Two build dirs: `build/` (Debug) and `build_release/` (Release).
*   **Dependencies:** Doctest (unit tests), Google Benchmark (benchmarks) — fetched automatically via `FetchContent`.
*   **Performance:** Focus on cache-friendly memory layouts and minimizing allocations in the hot path.

## Build & Run

### Tests
```bash
cd build && make unit_tests && ./unit_tests
```

### Eval (Recall@K + latency)
```bash
cd build && make eval_runner && ./eval_runner
```

### Benchmarks
```bash
cd build_release && make benchmarks && ./benchmarks
```
