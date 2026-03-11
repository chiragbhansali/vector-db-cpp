# Engineering Notes: Vector Index & SIMD Optimization

### Vector Database Fundamentals

- **Core Purpose**: Storing and retrieving high-dimensional vectors based on similarity rather than exact matches.
- **Metric Reduction**: Normalizing vectors during insertion allows Cosine Similarity to be calculated as a simple Dot Product, removing expensive square root and division operations from the search path.
- **Top-K Retrieval**: Utilizing a Max-Heap to track the K smallest distances. This allows for efficient "pruning" by comparing new candidates against the current "worst" candidate in the top-K set.

### SIMD (Single Instruction, Multiple Data)

- **Parallelism**: Leveraging hardware-specific registers (NEON for ARM64) to process multiple data points (4 floats) in a single CPU cycle.
- **Fused Multiply-Add (FMA)**: Using specialized instructions to perform multiplication and addition in one step, reducing rounding errors and increasing throughput.
- **Horizontal Summation**: The process of collapsing a SIMD register back into a scalar value after bulk processing.

### Data Layout & Memory Engineering

- **Contiguous Buffers**: Storing all vectors in one flat array to maximize CPU cache hits and prevent "pointer chasing."
- **Dimension Padding**: Rounding vector dimensions up to the nearest multiple of the SIMD width (e.g., 4).
- **Benefits of Padding**:
    - Ensures every vector starts at an optimal memory boundary.
    - Eliminates complex "tail loops" in SIMD kernels.
    - Simplifies indexing math (start index = ID \* Aligned Dimension).
- **Bit-Magic**: Using bitwise AND with a negated mask to efficiently "snap" numbers to power-of-two boundaries.

### Performance Evaluation

- **Benchmark Baselines**: Comparing SIMD vs. Scalar results in the same environment to calculate real-world speedup factors.
- **Dispatching**: Implementing toggleable execution paths (Scalar vs. SIMD) to verify correctness and measure performance gains without changing hardware.

TODO:
Sharding, Replication, Distributed query routing, Raft
tree-based indexing
diskann
Segment architecture (LSM-tree inspired)

Memory-mapped files

Segment format design

batch queries and caching
