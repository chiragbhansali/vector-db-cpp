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

# Product Quantization

## Step 1: Training

We split vectors in training data of dim D into M subspaces, each of which has vectors of dimension D/M. Each subspace will have K=256 centroids based on the given sub-vectors.

Training produces codebooks

## Step 2: Encoding

We map each sub-vector from the DB to its nearest centroid ID and store it in a flat vector (better for contiguous memory access).

Encoding produces centroid codes for each sub-vector

## Step 3: Search / Querying

During search time, you get the distance of a query's sub-vector and that centroid. This is done by calculating distance between the value in the distance table and the query sub-vector.

Example:
Original DB vector (dim=8, M=4, so 2 floats per subspace):
[ 0.3, 0.7, | 0.1, 0.9, | 0.5, 0.5, | 0.2, 0.8 ]
subspace_0 subspace_1 subspace_2 subspace_3

After encoding, you find the nearest centroid in each subspace and store just its ID:
[ 2, 1, 2, 0 ]

Distance table -
M=4, K=4:

             centroid_0  centroid_1  centroid_2  centroid_3

sub_vector_0 [ 0.12 0.45 0.89 0.23 ]
sub_vector_1 [ 0.67 0.11 0.34 0.78 ]
sub_vector_2 [ 0.55 0.92 0.08 0.41 ]
sub_vector_3 [ 0.33 0.19 0.76 0.62 ]

Each cell is the L2 distance between the query's sub-vector and that centroid.

Now a DB vector is stored as codes [2, 1, 2, 0] (one code per subspace). To get its approximate distance to the query, you just look up:

dist ≈ table[0][2] + table[1][1] + table[2][2] + table[3][0]
= 0.89 + 0.11 + 0.08 + 0.33
= 1.41

During the search process, we compare all DB vectors' against the query vector using the above process.
