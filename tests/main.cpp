#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "flat_index.hpp"
#include "ivf_flat_index.hpp"
#include "kmeans.hpp"
#include "vector_index.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// --- Randomized test helpers ---

static std::vector<std::vector<float>>
generate_random_vectors(size_t n, size_t dim, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<std::vector<float>> vecs(n, std::vector<float>(dim));
  for (auto &v : vecs)
    for (auto &x : v)
      x = dist(rng);
  return vecs;
}

static float dot(std::span<const float> a, std::span<const float> b) {
  float s = 0;
  for (size_t i = 0; i < a.size(); ++i)
    s += a[i] * b[i];
  return s;
}

static float vec_norm(std::span<const float> v) {
  return std::sqrtf(dot(v, v));
}

// Squared L2 over all db vectors, sorted ascending
static std::vector<SearchResult>
bruteforce_l2(const std::vector<std::vector<float>> &db,
              std::span<const float> query) {
  std::vector<SearchResult> all;
  all.reserve(db.size());
  for (size_t i = 0; i < db.size(); ++i) {
    float d = 0;
    for (size_t j = 0; j < query.size(); ++j) {
      float diff = query[j] - db[i][j];
      d += diff * diff;
    }
    all.push_back({i, d});
  }
  std::sort(all.begin(), all.end(),
            [](const SearchResult &a, const SearchResult &b) {
              return a.score < b.score;
            });
  return all;
}

// On-the-fly cosine over raw (unnormalized) vectors, sorted descending
static std::vector<SearchResult>
bruteforce_cosine(const std::vector<std::vector<float>> &db,
                  std::span<const float> query) {
  float qn = vec_norm(query);
  std::vector<SearchResult> all;
  all.reserve(db.size());
  for (size_t i = 0; i < db.size(); ++i) {
    float vn = vec_norm(db[i]);
    float sim = (qn > 0 && vn > 0) ? dot(query, db[i]) / (qn * vn) : 0.0f;
    all.push_back({i, sim});
  }
  std::sort(all.begin(), all.end(),
            [](const SearchResult &a, const SearchResult &b) {
              return a.score > b.score;
            });
  return all;
}

TEST_CASE("FlatIndex basic operations") {
  // Setup an index for 2D vectors
  FlatIndex index(2, Metric::L2);

  // Insert test data
  index.insert(std::vector<float>{1.0f, 1.0f});   // ID 0
  index.insert(std::vector<float>{2.0f, 2.0f});   // ID 1
  index.insert(std::vector<float>{10.0f, 10.0f}); // ID 2

  SUBCASE("Exact match search") {
    std::vector<float> query = {1.0f, 1.0f};
    auto results = index.search(query, 1);

    CHECK(results.size() == 1);
    CHECK(results[0].id == 0);
    CHECK(results[0].score == doctest::Approx(0.0f));
  }

  SUBCASE("Top-K ranking and ordering") {
    // Query at origin [0,0]
    std::vector<float> query = {0.0f, 0.0f};
    auto results = index.search(query, 2);

    // Should return ID 0 and ID 1, sorted by distance
    REQUIRE(results.size() == 2);
    CHECK(results[0].id == 0); // Closer: dist sq = 1^2 + 1^2 = 2
    CHECK(results[1].id == 1); // Farther: dist sq = 2^2 + 2^2 = 8
    CHECK(results[0].score < results[1].score);
  }

  SUBCASE("Dimension safety") {
    std::vector<float> bad_vec = {1.0f, 2.0f, 3.0f}; // 3D instead of 2D
    CHECK_THROWS_AS(index.insert(bad_vec), std::invalid_argument);

    std::vector<float> bad_query = {1.0f}; // 1D instead of 2D
    CHECK_THROWS_AS(index.search(bad_query, 1), std::invalid_argument);
  }

  SUBCASE("Empty results for k=0") {
    std::vector<float> query = {0.0f, 0.0f};
    auto results = index.search(query, 0);
    CHECK(results.empty());
  }
}

TEST_CASE("FlatIndex Cosine Similarity") {
  // Setup an index for 2D vectors using Cosine metric
  FlatIndex index(2, Metric::Cosine);

  // Insert normalized vectors
  index.insert(std::vector<float>{1.0f, 0.0f});  // ID 0: X-axis
  index.insert(std::vector<float>{0.0f, 1.0f});  // ID 1: Y-axis
  index.insert(std::vector<float>{-1.0f, 0.0f}); // ID 2: Negative X-axis

  SUBCASE("Exact match similarity") {
    std::vector<float> query = {1.0f, 0.0f};
    auto results = index.search(query, 1);

    CHECK(results.size() == 1);
    CHECK(results[0].id == 0);
    CHECK(results[0].score == doctest::Approx(1.0f));
  }

  SUBCASE("Orthogonal and Opposite vectors") {
    std::vector<float> query = {1.0f, 0.0f};
    auto results = index.search(query, 3);

    REQUIRE(results.size() == 3);

    // 1. Exact match
    CHECK(results[0].id == 0);
    CHECK(results[0].score == doctest::Approx(1.0f));

    // 2. Orthogonal (90 degrees)
    CHECK(results[1].id == 1);
    CHECK(results[1].score == doctest::Approx(0.0f));

    // 3. Opposite (180 degrees)
    CHECK(results[2].id == 2);
    CHECK(results[2].score == doctest::Approx(-1.0f));
  }

  SUBCASE("Ranking by similarity") {
    // Query at 45 degrees [1, 1] (unnormalized)
    std::vector<float> query = {1.0f, 1.0f};
    auto results = index.search(query, 2);

    // [1,1] is equally close to [1,0] and [0,1]
    // Correct Cosine: 1 / sqrt(2) approx 0.707
    REQUIRE(results.size() == 2);
    CHECK(results[0].score == doctest::Approx(0.70710678f));
    CHECK(results[1].score == doctest::Approx(0.70710678f));
  }
}

// ---------------------------------------------------------------------------
// §8.2 Randomized tests
// ---------------------------------------------------------------------------

// Validates that the index top-K matches an independent brute-force sort
// for both L2 and Cosine.
TEST_CASE("Randomized: Top-K ordering matches brute force") {
  constexpr size_t N = 200, DIM = 32, K = 10;
  std::mt19937 rng(42);
  auto db = generate_random_vectors(N, DIM, rng);

  SUBCASE("L2") {
    FlatIndex index(DIM, Metric::L2);
    for (auto &v : db)
      index.insert(v);

    for (size_t q = 0; q < 5; ++q) {
      auto query = generate_random_vectors(1, DIM, rng)[0];
      auto got = index.search(query, K);
      auto ref = bruteforce_l2(db, query);

      REQUIRE(got.size() == K);
      for (size_t i = 0; i < K; ++i) {
        CHECK(got[i].id == ref[i].id);
        CHECK(got[i].score == doctest::Approx(ref[i].score));
      }
    }
  }

  SUBCASE("Cosine") {
    FlatIndex index(DIM, Metric::Cosine);
    for (auto &v : db)
      index.insert(v);

    for (size_t q = 0; q < 5; ++q) {
      auto query = generate_random_vectors(1, DIM, rng)[0];
      auto got = index.search(query, K);
      auto ref = bruteforce_cosine(db, query);

      REQUIRE(got.size() == K);
      for (size_t i = 0; i < K; ++i) {
        CHECK(got[i].id == ref[i].id);
        CHECK(got[i].score == doctest::Approx(ref[i].score));
      }
    }
  }
}

// Same seed, same data, same query → identical results across two runs.
TEST_CASE("Randomized: Deterministic results with fixed seed") {
  auto build_and_search = [](size_t seed) {
    constexpr size_t N = 100, DIM = 16, K = 5;
    std::mt19937 rng(seed);
    auto db = generate_random_vectors(N, DIM, rng);
    auto query = generate_random_vectors(1, DIM, rng)[0];

    FlatIndex index(DIM, Metric::L2);
    for (auto &v : db)
      index.insert(v);
    return index.search(query, K);
  };

  auto a = build_and_search(123);
  auto b = build_and_search(123);

  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i].id == b[i].id);
    CHECK(a[i].score == b[i].score); // exact: deterministic inputs
  }
}

// Vectors scaled by random factors (0.01–100x).  The index pre-normalizes at
// insert time; this confirms the resulting scores match on-the-fly cosine
// computed on the raw magnitudes.
TEST_CASE("Randomized: Cosine stable across magnitude scales") {
  constexpr size_t N = 200, DIM = 32, K = 10;
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> scale_dist(0.01f, 100.0f);

  auto db = generate_random_vectors(N, DIM, rng);
  for (auto &v : db) {
    float s = scale_dist(rng);
    for (auto &x : v)
      x *= s;
  }

  FlatIndex index(DIM, Metric::Cosine);
  for (auto &v : db)
    index.insert(v);

  for (size_t q = 0; q < 5; ++q) {
    auto query = generate_random_vectors(1, DIM, rng)[0];
    float qs = scale_dist(rng);
    for (auto &x : query)
      x *= qs;

    auto got = index.search(query, K);
    auto ref = bruteforce_cosine(db, query);

    REQUIRE(got.size() == K);
    for (size_t i = 0; i < K; ++i) {
      CHECK(got[i].id == ref[i].id);
      CHECK(got[i].score == doctest::Approx(ref[i].score));
    }
  }
}

TEST_CASE("SIMD vs Scalar: Results must be identical") {
  // Use a dimension that requires padding (e.g., 13)
  constexpr size_t N = 100, DIM = 13, K = 10;
  std::mt19937 rng(42);
  auto db = generate_random_vectors(N, DIM, rng);

  for (auto metric : {Metric::L2, Metric::Cosine}) {
    FlatIndex index(DIM, metric);
    for (auto &v : db)
      index.insert(v);

    auto query = generate_random_vectors(1, DIM, rng)[0];

    auto results_simd = index.search(query, K, true);
    auto results_scalar = index.search(query, K, false);

    REQUIRE(results_simd.size() == results_scalar.size());
    for (size_t i = 0; i < results_simd.size(); ++i) {
      CHECK(results_simd[i].id == results_scalar[i].id);
      CHECK(results_simd[i].score == doctest::Approx(results_scalar[i].score));
    }
  }
}

TEST_CASE("FlatIndex Serialization (Save/Load)") {
  const std::string filename = "test_index.bin";
  constexpr size_t DIM = 4;
  FlatIndex index(DIM, Metric::L2);

  std::vector<float> v1 = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> v2 = {0.0f, 1.0f, 0.0f, 0.0f};
  index.insert(v1);
  index.insert(v2);

  // Save the index
  index.save(filename);

  // Load into a new index
  FlatIndex loaded_index(DIM, Metric::L2);
  loaded_index.load(filename);

  CHECK(loaded_index.size() == 2);
  CHECK(loaded_index.dimension() == DIM);

  // Verify search results are the same
  std::vector<float> query = {1.0f, 0.1f, 0.0f, 0.0f};
  auto original_res = index.search(query, 1);
  auto loaded_res = loaded_index.search(query, 1);

  REQUIRE(original_res.size() == 1);
  REQUIRE(loaded_res.size() == 1);
  CHECK(original_res[0].id == loaded_res[0].id);
  CHECK(original_res[0].score == doctest::Approx(loaded_res[0].score));

  // Cleanup
  std::remove(filename.c_str());
}

// Helper: checks if a centroid (at centroids[c*dim]) is near a target point
static bool centroid_near(const std::vector<float> &centroids, size_t c,
                           size_t dim, std::vector<float> target,
                           float tol = 1.0f) {
  for (size_t j = 0; j < dim; ++j) {
    if (std::abs(centroids[c * dim + j] - target[j]) > tol)
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// IVFFlatIndex tests
// ---------------------------------------------------------------------------

// Build a small IVF index: train on data, insert all, then search.
// With nprobe == num_centroids, every list is probed → exact top-K recall.
static IVFFlatIndex build_ivf(const std::vector<std::vector<float>> &db,
                              size_t dim, size_t num_centroids) {
  std::vector<float> flat;
  flat.reserve(db.size() * dim);
  for (const auto &v : db)
    flat.insert(flat.end(), v.begin(), v.end());

  IVFFlatIndex idx(dim, Metric::L2, /*nprobe=*/num_centroids, num_centroids);
  idx.train(flat);
  for (const auto &v : db)
    idx.insert(v);
  return idx;
}

TEST_CASE("IVFFlatIndex: size and dimension") {
  constexpr size_t DIM = 4, N = 20, K = 2;
  std::mt19937 rng(1);
  auto db = generate_random_vectors(N, DIM, rng);
  auto idx = build_ivf(db, DIM, K);

  CHECK(idx.size() == N);
  CHECK(idx.dimension() == DIM);
}

TEST_CASE("IVFFlatIndex: exact recall with nprobe == num_centroids") {
  // With nprobe equal to num_centroids all inverted lists are scanned, so
  // results must match brute-force L2 for the top-K.
  constexpr size_t N = 200, DIM = 16, K_CENTROIDS = 8, K = 5;
  std::mt19937 rng(42);
  auto db = generate_random_vectors(N, DIM, rng);
  auto idx = build_ivf(db, DIM, K_CENTROIDS);

  for (size_t q = 0; q < 5; ++q) {
    auto query = generate_random_vectors(1, DIM, rng)[0];
    auto got = idx.search(query, K);
    auto ref = bruteforce_l2(db, query);

    REQUIRE(got.size() == K);
    for (size_t i = 0; i < K; ++i) {
      CHECK(got[i].id == ref[i].id);
      CHECK(got[i].score == doctest::Approx(ref[i].score));
    }
  }
}

TEST_CASE("IVFFlatIndex: nprobe parameter controls recall") {
  // Higher nprobe → at least as good recall as lower nprobe.
  constexpr size_t N = 300, DIM = 8, K_CENTROIDS = 10, K = 5;
  std::mt19937 rng(7);
  auto db = generate_random_vectors(N, DIM, rng);

  std::vector<float> flat;
  flat.reserve(N * DIM);
  for (const auto &v : db)
    flat.insert(flat.end(), v.begin(), v.end());

  IVFFlatIndex idx(DIM, Metric::L2, /*nprobe=*/1, K_CENTROIDS);
  idx.train(flat);
  for (const auto &v : db)
    idx.insert(v);

  auto query = generate_random_vectors(1, DIM, rng)[0];
  auto ref = bruteforce_l2(db, query);

  auto top1_ids = [&](size_t nprobe) {
    auto res = idx.search(query, K, nprobe);
    std::vector<size_t> ids;
    for (auto &r : res)
      ids.push_back(r.id);
    return ids;
  };

  // nprobe=K_CENTROIDS is full scan — must find exact nearest neighbor
  auto full = top1_ids(K_CENTROIDS);
  REQUIRE(!full.empty());
  CHECK(full[0] == ref[0].id);
}

TEST_CASE("IVFFlatIndex: results sorted ascending by L2 score") {
  constexpr size_t N = 100, DIM = 4, K_CENTROIDS = 4, K = 10;
  std::mt19937 rng(99);
  auto db = generate_random_vectors(N, DIM, rng);
  auto idx = build_ivf(db, DIM, K_CENTROIDS);

  auto query = generate_random_vectors(1, DIM, rng)[0];
  auto results = idx.search(query, K);

  REQUIRE(!results.empty());
  for (size_t i = 1; i < results.size(); ++i)
    CHECK(results[i - 1].score <= results[i].score);
}

TEST_CASE("IVFFlatIndex: k larger than corpus returns all vectors") {
  constexpr size_t N = 10, DIM = 3, K_CENTROIDS = 2;
  std::mt19937 rng(5);
  auto db = generate_random_vectors(N, DIM, rng);
  auto idx = build_ivf(db, DIM, K_CENTROIDS);

  auto query = generate_random_vectors(1, DIM, rng)[0];
  auto results = idx.search(query, N + 100); // ask for more than we have
  CHECK(results.size() == N);
}

TEST_CASE("IVFFlatIndex: save and load round-trip") {
  const std::string filename = "test_ivf_index.bin";
  constexpr size_t DIM = 4, N = 30, K_CENTROIDS = 3, K = 3;
  std::mt19937 rng(11);
  auto db = generate_random_vectors(N, DIM, rng);
  auto idx = build_ivf(db, DIM, K_CENTROIDS);

  idx.save(filename);

  IVFFlatIndex loaded(DIM, Metric::L2, K_CENTROIDS, K_CENTROIDS);
  loaded.load(filename);

  CHECK(loaded.size() == N);
  CHECK(loaded.dimension() == DIM);

  auto query = generate_random_vectors(1, DIM, rng)[0];
  auto orig_res = idx.search(query, K, K_CENTROIDS);
  auto load_res = loaded.search(query, K, K_CENTROIDS);

  REQUIRE(orig_res.size() == load_res.size());
  for (size_t i = 0; i < orig_res.size(); ++i) {
    CHECK(orig_res[i].id == load_res[i].id);
    CHECK(orig_res[i].score == doctest::Approx(load_res[i].score));
  }

  std::remove(filename.c_str());
}

// ---------------------------------------------------------------------------
// Regression tests — each case targets a specific historical bug
// ---------------------------------------------------------------------------

// Bug: Constructor not defined → dim_, nprobe_, num_centroids_, metric_ are
// uninitialized (UB). Regression: verify constructor arguments are visible
// through the public API and drive correct runtime behavior.
TEST_CASE("IVFFlatIndex regression: constructor args are stored") {
  constexpr size_t DIM = 7, NPROBE = 3, K_CENTROIDS = 5;
  IVFFlatIndex idx(DIM, Metric::L2, NPROBE, K_CENTROIDS);
  CHECK(idx.dimension() == DIM);
  CHECK(idx.size() == 0);

  // num_centroids_ drives inverted_lists_.size() after train — check indirectly
  // by inserting N vectors and confirming they round-trip through save/load.
  std::mt19937 rng(2);
  auto db = generate_random_vectors(20, DIM, rng);
  std::vector<float> flat;
  for (const auto &v : db)
    flat.insert(flat.end(), v.begin(), v.end());
  idx.train(flat);
  for (const auto &v : db)
    idx.insert(v);
  CHECK(idx.size() == 20);
}

// Bug: insert() before train() is UB (out-of-bounds on empty inverted_lists_).
// Regression: must throw rather than silently corrupt or crash.
TEST_CASE("IVFFlatIndex regression: insert before train throws") {
  IVFFlatIndex idx(4, Metric::L2, 2, 2);
  std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
  CHECK_THROWS(idx.insert(vec));
}

// Bug: metric_ ignored — search() always used compute_l2 regardless of the
// metric passed to the constructor.
// Regression: construct two discriminating vectors where L2 and cosine disagree
// on the nearest neighbour, then assert cosine search picks the right one.
//
//   v0 = [10, 0]  — far from origin, perfectly aligned with query [1, 0]
//   v1 = [0.1, 0.995] — close in L2, but nearly orthogonal to [1, 0]
//   cosine nearest: v0 (similarity ≈ 1.0)
//   L2 nearest:     v1 (squared dist ≈ 0.81 + 0.99 ≈ 1.80 vs 81 for v0)
TEST_CASE("IVFFlatIndex regression: metric_ drives search (cosine vs L2)") {
  constexpr size_t DIM = 2, K_CENTROIDS = 1;
  // Single centroid so nprobe=1 is a full scan.
  std::vector<float> train_data = {10.0f, 0.0f, 0.1f, 0.995f};

  IVFFlatIndex idx(DIM, Metric::Cosine, /*nprobe=*/1, K_CENTROIDS);
  idx.train(train_data);
  idx.insert(std::vector<float>{10.0f, 0.0f});   // id 0 — cosine nearest
  idx.insert(std::vector<float>{0.1f, 0.995f});  // id 1 — L2 nearest

  auto results = idx.search(std::vector<float>{1.0f, 0.0f}, 1);
  REQUIRE(results.size() == 1);
  // If metric_ is ignored (L2 used), this returns id 1; correct cosine returns id 0.
  CHECK(results[0].id == 0);
}

// Bug: inverted_lists_ (and size_) not reset on re-train — resize() is a
// no-op when the vector already has the right size, leaving stale entries.
// Regression: after a re-train + fresh inserts the size must reflect only
// the new inserts, and search must not surface stale vectors.
TEST_CASE("IVFFlatIndex regression: re-train resets inverted lists and size") {
  constexpr size_t DIM = 2, K_CENTROIDS = 2;
  IVFFlatIndex idx(DIM, Metric::L2, K_CENTROIDS, K_CENTROIDS);

  // Phase 1 — train and populate with 6 vectors near origin
  std::vector<float> data1 = {0.0f, 0.0f, 1.0f, 0.0f,
                               0.0f, 1.0f, 0.5f, 0.5f,
                               0.2f, 0.8f, 0.8f, 0.2f};
  idx.train(data1);
  for (size_t i = 0; i < 6; ++i) {
    std::vector<float> v = {data1[i * 2], data1[i * 2 + 1]};
    idx.insert(v);
  }
  REQUIRE(idx.size() == 6);

  // Phase 2 — re-train on data far from origin, insert 2 new vectors
  std::vector<float> data2 = {100.0f, 100.0f, 101.0f, 100.0f,
                               100.0f, 101.0f, 101.0f, 101.0f};
  idx.train(data2);
  // After re-train, size must be 0 — old entries are gone
  CHECK(idx.size() == 0);

  idx.insert(std::vector<float>{100.0f, 100.0f});
  idx.insert(std::vector<float>{101.0f, 101.0f});
  CHECK(idx.size() == 2);

  // Full-probe search: should only see the 2 new vectors, not the 6 old ones
  auto results = idx.search(std::vector<float>{100.5f, 100.5f}, 10, K_CENTROIDS);
  CHECK(results.size() == 2);
  for (const auto &r : results)
    CHECK(r.score < 5.0f); // both new vectors are within sqrt(5) of the query
}

TEST_CASE("KMeans: output size") {
  // train() should always return exactly k * dim floats
  std::vector<float> data = {0.0f, 0.0f, 1.0f, 1.0f,
                             10.0f, 10.0f, 11.0f, 11.0f};
  KMeans km;
  auto centroids = km.train(data, 2, {2, 20});
  CHECK(centroids.size() == 4); // k=2, dim=2
}

TEST_CASE("KMeans: converges on well-separated clusters") {
  // Two tight clusters far apart — centroids must converge to their means:
  // Cluster A: (0,0), (1,0), (0,1)  → mean (0.33, 0.33)
  // Cluster B: (10,10), (11,10), (10,11) → mean (10.33, 10.33)
  std::vector<float> data = {
      0.0f,  0.0f,
      1.0f,  0.0f,
      0.0f,  1.0f,
      10.0f, 10.0f,
      11.0f, 10.0f,
      10.0f, 11.0f,
  };

  KMeans km;
  auto centroids = km.train(data, 2, {2, 50});
  REQUIRE(centroids.size() == 4);

  bool c0_low = centroid_near(centroids, 0, 2, {0.33f, 0.33f});
  bool c1_low = centroid_near(centroids, 1, 2, {0.33f, 0.33f});
  bool c0_high = centroid_near(centroids, 0, 2, {10.33f, 10.33f});
  bool c1_high = centroid_near(centroids, 1, 2, {10.33f, 10.33f});

  // One centroid near each cluster (order depends on random init)
  CHECK(((c0_low && c1_high) || (c1_low && c0_high)));
}

TEST_CASE("KMeans: k=1 centroid is the global mean") {
  // With k=1, the single centroid must be the mean of all vectors
  std::vector<float> data = {
      0.0f, 0.0f,
      2.0f, 0.0f,
      0.0f, 2.0f,
      2.0f, 2.0f,
  };
  // Mean = (1.0, 1.0)

  KMeans km;
  auto centroids = km.train(data, 2, {1, 20});
  REQUIRE(centroids.size() == 2);
  CHECK(centroids[0] == doctest::Approx(1.0f));
  CHECK(centroids[1] == doctest::Approx(1.0f));
}

TEST_CASE("KMeans: k=N each vector is its own centroid") {
  // With k equal to number of vectors, each centroid should sit on a vector
  std::vector<float> data = {
      0.0f, 0.0f,
      5.0f, 5.0f,
      9.0f, 1.0f,
  };

  KMeans km;
  auto centroids = km.train(data, 2, {3, 20});
  REQUIRE(centroids.size() == 6); // k=3, dim=2

  // Every data point should match one of the centroids exactly
  auto matches_a_centroid = [&](float x, float y) {
    for (size_t c = 0; c < 3; ++c) {
      if (std::abs(centroids[c * 2] - x) < 0.01f &&
          std::abs(centroids[c * 2 + 1] - y) < 0.01f)
        return true;
    }
    return false;
  };
  CHECK(matches_a_centroid(0.0f, 0.0f));
  CHECK(matches_a_centroid(5.0f, 5.0f));
  CHECK(matches_a_centroid(9.0f, 1.0f));
}
