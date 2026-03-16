#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "distance.hpp"
#include "flat_index.hpp"
#include "ivf_flat_index.hpp"
#include "ivf_pq.hpp"
#include "kmeans.hpp"
#include "vector_index.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <unordered_set>
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

// Raw inner product over vectors, sorted descending
static std::vector<SearchResult>
bruteforce_inner_product(const std::vector<std::vector<float>> &db,
                         std::span<const float> query) {
  std::vector<SearchResult> all;
  all.reserve(db.size());
  for (size_t i = 0; i < db.size(); ++i) {
    float sim = dot(query, db[i]);
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

TEST_CASE("FlatIndex InnerProduct") {
  // Setup an index for 2D unit vectors using InnerProduct metric
  FlatIndex index(2, Metric::InnerProduct);

  // Insert unit vectors (caller's responsibility to normalize for cosine semantics)
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

    // 1. Exact match: dot([1,0],[1,0]) = 1
    CHECK(results[0].id == 0);
    CHECK(results[0].score == doctest::Approx(1.0f));

    // 2. Orthogonal (90 degrees): dot([1,0],[0,1]) = 0
    CHECK(results[1].id == 1);
    CHECK(results[1].score == doctest::Approx(0.0f));

    // 3. Opposite (180 degrees): dot([1,0],[-1,0]) = -1
    CHECK(results[2].id == 2);
    CHECK(results[2].score == doctest::Approx(-1.0f));
  }

  SUBCASE("Ranking by similarity") {
    // Query [1,1]: dot with [1,0]=1, dot with [0,1]=1 (both equally close)
    std::vector<float> query = {1.0f, 1.0f};
    auto results = index.search(query, 2);

    REQUIRE(results.size() == 2);
    CHECK(results[0].score == doctest::Approx(1.0f));
    CHECK(results[1].score == doctest::Approx(1.0f));
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

  SUBCASE("InnerProduct") {
    FlatIndex index(DIM, Metric::InnerProduct);
    for (auto &v : db)
      index.insert(v);

    for (size_t q = 0; q < 5; ++q) {
      auto query = generate_random_vectors(1, DIM, rng)[0];
      auto got = index.search(query, K);
      auto ref = bruteforce_inner_product(db, query);

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

// Vectors scaled by random factors (0.01–100x), then pre-normalized before
// inserting. Scores are raw dot products on unit vectors == cosine similarity.
TEST_CASE("Randomized: InnerProduct stable across magnitude scales") {
  constexpr size_t N = 200, DIM = 32, K = 10;
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> scale_dist(0.01f, 100.0f);

  auto db = generate_random_vectors(N, DIM, rng);
  for (auto &v : db) {
    float s = scale_dist(rng);
    for (auto &x : v)
      x *= s;
  }

  // Pre-normalize before inserting; keep a normalized copy for brute-force ref
  FlatIndex index(DIM, Metric::InnerProduct);
  std::vector<std::vector<float>> db_norm = db;
  for (auto &v : db_norm) {
    normalize(std::span<float>(v.data(), v.size()));
    index.insert(v);
  }

  for (size_t q = 0; q < 5; ++q) {
    auto query = generate_random_vectors(1, DIM, rng)[0];
    float qs = scale_dist(rng);
    for (auto &x : query)
      x *= qs;

    auto got = index.search(query, K);
    auto ref = bruteforce_inner_product(db_norm, query);

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

  for (auto metric : {Metric::L2, Metric::InnerProduct}) {
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
// Regression: construct two vectors where L2 and InnerProduct rankings DIVERGE:
//
//   v0 = [3.0, 0.0]  — high magnitude, perfectly aligned with query [1,0]
//   v1 = [0.99, 0.1] — low magnitude, very close in L2 to query [1,0]
//
//   InnerProduct nearest: v0 (dot = 3.0 vs 0.99)
//   L2 nearest:           v1 (sq-dist ≈ 0.01 vs 4.0)
//
// If metric_ is ignored and L2 is used, the search returns id=1; the correct
// InnerProduct search returns id=0.
TEST_CASE("IVFFlatIndex regression: metric_ drives search (InnerProduct vs L2)") {
  constexpr size_t DIM = 2, K_CENTROIDS = 1;
  // Single centroid → nprobe=1 is a full scan.
  std::vector<float> train_data = {3.0f, 0.0f, 0.99f, 0.1f};

  IVFFlatIndex idx(DIM, Metric::InnerProduct, /*nprobe=*/1, K_CENTROIDS);
  idx.train(train_data);

  // Do NOT normalize — magnitude matters for raw InnerProduct.
  std::vector<float> v0 = {3.0f, 0.0f};   // id 0 — InnerProduct winner (dot=3.0)
  std::vector<float> v1 = {0.99f, 0.1f};  // id 1 — L2 winner (sq-dist≈0.01)
  idx.insert(v0);
  idx.insert(v1);

  auto results = idx.search(std::vector<float>{1.0f, 0.0f}, 1);
  REQUIRE(results.size() == 1);
  // Buggy (L2) returns id=1; correct InnerProduct returns id=0.
  CHECK(results[0].id == 0);
  // Score should be the dot product value (positive), not a negated L2.
  CHECK(results[0].score == doctest::Approx(3.0f));
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

// ---------------------------------------------------------------------------
// Edge case tests
// ---------------------------------------------------------------------------

TEST_CASE("IVFFlatIndex: search on empty corpus") {
  // Trained but no vectors inserted — search must return empty, not crash.
  constexpr size_t DIM = 4, K_CENTROIDS = 2;
  std::vector<float> train_data = {1.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 1.0f, 0.0f, 0.0f};
  IVFFlatIndex idx(DIM, Metric::L2, K_CENTROIDS, K_CENTROIDS);
  idx.train(train_data);
  // No inserts
  auto results = idx.search(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}, 5);
  CHECK(results.empty());
}

TEST_CASE("IVFFlatIndex: k=0 returns empty") {
  constexpr size_t DIM = 4, N = 10, K_CENTROIDS = 2;
  std::mt19937 rng(13);
  auto db = generate_random_vectors(N, DIM, rng);
  auto idx = build_ivf(db, DIM, K_CENTROIDS);
  auto results = idx.search(generate_random_vectors(1, DIM, rng)[0], /*k=*/0);
  CHECK(results.empty());
}

TEST_CASE("IVFFlatIndex: nprobe=0 returns empty") {
  constexpr size_t DIM = 4, N = 10, K_CENTROIDS = 2;
  std::mt19937 rng(14);
  auto db = generate_random_vectors(N, DIM, rng);
  auto idx = build_ivf(db, DIM, K_CENTROIDS);
  auto query = generate_random_vectors(1, DIM, rng)[0];
  auto results = idx.search(query, 5, /*nprobe=*/0);
  CHECK(results.empty());
}

TEST_CASE("IVFFlatIndex: nprobe > num_centroids handled gracefully") {
  // nprobe larger than the number of centroids should probe all centroids —
  // same result as nprobe == num_centroids, no crash.
  constexpr size_t DIM = 4, N = 50, K_CENTROIDS = 4, K = 5;
  std::mt19937 rng(15);
  auto db = generate_random_vectors(N, DIM, rng);
  auto idx = build_ivf(db, DIM, K_CENTROIDS);
  auto query = generate_random_vectors(1, DIM, rng)[0];

  auto full = idx.search(query, K, K_CENTROIDS);        // exact full scan
  auto over = idx.search(query, K, K_CENTROIDS * 100);  // nprobe >> num_centroids

  REQUIRE(full.size() == over.size());
  for (size_t i = 0; i < full.size(); ++i) {
    CHECK(full[i].id == over[i].id);
    CHECK(full[i].score == doctest::Approx(over[i].score));
  }
}

TEST_CASE("IVFFlatIndex: single vector corpus") {
  // With only one vector in the index, k=5 must return exactly 1 result.
  constexpr size_t DIM = 4, K_CENTROIDS = 1;
  std::vector<float> train_data = {1.0f, 2.0f, 3.0f, 4.0f};
  IVFFlatIndex idx(DIM, Metric::L2, 1, K_CENTROIDS);
  idx.train(train_data);
  idx.insert(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  auto results = idx.search(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}, 5);
  CHECK(results.size() == 1);
}

// ---------------------------------------------------------------------------
// Approximate recall test
// ---------------------------------------------------------------------------

TEST_CASE("IVFFlatIndex: approximate recall degrades gracefully with nprobe") {
  // Verify IVF is a useful ANN index: full probe == exact recall, partial probe
  // still achieves reasonable recall.
  constexpr size_t N = 500, DIM = 16, K_CENTROIDS = 20, K = 10;
  std::mt19937 rng(77);
  auto db = generate_random_vectors(N, DIM, rng);

  std::vector<float> flat;
  flat.reserve(N * DIM);
  for (const auto &v : db)
    flat.insert(flat.end(), v.begin(), v.end());

  IVFFlatIndex idx(DIM, Metric::L2, K_CENTROIDS, K_CENTROIDS);
  idx.train(flat);
  for (const auto &v : db)
    idx.insert(v);

  // Helper: compute Recall@K for one query.
  auto recall_at_k = [&](std::span<const float> query, size_t nprobe) -> float {
    auto approx = idx.search(query, K, nprobe);
    auto exact  = bruteforce_l2(db, query);
    exact.resize(K);  // keep top-K only

    std::unordered_set<size_t> truth_ids;
    for (const auto &r : exact)
      truth_ids.insert(r.id);

    size_t hits = 0;
    for (const auto &r : approx)
      hits += truth_ids.count(r.id);
    return static_cast<float>(hits) / K;
  };

  constexpr size_t NQ = 20;
  auto queries = generate_random_vectors(NQ, DIM, rng);

  float recall_full = 0, recall_partial = 0;
  for (const auto &q : queries) {
    recall_full    += recall_at_k(q, K_CENTROIDS);  // full scan
    recall_partial += recall_at_k(q, 5);            // partial scan (5/20 lists)
  }
  recall_full    /= NQ;
  recall_partial /= NQ;

  // Full probe must be exact.
  CHECK(recall_full == doctest::Approx(1.0f));
  // Partial probe should still be reasonably good.
  CHECK(recall_partial >= 0.5f);
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

// ===========================================================================
// ProductQuantizer and IVFPQIndex test helpers
// ===========================================================================

// Build identity codebooks: centroid c = float(c) in every dimension.
// Requires subspace_dim = dim/M = 1 (i.e. dim == M).
static std::vector<float> make_pq_identity_codebooks(size_t M, size_t K,
                                                     size_t dim) {
  size_t sdim = dim / M;
  std::vector<float> cb(M * K * sdim);
  for (size_t m = 0; m < M; ++m)
    for (size_t c = 0; c < K; ++c)
      for (size_t j = 0; j < sdim; ++j)
        cb[m * K * sdim + c * sdim + j] = static_cast<float>(c);
  return cb;
}

// Generate flat cluster data. Cluster c is centered at [100*(c+1), 0,...,0]
// with Gaussian-like noise ±spread on every dimension.
static std::vector<float> make_cluster_data(size_t n_clusters,
                                            size_t n_per_cluster, size_t dim,
                                            float spread, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> noise(-spread, spread);
  std::vector<float> data;
  data.reserve(n_clusters * n_per_cluster * dim);
  for (size_t c = 0; c < n_clusters; ++c) {
    float center = static_cast<float>(c + 1) * 100.0f;
    for (size_t i = 0; i < n_per_cluster; ++i)
      for (size_t j = 0; j < dim; ++j)
        data.push_back((j == 0) ? center + noise(rng) : noise(rng));
  }
  return data;
}

// Train and populate an IVFPQIndex from a flat data buffer.
static IVFPQIndex build_ivfpq(std::span<const float> db, size_t dim,
                               size_t num_centroids, size_t nprobe) {
  IVFPQIndex idx(dim, Metric::L2, nprobe, num_centroids);
  idx.train(db);
  for (size_t i = 0; i + dim <= db.size(); i += dim)
    idx.insert(db.subspan(i, dim));
  return idx;
}

// Recall@K: fraction of `exact` IDs that appear in `approx`.
static float compute_recall(const std::vector<SearchResult> &approx,
                            const std::vector<SearchResult> &exact) {
  if (exact.empty())
    return 1.0f;
  std::unordered_set<size_t> truth;
  for (const auto &r : exact)
    truth.insert(r.id);
  size_t hits = 0;
  for (const auto &r : approx)
    hits += truth.count(r.id);
  return static_cast<float>(hits) / static_cast<float>(exact.size());
}

// ===========================================================================
// Section 1: ProductQuantizer unit tests
// ===========================================================================

TEST_CASE("ProductQuantizer - constructor and state") {
  SUBCASE("encode before train throws") {
    ProductQuantizer pq(8, 256, 8);
    std::vector<float> vec(8, 0.0f);
    CHECK_THROWS_AS(pq.encode(vec), std::runtime_error);
  }
  SUBCASE("codebooks empty before train") {
    ProductQuantizer pq(8, 256, 8);
    CHECK(pq.get_codebooks().empty());
  }
}

TEST_CASE("ProductQuantizer - train output") {
  SUBCASE("codebook size") {
    // dim=16, M=8, K=256 → codebook = 8 * 256 * 2 = 4096 floats
    constexpr size_t M = 8, K = 256, DIM = 16, N = 300;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(N * DIM);
    for (auto &x : data)
      x = dist(rng);

    ProductQuantizer pq(M, K, DIM);
    pq.train(data);
    CHECK(pq.get_codebooks().size() == M * K * (DIM / M));
  }

  SUBCASE("subspace independence") {
    // Datasets A and B differ only in subspace 0 (A: near 0, B: near 100).
    // Codebook slab 0 must differ; slabs 1-7 are similar (same data).
    constexpr size_t M = 8, K = 4, DIM = 8, N = 100;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> dataA(N * DIM), dataB(N * DIM);
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < DIM; ++j) {
        float v = (j == 0) ? 0.0f : dist(rng);
        dataA[i * DIM + j] = v;
        dataB[i * DIM + j] = (j == 0) ? 100.0f : v;
      }

    ProductQuantizer pqA(M, K, DIM), pqB(M, K, DIM);
    pqA.train(dataA);
    pqB.train(dataB);

    auto cbA = pqA.get_codebooks();
    auto cbB = pqB.get_codebooks();
    size_t sdim = DIM / M; // == 1

    // Mean of slab 0 centroids: A near 0, B near 100
    float meanA0 = 0, meanB0 = 0;
    for (size_t c = 0; c < K; ++c) {
      meanA0 += cbA[0 * K * sdim + c * sdim];
      meanB0 += cbB[0 * K * sdim + c * sdim];
    }
    meanA0 /= K;
    meanB0 /= K;
    CHECK(meanA0 < 5.0f);
    CHECK(meanB0 > 95.0f);

    // Mean of slab 1 centroids: both near 0 (same data in subspace 1)
    float meanA1 = 0, meanB1 = 0;
    for (size_t c = 0; c < K; ++c) {
      meanA1 += cbA[1 * K * sdim + c * sdim];
      meanB1 += cbB[1 * K * sdim + c * sdim];
    }
    meanA1 /= K;
    meanB1 /= K;
    CHECK(std::abs(meanA1) < 5.0f);
    CHECK(std::abs(meanB1) < 5.0f);
  }
}

TEST_CASE("ProductQuantizer - encode") {
  SUBCASE("output size") {
    constexpr size_t M = 8, K = 256, DIM = 8;
    auto cb = make_pq_identity_codebooks(M, K, DIM);
    ProductQuantizer pq(M, K, DIM);
    pq.set_codebooks(cb);
    std::vector<float> vec(DIM, 0.0f);
    CHECK(pq.encode(vec).size() == M);
  }

  SUBCASE("codes in range") {
    constexpr size_t M = 8, K = 256, DIM = 8;
    auto cb = make_pq_identity_codebooks(M, K, DIM);
    ProductQuantizer pq(M, K, DIM);
    pq.set_codebooks(cb);
    std::vector<float> vec(DIM, 5.0f);
    for (auto c : pq.encode(vec))
      CHECK(static_cast<size_t>(c) < K);
  }

  SUBCASE("deterministic") {
    constexpr size_t M = 8, K = 256, DIM = 8;
    auto cb = make_pq_identity_codebooks(M, K, DIM);
    ProductQuantizer pq(M, K, DIM);
    pq.set_codebooks(cb);
    std::vector<float> vec = {1.0f, 5.0f, 3.0f, 7.0f, 2.0f, 4.0f, 6.0f, 0.0f};
    auto c1 = pq.encode(vec);
    auto c2 = pq.encode(vec);
    REQUIRE(c1.size() == c2.size());
    for (size_t i = 0; i < c1.size(); ++i)
      CHECK(c1[i] == c2[i]);
  }

  SUBCASE("codes point to nearest centroid") {
    // Identity codebook: centroid c = float(c). vec[m] = float(c) → code c.
    constexpr size_t M = 8, K = 256, DIM = 8;
    auto cb = make_pq_identity_codebooks(M, K, DIM);
    ProductQuantizer pq(M, K, DIM);
    pq.set_codebooks(cb);
    std::vector<float> vec = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
    auto codes = pq.encode(vec);
    CHECK(codes[0] == 3);
    CHECK(codes[1] == 1);
    CHECK(codes[2] == 4);
    CHECK(codes[5] == 9);
  }
}

TEST_CASE("ProductQuantizer - codebook get/set round-trip") {
  SUBCASE("identity") {
    constexpr size_t M = 8, K = 256, DIM = 8;
    auto cb = make_pq_identity_codebooks(M, K, DIM);
    ProductQuantizer pq(M, K, DIM);
    pq.set_codebooks(cb);
    auto result = pq.get_codebooks();
    REQUIRE(result.size() == cb.size());
    for (size_t i = 0; i < cb.size(); ++i)
      CHECK(result[i] == cb[i]);
  }

  SUBCASE("encode uses new codebook") {
    // All-zero codebook: every centroid is at 0. find_closest_centroid keeps
    // the first tie (code 0) since it uses strict <.
    constexpr size_t M = 8, K = 256, DIM = 8;
    std::vector<float> zeros(M * K * (DIM / M), 0.0f);
    ProductQuantizer pq(M, K, DIM);
    pq.set_codebooks(zeros);
    std::vector<float> vec = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
    for (auto c : pq.encode(vec))
      CHECK(c == 0);
  }
}

TEST_CASE("ProductQuantizer - compute_distance_table") {
  SUBCASE("output size") {
    constexpr size_t M = 8, K = 256, DIM = 8;
    auto cb = make_pq_identity_codebooks(M, K, DIM);
    ProductQuantizer pq(M, K, DIM);
    pq.set_codebooks(cb);
    std::vector<float> query(DIM, 0.0f);
    CHECK(pq.compute_distance_table(query).size() == M * K);
  }

  SUBCASE("self-distance of centroid is zero") {
    // Centroid 0 of subspace 0 = [0.0]. Query sub-vector 0 = [0.0].
    // table[0*K + 0] = l2([0],[0]) = 0.
    constexpr size_t M = 8, K = 256, DIM = 8;
    auto cb = make_pq_identity_codebooks(M, K, DIM);
    ProductQuantizer pq(M, K, DIM);
    pq.set_codebooks(cb);
    std::vector<float> query(DIM, 0.0f);
    auto table = pq.compute_distance_table(query);
    CHECK(table[0 * K + 0] == doctest::Approx(0.0f));
  }

  SUBCASE("layout correctness") {
    // Identity codebook, subspace_dim=1.
    // table[m*K + c] = (query[m] - float(c))^2
    constexpr size_t M = 8, K = 256, DIM = 8;
    auto cb = make_pq_identity_codebooks(M, K, DIM);
    ProductQuantizer pq(M, K, DIM);
    pq.set_codebooks(cb);
    std::vector<float> query = {2.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
    auto table = pq.compute_distance_table(query);

    for (size_t m = 0; m < M; ++m)
      for (size_t c : {0UL, 1UL, 5UL, 10UL}) {
        float diff = query[m] - static_cast<float>(c);
        CHECK(table[m * K + c] == doctest::Approx(diff * diff));
      }
  }
}

// ===========================================================================
// Section 3: ADC correctness (placed before IVFPQIndex to be self-contained)
// ===========================================================================

TEST_CASE("ProductQuantizer - ADC arithmetic") {
  SUBCASE("end-to-end ADC score matches manual calculation") {
    // Identity codebook, subspace_dim=1, M=8, K=256.
    // vec   = [2,1,4,3,0,5,7,2] → codes = [2,1,4,3,0,5,7,2]
    // query = [3,2,5,4,1,6,8,3]
    // score = Σ (query[m] - float(code[m]))^2 = 8 × 1.0 = 8.0
    constexpr size_t M = 8, K = 256, DIM = 8;
    auto cb = make_pq_identity_codebooks(M, K, DIM);
    ProductQuantizer pq(M, K, DIM);
    pq.set_codebooks(cb);

    std::vector<float> vec = {2.0f, 1.0f, 4.0f, 3.0f, 0.0f, 5.0f, 7.0f, 2.0f};
    std::vector<float> query = {3.0f, 2.0f, 5.0f, 4.0f, 1.0f, 6.0f, 8.0f, 3.0f};

    auto codes = pq.encode(vec);
    REQUIRE(codes.size() == M);
    CHECK(codes[0] == 2);
    CHECK(codes[1] == 1);
    CHECK(codes[4] == 0);

    auto table = pq.compute_distance_table(query);
    float adc_score = 0.0f;
    for (size_t m = 0; m < M; ++m)
      adc_score += table[K * m + codes[m]];
    CHECK(adc_score == doctest::Approx(8.0f));
  }

  SUBCASE("self-ADC score is near zero") {
    // Encode vector v and compute distance table with v as query.
    // Summing table[K*m + code[m]] gives the quantization error, which
    // should be small (well below 0.5 per dimension).
    constexpr size_t M = 8, K = 256, DIM = 8, N = 300;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    std::vector<float> data(N * DIM);
    for (auto &x : data)
      x = dist(rng);

    ProductQuantizer pq(M, K, DIM);
    pq.train(data);

    std::vector<float> v(data.begin(), data.begin() + DIM);
    auto codes = pq.encode(v);
    auto table = pq.compute_distance_table(v);

    float score = 0.0f;
    for (size_t m = 0; m < M; ++m)
      score += table[K * m + codes[m]];
    CHECK(score < 0.5f * DIM);
  }
}

// ===========================================================================
// Section 2: IVFPQIndex unit tests
// ===========================================================================

TEST_CASE("IVFPQIndex - constructor") {
  SUBCASE("size and dimension") {
    IVFPQIndex idx(8, Metric::L2, 2, 4);
    CHECK(idx.size() == 0);
    CHECK(idx.dimension() == 8);
  }
}

TEST_CASE("IVFPQIndex - error paths") {
  SUBCASE("insert before train throws") {
    IVFPQIndex idx(8, Metric::L2, 2, 4);
    std::vector<float> vec(8, 0.0f);
    CHECK_THROWS_AS(idx.insert(vec), std::runtime_error);
  }

  SUBCASE("save to invalid path throws") {
    IVFPQIndex idx(8, Metric::L2, 2, 4);
    CHECK_THROWS(idx.save("/nonexistent/path/file.bin"));
  }

  SUBCASE("load from missing file throws") {
    IVFPQIndex idx(8, Metric::L2, 2, 4);
    CHECK_THROWS(idx.load("/no_such_file.bin"));
  }
}

TEST_CASE("IVFPQIndex - size tracking") {
  constexpr size_t DIM = 8;
  // 4 clusters × 75 = 300 training vectors (enough for K=256 k-means)
  auto train_data = make_cluster_data(4, 75, DIM, 0.5f, 42);

  SUBCASE("increments on insert") {
    IVFPQIndex idx(DIM, Metric::L2, 4, 4);
    idx.train(train_data);
    CHECK(idx.size() == 0);
    for (size_t i = 0; i < 5; ++i) {
      idx.insert(std::span<const float>(train_data.data() + i * DIM, DIM));
      CHECK(idx.size() == i + 1);
    }
  }

  SUBCASE("resets on retrain") {
    IVFPQIndex idx(DIM, Metric::L2, 4, 4);
    idx.train(train_data);
    for (size_t i = 0; i < 5; ++i)
      idx.insert(std::span<const float>(train_data.data() + i * DIM, DIM));
    REQUIRE(idx.size() == 5);
    idx.train(train_data);
    CHECK(idx.size() == 0);
  }

  SUBCASE("IDs are sequential") {
    constexpr size_t N = 10;
    IVFPQIndex idx(DIM, Metric::L2, 4, 4);
    idx.train(train_data);
    for (size_t i = 0; i < N; ++i)
      idx.insert(std::span<const float>(train_data.data() + i * DIM, DIM));

    // Full search: verify returned IDs are exactly {0,...,N-1}
    std::vector<float> q(DIM, 50.0f);
    auto res = idx.search(q, N, 4);
    REQUIRE(res.size() == N);
    std::unordered_set<size_t> ids;
    for (const auto &r : res)
      ids.insert(r.id);
    for (size_t i = 0; i < N; ++i)
      CHECK(ids.count(i) == 1);
  }
}

TEST_CASE("IVFPQIndex - search edge cases") {
  constexpr size_t DIM = 8;
  auto train_data = make_cluster_data(4, 75, DIM, 0.5f, 42);

  SUBCASE("k=0 returns empty") {
    IVFPQIndex idx(DIM, Metric::L2, 4, 4);
    idx.train(train_data);
    idx.insert(std::span<const float>(train_data.data(), DIM));
    CHECK(idx.search(std::vector<float>(DIM, 0.0f), 0).empty());
  }

  SUBCASE("nprobe=0 returns empty") {
    IVFPQIndex idx(DIM, Metric::L2, 4, 4);
    idx.train(train_data);
    idx.insert(std::span<const float>(train_data.data(), DIM));
    CHECK(idx.search(std::vector<float>(DIM, 0.0f), 5, 0).empty());
  }

  SUBCASE("empty corpus returns empty") {
    IVFPQIndex idx(DIM, Metric::L2, 4, 4);
    idx.train(train_data);
    CHECK(idx.search(std::vector<float>(DIM, 0.0f), 5).empty());
  }

  SUBCASE("k > corpus returns all vectors") {
    constexpr size_t N = 10;
    IVFPQIndex idx(DIM, Metric::L2, 4, 4);
    idx.train(train_data);
    for (size_t i = 0; i < N; ++i)
      idx.insert(std::span<const float>(train_data.data() + i * DIM, DIM));
    auto results = idx.search(std::vector<float>(DIM, 0.0f), 1000, 4);
    CHECK(results.size() == N);
  }

  SUBCASE("nprobe > num_centroids is handled") {
    constexpr size_t N = 10, NC = 4;
    IVFPQIndex idx(DIM, Metric::L2, NC, NC);
    idx.train(train_data);
    for (size_t i = 0; i < N; ++i)
      idx.insert(std::span<const float>(train_data.data() + i * DIM, DIM));
    std::vector<float> q(DIM, 50.0f);
    auto full = idx.search(q, 5, NC);
    auto over = idx.search(q, 5, NC * 1000);
    REQUIRE(full.size() == over.size());
    for (size_t i = 0; i < full.size(); ++i) {
      CHECK(full[i].id == over[i].id);
      CHECK(full[i].score == doctest::Approx(over[i].score));
    }
  }
}

TEST_CASE("IVFPQIndex - search results ordering") {
  SUBCASE("sorted ascending by score") {
    constexpr size_t DIM = 8, N = 4 * 75;
    auto data = make_cluster_data(4, 75, DIM, 0.5f, 7);
    IVFPQIndex idx(DIM, Metric::L2, 4, 4);
    idx.train(data);
    for (size_t i = 0; i < N; ++i)
      idx.insert(std::span<const float>(data.data() + i * DIM, DIM));
    std::vector<float> q(DIM, 50.0f);
    auto results = idx.search(q, N, 4);
    for (size_t i = 1; i < results.size(); ++i)
      CHECK(results[i - 1].score <= results[i].score);
  }
}

TEST_CASE("IVFPQIndex regression - retrain clears state") {
  constexpr size_t DIM = 8;
  // 4 clusters × 75 = 300 vectors: enough for K=256 PQ k-means per subspace
  auto dataA = make_cluster_data(4, 75, DIM, 0.1f, 42);

  IVFPQIndex idx(DIM, Metric::L2, 4, 4);
  idx.train(dataA);
  for (size_t i = 0; i < 5; ++i)
    idx.insert(std::span<const float>(dataA.data() + i * DIM, DIM));
  REQUIRE(idx.size() == 5);

  // Retrain on data far from cluster A (shift center from ~100 to ~500)
  auto dataB = make_cluster_data(4, 75, DIM, 0.1f, 99);
  for (auto &x : dataB)
    x += 400.0f;

  idx.train(dataB);
  CHECK(idx.size() == 0);

  for (size_t i = 0; i < 5; ++i)
    idx.insert(std::span<const float>(dataB.data() + i * DIM, DIM));
  CHECK(idx.size() == 5);

  // All returned IDs must be from the new batch (0–4), not stale entries
  std::vector<float> q(DIM, 0.0f);
  q[0] = 500.0f;
  auto results = idx.search(q, 10, 4);
  CHECK(results.size() == 5);
  for (const auto &r : results)
    CHECK(r.id < 5);
}

// ===========================================================================
// Section 4: Integration tests
// ===========================================================================

TEST_CASE("IVFPQIndex integration - exact recall on well-separated clusters") {
  // 4 clusters at [100,0,...], [200,0,...], [300,0,...], [400,0,...].
  // Spread << inter-cluster distance → PQ encoding is deterministic.
  // Use NPE=75 so total N=300 >= K=256 (avoids KMeans k>N crash).
  constexpr size_t NC = 4, NPE = 75, DIM = 8, N = NC * NPE;
  auto data = make_cluster_data(NC, NPE, DIM, 0.05f, 42);

  IVFPQIndex idx(DIM, Metric::L2, NC, NC);
  idx.train(data);
  for (size_t i = 0; i < N; ++i)
    idx.insert(std::span<const float>(data.data() + i * DIM, DIM));

  size_t correct = 0;
  for (size_t c = 0; c < NC; ++c) {
    std::vector<float> query(DIM, 0.0f);
    query[0] = static_cast<float>(c + 1) * 100.0f;
    auto results = idx.search(query, 1, NC);
    if (!results.empty() && results[0].id / NPE == c)
      ++correct;
  }
  CHECK(correct == NC);
}

TEST_CASE("IVFPQIndex integration - recall degrades gracefully with nprobe") {
  constexpr size_t N = 300, DIM = 16, NC = 16, K = 10, NQ = 20;
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(N * DIM);
  for (auto &x : data)
    x = dist(rng);

  IVFPQIndex idx(DIM, Metric::L2, NC, NC);
  idx.train(data);
  for (size_t i = 0; i < N; ++i)
    idx.insert(std::span<const float>(data.data() + i * DIM, DIM));

  std::vector<std::vector<float>> db(N, std::vector<float>(DIM));
  for (size_t i = 0; i < N; ++i)
    std::copy(data.begin() + i * DIM, data.begin() + (i + 1) * DIM,
              db[i].begin());

  std::vector<float> qdata(NQ * DIM);
  for (auto &x : qdata)
    x = dist(rng);

  float recall_full = 0, recall_half = 0, recall_one = 0;
  for (size_t q = 0; q < NQ; ++q) {
    std::span<const float> qvec(qdata.data() + q * DIM, DIM);
    auto exact = bruteforce_l2(db, qvec);
    exact.resize(K);
    recall_full += compute_recall(idx.search(qvec, K, NC), exact);
    recall_half += compute_recall(idx.search(qvec, K, NC / 2), exact);
    recall_one += compute_recall(idx.search(qvec, K, 1), exact);
  }
  recall_full /= NQ;
  recall_half /= NQ;
  recall_one /= NQ;

  CHECK(recall_full >= recall_half);
  CHECK(recall_half >= recall_one);
  CHECK(recall_full >= 0.7f);
}

// ===========================================================================
// Section 5: Serialization tests
// ===========================================================================

TEST_CASE("IVFPQIndex serialization") {
  const std::string fname = "test_ivfpq_index.bin";
  constexpr size_t DIM = 8, NC = 4, NPE = 75, N = NC * NPE;
  auto data = make_cluster_data(NC, NPE, DIM, 0.5f, 42);

  IVFPQIndex idx(DIM, Metric::L2, NC, NC);
  idx.train(data);
  for (size_t i = 0; i < N; ++i)
    idx.insert(std::span<const float>(data.data() + i * DIM, DIM));

  SUBCASE("save-load round-trip") {
    idx.save(fname);
    IVFPQIndex loaded(DIM, Metric::L2, NC, NC);
    loaded.load(fname);

    CHECK(loaded.size() == idx.size());
    CHECK(loaded.dimension() == idx.dimension());

    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(0.0f, 200.0f);
    for (size_t q = 0; q < 3; ++q) {
      std::vector<float> query(DIM);
      for (auto &x : query)
        x = d(rng);
      auto orig = idx.search(query, 5, NC);
      auto load = loaded.search(query, 5, NC);
      REQUIRE(orig.size() == load.size());
      for (size_t i = 0; i < orig.size(); ++i) {
        CHECK(orig[i].id == load[i].id);
        CHECK(orig[i].score == doctest::Approx(load[i].score));
      }
    }
    std::remove(fname.c_str());
  }

  SUBCASE("codebooks survive round-trip") {
    idx.save(fname);
    IVFPQIndex loaded(DIM, Metric::L2, NC, NC);
    loaded.load(fname);

    // ADC scores must be bitwise identical (same codebooks, same computation)
    std::vector<float> q(DIM, 0.0f);
    q[0] = 150.0f;
    auto before = idx.search(q, 5, NC);
    auto after = loaded.search(q, 5, NC);
    REQUIRE(before.size() == after.size());
    for (size_t i = 0; i < before.size(); ++i)
      CHECK(before[i].score == after[i].score);
    std::remove(fname.c_str());
  }

  SUBCASE("all PQEntry data survives") {
    idx.save(fname);
    IVFPQIndex loaded(DIM, Metric::L2, NC, NC);
    loaded.load(fname);

    std::vector<float> q(DIM, 0.0f);
    q[0] = 50.0f;
    auto before = idx.search(q, N, NC);
    auto after = loaded.search(q, N, NC);

    auto by_id = [](const SearchResult &a, const SearchResult &b) {
      return a.id < b.id;
    };
    std::sort(before.begin(), before.end(), by_id);
    std::sort(after.begin(), after.end(), by_id);

    REQUIRE(before.size() == after.size());
    for (size_t i = 0; i < before.size(); ++i) {
      CHECK(before[i].id == after[i].id);
      CHECK(before[i].score == doctest::Approx(after[i].score));
    }
    std::remove(fname.c_str());
  }
}

// ===========================================================================
// Section 6: Edge cases
// ===========================================================================

TEST_CASE("IVFPQIndex edge cases") {
  SUBCASE("minimum dimension (dim=8)") {
    // Need N >= K=256 training sub-vectors (300 vectors × 1 dim/subspace = 300 each)
    constexpr size_t DIM = 8;
    auto data = make_cluster_data(4, 75, DIM, 0.5f, 42); // 300 vectors
    IVFPQIndex idx(DIM, Metric::L2, 4, 4);
    idx.train(data);
    for (size_t i = 0; i < 10; ++i)
      idx.insert(std::span<const float>(data.data() + i * DIM, DIM));
    CHECK(!idx.search(std::vector<float>(DIM, 50.0f), 5, 4).empty());
  }

  SUBCASE("single vector per centroid") {
    // Train on abundant data so PQ k-means (K=256) has enough sub-vectors.
    // Then insert exactly one vector per IVF centroid and verify each is
    // findable as its own top-1 result.
    constexpr size_t DIM = 8, NC = 4;
    auto train_data = make_cluster_data(NC, 75, DIM, 0.5f, 42); // 300 vecs

    IVFPQIndex idx(DIM, Metric::L2, NC, NC);
    idx.train(train_data);

    // Insert one representative per cluster (first vector of each cluster block)
    for (size_t c = 0; c < NC; ++c)
      idx.insert(std::span<const float>(train_data.data() + c * 75 * DIM, DIM));

    for (size_t c = 0; c < NC; ++c) {
      auto results = idx.search(
          std::span<const float>(train_data.data() + c * 75 * DIM, DIM), 1, NC);
      REQUIRE(!results.empty());
      CHECK(results[0].id == c);
    }
  }

  SUBCASE("duplicate vectors get distinct IDs") {
    constexpr size_t DIM = 8;
    auto train = make_cluster_data(4, 75, DIM, 0.5f, 42); // 300 vecs, K=256 safe
    IVFPQIndex idx(DIM, Metric::L2, 4, 4);
    idx.train(train);

    std::vector<float> v(DIM, 0.0f);
    v[0] = 50.0f;
    idx.insert(v); // ID 0
    idx.insert(v); // ID 1

    auto results = idx.search(v, 2, 4);
    REQUIRE(results.size() == 2);
    std::unordered_set<size_t> ids;
    for (const auto &r : results)
      ids.insert(r.id);
    CHECK(ids.count(0) == 1);
    CHECK(ids.count(1) == 1);
    CHECK(results[0].score == doctest::Approx(results[1].score));
  }

  SUBCASE("all vectors in one list") {
    // NC=1: single centroid guarantees all vectors land in one list.
    // Use 300 training vectors so PQ k-means (K=256) has enough data.
    constexpr size_t DIM = 8, N = 20, NC = 1;
    auto train_data = make_cluster_data(4, 75, DIM, 0.5f, 42); // 300 vecs

    IVFPQIndex idx(DIM, Metric::L2, NC, NC);
    idx.train(train_data);

    // Insert 20 near-origin vectors into the single list
    std::vector<float> ins_data(N * DIM, 0.0f);
    for (size_t i = 0; i < N; ++i)
      ins_data[i * DIM] = static_cast<float>(i) * 0.001f;
    for (size_t i = 0; i < N; ++i)
      idx.insert(std::span<const float>(ins_data.data() + i * DIM, DIM));

    // Must not crash; results should be non-empty
    auto results = idx.search(std::vector<float>(DIM, 0.0f), 5, NC);
    CHECK(!results.empty());
  }
}

// ===========================================================================
// Section 7: Regression tests
// ===========================================================================

TEST_CASE("IVFPQIndex regression - metric does not affect ADC scoring") {
  SUBCASE("InnerProduct index has non-negative ascending scores") {
    constexpr size_t DIM = 8, N = 4 * 75;
    auto data = make_cluster_data(4, 75, DIM, 0.5f, 42);
    IVFPQIndex idx(DIM, Metric::InnerProduct, 4, 4);
    idx.train(data);
    for (size_t i = 0; i < N; ++i)
      idx.insert(std::span<const float>(data.data() + i * DIM, DIM));

    std::vector<float> q(DIM, 50.0f);
    auto results = idx.search(q, N, 4);
    REQUIRE(!results.empty());
    for (const auto &r : results)
      CHECK(r.score >= 0.0f);
    for (size_t i = 1; i < results.size(); ++i)
      CHECK(results[i - 1].score <= results[i].score);
  }
}

TEST_CASE("IVFPQIndex regression - centroid routing always uses L2") {
  // With InnerProduct metric, centroid assignment must still use L2.
  // vec=[0.2,0,...]: L2-nearest centroid is the one near 0.1, not near 10.
  // If routing used IP instead, the high-magnitude centroid would win and
  // the vector would not be found with nprobe=1.
  constexpr size_t DIM = 8, NC = 2;

  // 150 vecs near 0.1 + 150 near 10.0 = 300 total (>= K=256 for PQ)
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> noise(-0.01f, 0.01f);
  std::vector<float> train(300 * DIM, 0.0f);
  for (size_t i = 0; i < 150; ++i)
    train[i * DIM] = 0.1f + noise(rng);
  for (size_t i = 150; i < 300; ++i)
    train[i * DIM] = 10.0f + noise(rng);

  IVFPQIndex idx(DIM, Metric::InnerProduct, 1, NC);
  idx.train(train);

  std::vector<float> v(DIM, 0.0f);
  v[0] = 0.2f; // L2-nearest to centroid ~0.1, IP-nearest to centroid ~10
  idx.insert(v);

  auto results = idx.search(v, 1, 1);
  REQUIRE(!results.empty());
  CHECK(results[0].id == 0); // found: routing used L2 (correct)
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

TEST_CASE("KMeans: N < K throws") {
  // k-means is undefined when there are fewer data points than centroids.
  // The implementation must throw rather than segfault or silently corrupt.
  KMeans km;
  std::vector<float> data = {0.0f, 0.0f, 1.0f, 1.0f}; // 2 vectors, dim=2
  // Requesting k=5 > N=2 must throw
  CHECK_THROWS_AS(km.train(data, 2, KMeans::Config{5, 20}),
                  std::runtime_error);
}
