#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "vector_index.hpp"
#include <algorithm>
#include <cmath>
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

static float vec_norm(std::span<const float> v) { return std::sqrtf(dot(v, v)); }

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