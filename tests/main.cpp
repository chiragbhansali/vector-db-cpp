#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "vector_index.hpp"
#include <vector>

TEST_CASE("FlatIndex basic operations") {
  // Setup an index for 2D vectors
  FlatIndex index(2);

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