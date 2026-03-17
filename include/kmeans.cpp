#include "kmeans.hpp"
#include "distance.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <span>
#include <vector>

std::vector<float> KMeans::train(std::span<const float> data, size_t dim,
                                 const Config &config) {
  if (data.size() / dim < config.k) {
    throw std::runtime_error("KMeans: num_vectors (N) must be >= K");
  }
  // Pick K random vectors from 'data' and store them in a 'centroids' vector
  std::vector<float> centroids =
      pick_random_centroids(data, dim, config.k, config.seed);
  for (size_t i = 0; i < config.max_iterations; ++i) {
    std::vector<float> old_centroids = centroids; // Save to check movement
    auto assignments = assign(data, dim, centroids);
    update(data, dim, assignments, centroids);

    // Calculate how much the centroids moved (L2 distance)
    float shift = 0;
    for (size_t j = 0; j < centroids.size(); ++j) {
      float diff = centroids[j] - old_centroids[j];
      shift += diff * diff;
    }

    if (std::sqrt(shift) < config.tolerance) {
      break;
    }
  }

  return centroids;
}

std::vector<float> KMeans::pick_random_centroids(std::span<const float> data,
                                                 size_t dim, size_t k,
                                                 uint32_t seed) {
  // Create a list of all possible indices
  size_t num_vectors = data.size() / dim;
  std::vector<float> centroids(k * dim);

  std::vector<size_t> indices(num_vectors);
  std::iota(indices.begin(), indices.end(), 0);

  // Shuffle the indices using given seed
  std::mt19937 g(seed);
  std::shuffle(indices.begin(), indices.end(), g);

  // Copy the vectors at the first 'k' shuffled indices
  for (size_t i = 0; i < k; ++i) {
    size_t src_idx = indices[i];

    // Copy one vector (length 'dim') into our centroids buffer
    std::copy(data.begin() + src_idx * dim, data.begin() + (src_idx + 1) * dim,
              centroids.begin() + i * dim);
  }

  return centroids;
}
/// @brief Assigns each vector in `data` to its nearest centroid.
///
/// Performs a full scan over all centroids per vector and returns the index
/// of the closest centroid for every input vector.
///
/// @param data Flat buffer of input vectors.
/// @param dim Dimensionality of each vector.
/// @param centroids Flat buffer of centroid vectors.
/// @return Vector of centroid indices; `assignments[i]` is the nearest
/// centroid for input vector `i`.
std::vector<size_t> KMeans::assign(std::span<const float> data, size_t dim,
                                   const std::vector<float> &centroids) {
  size_t num_vectors = data.size() / dim;
  size_t k = centroids.size() / dim;
  std::vector<size_t> assignments(num_vectors);

  // Match every vector in 'data' to its closest centroid
  for (size_t i = 0; i < num_vectors; ++i) {
    std::span<const float> vec(data.data() + i * dim, dim);
    assignments[i] = find_closest_centroid(vec, centroids, dim);
  }

  return assignments;
}

/// @brief Moves each centroid to the means of its assigned vectors
///
/// 1. Sums the coordinates and counts assignments per centroid
/// 2. Finds new centroid which is the mean of vectors in the centroid.
/// Note - centroids with no assignments are skipped, preserving their previous
/// coordinates
///
/// @param data flat vector buffer
/// @param dim dimensionality of each vector
/// @param assignments output of assign(); assignments[i] is the centroid index
/// for vector i
/// @param centroids modified in place; each centroid is replaced by the mean of
/// its assigned vectors
void KMeans::update(std::span<const float> data, size_t dim,
                    const std::vector<size_t> &assignments,
                    std::vector<float> &centroids) {
  size_t k = centroids.size() / dim;
  size_t num_vectors = data.size() / dim;
  bool dead_clusters_exist = false;

  std::vector<float> sums(k * dim, 0.0f);
  std::vector<size_t> counts(k, 0);

  for (size_t i = 0; i < num_vectors; ++i) {
    size_t centroid_id = assignments[i];
    for (size_t j = 0; j < dim; ++j) {
      sums[centroid_id * dim + j] += data[i * dim + j];
    }
    counts[centroid_id] += 1;
  }

  for (size_t i = 0; i < centroids.size(); ++i) {
    size_t centroid_id = i / dim;
    if (counts[centroid_id] == 0) {
      dead_clusters_exist = true;
      continue;
    }
    centroids[i] = sums[i] / counts[centroid_id];
  }

  if (!dead_clusters_exist)
    return;
  auto it = std::max_element(counts.begin(), counts.end());
  size_t largest = std::distance(counts.begin(), it);
  std::vector<size_t> candidates;

  for (size_t i = 0; i < assignments.size(); ++i) {
    if (assignments[i] == largest) {
      candidates.push_back(i);
    }
  }

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
  for (size_t i = 0; i < k; ++i) {
    if (counts[i] == 0) {
      size_t chosen = candidates[dist(rng)];
      std::copy(data.begin() + chosen * dim, data.begin() + chosen * dim + dim,
                centroids.begin() + i * dim);
    }
  }
}