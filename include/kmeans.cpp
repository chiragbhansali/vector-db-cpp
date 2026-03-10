#include "kmeans.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <span>
#include <vector>

std::vector<float> KMeans::train(std::span<const float> data, size_t dim,
                                 const Config &config) {
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

std::vector<size_t> KMeans::assign(std::span<const float> data, size_t dim,
                                   const std::vector<float> &centroids) {
  size_t num_vectors = data.size() / dim;
  size_t k = centroids.size() / dim;
  std::vector<size_t> assignments(num_vectors);

  // Match every vector in 'data' to its closest centroid
  for (size_t i = 0; i < num_vectors; ++i) {
    std::span<const float> vec(data.data() + i * dim, dim);
    float min_dist = std::numeric_limits<float>::max();
    size_t best_centroid = 0;

    for (size_t c = 0; c < k; ++c) {
      std::span<const float> centroid(centroids.data() + c * dim, dim);
      float dist = 0;
      for (size_t j = 0; j < dim; ++j) {
        float diff = (vec[j] - centroid[j]);
        dist += diff * diff;
      }
      if (dist < min_dist) {
        min_dist = dist;
        best_centroid = c;
      }
    }

    assignments[i] = best_centroid;
  }

  return assignments;
}

void KMeans::update(std::span<const float> data, size_t dim,
                    const std::vector<size_t> &assignments,
                    std::vector<float> &centroids) {
  size_t k = centroids.size() / dim;
  size_t num_vectors = data.size() / dim;

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
    if (counts[centroid_id] == 0)
      continue;
    centroids[i] = sums[i] / counts[centroid_id];
  }
}