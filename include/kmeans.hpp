#pragma once
#include <span>
#include <vector>

class KMeans {
public:
  struct Config {
    size_t k; // number of centroids
    size_t max_iterations;
    float tolerance = 1e-4; // Stop if centroids move less than this
    uint32_t seed = 42;     // fixed default, caller can override
  };

  // Returns a flat buffer of k * dim centroids
  std::vector<float> train(std::span<const float> data, size_t dim,
                           const Config &config);

private:
  // Helper to find the closest vector for a bunch of centroids
  std::vector<size_t> assign(std::span<const float> data, size_t dim,
                             const std::vector<float> &centroids);
  // Helper to move centroids to the average of their assigned points
  void update(std::span<const float> data, size_t dim,
              const std::vector<size_t> &assignments,
              std::vector<float> &centroids);

  std::vector<float> pick_random_centroids(std::span<const float> data,
                                           size_t dim, size_t k, uint32_t seed);
};