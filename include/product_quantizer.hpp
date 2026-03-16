#pragma once
#include <span>
#include <vector>

class ProductQuantizer {
public:
  explicit ProductQuantizer(size_t M_, size_t K_, size_t dim_);
  void train(std::span<const float> data);
  std::vector<uint8_t> encode(std::span<const float> vec);
  std::vector<float> compute_distance_table(std::span<const float> query);

private:
  size_t M_; // number of subspaces
  size_t K_; // number of centroids per subspace
  size_t dim_;
  std::vector<float> codebooks_;
};
