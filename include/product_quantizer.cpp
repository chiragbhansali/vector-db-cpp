#include "product_quantizer.hpp"
#include "distance.hpp"
#include "kmeans.hpp"
#include <span>
#include <vector>

ProductQuantizer::ProductQuantizer(size_t M_, size_t K_, size_t dim_)
    : M_(M_), K_(K_), dim_(dim_) {
  if (K_ > 256)
    throw std::invalid_argument("K must be <= 256 to fit in uint8_t codes");
}

void ProductQuantizer::train(std::span<const float> data) {
  size_t num_vectors = data.size() / dim_;
  //   For each subspace
  for (size_t m = 0; m < M_; ++m) {
    std::vector<float> subspace;
    subspace.reserve((dim_ / M_) * num_vectors);
    // For each sub-vector belonging to this subspace m
    for (size_t i = 0; i < num_vectors; ++i) {
      std::span<const float> subspace_vector(
          data.data() + i * dim_ + m * (dim_ / M_), dim_ / M_);
      subspace.insert(subspace.end(), subspace_vector.data(),
                      subspace_vector.data() + subspace_vector.size());
    }

    KMeans cluster;
    std::vector<float> centroids =
        cluster.train(subspace, dim_ / M_, KMeans::Config{K_, 100});
    codebooks_.insert(codebooks_.end(), centroids.data(),
                      centroids.data() + centroids.size());
  }
}

std::vector<uint8_t> ProductQuantizer::encode(std::span<const float> vec) {
  if (codebooks_.empty()) {
    throw std::runtime_error(
        "Subspace centroids must be trained before inserting");
  }
  std::vector<uint8_t> vector_centroid_map;
  vector_centroid_map.reserve(M_);

  for (size_t m = 0; m < M_; ++m) {
    std::span<const float> subspace_centroids(
        codebooks_.data() + m * K_ * (dim_ / M_), K_ * (dim_ / M_));
    std::span<const float> subspace_vector(vec.data() + m * (dim_ / M_),
                                           dim_ / M_);

    vector_centroid_map.push_back(find_closest_centroid(
        subspace_vector, subspace_centroids, (dim_ / M_)));
  }
  return vector_centroid_map;
}

std::vector<float>
ProductQuantizer::compute_distance_table(std::span<const float> query) const {
  std::vector<float> sub_vector_centroid_map;
  sub_vector_centroid_map.reserve(M_ * K_);

  for (size_t m = 0; m < M_; ++m) {
    std::span<const float> subspace_centroids(
        codebooks_.data() + m * K_ * (dim_ / M_), K_ * (dim_ / M_));

    std::span<const float> sub_vector(query.data() + m * (dim_ / M_),
                                      dim_ / M_);
    for (size_t c = 0; c < K_; ++c) {
      std::span<const float> centroid(subspace_centroids.data() + c * dim_ / M_,
                                      dim_ / M_);
      sub_vector_centroid_map.push_back(compute_l2(sub_vector, centroid));
    }
  }
  return sub_vector_centroid_map;
}

std::vector<float> ProductQuantizer::get_codebooks() const {
  return codebooks_;
}

void ProductQuantizer::set_codebooks(std::vector<float> codebooks) {
  codebooks_ = codebooks;
}