#pragma once
#include "product_quantizer.hpp"
#include "vector_index.hpp"
#include <span>
#include <stdexcept>
#include <vector>

struct PQEntry {
  size_t id;
  std::vector<uint8_t> vec_centroid_codes;
};

class IVFPQIndex : public VectorIndex {
public:
  // Note: only Metric::L2 is supported. InnerProduct is not implemented
  // in the ADC distance table and will throw at construction time.
  explicit IVFPQIndex(size_t dimension, Metric metric, size_t nprobe,
                      size_t num_centroids, size_t M_, size_t K_);
  void train(std::span<const float> data) override;
  void insert(std::span<const float> vec) override;
  std::vector<SearchResult> search(std::span<const float> query,
                                   size_t k) const override;
  std::vector<SearchResult> search(std::span<const float> query, size_t k,
                                   size_t nprobe) const;
  size_t size() const override;
  size_t dimension() const override;
  void save(const std::string &path) const override;
  void load(const std::string &path) override;

private:
  size_t M_ = 8;   // number of subspaces
  size_t K_ = 256; // number of centroids per subspace
  size_t dim_;
  size_t nprobe_;
  size_t num_centroids_;
  size_t size_;
  std::vector<float> centroids_;
  std::vector<std::vector<PQEntry>> inverted_lists_;
  ProductQuantizer pq_;
  Metric metric_;
};