#pragma once
#include "distance.hpp"
#include "vector_index.hpp"
#include <span>
#include <stdexcept>
#include <vector>

class FlatIndex : public VectorIndex {
public:
  explicit FlatIndex(size_t dimension, Metric metric);
  void train(std::span<const float> data) override;
  void insert(std::span<const float> vec) override;
  std::vector<SearchResult> search(std::span<const float> query,
                                   size_t k) const override;
  std::vector<SearchResult> search(std::span<const float> query, size_t k,
                                   bool use_simd) const;
  void save(const std::string &path) const override;
  void load(const std::string &path) override;
  size_t size() const override;
  size_t dimension() const override;

private:
  size_t dim_;
  size_t size_ = 0;
  size_t aligned_dim_;
  Metric metric_;
  std::vector<float> data_;
};