#pragma once
#include "vector_index.hpp"
#include <span>
#include <stdexcept>
#include <vector>

struct Entry {
  size_t id;
  std::vector<float> vec;
};

class IVFFlatIndex : public VectorIndex {
public:
  explicit IVFFlatIndex(size_t dimension, Metric metric, size_t nprobe,
                        size_t num_centroids);
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
  std::vector<float> centroids_;
  std::vector<std::vector<Entry>> inverted_lists_;
  size_t dim_;
  size_t size_ = 0;
  size_t nprobe_ = 10;
  size_t num_centroids_;
  Metric metric_;
};