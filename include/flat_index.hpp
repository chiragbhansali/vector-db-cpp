#pragma once
#include "vector_index.hpp"
#include <arm_neon.h>
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
  void save(const std::string &path) const override;
  void load(const std::string &path) override;
  size_t size() const override;
  size_t dimension() const override;

private:
  static void normalize(std::span<float> vec);

  float compute_l2(std::span<const float> vec1,
                   std::span<const float> vec2) const;
  float compute_dot(std::span<const float> vec1,
                    std::span<const float> vec2) const;
  float compute_dot_simd(std::span<const float> a,
                         std::span<const float> b) const;
  float compute_l2_simd(std::span<const float> a,
                        std::span<const float> b) const;

private:
  size_t dim_;
  size_t size_ = 0;
  size_t aligned_dim_;
  Metric metric_;
  std::vector<float> data_;
};