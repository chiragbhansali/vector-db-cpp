#pragma once
#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <queue>
#include <span>
#include <stdexcept>
#include <vector>

struct SearchResult {
  size_t id;
  float score;
};

enum class Metric { L2, InnerProduct };

class VectorIndex {
public:
  virtual ~VectorIndex() = default;
  // Training is required for IVF/PQ.
  virtual void train(std::span<const float> data) = 0;
  virtual void insert(std::span<const float> vec) = 0;

  virtual std::vector<SearchResult> search(std::span<const float> query,
                                           size_t k) const = 0;
  virtual void save(const std::string &path) const = 0;
  virtual void load(const std::string &path) = 0;

  virtual size_t size() const = 0;
  virtual size_t dimension() const = 0;
};
