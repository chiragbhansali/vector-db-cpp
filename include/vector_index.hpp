#pragma once
#include <algorithm>
#include <queue>
#include <span>
#include <stdexcept>
#include <vector>

struct SearchResult {
  size_t id;
  float score;

  bool operator>(const SearchResult &other) const {
    return score > other.score;
  }
};

class FlatIndex {
public:
  explicit FlatIndex(size_t dimension) : dim_(dimension) {}
  void insert(std::span<const float> vec) {
    if (vec.size() != dim_) {
      throw std::invalid_argument("Vector dimension mismatch");
    }
    data_.insert(data_.end(), vec.begin(), vec.end());
    ++size_;
  };

  // Perform exact k-NN search using a Max-Heap
  std::vector<SearchResult> search(std::span<const float> query,
                                   size_t k) const {
    if (query.size() != dim_) {
      throw std::invalid_argument("Query dimension mismatch");
    }
    if (k == 0)
      return {};

    // Max-Heap: Keeps the K smallest distances.
    auto comp = [](const SearchResult &a, const SearchResult &b) {
      return a.score < b.score;
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(comp)>
        pq(comp);

    for (size_t i = 0; i < size_; ++i) {
      // Calculate start of current vector in the flat buffer
      std::span<const float> vec_i(data_.data() + (i * dim_), dim_);

      float dist = compute_l2(query, vec_i);

      if (pq.size() < k) {
        pq.push({i, dist});
      } else if (dist < pq.top().score) {
        // Found a closer neighbor than the worst one in our top-k
        pq.pop();
        pq.push({i, dist});
      }
    }

    // Extract results (Worst -> Best)
    std::vector<SearchResult> results;
    results.reserve(pq.size());
    while (!pq.empty()) {
      results.push_back(pq.top());
      pq.pop();
    }

    // Reverse to get Best -> Worst (Ascending distance)
    std::reverse(results.begin(), results.end());
    return results;
  }

  size_t size() const { return size_; }
  size_t dimension() const { return dim_; }

private:
  size_t dim_;
  size_t size_ = 0;
  std::vector<float>
      data_; // Contiguous layout: [v0_0, v0_1, ..., v1_0, v1_1, ...]

  float compute_l2(std::span<const float> vec1,
                   std::span<const float> vec2) const {
    if (vec1.size() != vec2.size())
      throw std::runtime_error("size mismatch");

    float dst_sum = 0;

    for (size_t i = 0; i < vec1.size(); ++i) {
      dst_sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }

    return dst_sum;
  }
};