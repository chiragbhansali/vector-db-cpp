#pragma once
#include <algorithm>
#include <cmath>
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

enum class Metric { L2, Cosine };

class FlatIndex {
public:
  explicit FlatIndex(size_t dimension, Metric metric)
      : dim_(dimension), metric_(metric) {}
  void insert(std::span<const float> vec) {
    if (vec.size() != dim_) {
      throw std::invalid_argument("Vector dimension mismatch");
    }

    if (metric_ == Metric::Cosine) {
      std::vector<float> tmp(vec.begin(), vec.end());
      normalize(tmp);
      data_.insert(data_.end(), tmp.begin(), tmp.end());
    } else {
      data_.insert(data_.end(), vec.begin(), vec.end());
    }
    ++size_;
  };

  static void normalize(std::span<float> vec) {
    float vec_len = 0.0f;
    for (size_t i = 0; i < vec.size(); i++) {
      vec_len += vec[i] * vec[i];
    }
    vec_len = std::sqrtf(vec_len);
    if (vec_len <= 0.0f)
      return;
    float inv = 1.0f / vec_len;
    for (size_t i = 0; i < vec.size(); ++i) {
      vec[i] *= inv;
    }
  }

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

    // For cosine: stored vecs are already normalized; normalize query once
    // here.
    std::vector<float> q_buf;
    std::span<const float> q = query;
    if (metric_ == Metric::Cosine) {
      q_buf.assign(query.begin(), query.end());
      normalize(q_buf);
      q = q_buf;
    }

    for (size_t i = 0; i < size_; ++i) {
      // Calculate start of current vector in the flat buffer
      std::span<const float> vec_i(data_.data() + (i * dim_), dim_);

      float score = metric_ == Metric::L2 ? compute_l2(q, vec_i)
                                          : -compute_dot(q, vec_i);

      if (pq.size() < k) {
        pq.push({i, score});
      } else if (score < pq.top().score) {
        // Found a closer neighbor than the worst one in our top-k
        pq.pop();
        pq.push({i, score});
      }
    }

    // Extract results (Worst -> Best)
    std::vector<SearchResult> results;
    results.reserve(pq.size());
    while (!pq.empty()) {
      float final_score = pq.top().score;
      if (metric_ == Metric::Cosine)
        final_score = -final_score;
      results.push_back({pq.top().id, final_score});
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
  Metric metric_;

  std::vector<float>
      data_; // Contiguous layout: [v0_0, v0_1, ..., v1_0, v1_1, ...]

  float compute_l2(std::span<const float> vec1,
                   std::span<const float> vec2) const {
    float dst_sum = 0;

    for (size_t i = 0; i < vec1.size(); ++i) {
      dst_sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }

    return dst_sum;
  }

  float compute_dot(std::span<const float> vec1,
                    std::span<const float> vec2) const {
    if (vec1.size() != vec2.size())
      throw std::runtime_error("size mismatch");

    float dot_prod = 0;
    for (size_t i = 0; i < vec1.size(); ++i) {
      dot_prod += vec1[i] * vec2[i];
    }

    return dot_prod;
  }
};