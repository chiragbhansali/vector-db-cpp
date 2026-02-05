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

  bool operator>(const SearchResult &other) const {
    return score > other.score;
  }
};

enum class Metric { L2, Cosine };

class FlatIndex {
public:
  explicit FlatIndex(size_t dimension, Metric metric)
      : dim_(dimension),
        aligned_dim_((dimension + 3) & ~static_cast<size_t>(3)),
        metric_(metric) {}
  void insert(std::span<const float> vec) {
    if (vec.size() != dim_) {
      throw std::invalid_argument("Vector dimension mismatch");
    }

    size_t start_idx = data_.size();

    // Resize with padding (aligned_dim_), initialized to 0.0f
    data_.resize(start_idx + aligned_dim_, 0.0f);

    // Copy the actual data into the first 'dim_' slots
    std::copy(vec.begin(), vec.end(), data_.begin() + start_idx);

    if (metric_ == Metric::Cosine) {
      std::span<float> inserted_vec(data_.data() + start_idx, aligned_dim_);
      normalize(inserted_vec);
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
  std::vector<SearchResult> search(std::span<const float> query, size_t k,
                                   bool use_simd = true) const {
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

    // Prepare a padded query buffer
    std::vector<float> q_padded(aligned_dim_, 0.0f);
    std::copy(query.begin(), query.end(), q_padded.begin());

    if (metric_ == Metric::Cosine) {
      normalize(q_padded);
    }

    for (size_t i = 0; i < size_; ++i) {
      // Calculate start of current vector in the flat buffer
      std::span<const float> vec_i(data_.data() + (i * aligned_dim_),
                                   aligned_dim_);

      float score;
      if (use_simd) {
        score = metric_ == Metric::L2 ? compute_l2_simd(q_padded, vec_i)
                                      : -compute_dot_simd(q_padded, vec_i);
      } else {
        score = metric_ == Metric::L2 ? compute_l2(q_padded, vec_i)
                                      : -compute_dot(q_padded, vec_i);
      }

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
  size_t aligned_dim_;
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

  float compute_dot_simd(std::span<const float> a,
                         std::span<const float> b) const {
    size_t n = a.size();
    float32x4_t sum_vec = vdupq_n_f32(
        0.0f); // 1. Initialize a vector of 4 zeros to hold partial sums
    size_t i = 0;

    // 2. Main loop: process 4 floats at a time
    for (; i + 3 < n; i += 4) {
      float32x4_t va = vld1q_f32(&a[i]);
      float32x4_t vb = vld1q_f32(&b[i]);

      // Fused Multiply-Add: sum_vec += va * vb
      sum_vec = vfmaq_f32(sum_vec, va, vb);
    }

    // 3. Horizontal sum: add the 4 floats inside sum_vec together
    return vaddvq_f32(sum_vec);
  }

  float compute_l2_simd(std::span<const float> a,
                        std::span<const float> b) const {
    size_t n = a.size();
    float32x4_t diff_sq_sum = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 3 < n; i += 4) {
      float32x4_t va = vld1q_f32(&a[i]);
      float32x4_t vb = vld1q_f32(&b[i]);

      // va = va - vb
      float32x4_t diff = vsubq_f32(va, vb);

      // diff_sq_sum += diff * diff
      diff_sq_sum = vfmaq_f32(diff_sq_sum, diff, diff);
    }

    return vaddvq_f32(diff_sq_sum);
  }
};