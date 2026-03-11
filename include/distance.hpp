#pragma once
#include "vector_index.hpp"
#include <arm_neon.h>
#include <cmath>
#include <limits>
#include <queue>
#include <span>
#include <vector>

inline float compute_l2(std::span<const float> vec1,
                        std::span<const float> vec2) {
  float dst_sum = 0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    float diff = (vec1[i] - vec2[i]);
    dst_sum += diff * diff;
  }
  return dst_sum;
}

inline size_t find_closest_centroid(std::span<const float> vec,
                                    std::span<const float> centroids,
                                    size_t dim) {
  float min_dist = std::numeric_limits<float>::max();
  size_t best_centroid = 0;
  size_t k = centroids.size() / dim;

  for (size_t c = 0; c < k; ++c) {
    std::span<const float> centroid(centroids.data() + c * dim, dim);
    float dist = compute_l2(vec, centroid);
    if (dist < min_dist) {
      min_dist = dist;
      best_centroid = c;
    }
  }
  return best_centroid;
}

inline std::vector<SearchResult>
find_closest_centroids(std::span<const float> vec,
                       std::span<const float> centroids, size_t dim,
                       size_t nprobe) {
  auto comp = [](const SearchResult &a, const SearchResult &b) {
    return a.score < b.score;
  };
  std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(comp)>
      pq(comp);

  for (size_t i = 0; i < centroids.size() / dim; ++i) {
    std::span<const float> centroid(centroids.data() + (i * dim), dim);

    float score = compute_l2(vec, centroid);
    if (pq.size() < nprobe) {
      pq.push({i, score});
    } else if (score < pq.top().score) {
      pq.pop();
      pq.push({i, score});
    }
  }

  std::vector<SearchResult> results;
  results.reserve(pq.size());
  while (!pq.empty()) {
    results.push_back({pq.top().id, pq.top().score});
    pq.pop();
  }

  return results;
}

inline float compute_dot(std::span<const float> vec1,
                         std::span<const float> vec2) {
  float dot_prod = 0;
  for (size_t i = 0; i < vec1.size(); ++i)
    dot_prod += vec1[i] * vec2[i];
  return dot_prod;
}

inline float compute_dot_simd(std::span<const float> a,
                               std::span<const float> b) {
  size_t n = a.size();
  float32x4_t sum_vec = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 3 < n; i += 4) {
    float32x4_t va = vld1q_f32(&a[i]);
    float32x4_t vb = vld1q_f32(&b[i]);
    sum_vec = vfmaq_f32(sum_vec, va, vb);
  }
  return vaddvq_f32(sum_vec);
}

inline float compute_l2_simd(std::span<const float> a,
                              std::span<const float> b) {
  size_t n = a.size();
  float32x4_t diff_sq_sum = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 3 < n; i += 4) {
    float32x4_t va = vld1q_f32(&a[i]);
    float32x4_t vb = vld1q_f32(&b[i]);
    float32x4_t diff = vsubq_f32(va, vb);
    diff_sq_sum = vfmaq_f32(diff_sq_sum, diff, diff);
  }
  return vaddvq_f32(diff_sq_sum);
}

inline void normalize(std::span<float> vec) {
  float vec_len = 0.0f;
  for (float x : vec)
    vec_len += x * x;
  vec_len = std::sqrtf(vec_len);
  if (vec_len <= 0.0f)
    return;
  float inv = 1.0f / vec_len;
  for (float &x : vec)
    x *= inv;
}