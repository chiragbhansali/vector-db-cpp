#pragma once
#include "vector_index.hpp"
#include <arm_neon.h>
#include <cmath>
#include <limits>
#include <queue>
#include <span>
#include <vector>

/// @brief Computes the squared L2 (Euclidean) distance between two vectors.
///
/// Squared distance is used throughout the hot path to avoid the cost of
/// `sqrt`, which is unnecessary for nearest-neighbor comparisons (ordering
/// is preserved under squaring).
///
/// @param vec1 First vector.
/// @param vec2 Second vector. Must have the same length as @p vec1.
/// @return Sum of squared element-wise differences: Σ(vec1[i] - vec2[i])².
inline float compute_l2(std::span<const float> vec1,
                        std::span<const float> vec2) {
  float dst_sum = 0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    float diff = (vec1[i] - vec2[i]);
    dst_sum += diff * diff;
  }
  return dst_sum;
}

/// @brief Returns the index of the centroid closest to @p vec under squared L2.
///
/// Used during IVF insert to assign a vector to its Voronoi cell, and during
/// KMeans training for the assignment step. Always uses L2 regardless of the
/// index metric — centroid assignment is a geometric partition, not a
/// similarity measure.
///
/// @param vec     Query vector of length @p dim.
/// @param centroids Flat buffer of @p k centroids laid out as
///                  [c0 | c1 | ... | c_{k-1}], each of length @p dim.
/// @param dim     Dimensionality of each centroid and of @p vec.
/// @return Index into @p centroids (0-based) of the nearest centroid.
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

/// @brief Returns the @p nprobe centroids nearest to @p vec under squared L2.
///
/// Used at IVF search time to select which inverted lists to probe. Maintains
/// a max-heap of size @p nprobe so that each centroid is considered in O(log
/// nprobe) time, giving an overall complexity of O(k · log nprobe) where k is
/// the total number of centroids.
///
/// If @p nprobe >= k the function returns all centroids (effectively a full
/// scan). Results are returned in unspecified order — callers that need sorted
/// order should sort the output themselves.
///
/// @param vec       Query vector of length @p dim.
/// @param centroids Flat centroid buffer (same layout as
/// find_closest_centroid).
/// @param dim       Dimensionality of each centroid and of @p vec.
/// @param nprobe    Number of nearest centroids to return.
/// @return Up to @p nprobe SearchResult entries, each holding the centroid
///         index as `id` and its squared L2 distance to @p vec as `score`.
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

/// @brief Computes the dot product (inner product) of two vectors.
///
/// Used as the raw similarity score for `Metric::InnerProduct`. For cosine
/// similarity, callers are expected to pre-normalize vectors at insert time so
/// that this dot product equals the cosine.
///
/// @param vec1 First vector.
/// @param vec2 Second vector. Must have the same length as @p vec1.
/// @return Σ vec1[i] · vec2[i].
inline float compute_dot(std::span<const float> vec1,
                         std::span<const float> vec2) {
  float dot_prod = 0;
  for (size_t i = 0; i < vec1.size(); ++i)
    dot_prod += vec1[i] * vec2[i];
  return dot_prod;
}

/// @brief SIMD-accelerated dot product using ARM NEON intrinsics.
///
/// Processes four floats per iteration via 128-bit NEON registers. Remainder
/// elements (when `n % 4 != 0`) are not accumulated — callers must ensure
/// vectors are padded to a multiple of 4, or use `compute_dot` for the tail.
///
/// @param a First vector.
/// @param b Second vector. Must have the same length as @p a.
/// @return Approximate dot product (may differ from scalar version by floating-
///         point rounding due to reordered additions).
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

/// @brief SIMD-accelerated squared L2 distance using ARM NEON intrinsics.
///
/// Processes four floats per iteration via 128-bit NEON registers. Same
/// remainder caveat as `compute_dot_simd` — elements beyond the last full
/// group of 4 are ignored.
///
/// @param a First vector.
/// @param b Second vector. Must have the same length as @p a.
/// @return Approximate squared L2 distance.
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

/// @brief Normalizes a vector to unit length in-place.
///
/// Divides each element by the L2 norm of the vector. After this call the
/// vector lies on the unit hypersphere, so a dot product against another
/// normalized vector equals their cosine similarity.
///
/// No-op if the vector is zero (norm ≤ 0) to avoid division by zero.
///
/// @param vec Vector to normalize. Modified in place.
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
