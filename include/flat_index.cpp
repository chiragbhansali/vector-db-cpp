#include "flat_index.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <queue>

FlatIndex::FlatIndex(size_t dimension, Metric metric)
    : dim_(dimension), aligned_dim_((dimension + 3) & ~static_cast<size_t>(3)),
      metric_(metric) {}

void FlatIndex::train(std::span<const float> /*data*/) {
  // FlatIndex does not require training
}

void FlatIndex::insert(std::span<const float> vec) {
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
}

void FlatIndex::normalize(std::span<float> vec) {
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

std::vector<SearchResult> FlatIndex::search(std::span<const float> query,
                                            size_t k) const {
  return search(query, k, true);
}

std::vector<SearchResult> FlatIndex::search(std::span<const float> query,
                                            size_t k, bool use_simd) const {
  if (query.size() != dim_) {
    throw std::invalid_argument("Query dimension mismatch");
  }
  if (k == 0)
    return {};

  auto comp = [](const SearchResult &a, const SearchResult &b) {
    return a.score < b.score;
  };
  std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(comp)>
      pq(comp);

  std::vector<float> q_padded(aligned_dim_, 0.0f);
  std::copy(query.begin(), query.end(), q_padded.begin());

  if (metric_ == Metric::Cosine) {
    normalize(q_padded);
  }

  for (size_t i = 0; i < size_; ++i) {
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
      pq.pop();
      pq.push({i, score});
    }
  }

  std::vector<SearchResult> results;
  results.reserve(pq.size());
  while (!pq.empty()) {
    float final_score = pq.top().score;
    if (metric_ == Metric::Cosine)
      final_score = -final_score;
    results.push_back({pq.top().id, final_score});
    pq.pop();
  }

  std::reverse(results.begin(), results.end());
  return results;
}

void FlatIndex::save(const std::string &path) const {
  std::ofstream os(path, std::ios::binary);
  if (!os)
    throw std::runtime_error("Failed to open file for saving: " + path);

  os.write(reinterpret_cast<const char *>(&dim_), sizeof(dim_));
  os.write(reinterpret_cast<const char *>(&size_), sizeof(size_));
  os.write(reinterpret_cast<const char *>(&aligned_dim_), sizeof(aligned_dim_));
  os.write(reinterpret_cast<const char *>(&metric_), sizeof(metric_));
  os.write(reinterpret_cast<const char *>(data_.data()),
           data_.size() * sizeof(float));
}

void FlatIndex::load(const std::string &path) {
  std::ifstream is(path, std::ios::binary);
  if (!is)
    throw std::runtime_error("Failed to open file for loading: " + path);

  is.read(reinterpret_cast<char *>(&dim_), sizeof(dim_));
  is.read(reinterpret_cast<char *>(&size_), sizeof(size_));
  is.read(reinterpret_cast<char *>(&aligned_dim_), sizeof(aligned_dim_));
  is.read(reinterpret_cast<char *>(&metric_), sizeof(metric_));

  data_.resize(size_ * aligned_dim_);
  is.read(reinterpret_cast<char *>(data_.data()), data_.size() * sizeof(float));
}

size_t FlatIndex::size() const { return size_; }
size_t FlatIndex::dimension() const { return dim_; }

float FlatIndex::compute_l2(std::span<const float> vec1,
                            std::span<const float> vec2) const {
  float dst_sum = 0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    dst_sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
  }
  return dst_sum;
}

float FlatIndex::compute_dot(std::span<const float> vec1,
                             std::span<const float> vec2) const {
  float dot_prod = 0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    dot_prod += vec1[i] * vec2[i];
  }
  return dot_prod;
}

float FlatIndex::compute_dot_simd(std::span<const float> a,
                                  std::span<const float> b) const {
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

float FlatIndex::compute_l2_simd(std::span<const float> a,
                                 std::span<const float> b) const {
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
