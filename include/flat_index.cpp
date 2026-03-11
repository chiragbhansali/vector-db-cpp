#include "flat_index.hpp"
#include "distance.hpp"
#include <algorithm>
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
