#include "ivf_flat_index.hpp"
#include "distance.hpp"
#include "kmeans.hpp"
#include <fstream>
#include <span>
#include <vector>

IVFFlatIndex::IVFFlatIndex(size_t dimension, Metric metric, size_t nprobe,
                           size_t num_centroids)
    : dim_(dimension), metric_(metric), nprobe_(nprobe),
      num_centroids_(num_centroids) {}

void IVFFlatIndex::train(std::span<const float> data) {
  KMeans cluster;
  centroids_ = cluster.train(data, dim_, KMeans::Config{num_centroids_, 100});
  inverted_lists_.assign(num_centroids_, {});
  size_ = 0;
}

void IVFFlatIndex::insert(std::span<const float> vec) {
  if (inverted_lists_.empty()) {
    throw std::runtime_error("Index must be trained before inserting");
  }
  size_t best_centroid = find_closest_centroid(vec, centroids_, dim_);
  std::vector<float> vec_modified(vec.begin(),
                                  vec.end()); // convert span to vector
  inverted_lists_[best_centroid].push_back(
      Entry{size_, std::move(vec_modified)});
  ++size_;
}

std::vector<SearchResult> IVFFlatIndex::search(std::span<const float> query,
                                               size_t k) const {
  return search(query, k, nprobe_);
}

std::vector<SearchResult> IVFFlatIndex::search(std::span<const float> query,
                                               size_t k, size_t nprobe) const {
  if (nprobe == 0 || k == 0)
    return {};

  std::vector<SearchResult> n_centroids =
      find_closest_centroids(query, centroids_, dim_, nprobe);
  auto comp = [](const SearchResult &a, const SearchResult &b) {
    return a.score < b.score;
  };
  std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(comp)>
      pq(comp);

  for (size_t i = 0; i < n_centroids.size(); ++i) {
    for (const Entry &entry : inverted_lists_[n_centroids[i].id]) {
      float score = (metric_ == Metric::InnerProduct)
                        ? -compute_dot(entry.vec, query)
                        : compute_l2(entry.vec, query);

      if (pq.size() < k) {
        pq.push({entry.id, score});
      } else if (score < pq.top().score) {
        pq.pop();
        pq.push({entry.id, score});
      }
    }
  }

  std::vector<SearchResult> results;
  results.reserve(pq.size());
  while (!pq.empty()) {
    float final_score = pq.top().score;
    if (metric_ == Metric::InnerProduct)
      final_score = -final_score;
    results.push_back({pq.top().id, final_score});
    pq.pop();
  }

  std::reverse(results.begin(), results.end());

  return results;
}

size_t IVFFlatIndex::size() const { return size_; }
size_t IVFFlatIndex::dimension() const { return dim_; }

void IVFFlatIndex::save(const std::string &path) const {
  std::ofstream os(path, std::ios::binary);
  if (!os)
    throw std::runtime_error("Failed to open file for saving: " + path);

  os.write(reinterpret_cast<const char *>(&dim_), sizeof(dim_));
  os.write(reinterpret_cast<const char *>(&size_), sizeof(size_));
  os.write(reinterpret_cast<const char *>(&nprobe_), sizeof(nprobe_));
  os.write(reinterpret_cast<const char *>(&num_centroids_),
           sizeof(num_centroids_));
  os.write(reinterpret_cast<const char *>(&metric_), sizeof(metric_));
  os.write(reinterpret_cast<const char *>(centroids_.data()),
           centroids_.size() * sizeof(float));
  for (const std::vector<Entry> &inverted_list : inverted_lists_) {
    size_t list_size = inverted_list.size();
    os.write(reinterpret_cast<const char *>(&list_size), sizeof(list_size));
    for (const Entry &entry : inverted_list) {
      os.write(reinterpret_cast<const char *>(&entry.id), sizeof(entry.id));
      os.write(reinterpret_cast<const char *>(entry.vec.data()),
               entry.vec.size() * sizeof(float));
    }
  }
}

void IVFFlatIndex::load(const std::string &path) {
  std::ifstream is(path, std::ios::binary);
  if (!is)
    throw std::runtime_error("Failed to open file for loading: " + path);

  is.read(reinterpret_cast<char *>(&dim_), sizeof(dim_));
  is.read(reinterpret_cast<char *>(&size_), sizeof(size_));
  is.read(reinterpret_cast<char *>(&nprobe_), sizeof(nprobe_));
  is.read(reinterpret_cast<char *>(&num_centroids_), sizeof(num_centroids_));
  is.read(reinterpret_cast<char *>(&metric_), sizeof(metric_));

  centroids_.resize(num_centroids_ * dim_);
  is.read(reinterpret_cast<char *>(centroids_.data()),
          centroids_.size() * sizeof(float));

  inverted_lists_.resize(num_centroids_);
  for (size_t i = 0; i < num_centroids_; ++i) {
    size_t list_size;
    is.read(reinterpret_cast<char *>(&list_size), sizeof(list_size));
    inverted_lists_[i].resize(list_size);
    for (Entry &entry : inverted_lists_[i]) {
      is.read(reinterpret_cast<char *>(&entry.id), sizeof(entry.id));
      entry.vec.resize(dim_);
      is.read(reinterpret_cast<char *>(entry.vec.data()), dim_ * sizeof(float));
    }
  }
}