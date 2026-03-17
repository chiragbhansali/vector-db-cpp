#include "ivf_pq.hpp"
#include "distance.hpp"
#include "kmeans.hpp"
#include <fstream>
#include <span>
#include <vector>

IVFPQIndex::IVFPQIndex(size_t dimension, Metric metric, size_t nprobe,
                       size_t num_centroids)
    : dim_(dimension), metric_(metric), nprobe_(nprobe),
      num_centroids_(num_centroids), size_(0), pq_(M_, K_, dim_) {}

void IVFPQIndex::train(std::span<const float> data) {
  KMeans cluster;
  centroids_ = cluster.train(data, dim_, KMeans::Config{num_centroids_, 100});
  inverted_lists_.assign(num_centroids_, {});
  pq_.train(data);
  size_ = 0;
}

void IVFPQIndex::insert(std::span<const float> vec) {
  if (inverted_lists_.empty()) {
    throw std::runtime_error("Index must be trained before inserting");
  }
  size_t best_centroid = find_closest_centroid(vec, centroids_, dim_);
  std::vector<uint8_t> vec_centroids = pq_.encode(vec);
  inverted_lists_[best_centroid].push_back(PQEntry{size_, vec_centroids});
  ++size_;
}

std::vector<SearchResult> IVFPQIndex::search(std::span<const float> query,
                                             size_t k) const {
  return search(query, k, nprobe_);
}

std::vector<SearchResult> IVFPQIndex::search(std::span<const float> query,
                                             size_t k, size_t nprobe) const {
  if (nprobe == 0 || k == 0)
    return {};

  std::vector<SearchResult> n_centroids =
      find_closest_centroids(query, centroids_, dim_, nprobe);
  std::vector<float> adc_table = pq_.compute_distance_table(query);
  auto comp = [](const SearchResult &a, const SearchResult &b) {
    return a.score < b.score;
  };
  std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(comp)>
      pq(comp);

  for (size_t i = 0; i < n_centroids.size(); ++i) {
    for (const PQEntry &entry : inverted_lists_[n_centroids[i].id]) {
      float score = 0;
      for (size_t j = 0; j < entry.vec_centroid_codes.size(); ++j) {
        score += adc_table[K_ * j + entry.vec_centroid_codes[j]];
      }

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
    results.push_back({pq.top().id, pq.top().score});
    pq.pop();
  }

  std::reverse(results.begin(), results.end());

  return results;
}

size_t IVFPQIndex::size() const { return size_; }
size_t IVFPQIndex::dimension() const { return dim_; }

void IVFPQIndex::save(const std::string &path) const {
  std::ofstream os(path, std::ios::binary);
  if (!os)
    throw std::runtime_error("Failed to open file for saving: " + path);

  os.write(reinterpret_cast<const char *>(&dim_), sizeof(dim_));
  os.write(reinterpret_cast<const char *>(&size_), sizeof(size_));
  os.write(reinterpret_cast<const char *>(&nprobe_), sizeof(nprobe_));
  os.write(reinterpret_cast<const char *>(&num_centroids_),
           sizeof(num_centroids_));
  os.write(reinterpret_cast<const char *>(&metric_), sizeof(metric_));
  os.write(reinterpret_cast<const char *>(&K_), sizeof(K_));
  os.write(reinterpret_cast<const char *>(&M_), sizeof(M_));
  os.write(reinterpret_cast<const char *>(centroids_.data()),
           centroids_.size() * sizeof(float));
  std::vector<float> codebooks = pq_.get_codebooks();
  os.write(reinterpret_cast<const char *>(codebooks.data()),
           codebooks.size() * sizeof(float));
  for (const std::vector<PQEntry> &inverted_list : inverted_lists_) {
    size_t list_size = inverted_list.size();
    os.write(reinterpret_cast<const char *>(&list_size), sizeof(list_size));
    for (const PQEntry &entry : inverted_list) {
      os.write(reinterpret_cast<const char *>(&entry.id), sizeof(entry.id));
      os.write(reinterpret_cast<const char *>(entry.vec_centroid_codes.data()),
               entry.vec_centroid_codes.size() * sizeof(uint8_t));
    }
  }
}

void IVFPQIndex::load(const std::string &path) {
  std::ifstream is(path, std::ios::binary);
  if (!is)
    throw std::runtime_error("Failed to open file for loading: " + path);

  is.read(reinterpret_cast<char *>(&dim_), sizeof(dim_));
  is.read(reinterpret_cast<char *>(&size_), sizeof(size_));
  is.read(reinterpret_cast<char *>(&nprobe_), sizeof(nprobe_));
  is.read(reinterpret_cast<char *>(&num_centroids_), sizeof(num_centroids_));
  is.read(reinterpret_cast<char *>(&metric_), sizeof(metric_));
  is.read(reinterpret_cast<char *>(&K_), sizeof(K_));
  is.read(reinterpret_cast<char *>(&M_), sizeof(M_));

  centroids_.resize(num_centroids_ * dim_);
  is.read(reinterpret_cast<char *>(centroids_.data()),
          centroids_.size() * sizeof(float));
  std::vector<float> codebooks;
  codebooks.resize(K_ * dim_);
  is.read(reinterpret_cast<char *>(codebooks.data()),
          codebooks.size() * sizeof(float));
  pq_.set_codebooks(codebooks);

  inverted_lists_.resize(num_centroids_);
  for (size_t i = 0; i < num_centroids_; ++i) {
    size_t list_size;
    is.read(reinterpret_cast<char *>(&list_size), sizeof(list_size));
    inverted_lists_[i].resize(list_size);
    for (PQEntry &entry : inverted_lists_[i]) {
      is.read(reinterpret_cast<char *>(&entry.id), sizeof(entry.id));
      entry.vec_centroid_codes.resize(M_);
      is.read(reinterpret_cast<char *>(entry.vec_centroid_codes.data()),
              M_ * sizeof(uint8_t));
    }
  }
}