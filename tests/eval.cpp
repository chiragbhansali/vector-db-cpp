#include "flat_index.hpp"
#include "ivf_flat_index.hpp"
#include "ivf_pq.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

// Helper to generate random vectors
std::vector<std::vector<float>> generate_data(size_t n, size_t dim,
                                              uint32_t seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<std::vector<float>> vecs(n, std::vector<float>(dim));
  for (auto &v : vecs)
    for (auto &x : v)
      x = dist(rng);
  return vecs;
}

// Optimized Recall@K Calculator
float calculate_recall(const std::vector<SearchResult> &truth,
                       const std::vector<SearchResult> &candidates) {
  if (truth.empty())
    return 0.0f;

  std::unordered_set<size_t> truth_set;
  for (const auto &res : truth) {
    truth_set.insert(res.id);
  }

  size_t matches = 0;
  for (const auto &cand : candidates) {
    if (truth_set.find(cand.id) != truth_set.end()) {
      matches++;
    }
  }

  return static_cast<float>(matches) / truth.size();
}

// Flatten a 2D vector into a contiguous float buffer
std::vector<float> flatten(const std::vector<std::vector<float>> &vecs) {
  if (vecs.empty())
    return {};
  std::vector<float> flat;
  flat.reserve(vecs.size() * vecs[0].size());
  for (const auto &v : vecs)
    flat.insert(flat.end(), v.begin(), v.end());
  return flat;
}

// Evaluate IVFFlatIndex at a sweep of nprobe values for a given dataset.
// Prints a table with columns: nprobe | Recall@K | QPS
// build_ms is passed in and printed once before the table.
void eval_ivf_block(const std::string &label, size_t NB, size_t NQ, size_t K,
                    size_t DIM, size_t num_centroids,
                    const std::vector<size_t> &nprobe_levels,
                    const std::vector<std::vector<float>> &base_data,
                    const std::vector<std::vector<float>> &queries,
                    const std::vector<std::vector<SearchResult>> &all_truth) {
  auto flat_data = flatten(base_data);

  // Time train + insert
  auto build_start = std::chrono::high_resolution_clock::now();
  IVFFlatIndex ivf_idx(DIM, Metric::L2, /*nprobe=*/num_centroids,
                       num_centroids);
  ivf_idx.train(flat_data);
  for (const auto &v : base_data)
    ivf_idx.insert(v);
  auto build_end = std::chrono::high_resolution_clock::now();
  double build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_start)
          .count();

  std::cout << "\n[" << label << "] num_centroids=" << num_centroids
            << "  build_ms=" << build_ms << std::endl;
  std::cout << "  nprobe | Recall@" << K << " | QPS" << std::endl;
  std::cout << "  -------|---------|----" << std::endl;

  size_t last_nprobe = 0;
  for (size_t nprobe : nprobe_levels) {
    size_t effective_nprobe = std::min(nprobe, num_centroids);
    if (effective_nprobe == last_nprobe)
      continue; // already printed this value after clamping
    last_nprobe = effective_nprobe;

    float total_recall = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NQ; ++i) {
      auto results = ivf_idx.search(queries[i], K, effective_nprobe);
      total_recall += calculate_recall(all_truth[i], results);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    double qps = NQ / (elapsed.count() / 1000.0);
    std::cout << "  " << effective_nprobe << "      | " << (total_recall / NQ)
              << "    | " << qps << std::endl;
  }
}

void eval_ivfpq_block(const std::string &label, size_t NB, size_t NQ, size_t K,
                      size_t DIM, size_t num_centroids,
                      const std::vector<size_t> &nprobe_levels,
                      const std::vector<std::vector<float>> &base_data,
                      const std::vector<std::vector<float>> &queries,
                      const std::vector<std::vector<SearchResult>> &all_truth) {
  auto flat_data = flatten(base_data);

  auto build_start = std::chrono::high_resolution_clock::now();
  IVFPQIndex ivfpq_idx(DIM, Metric::L2, num_centroids, num_centroids, /*M=*/8, /*K=*/256);
  ivfpq_idx.train(flat_data);
  for (const auto &v : base_data)
    ivfpq_idx.insert(v);
  auto build_end = std::chrono::high_resolution_clock::now();
  double build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_start)
          .count();

  std::cout << "\n[" << label << "] num_centroids=" << num_centroids
            << "  build_ms=" << build_ms << std::endl;
  std::cout << "  nprobe | Recall@" << K << " | QPS" << std::endl;
  std::cout << "  -------|---------|----" << std::endl;

  size_t last_nprobe = 0;
  for (size_t nprobe : nprobe_levels) {
    size_t effective_nprobe = std::min(nprobe, num_centroids);
    if (effective_nprobe == last_nprobe)
      continue;
    last_nprobe = effective_nprobe;

    float total_recall = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NQ; ++i) {
      auto results = ivfpq_idx.search(queries[i], K, effective_nprobe);
      total_recall += calculate_recall(all_truth[i], results);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    double qps = NQ / (elapsed.count() / 1000.0);
    std::cout << "  " << effective_nprobe << "      | " << (total_recall / NQ)
              << "    | " << qps << std::endl;
  }
}

int main() {
  const size_t DIM = 128;
  const size_t NQ = 100; // Number of queries
  const size_t K = 10;

  // Logarithmically-spaced nprobe sweep values (clamped per block)
  const std::vector<size_t> NPROBE_SWEEP = {1,  2,   5,   10,  20,
                                            50, 100, 200, 500, 1000};

  // -------------------------------------------------------------------------
  // Block 1: N = 10k
  // -------------------------------------------------------------------------
  {
    const size_t NB = 10000;

    std::cout << "=== N=" << NB << " ===" << std::endl;
    std::cout << "Generating " << NB << " base vectors (dim=" << DIM << ")..."
              << std::endl;
    auto base_data = generate_data(NB, DIM, /*seed=*/42);
    auto queries = generate_data(NQ, DIM, /*seed=*/99);

    // Ground truth
    FlatIndex ground_truth_index(DIM, Metric::L2);
    for (const auto &v : base_data)
      ground_truth_index.insert(v);
    std::vector<std::vector<SearchResult>> all_truth;
    for (const auto &q : queries)
      all_truth.push_back(ground_truth_index.search(q, K));

    // FlatIndex baseline
    {
      FlatIndex fi(DIM, Metric::L2);
      for (const auto &v : base_data)
        fi.insert(v);

      float total_recall = 0;
      auto start = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < NQ; ++i) {
        auto results = fi.search(queries[i], K);
        total_recall += calculate_recall(all_truth[i], results);
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end - start;

      std::cout << "\n[FlatIndex] N=" << NB << std::endl;
      std::cout << "  Recall@" << K << ": " << (total_recall / NQ) << std::endl;
      std::cout << "  QPS: " << (NQ / (elapsed.count() / 1000.0)) << std::endl;
    }

    // IVF sweep: num_centroids in {50, 100, 200, 500}
    for (size_t nc : {50, 100, 200, 500}) {
      eval_ivf_block("IVFFlatIndex N=10k", NB, NQ, K, DIM, nc, NPROBE_SWEEP,
                     base_data, queries, all_truth);
    }

    for (size_t nc : {50, 100, 200, 500})
      eval_ivfpq_block("IVFPQIndex N=10k", NB, NQ, K, DIM, nc, NPROBE_SWEEP,
                       base_data, queries, all_truth);
  }

  // -------------------------------------------------------------------------
  // Block 2: N = 100k
  // -------------------------------------------------------------------------
  {
    const size_t NB = 100000;

    std::cout << "\n=== N=" << NB << " ===" << std::endl;
    std::cout << "Generating " << NB << " base vectors (dim=" << DIM << ")..."
              << std::endl;
    auto base_data = generate_data(NB, DIM, /*seed=*/42);
    auto queries = generate_data(NQ, DIM, /*seed=*/99);

    // Ground truth (FlatIndex exact search)
    FlatIndex ground_truth_index(DIM, Metric::L2);
    for (const auto &v : base_data)
      ground_truth_index.insert(v);
    std::vector<std::vector<SearchResult>> all_truth;
    for (const auto &q : queries)
      all_truth.push_back(ground_truth_index.search(q, K));

    // FlatIndex baseline
    {
      FlatIndex fi(DIM, Metric::L2);
      for (const auto &v : base_data)
        fi.insert(v);

      float total_recall = 0;
      auto start = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < NQ; ++i) {
        auto results = fi.search(queries[i], K);
        total_recall += calculate_recall(all_truth[i], results);
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end - start;

      std::cout << "\n[FlatIndex] N=" << NB << std::endl;
      std::cout << "  Recall@" << K << ": " << (total_recall / NQ) << std::endl;
      std::cout << "  QPS: " << (NQ / (elapsed.count() / 1000.0)) << std::endl;
    }

    // IVF sweep: num_centroids in {100, 200, 316, 500, 1000}  (316 ≈ √100k)
    for (size_t nc : {100, 200, 316, 500, 1000}) {
      eval_ivf_block("IVFFlatIndex N=100k", NB, NQ, K, DIM, nc, NPROBE_SWEEP,
                     base_data, queries, all_truth);
    }

    for (size_t nc : {100, 200, 316, 500, 1000})
      eval_ivfpq_block("IVFPQIndex N=100k", NB, NQ, K, DIM, nc, NPROBE_SWEEP,
                       base_data, queries, all_truth);
  }

  return 0;
}
