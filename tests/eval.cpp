#include "flat_index.hpp"
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

// Helper to generate random vectors
std::vector<std::vector<float>> generate_data(size_t n, size_t dim) {
  std::mt19937 rng(42);
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

int main() {
  const size_t DIM = 128;
  const size_t NB = 10000; // Database size
  const size_t NQ = 100;   // Number of queries
  const size_t K = 10;

  std::cout << "--- Vector Index Evaluation Rig ---" << std::endl;
  std::cout << "Generating " << NB << " vectors of dimension " << DIM << "..."
            << std::endl;
  auto base_data = generate_data(NB, DIM);
  auto queries = generate_data(NQ, DIM);

  // 1. Setup Ground Truth
  FlatIndex ground_truth_index(DIM, Metric::L2);
  for (const auto &v : base_data)
    ground_truth_index.insert(v);

  std::cout << "Computing ground truth for " << NQ << " queries..."
            << std::endl;
  std::vector<std::vector<SearchResult>> all_truth;
  for (const auto &q : queries) {
    all_truth.push_back(ground_truth_index.search(q, K));
  }

  // 2. Setup Candidate Index (Polymorphic unique_ptr)
  std::unique_ptr<VectorIndex> candidate_index =
      std::make_unique<FlatIndex>(DIM, Metric::L2);

  std::cout << "Inserting " << NB << " vectors into candidate index..."
            << std::endl;
  for (const auto &v : base_data)
    candidate_index->insert(v);

  std::cout << "Evaluating candidate index..." << std::endl;
  float total_recall = 0;

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < NQ; ++i) {
    auto results = candidate_index->search(queries[i], K);
    total_recall += calculate_recall(all_truth[i], results);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;

  std::cout << "\nResults:" << std::endl;
  std::cout << "  Average Recall@" << K << ": " << (total_recall / NQ)
            << std::endl;
  std::cout << "  Average Latency: " << (elapsed.count() / NQ) << " ms/query"
            << std::endl;
  std::cout << "  Total Time: " << elapsed.count() << " ms" << std::endl;

  return 0;
}