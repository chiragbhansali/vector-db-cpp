cmake_minimum_required(VERSION 3.16)
project(vector_db_cpp)

# Fix for dependencies using old CMake versions (like doctest asking for 3.0)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# --- Doctest ---
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)

# --- Google Benchmark ---
FetchContent_Declare(
  google_benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(google_benchmark)

# --- Library ---
add_library(vector_index include/flat_index.cpp include/kmeans.cpp include/ivf_flat_index.cpp include/ivf_pq.cpp include/product_quantizer.cpp)
target_include_directories(vector_index PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(vector_index INTERFACE cxx_std_20)

# --- Tests ---
add_executable(unit_tests tests/main.cpp)
target_link_libraries(unit_tests PRIVATE vector_index doctest::doctest)

add_executable(eval_runner tests/eval.cpp)
target_link_libraries(eval_runner PRIVATE vector_index)

# --- Benchmarks ---
add_executable(benchmarks benchmarks/main.cpp)
target_link_libraries(benchmarks PRIVATE vector_index benchmark::benchmark)
