#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <vector>

namespace {

constexpr size_t kBytes = 64u * 1024u * 1024u;
constexpr int kBatchRepeats = 50;
constexpr double kDefaultTotalSeconds = 10.0;
constexpr double kMinimumSecondsPerPair = 0.25;

bool check_cuda(cudaError_t status, const char* message) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(status));
  return false;
}

double parse_total_seconds(int argc, char** argv) {
  double seconds = kDefaultTotalSeconds;
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--seconds") == 0 || std::strcmp(argv[index], "-s") == 0) {
      if (index + 1 < argc) {
        seconds = std::atof(argv[index + 1]);
        ++index;
      }
      continue;
    }
    if (std::strncmp(argv[index], "--seconds=", 10) == 0) {
      seconds = std::atof(argv[index] + 10);
    }
  }
  return seconds > 0.0 ? seconds : kDefaultTotalSeconds;
}

double benchmark_copy(int src_gpu, int dst_gpu, void* src_ptr, void* dst_ptr, double target_seconds) {
  if (!check_cuda(cudaSetDevice(dst_gpu), "cudaSetDevice(dst)")) {
    return 0.0;
  }

  auto started = std::chrono::steady_clock::now();
  int total_repeats = 0;
  double elapsed = 0.0;
  do {
    for (int repeat = 0; repeat < kBatchRepeats; ++repeat) {
      if (!check_cuda(cudaMemcpyPeer(dst_ptr, dst_gpu, src_ptr, src_gpu, kBytes), "cudaMemcpyPeer")) {
        return 0.0;
      }
    }
    total_repeats += kBatchRepeats;
    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
      return 0.0;
    }

    const auto ended = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(ended - started).count();
  } while (elapsed < target_seconds);
  if (elapsed <= 0.0) {
    return 0.0;
  }
  return (static_cast<double>(kBytes) * total_repeats) / elapsed / 1.0e9;
}

}  // namespace

int main(int argc, char** argv) {
  int device_count = 0;
  if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
    return 1;
  }
  if (device_count <= 0) {
    std::fprintf(stderr, "No CUDA devices detected\n");
    return 1;
  }
  const double total_seconds = parse_total_seconds(argc, argv);

  std::vector<void*> buffers(device_count, nullptr);
  for (int gpu_index = 0; gpu_index < device_count; ++gpu_index) {
    if (!check_cuda(cudaSetDevice(gpu_index), "cudaSetDevice")) {
      return 1;
    }
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers[gpu_index]), kBytes), "cudaMalloc")) {
      return 1;
    }
  }

  std::vector<std::pair<int, int>> enabled_pairs;
  for (int src = 0; src < device_count; ++src) {
    for (int dst = 0; dst < device_count; ++dst) {
      if (src == dst) {
        continue;
      }
      int can_access = 0;
      if (!check_cuda(cudaDeviceCanAccessPeer(&can_access, dst, src), "cudaDeviceCanAccessPeer")) {
        return 1;
      }
      if (!can_access) {
        continue;
      }
      enabled_pairs.emplace_back(src, dst);
      if (!check_cuda(cudaSetDevice(dst), "cudaSetDevice(enable peer)")) {
        return 1;
      }
      cudaError_t access_status = cudaDeviceEnablePeerAccess(src, 0);
      if (access_status != cudaSuccess && access_status != cudaErrorPeerAccessAlreadyEnabled) {
        std::fprintf(stderr, "cudaDeviceEnablePeerAccess(%d->%d): %s\n", src, dst,
                     cudaGetErrorString(access_status));
        return 1;
      }
    }
  }

  const double seconds_per_pair = enabled_pairs.empty()
                                      ? total_seconds
                                      : std::max(total_seconds / enabled_pairs.size(), kMinimumSecondsPerPair);
  std::vector<std::vector<double>> matrix(device_count, std::vector<double>(device_count, 0.0));
  for (int src = 0; src < device_count; ++src) {
    for (int dst = 0; dst < device_count; ++dst) {
      if (src == dst) {
        continue;
      }
      int can_access = 0;
      cudaDeviceCanAccessPeer(&can_access, dst, src);
      if (!can_access) {
        continue;
      }
      matrix[src][dst] = benchmark_copy(src, dst, buffers[src], buffers[dst], seconds_per_pair);
    }
  }

  std::printf("Requested Test Duration (s): %.2f\n", total_seconds);
  std::printf("Bandwidth Matrix (GB/s)\n");
  std::printf("D\\D");
  for (int gpu_index = 0; gpu_index < device_count; ++gpu_index) {
    std::printf(" %d", gpu_index);
  }
  std::printf("\n");

  for (int src = 0; src < device_count; ++src) {
    std::printf("%d", src);
    for (int dst = 0; dst < device_count; ++dst) {
      std::printf(" %.2f", matrix[src][dst]);
    }
    std::printf("\n");
  }

  for (int gpu_index = 0; gpu_index < device_count; ++gpu_index) {
    cudaSetDevice(gpu_index);
    cudaFree(buffers[gpu_index]);
  }

  return 0;
}
