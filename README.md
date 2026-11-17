# BlockStride

BlockStride is a standalone C++20 simulator project for exploring LLM inference
scheduling concepts. It is intended to demonstrate continuous batching,
metadata-level paged KV-cache management, request lifecycles, deterministic
execution simulation, and reproducible benchmarking. It does not run LLM
inference or require a GPU.

## Dependencies

- CMake 3.20 or newer
- A C++20 compiler (Apple Clang, Clang, GCC, or MSVC)
- GoogleTest 1.15.2 for tests; CMake fetches it if it is not already available

## Build and test

Run these commands from the repository root:

```sh
cmake -S . -B build
cmake --build build
./build/blockstride_bench
ctest --test-dir build --output-on-failure
```

To enable AddressSanitizer and UndefinedBehaviorSanitizer on supported GCC or
Clang toolchains:

```sh
cmake -S . -B build-sanitize -DBLOCKSTRIDE_ENABLE_SANITIZERS=ON
cmake --build build-sanitize
ctest --test-dir build-sanitize --output-on-failure
```
