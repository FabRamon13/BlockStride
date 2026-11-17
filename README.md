# BlockStride

BlockStride is a standalone C++20 simulator project for exploring LLM inference
scheduling concepts. It is intended to demonstrate continuous batching,
metadata-level paged KV-cache management, request lifecycles, deterministic
execution simulation, and reproducible benchmarking. It does not run LLM
inference or require a GPU.

## Request lifecycle

Requests move through `WAITING`, `RUNNING`, and one terminal state:
`COMPLETED` or `CANCELLED`. The valid transitions are `WAITING` to `RUNNING`
or `CANCELLED`, and `RUNNING` to `COMPLETED` or `CANCELLED`. Generation requires
completed prefill, cannot exceed the configured token limit, and completion
requires that limit to be reached.
Both successful completion and cancellation record a terminal timestamp, while
the request state distinguishes the terminal outcome.

The FCFS request manager rejects duplicate IDs, orders requests by arrival time
with stable submission order for ties, and admits only requests that have
arrived at the current simulated time. It tracks waiting, running, and terminal
requests and supports cancellation. It does not schedule based on token budgets
or KV-cache capacity; a scheduler is not implemented yet.

## Paged KV-cache block manager

The block manager tracks request ownership of a bounded pool of fixed-size
logical token blocks. Configure the total physical block capacity and tokens
per block; sequence lengths are rounded up to the number of blocks required.
Requests can grow their allocation as total sequence length increases, and
release returns all their blocks to the reusable pool. Capacity exhaustion is
reported as a recoverable allocation result, without changing an existing
allocation.

This is metadata-only simulation: it does not allocate GPU memory, KV tensors,
or implement attention or scheduling.

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
