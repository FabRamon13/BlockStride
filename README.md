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

The FCFS request manager rejects duplicate IDs, orders requests by arrival time with
stable submission order for ties, and admits only requests that have arrived
at the current simulated time. It tracks waiting, running, and terminal
requests and supports cancellation. It remains independent of token budgets
and KV-cache capacity; those decisions belong to the scheduler.

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

## Scheduling policies

The shared deterministic scheduling engine offers three policies, selected
through `SchedulerConfig::policy` (default: `CONTINUOUS_BATCHING`):

- **Sequential** processes one request at a time in FCFS order, completing its
  prefill and all generation before starting the next request.
- **Static batching** forms an FCFS cohort at an admission boundary, up to the
  configured active limit and currently available prompt-block capacity. No
  replacement enters after cohort formation, even if a member completes
  early. The next cohort forms only after every member completes or is
  cancelled. Arrivals during a cohort wait for that next cohort boundary.
- **Continuous batching** admits eligible FCFS requests between execution
  steps and admits replacements as active slots and KV blocks become available.
  A memory-blocked request does not prevent other active requests from
  progressing when they can.

All policies share the configured active-request limit (sequential applies the
stricter one-request limit), per-step prompt-plus-decode token budget, KV-block
capacity and size, request lifecycle, arrival handling, and cost model. A
blocked oldest eligible request is not bypassed. Requests that cannot ever fit
are reported as permanently unserviceable; capacity-induced deadlock is
reported rather than resolved by changing limits or cancelling requests.

`compare_scheduling_policies` accepts one immutable workload description and
creates independent request-manager, block-manager, and clock state for each
policy. The `blockstride_policy_demo` executable prints a small synthetic
comparison; it is not the final benchmark suite.

## Deterministic execution simulator

The discrete-event simulator coordinates the scheduler and a simulated clock.
It serializes prefill and decode stages and advances time only by configured
synthetic costs or directly to the next future arrival; it never sleeps or
uses wall-clock time. Arrival events during a scheduler step are considered at
the next scheduler boundary, not in the middle of an executing batch.

Each request's prefill is one operation with cost
`prefill_fixed_us + prefill_per_token_us * prompt_tokens`. Decode work is
grouped into round-robin batches with cost
`decode_fixed_batch_us + decode_per_token_us * generated_tokens_in_batch`;
the fixed decode cost is charged once per batch. Zero-work operations cost
nothing. The existing per-step token budget counts both prompt tokens in
prefill and generated tokens in decode.

A request starts when it is admitted for processing. Prefill completes before
decode begins. First-token time is recorded at the end of the decode batch
that produces the first token, and completion time is recorded at the end of
the final decode batch. Time-to-first-token and completion latency are measured
from the request's arrival timestamp. Temporary KV-block pressure is reported
while other work is allowed to progress; permanent unserviceability,
deadlock, and simulated-time overflow produce explicit outcomes. All modeled
costs are synthetic: no LLM inference or GPU work is executed.
These simulated times do not measure CPU scheduling or execution performance,
and policy results must not be interpreted as measured speedups.

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
./build/blockstride_policy_demo
ctest --test-dir build --output-on-failure
```

To enable AddressSanitizer and UndefinedBehaviorSanitizer on supported GCC or
Clang toolchains:

```sh
cmake -S . -B build-sanitize -DBLOCKSTRIDE_ENABLE_SANITIZERS=ON
cmake --build build-sanitize
ctest --test-dir build-sanitize --output-on-failure
```
