#pragma once

#include "blockstride/scheduler.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace blockstride {

struct SimulationConfig {
    std::uint64_t prefill_fixed_us = 0;
    std::uint64_t prefill_per_token_us = 0;
    std::uint64_t decode_fixed_batch_us = 0;
    std::uint64_t decode_per_token_us = 0;
};

class SimulationCostModel {
public:
    explicit SimulationCostModel(SimulationConfig config);

    [[nodiscard]] std::uint64_t prefill_cost(std::size_t prompt_tokens) const;
    [[nodiscard]] std::uint64_t decode_batch_cost(std::size_t generated_tokens) const;

private:
    SimulationConfig config_;
};

class SimulationClock {
public:
    explicit SimulationClock(SimulatedTimestamp initial_timestamp = 0) noexcept;

    [[nodiscard]] SimulatedTimestamp now() const noexcept;
    [[nodiscard]] bool try_advance_by(std::uint64_t duration_us) noexcept;
    void advance_by(std::uint64_t duration_us);
    void advance_to(SimulatedTimestamp timestamp);

private:
    SimulatedTimestamp timestamp_;
};

enum class SimulationStatus {
    COMPLETED,
    EMPTY_WORKLOAD,
    MEMORY_BLOCKED,
    PERMANENTLY_UNSERVICEABLE,
    DEADLOCKED,
    TIME_OVERFLOW,
    NO_PROGRESS
};

[[nodiscard]] std::string_view simulation_status_name(SimulationStatus status) noexcept;

struct WorkloadRequest {
    RequestId id;
    std::size_t prompt_token_count;
    std::size_t max_new_tokens;
    SimulatedTimestamp arrival_timestamp;
};

struct RequestSimulationResult {
    RequestId request_id;
    SimulatedTimestamp arrival_timestamp;
    RequestState final_state = RequestState::WAITING;
    std::size_t generated_tokens = 0;
    std::optional<SimulatedTimestamp> start_timestamp;
    std::optional<SimulatedTimestamp> first_token_timestamp;
    std::optional<SimulatedTimestamp> completion_timestamp;
    std::optional<std::uint64_t> time_to_first_token_us;
    std::optional<std::uint64_t> completion_latency_us;
};

struct RejectedRequest {
    RequestId request_id;
    SubmissionResult reason;
};

struct SimulationStatistics {
    std::uint64_t scheduler_steps = 0;
    std::uint64_t prefill_operations = 0;
    std::uint64_t prefill_tokens = 0;
    std::uint64_t decode_batches = 0;
    std::uint64_t generated_tokens = 0;
    std::uint64_t completed_requests = 0;
    std::uint64_t cancelled_requests = 0;
    std::uint64_t prefill_execution_time_us = 0;
    std::uint64_t decode_execution_time_us = 0;
};

struct SimulationOutcome {
    SimulationStatus status = SimulationStatus::EMPTY_WORKLOAD;
    SimulatedTimestamp final_timestamp = 0;
    SimulationStatistics statistics;
    std::vector<RequestSimulationResult> requests;
    std::vector<RejectedRequest> rejected_requests;
    std::vector<RequestId> memory_blocked_requests;
    std::vector<RequestId> permanently_unserviceable_requests;
    std::vector<RequestId> outstanding_requests;
};

struct PolicySimulationResult {
    SchedulingPolicy policy;
    SimulationOutcome outcome;
};

class ExecutionSimulator {
public:
    ExecutionSimulator(SchedulerConfig scheduler_config, std::size_t total_physical_blocks,
                       std::size_t tokens_per_block, SimulationConfig simulation_config);

    [[nodiscard]] SubmissionResult submit(Request request);
    [[nodiscard]] std::vector<SubmissionResult> submit_workload(std::vector<Request> requests);
    void cancel(RequestId id);
    [[nodiscard]] SimulationOutcome run();

    [[nodiscard]] const SimulationClock& clock() const noexcept;
    [[nodiscard]] const Scheduler& scheduler() const noexcept;

private:
    [[nodiscard]] std::optional<SimulatedTimestamp> finish_prefill(std::size_t tokens);
    [[nodiscard]] std::optional<SimulatedTimestamp> finish_decode_batch(std::size_t tokens);
    [[nodiscard]] SimulationOutcome make_outcome(
        SimulationStatus status, const std::vector<RequestId>& memory_blocked,
        const std::vector<RequestId>& permanently_unserviceable) const;

    Scheduler scheduler_;
    SimulationCostModel cost_model_;
    SimulationClock clock_;
    SimulationStatistics statistics_;
    std::vector<RequestSimulationResult> request_results_;
    std::unordered_map<RequestId, std::size_t> request_result_indices_;
    std::vector<RejectedRequest> rejected_requests_;
};

[[nodiscard]] std::vector<PolicySimulationResult> compare_scheduling_policies(
    const std::vector<WorkloadRequest>& workload, SchedulerConfig scheduler_config,
    std::size_t total_physical_blocks, std::size_t tokens_per_block,
    SimulationConfig simulation_config);

}  // namespace blockstride
