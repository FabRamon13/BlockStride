#pragma once

#include "blockstride/block_manager.hpp"
#include "blockstride/request_manager.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

namespace blockstride {

enum class SchedulingPolicy {
    SEQUENTIAL,
    STATIC_BATCHING,
    CONTINUOUS_BATCHING
};

[[nodiscard]] std::string_view scheduling_policy_name(SchedulingPolicy policy) noexcept;

struct SchedulerConfig {
    std::size_t max_active_requests;
    std::size_t max_tokens_per_step;
    SchedulingPolicy policy = SchedulingPolicy::CONTINUOUS_BATCHING;
};

enum class SubmissionResult {
    ACCEPTED,
    PROMPT_EXCEEDS_STEP_TOKEN_BUDGET
};

enum class SchedulerStepStatus {
    PROGRESSED,
    EMPTY_WORKLOAD,
    WAITING_FOR_ARRIVAL,
    MEMORY_BLOCKED,
    PERMANENTLY_UNSERVICEABLE,
    DEADLOCKED,
    TIME_OVERFLOW
};

struct RequestAdmissionEvent {
    RequestId request_id;
    SimulatedTimestamp start_timestamp;
};

struct SchedulerStepResult {
    SchedulerStepStatus status = SchedulerStepStatus::EMPTY_WORKLOAD;
    std::vector<RequestAdmissionEvent> admission_events;
    std::vector<RequestId> admitted_requests;
    std::vector<RequestId> prefill_completed_requests;
    std::vector<RequestId> generated_tokens;
    std::vector<RequestId> completed_requests;
    std::vector<RequestId> cancelled_requests;
    std::vector<RequestId> memory_blocked_requests;
    std::vector<RequestId> permanently_unserviceable_requests;
    std::size_t tokens_processed = 0;
    std::size_t prefill_tokens_processed = 0;
    std::size_t decode_tokens_processed = 0;
    std::size_t decode_batch_count = 0;
    SimulatedTimestamp step_start_timestamp = 0;
    SimulatedTimestamp step_end_timestamp = 0;
    bool made_progress = false;
    bool time_overflow = false;
    bool future_arrivals_remain = false;
    std::optional<SimulatedTimestamp> next_arrival_timestamp;
};

class Scheduler {
public:
    using StageCompletionCallback =
        std::function<std::optional<SimulatedTimestamp>(std::size_t)>;

    Scheduler(SchedulerConfig config, std::size_t total_physical_blocks,
              std::size_t tokens_per_block);

    [[nodiscard]] SubmissionResult submit(Request request);
    [[nodiscard]] SchedulerStepResult step(SimulatedTimestamp timestamp);
    [[nodiscard]] SchedulerStepResult step(SimulatedTimestamp timestamp,
                                           const StageCompletionCallback& prefill_completion,
                                           const StageCompletionCallback& decode_batch_completion);
    void cancel(RequestId id, SimulatedTimestamp timestamp);

    [[nodiscard]] const RequestManager& request_manager() const noexcept;
    [[nodiscard]] const BlockManager& block_manager() const noexcept;
    [[nodiscard]] const SchedulerConfig& config() const noexcept;

private:
    [[nodiscard]] bool permanently_unserviceable(const Request& request) const noexcept;
    [[nodiscard]] bool active_work_can_progress() const;
    [[nodiscard]] std::optional<SimulatedTimestamp> next_arrival_after(
        SimulatedTimestamp timestamp) const;
    static void append_unique(std::vector<RequestId>& requests, RequestId id);

    SchedulerConfig config_;
    RequestManager request_manager_;
    BlockManager block_manager_;
    std::optional<SimulatedTimestamp> current_timestamp_;
    std::vector<RequestId> pending_cancelled_requests_;
    std::vector<RequestId> static_cohort_;
    bool static_cohort_active_ = false;
};

}  // namespace blockstride
