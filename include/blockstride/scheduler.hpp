#pragma once

#include "blockstride/block_manager.hpp"
#include "blockstride/request_manager.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace blockstride {

struct SchedulerConfig {
    std::size_t max_active_requests;
    std::size_t max_tokens_per_step;
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
    DEADLOCKED
};

struct SchedulerStepResult {
    SchedulerStepStatus status = SchedulerStepStatus::EMPTY_WORKLOAD;
    std::vector<RequestId> admitted_requests;
    std::vector<RequestId> prefill_completed_requests;
    std::vector<RequestId> generated_tokens;
    std::vector<RequestId> completed_requests;
    std::vector<RequestId> cancelled_requests;
    std::vector<RequestId> memory_blocked_requests;
    std::vector<RequestId> permanently_unserviceable_requests;
    std::size_t tokens_processed = 0;
    bool made_progress = false;
    bool future_arrivals_remain = false;
    std::optional<SimulatedTimestamp> next_arrival_timestamp;
};

class Scheduler {
public:
    Scheduler(SchedulerConfig config, std::size_t total_physical_blocks,
              std::size_t tokens_per_block);

    [[nodiscard]] SubmissionResult submit(Request request);
    [[nodiscard]] SchedulerStepResult step(SimulatedTimestamp timestamp);
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
};

}  // namespace blockstride
