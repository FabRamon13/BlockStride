#include "blockstride/simulator.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace blockstride {
namespace {

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("simulated execution cost overflow");
    }
    return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error("simulated execution cost overflow");
    }
    return left * right;
}

std::uint64_t to_uint64(std::size_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("work count exceeds simulated counter range");
    }
    return static_cast<std::uint64_t>(value);
}

bool can_add(std::uint64_t current, std::uint64_t increment) noexcept {
    return increment <= std::numeric_limits<std::uint64_t>::max() - current;
}

}  // namespace

SimulationCostModel::SimulationCostModel(SimulationConfig config) : config_(config) {
    static_cast<void>(checked_add(config_.prefill_fixed_us, config_.prefill_per_token_us));
    static_cast<void>(
        checked_add(config_.decode_fixed_batch_us, config_.decode_per_token_us));
}

std::uint64_t SimulationCostModel::prefill_cost(std::size_t prompt_tokens) const {
    if (prompt_tokens == 0) {
        return 0;
    }
    return checked_add(config_.prefill_fixed_us,
                       checked_multiply(config_.prefill_per_token_us,
                                        to_uint64(prompt_tokens)));
}

std::uint64_t SimulationCostModel::decode_batch_cost(std::size_t generated_tokens) const {
    if (generated_tokens == 0) {
        return 0;
    }
    return checked_add(config_.decode_fixed_batch_us,
                       checked_multiply(config_.decode_per_token_us,
                                        to_uint64(generated_tokens)));
}

SimulationClock::SimulationClock(SimulatedTimestamp initial_timestamp) noexcept
    : timestamp_(initial_timestamp) {}

SimulatedTimestamp SimulationClock::now() const noexcept {
    return timestamp_;
}

bool SimulationClock::try_advance_by(std::uint64_t duration_us) noexcept {
    if (duration_us > std::numeric_limits<SimulatedTimestamp>::max() - timestamp_) {
        return false;
    }
    timestamp_ += duration_us;
    return true;
}

void SimulationClock::advance_by(std::uint64_t duration_us) {
    if (!try_advance_by(duration_us)) {
        throw std::overflow_error("simulated clock overflow");
    }
}

void SimulationClock::advance_to(SimulatedTimestamp timestamp) {
    if (timestamp < timestamp_) {
        throw std::invalid_argument("simulated clock cannot move backwards");
    }
    timestamp_ = timestamp;
}

ExecutionSimulator::ExecutionSimulator(SchedulerConfig scheduler_config,
                                       std::size_t total_physical_blocks,
                                       std::size_t tokens_per_block,
                                       SimulationConfig simulation_config)
    : scheduler_(scheduler_config, total_physical_blocks, tokens_per_block),
      cost_model_(simulation_config) {}

SubmissionResult ExecutionSimulator::submit(Request request) {
    if (request.arrival_timestamp() < clock_.now()) {
        throw std::invalid_argument("request arrival cannot precede the simulation clock");
    }
    const auto id = request.id();
    const auto arrival = request.arrival_timestamp();
    const auto result = scheduler_.submit(std::move(request));
    if (result == SubmissionResult::ACCEPTED) {
        const auto index = request_results_.size();
        RequestSimulationResult request_result{};
        request_result.request_id = id;
        request_result.arrival_timestamp = arrival;
        request_results_.push_back(std::move(request_result));
        request_result_indices_.emplace(id, index);
    } else {
        rejected_requests_.push_back({id, result});
    }
    return result;
}

std::vector<SubmissionResult> ExecutionSimulator::submit_workload(
    std::vector<Request> requests) {
    std::vector<SubmissionResult> results;
    results.reserve(requests.size());
    for (auto& request : requests) {
        results.push_back(submit(std::move(request)));
    }
    return results;
}

void ExecutionSimulator::cancel(RequestId id) {
    const auto& request = scheduler_.request_manager().get(id);
    if (request.state() != RequestState::WAITING && request.state() != RequestState::RUNNING) {
        throw std::logic_error("only a waiting or running request can be cancelled");
    }
    if (statistics_.cancelled_requests == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("cancelled request counter overflow");
    }
    const auto cancellation_time = std::max(clock_.now(), request.arrival_timestamp());
    clock_.advance_to(cancellation_time);
    scheduler_.cancel(id, cancellation_time);
    ++statistics_.cancelled_requests;
}

SimulationOutcome ExecutionSimulator::run() {
    if (request_results_.empty()) {
        return make_outcome(SimulationStatus::EMPTY_WORKLOAD, {}, {});
    }

    const Scheduler::StageCompletionCallback prefill_completion =
        [this](std::size_t tokens) { return finish_prefill(tokens); };
    const Scheduler::StageCompletionCallback decode_completion =
        [this](std::size_t tokens) { return finish_decode_batch(tokens); };

    auto memory_blocked = std::vector<RequestId>{};
    auto permanently_unserviceable = std::vector<RequestId>{};
    while (true) {
        if (scheduler_.request_manager().waiting_requests().empty() &&
            scheduler_.request_manager().running_requests().empty()) {
            return make_outcome(SimulationStatus::COMPLETED, memory_blocked,
                                permanently_unserviceable);
        }

        if (statistics_.scheduler_steps == std::numeric_limits<std::uint64_t>::max()) {
            return make_outcome(SimulationStatus::TIME_OVERFLOW, memory_blocked,
                                permanently_unserviceable);
        }
        ++statistics_.scheduler_steps;
        const auto step = scheduler_.step(clock_.now(), prefill_completion, decode_completion);
        for (const auto& event : step.admission_events) {
            const auto index = request_result_indices_.find(event.request_id);
            if (index == request_result_indices_.end()) {
                throw std::logic_error("scheduler admitted a request absent from the workload");
            }
            auto& request_result = request_results_.at(index->second);
            if (request_result.start_timestamp) {
                throw std::logic_error("request received more than one start timestamp");
            }
            request_result.start_timestamp = event.start_timestamp;
        }

        if (!can_add(statistics_.prefill_operations,
                     to_uint64(step.prefill_completed_requests.size())) ||
            !can_add(statistics_.prefill_tokens, to_uint64(step.prefill_tokens_processed)) ||
            !can_add(statistics_.decode_batches, to_uint64(step.decode_batch_count)) ||
            !can_add(statistics_.generated_tokens, to_uint64(step.decode_tokens_processed)) ||
            !can_add(statistics_.completed_requests,
                     to_uint64(step.completed_requests.size()))) {
            return make_outcome(SimulationStatus::TIME_OVERFLOW, step.memory_blocked_requests,
                                step.permanently_unserviceable_requests);
        }
        statistics_.prefill_operations +=
            to_uint64(step.prefill_completed_requests.size());
        statistics_.prefill_tokens += to_uint64(step.prefill_tokens_processed);
        statistics_.decode_batches += to_uint64(step.decode_batch_count);
        statistics_.generated_tokens += to_uint64(step.decode_tokens_processed);
        statistics_.completed_requests += to_uint64(step.completed_requests.size());
        memory_blocked = step.memory_blocked_requests;
        permanently_unserviceable = step.permanently_unserviceable_requests;

        if (step.time_overflow || step.status == SchedulerStepStatus::TIME_OVERFLOW) {
            return make_outcome(SimulationStatus::TIME_OVERFLOW, memory_blocked,
                                permanently_unserviceable);
        }
        if (step.status == SchedulerStepStatus::DEADLOCKED) {
            return make_outcome(SimulationStatus::DEADLOCKED, memory_blocked,
                                permanently_unserviceable);
        }
        if (step.status == SchedulerStepStatus::MEMORY_BLOCKED && !step.made_progress) {
            return make_outcome(SimulationStatus::MEMORY_BLOCKED, memory_blocked,
                                permanently_unserviceable);
        }
        if (step.status == SchedulerStepStatus::PERMANENTLY_UNSERVICEABLE &&
            !step.made_progress) {
            return make_outcome(SimulationStatus::PERMANENTLY_UNSERVICEABLE, memory_blocked,
                                permanently_unserviceable);
        }
        if (step.status == SchedulerStepStatus::WAITING_FOR_ARRIVAL) {
            if (!step.next_arrival_timestamp) {
                return make_outcome(SimulationStatus::NO_PROGRESS, memory_blocked,
                                    permanently_unserviceable);
            }
            clock_.advance_to(*step.next_arrival_timestamp);
            continue;
        }
        if (step.status == SchedulerStepStatus::PROGRESSED) {
            const auto running = scheduler_.request_manager().running_requests();
            if (running.empty() && step.next_arrival_timestamp) {
                clock_.advance_to(*step.next_arrival_timestamp);
                continue;
            }
            if (!step.made_progress) {
                return make_outcome(SimulationStatus::NO_PROGRESS, memory_blocked,
                                    permanently_unserviceable);
            }
            continue;
        }
        if (step.status == SchedulerStepStatus::EMPTY_WORKLOAD) {
            return make_outcome(SimulationStatus::COMPLETED, memory_blocked,
                                permanently_unserviceable);
        }
        return make_outcome(SimulationStatus::NO_PROGRESS, memory_blocked,
                            permanently_unserviceable);
    }
}

const SimulationClock& ExecutionSimulator::clock() const noexcept {
    return clock_;
}

const Scheduler& ExecutionSimulator::scheduler() const noexcept {
    return scheduler_;
}

std::optional<SimulatedTimestamp> ExecutionSimulator::finish_prefill(std::size_t tokens) {
    try {
        const auto cost = cost_model_.prefill_cost(tokens);
        if (!can_add(statistics_.prefill_execution_time_us, cost) ||
            !clock_.try_advance_by(cost)) {
            return std::nullopt;
        }
        statistics_.prefill_execution_time_us += cost;
        return clock_.now();
    } catch (const std::overflow_error&) {
        return std::nullopt;
    }
}

std::optional<SimulatedTimestamp> ExecutionSimulator::finish_decode_batch(
    std::size_t tokens) {
    try {
        const auto cost = cost_model_.decode_batch_cost(tokens);
        if (!can_add(statistics_.decode_execution_time_us, cost) ||
            !clock_.try_advance_by(cost)) {
            return std::nullopt;
        }
        statistics_.decode_execution_time_us += cost;
        return clock_.now();
    } catch (const std::overflow_error&) {
        return std::nullopt;
    }
}

SimulationOutcome ExecutionSimulator::make_outcome(
    SimulationStatus status, const std::vector<RequestId>& memory_blocked,
    const std::vector<RequestId>& permanently_unserviceable) const {
    SimulationOutcome outcome;
    outcome.status = status;
    outcome.final_timestamp = clock_.now();
    outcome.statistics = statistics_;
    outcome.requests = request_results_;
    outcome.rejected_requests = rejected_requests_;
    outcome.memory_blocked_requests = memory_blocked;
    outcome.permanently_unserviceable_requests = permanently_unserviceable;

    for (auto& request_result : outcome.requests) {
        const auto& request = scheduler_.request_manager().get(request_result.request_id);
        request_result.final_state = request.state();
        request_result.generated_tokens = request.generated_token_count();
        request_result.first_token_timestamp = request.first_token_timestamp();
        request_result.completion_timestamp = request.completion_timestamp();
        if (request_result.first_token_timestamp) {
            request_result.time_to_first_token_us =
                *request_result.first_token_timestamp - request_result.arrival_timestamp;
        }
        if (request_result.completion_timestamp) {
            request_result.completion_latency_us =
                *request_result.completion_timestamp - request_result.arrival_timestamp;
        }
    }

    outcome.outstanding_requests = scheduler_.request_manager().waiting_requests();
    const auto running = scheduler_.request_manager().running_requests();
    outcome.outstanding_requests.insert(outcome.outstanding_requests.end(), running.begin(),
                                        running.end());
    return outcome;
}

}  // namespace blockstride
