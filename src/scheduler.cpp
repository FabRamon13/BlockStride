#include "blockstride/scheduler.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace blockstride {

std::string_view scheduling_policy_name(SchedulingPolicy policy) noexcept {
    switch (policy) {
        case SchedulingPolicy::SEQUENTIAL:
            return "sequential";
        case SchedulingPolicy::STATIC_BATCHING:
            return "static-batching";
        case SchedulingPolicy::CONTINUOUS_BATCHING:
            return "continuous-batching";
    }
    return "unknown";
}

Scheduler::Scheduler(SchedulerConfig config, std::size_t total_physical_blocks,
                     std::size_t tokens_per_block)
    : config_(config), block_manager_(total_physical_blocks, tokens_per_block) {
    if (config_.max_active_requests == 0) {
        throw std::invalid_argument("maximum active requests must be greater than zero");
    }
    if (config_.max_tokens_per_step == 0) {
        throw std::invalid_argument("maximum tokens per step must be greater than zero");
    }
    if (config_.policy != SchedulingPolicy::SEQUENTIAL &&
        config_.policy != SchedulingPolicy::STATIC_BATCHING &&
        config_.policy != SchedulingPolicy::CONTINUOUS_BATCHING) {
        throw std::invalid_argument("unsupported scheduling policy");
    }
}

SubmissionResult Scheduler::submit(Request request) {
    if (request.prompt_token_count() > config_.max_tokens_per_step) {
        return SubmissionResult::PROMPT_EXCEEDS_STEP_TOKEN_BUDGET;
    }
    request_manager_.submit(std::move(request));
    return SubmissionResult::ACCEPTED;
}

SchedulerStepResult Scheduler::step(SimulatedTimestamp timestamp) {
    const auto instantaneous = [timestamp](std::size_t) {
        return std::optional<SimulatedTimestamp>{timestamp};
    };
    return step(timestamp, instantaneous, instantaneous);
}

SchedulerStepResult Scheduler::step(
    SimulatedTimestamp timestamp, const StageCompletionCallback& prefill_completion,
    const StageCompletionCallback& decode_batch_completion) {
    if (!prefill_completion || !decode_batch_completion) {
        throw std::invalid_argument("scheduler stage callbacks must be provided");
    }
    if (current_timestamp_ && timestamp < *current_timestamp_) {
        throw std::invalid_argument("scheduler timestamp cannot move backwards");
    }
    current_timestamp_ = timestamp;

    SchedulerStepResult result;
    result.step_start_timestamp = timestamp;
    auto execution_timestamp = timestamp;
    result.cancelled_requests = std::move(pending_cancelled_requests_);
    pending_cancelled_requests_.clear();
    result.made_progress = !result.cancelled_requests.empty();

    if (config_.policy == SchedulingPolicy::STATIC_BATCHING && static_cohort_active_) {
        const auto cohort_terminated = std::all_of(
            static_cohort_.begin(), static_cohort_.end(), [this](RequestId id) {
                const auto state = request_manager_.get(id).state();
                return state == RequestState::COMPLETED || state == RequestState::CANCELLED;
            });
        if (cohort_terminated) {
            static_cohort_.clear();
            static_cohort_active_ = false;
        }
    }
    auto static_admission_closed =
        config_.policy == SchedulingPolicy::STATIC_BATCHING && static_cohort_active_;

    const auto waiting_at_start = request_manager_.waiting_requests();
    for (const auto id : waiting_at_start) {
        if (permanently_unserviceable(request_manager_.get(id))) {
            append_unique(result.permanently_unserviceable_requests, id);
        }
    }

    auto remaining_tokens = config_.max_tokens_per_step;
    bool time_overflow = false;
    while (remaining_tokens > 0 && !time_overflow) {
        bool pass_made_progress = false;

        const auto active_limit =
            config_.policy == SchedulingPolicy::SEQUENTIAL
                ? std::size_t{1}
                : config_.max_active_requests;
        while (!static_admission_closed &&
               request_manager_.running_requests().size() < active_limit) {
            const auto waiting = request_manager_.waiting_requests();
            if (waiting.empty()) {
                break;
            }

            const auto id = waiting.front();
            const auto& request = request_manager_.get(id);
            if (request.arrival_timestamp() > timestamp) {
                break;
            }
            if (permanently_unserviceable(request)) {
                append_unique(result.permanently_unserviceable_requests, id);
                break;
            }

            const auto allocation =
                block_manager_.allocate_or_grow(id, request.prompt_token_count());
            if (allocation == AllocationResult::CAPACITY_EXHAUSTED) {
                append_unique(result.memory_blocked_requests, id);
                break;
            }

            try {
                const auto admitted = request_manager_.admit_next(timestamp, execution_timestamp);
                if (!admitted || *admitted != id) {
                    throw std::logic_error("request manager admitted an unexpected request");
                }
            } catch (...) {
                block_manager_.release(id);
                throw;
            }
            result.admission_events.push_back({id, execution_timestamp});
            result.admitted_requests.push_back(id);
            if (config_.policy == SchedulingPolicy::STATIC_BATCHING) {
                static_cohort_.push_back(id);
            }
            pass_made_progress = true;
        }
        if (config_.policy == SchedulingPolicy::STATIC_BATCHING &&
            !static_cohort_.empty()) {
            static_cohort_active_ = true;
            static_admission_closed = true;
        }

        const auto running_for_prefill = request_manager_.running_requests();
        for (const auto id : running_for_prefill) {
            const auto& request = request_manager_.get(id);
            if (request.prefill_completed()) {
                continue;
            }
            const auto prompt_tokens = request.prompt_token_count();
            if (prompt_tokens > remaining_tokens) {
                break;
            }

            const auto completion_time = prefill_completion(prompt_tokens);
            if (!completion_time) {
                time_overflow = true;
                break;
            }
            if (*completion_time < execution_timestamp) {
                throw std::invalid_argument("prefill completion time cannot move backwards");
            }
            execution_timestamp = *completion_time;
            current_timestamp_ = execution_timestamp;
            request_manager_.mark_prefill_completed(id);
            remaining_tokens -= prompt_tokens;
            result.tokens_processed += prompt_tokens;
            result.prefill_tokens_processed += prompt_tokens;
            result.prefill_completed_requests.push_back(id);
            pass_made_progress = true;
        }

        if (time_overflow) {
            result.made_progress = result.made_progress || pass_made_progress;
            break;
        }

        const auto running_for_decode = request_manager_.running_requests();
        while (remaining_tokens > 0) {
            std::vector<RequestId> decode_batch;
            auto available_blocks = block_manager_.free_block_count();
            for (const auto id : running_for_decode) {
                const auto& request = request_manager_.get(id);
                if (request.state() != RequestState::RUNNING || !request.prefill_completed() ||
                    request.generated_token_count() >= request.max_new_tokens()) {
                    continue;
                }

                const auto total_tokens =
                    request.prompt_token_count() + request.generated_token_count() + 1;
                const auto needed_blocks = block_manager_.blocks_needed(total_tokens);
                const auto allocated_blocks = block_manager_.allocated_blocks(id).size();
                const auto additional_blocks =
                    needed_blocks > allocated_blocks ? needed_blocks - allocated_blocks : 0;
                if (additional_blocks > available_blocks) {
                    append_unique(result.memory_blocked_requests, id);
                    continue;
                }
                available_blocks -= additional_blocks;
                decode_batch.push_back(id);
                if (decode_batch.size() == remaining_tokens) {
                    break;
                }
            }

            if (decode_batch.empty()) {
                break;
            }

            const auto completion_time = decode_batch_completion(decode_batch.size());
            if (!completion_time) {
                time_overflow = true;
                break;
            }
            if (*completion_time < execution_timestamp) {
                throw std::invalid_argument("decode completion time cannot move backwards");
            }
            execution_timestamp = *completion_time;
            current_timestamp_ = execution_timestamp;
            ++result.decode_batch_count;

            for (const auto id : decode_batch) {
                const auto& request = request_manager_.get(id);
                const auto total_tokens =
                    request.prompt_token_count() + request.generated_token_count() + 1;
                if (block_manager_.allocate_or_grow(id, total_tokens) !=
                    AllocationResult::SUCCESS) {
                    throw std::logic_error("planned decode allocation unexpectedly failed");
                }

                request_manager_.record_generated_token(id, execution_timestamp);
                result.generated_tokens.push_back(id);
                ++result.tokens_processed;
                ++result.decode_tokens_processed;
                --remaining_tokens;
                pass_made_progress = true;

                const auto& updated_request = request_manager_.get(id);
                if (updated_request.generated_token_count() ==
                    updated_request.max_new_tokens()) {
                    request_manager_.complete(id, execution_timestamp);
                    block_manager_.release(id);
                    result.completed_requests.push_back(id);
                }
            }
        }

        result.made_progress = result.made_progress || pass_made_progress;
        if (time_overflow || !pass_made_progress || remaining_tokens == 0) {
            break;
        }
    }

    result.step_end_timestamp = execution_timestamp;
    result.time_overflow = time_overflow;
    result.next_arrival_timestamp = next_arrival_after(execution_timestamp);
    result.future_arrivals_remain = result.next_arrival_timestamp.has_value();

    const auto waiting = request_manager_.waiting_requests();
    const auto running = request_manager_.running_requests();
    if (result.time_overflow) {
        result.status = SchedulerStepStatus::TIME_OVERFLOW;
    } else if (result.made_progress) {
        result.status = SchedulerStepStatus::PROGRESSED;
    } else if (waiting.empty() && running.empty()) {
        result.status = SchedulerStepStatus::EMPTY_WORKLOAD;
    } else if (!result.permanently_unserviceable_requests.empty()) {
        result.status = SchedulerStepStatus::PERMANENTLY_UNSERVICEABLE;
    } else if (!result.memory_blocked_requests.empty()) {
        result.status = active_work_can_progress() ? SchedulerStepStatus::MEMORY_BLOCKED
                                                  : SchedulerStepStatus::DEADLOCKED;
    } else if (running.empty() && result.future_arrivals_remain) {
        result.status = SchedulerStepStatus::WAITING_FOR_ARRIVAL;
    } else {
        result.status = SchedulerStepStatus::DEADLOCKED;
    }

    return result;
}

void Scheduler::cancel(RequestId id, SimulatedTimestamp timestamp) {
    if (current_timestamp_ && timestamp < *current_timestamp_) {
        throw std::invalid_argument("cancellation timestamp cannot precede scheduler time");
    }

    const auto& request = request_manager_.get(id);
    const auto running = request.state() == RequestState::RUNNING;
    if (running) {
        static_cast<void>(block_manager_.allocated_blocks(id));
    }

    pending_cancelled_requests_.push_back(id);
    try {
        request_manager_.cancel(id, timestamp);
    } catch (...) {
        pending_cancelled_requests_.pop_back();
        throw;
    }

    if (running) {
        block_manager_.release(id);
    }
    current_timestamp_ = timestamp;
}

const RequestManager& Scheduler::request_manager() const noexcept {
    return request_manager_;
}

const BlockManager& Scheduler::block_manager() const noexcept {
    return block_manager_;
}

const SchedulerConfig& Scheduler::config() const noexcept {
    return config_;
}

bool Scheduler::permanently_unserviceable(const Request& request) const noexcept {
    const auto prompt_tokens = request.prompt_token_count();
    const auto generation_tokens = request.max_new_tokens();
    if (generation_tokens > std::numeric_limits<std::size_t>::max() - prompt_tokens) {
        return true;
    }
    const auto maximum_sequence_tokens = prompt_tokens + generation_tokens;
    return block_manager_.blocks_needed(maximum_sequence_tokens) >
           block_manager_.total_physical_blocks();
}

bool Scheduler::active_work_can_progress() const {
    for (const auto id : request_manager_.running_requests()) {
        const auto& request = request_manager_.get(id);
        if (!request.prefill_completed()) {
            return true;
        }
        if (request.generated_token_count() >= request.max_new_tokens()) {
            continue;
        }

        const auto current_sequence_tokens =
            request.prompt_token_count() + request.generated_token_count();
        const auto next_blocks =
            block_manager_.blocks_needed(current_sequence_tokens + 1);
        const auto allocated_blocks = block_manager_.allocated_blocks(id).size();
        if (next_blocks <= allocated_blocks ||
            next_blocks - allocated_blocks <= block_manager_.free_block_count()) {
            return true;
        }
    }
    return false;
}

std::optional<SimulatedTimestamp> Scheduler::next_arrival_after(
    SimulatedTimestamp timestamp) const {
    std::optional<SimulatedTimestamp> next_arrival;
    for (const auto id : request_manager_.waiting_requests()) {
        const auto arrival = request_manager_.get(id).arrival_timestamp();
        if (arrival > timestamp && (!next_arrival || arrival < *next_arrival)) {
            next_arrival = arrival;
        }
    }
    return next_arrival;
}

void Scheduler::append_unique(std::vector<RequestId>& requests, RequestId id) {
    if (std::find(requests.begin(), requests.end(), id) == requests.end()) {
        requests.push_back(id);
    }
}

}  // namespace blockstride
