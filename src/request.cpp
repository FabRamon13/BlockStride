#include "blockstride/request.hpp"

#include <stdexcept>

namespace blockstride {

Request::Request(RequestId id, std::size_t prompt_token_count, std::size_t max_new_tokens,
                 SimulatedTimestamp arrival_timestamp)
    : id_(id),
      prompt_token_count_(prompt_token_count),
      max_new_tokens_(max_new_tokens),
      arrival_timestamp_(arrival_timestamp) {
    if (prompt_token_count_ == 0) {
        throw std::invalid_argument("prompt token count must be greater than zero");
    }
    if (max_new_tokens_ == 0) {
        throw std::invalid_argument("maximum generation tokens must be greater than zero");
    }
}

RequestId Request::id() const noexcept {
    return id_;
}

std::size_t Request::prompt_token_count() const noexcept {
    return prompt_token_count_;
}

std::size_t Request::max_new_tokens() const noexcept {
    return max_new_tokens_;
}

SimulatedTimestamp Request::arrival_timestamp() const noexcept {
    return arrival_timestamp_;
}

RequestState Request::state() const noexcept {
    return state_;
}

std::size_t Request::generated_token_count() const noexcept {
    return generated_token_count_;
}

bool Request::prefill_completed() const noexcept {
    return prefill_completed_;
}

const std::optional<SimulatedTimestamp>& Request::first_token_timestamp() const noexcept {
    return first_token_timestamp_;
}

const std::optional<SimulatedTimestamp>& Request::completion_timestamp() const noexcept {
    return completion_timestamp_;
}

void Request::validate_timestamp(SimulatedTimestamp timestamp) const {
    if (timestamp < arrival_timestamp_) {
        throw std::invalid_argument("timestamp cannot precede request arrival");
    }
}

void Request::start(SimulatedTimestamp timestamp) {
    validate_timestamp(timestamp);
    if (state_ != RequestState::WAITING) {
        throw std::logic_error("only a waiting request can start");
    }
    state_ = RequestState::RUNNING;
}

void Request::mark_prefill_completed() {
    if (state_ != RequestState::RUNNING) {
        throw std::logic_error("prefill can only complete while running");
    }
    if (prefill_completed_) {
        throw std::logic_error("prefill has already completed");
    }
    prefill_completed_ = true;
}

void Request::record_generated_token(SimulatedTimestamp timestamp) {
    validate_timestamp(timestamp);
    if (state_ != RequestState::RUNNING) {
        throw std::logic_error("tokens can only be generated while running");
    }
    if (!prefill_completed_) {
        throw std::logic_error("generation requires completed prefill");
    }
    if (generated_token_count_ >= max_new_tokens_) {
        throw std::overflow_error("generation limit has been reached");
    }

    if (!first_token_timestamp_) {
        first_token_timestamp_ = timestamp;
    }
    ++generated_token_count_;
}

void Request::complete(SimulatedTimestamp timestamp) {
    validate_timestamp(timestamp);
    if (state_ != RequestState::RUNNING) {
        throw std::logic_error("only a running request can complete");
    }
    if (!prefill_completed_) {
        throw std::logic_error("completion requires completed prefill");
    }
    if (generated_token_count_ != max_new_tokens_) {
        throw std::logic_error("completion requires the generation limit to be reached");
    }
    if (first_token_timestamp_ && timestamp < *first_token_timestamp_) {
        throw std::invalid_argument("completion timestamp cannot precede the first token");
    }

    completion_timestamp_ = timestamp;
    state_ = RequestState::COMPLETED;
}

void Request::cancel(SimulatedTimestamp timestamp) {
    validate_timestamp(timestamp);
    if (state_ != RequestState::WAITING && state_ != RequestState::RUNNING) {
        throw std::logic_error("only a waiting or running request can be cancelled");
    }
    if (first_token_timestamp_ && timestamp < *first_token_timestamp_) {
        throw std::invalid_argument("cancellation timestamp cannot precede the first token");
    }

    state_ = RequestState::CANCELLED;
}

}  // namespace blockstride
