#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace blockstride {

using RequestId = std::uint64_t;
using SimulatedTimestamp = std::uint64_t;

enum class RequestState {
    WAITING,
    RUNNING,
    COMPLETED,
    CANCELLED
};

class Request {
public:
    Request(RequestId id, std::size_t prompt_token_count, std::size_t max_new_tokens,
            SimulatedTimestamp arrival_timestamp);

    [[nodiscard]] RequestId id() const noexcept;
    [[nodiscard]] std::size_t prompt_token_count() const noexcept;
    [[nodiscard]] std::size_t max_new_tokens() const noexcept;
    [[nodiscard]] SimulatedTimestamp arrival_timestamp() const noexcept;
    [[nodiscard]] RequestState state() const noexcept;
    [[nodiscard]] std::size_t generated_token_count() const noexcept;
    [[nodiscard]] bool prefill_completed() const noexcept;
    [[nodiscard]] const std::optional<SimulatedTimestamp>& first_token_timestamp() const noexcept;
    [[nodiscard]] const std::optional<SimulatedTimestamp>& completion_timestamp() const noexcept;

    void start(SimulatedTimestamp timestamp);
    void mark_prefill_completed();
    void record_generated_token(SimulatedTimestamp timestamp);
    void complete(SimulatedTimestamp timestamp);
    void cancel(SimulatedTimestamp timestamp);

private:
    void validate_timestamp(SimulatedTimestamp timestamp) const;

    RequestId id_;
    std::size_t prompt_token_count_;
    std::size_t max_new_tokens_;
    SimulatedTimestamp arrival_timestamp_;
    RequestState state_ = RequestState::WAITING;
    std::size_t generated_token_count_ = 0;
    bool prefill_completed_ = false;
    std::optional<SimulatedTimestamp> first_token_timestamp_;
    std::optional<SimulatedTimestamp> completion_timestamp_;
};

}  // namespace blockstride
