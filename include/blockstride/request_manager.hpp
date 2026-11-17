#pragma once

#include "blockstride/request.hpp"

#include <optional>
#include <unordered_map>
#include <vector>

namespace blockstride {

class RequestManager {
public:
    void submit(Request request);
    [[nodiscard]] std::optional<RequestId> admit_next(SimulatedTimestamp current_timestamp);
    [[nodiscard]] std::optional<RequestId> admit_next(
        SimulatedTimestamp eligible_timestamp, SimulatedTimestamp start_timestamp);

    void mark_prefill_completed(RequestId id);
    void record_generated_token(RequestId id, SimulatedTimestamp timestamp);
    void complete(RequestId id, SimulatedTimestamp timestamp);
    void cancel(RequestId id, SimulatedTimestamp timestamp);

    [[nodiscard]] const Request& get(RequestId id) const;
    [[nodiscard]] std::vector<RequestId> waiting_requests() const;
    [[nodiscard]] std::vector<RequestId> running_requests() const;
    [[nodiscard]] std::vector<RequestId> terminal_requests() const;

private:
    [[nodiscard]] Request& get_mutable(RequestId id);

    std::unordered_map<RequestId, Request> requests_;
    std::vector<RequestId> waiting_requests_;
    std::vector<RequestId> running_requests_;
    std::vector<RequestId> terminal_requests_;
};

}  // namespace blockstride
