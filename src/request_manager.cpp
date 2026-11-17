#include "blockstride/request_manager.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace blockstride {

void RequestManager::submit(Request request) {
    if (request.state() != RequestState::WAITING) {
        throw std::invalid_argument("only waiting requests can be submitted");
    }
    if (requests_.contains(request.id())) {
        throw std::invalid_argument("request ID has already been submitted");
    }

    const auto id = request.id();
    const auto arrival_timestamp = request.arrival_timestamp();
    const auto position = std::upper_bound(
        waiting_requests_.begin(), waiting_requests_.end(), arrival_timestamp,
        [this](SimulatedTimestamp arrival, RequestId existing_id) {
            return arrival < requests_.at(existing_id).arrival_timestamp();
        });
    const auto offset = std::distance(waiting_requests_.begin(), position);
    waiting_requests_.insert(position, id);

    try {
        const auto [it, inserted] = requests_.emplace(id, std::move(request));
        static_cast<void>(it);
        if (!inserted) {
            throw std::invalid_argument("request ID has already been submitted");
        }
    } catch (...) {
        waiting_requests_.erase(waiting_requests_.begin() + offset);
        throw;
    }
}

std::optional<RequestId> RequestManager::admit_next(SimulatedTimestamp current_timestamp) {
    return admit_next(current_timestamp, current_timestamp);
}

std::optional<RequestId> RequestManager::admit_next(SimulatedTimestamp eligible_timestamp,
                                                    SimulatedTimestamp start_timestamp) {
    if (start_timestamp < eligible_timestamp) {
        throw std::invalid_argument("request start time cannot precede eligibility time");
    }
    if (waiting_requests_.empty()) {
        return std::nullopt;
    }

    const auto id = waiting_requests_.front();
    auto& request = get_mutable(id);
    if (request.arrival_timestamp() > eligible_timestamp) {
        return std::nullopt;
    }

    running_requests_.push_back(id);
    try {
        request.start(start_timestamp);
    } catch (...) {
        running_requests_.pop_back();
        throw;
    }
    waiting_requests_.erase(waiting_requests_.begin());
    return id;
}

void RequestManager::mark_prefill_completed(RequestId id) {
    get_mutable(id).mark_prefill_completed();
}

void RequestManager::record_generated_token(RequestId id, SimulatedTimestamp timestamp) {
    get_mutable(id).record_generated_token(timestamp);
}

void RequestManager::complete(RequestId id, SimulatedTimestamp timestamp) {
    auto& request = get_mutable(id);
    if (request.state() != RequestState::RUNNING) {
        throw std::logic_error("only a running request can complete");
    }
    const auto position = std::find(running_requests_.begin(), running_requests_.end(), id);
    if (position == running_requests_.end()) {
        throw std::logic_error("running request is missing from the running collection");
    }

    terminal_requests_.push_back(id);
    try {
        request.complete(timestamp);
    } catch (...) {
        terminal_requests_.pop_back();
        throw;
    }
    running_requests_.erase(position);
}

void RequestManager::cancel(RequestId id, SimulatedTimestamp timestamp) {
    auto& request = get_mutable(id);
    std::vector<RequestId>* active_requests = nullptr;
    if (request.state() == RequestState::WAITING) {
        active_requests = &waiting_requests_;
    } else if (request.state() == RequestState::RUNNING) {
        active_requests = &running_requests_;
    } else {
        throw std::logic_error("only a waiting or running request can be cancelled");
    }

    const auto position = std::find(active_requests->begin(), active_requests->end(), id);
    if (position == active_requests->end()) {
        throw std::logic_error("active request is missing from its collection");
    }

    terminal_requests_.push_back(id);
    try {
        request.cancel(timestamp);
    } catch (...) {
        terminal_requests_.pop_back();
        throw;
    }
    active_requests->erase(position);
}

const Request& RequestManager::get(RequestId id) const {
    const auto position = requests_.find(id);
    if (position == requests_.end()) {
        throw std::out_of_range("request ID was not submitted");
    }
    return position->second;
}

std::vector<RequestId> RequestManager::waiting_requests() const {
    return waiting_requests_;
}

std::vector<RequestId> RequestManager::running_requests() const {
    return running_requests_;
}

std::vector<RequestId> RequestManager::terminal_requests() const {
    return terminal_requests_;
}

Request& RequestManager::get_mutable(RequestId id) {
    const auto position = requests_.find(id);
    if (position == requests_.end()) {
        throw std::out_of_range("request ID was not submitted");
    }
    return position->second;
}

}  // namespace blockstride
