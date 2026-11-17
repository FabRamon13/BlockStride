#include "blockstride/request_manager.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace blockstride {
namespace {

TEST(RequestManagerTest, RejectsDuplicateRequestIds) {
    RequestManager manager;
    manager.submit(Request(7, 2, 1, 0));

    EXPECT_THROW(manager.submit(Request(7, 3, 1, 1)), std::invalid_argument);
    EXPECT_EQ(manager.waiting_requests(), (std::vector<RequestId>{7}));
}

TEST(RequestManagerTest, OrdersWaitingRequestsByArrivalTime) {
    RequestManager manager;
    manager.submit(Request(1, 2, 1, 30));
    manager.submit(Request(2, 2, 1, 10));
    manager.submit(Request(3, 2, 1, 20));

    EXPECT_EQ(manager.waiting_requests(), (std::vector<RequestId>{2, 3, 1}));
}

TEST(RequestManagerTest, KeepsSubmissionOrderForEqualArrivalTimes) {
    RequestManager manager;
    manager.submit(Request(8, 2, 1, 10));
    manager.submit(Request(3, 2, 1, 10));
    manager.submit(Request(5, 2, 1, 10));

    EXPECT_EQ(manager.waiting_requests(), (std::vector<RequestId>{8, 3, 5}));
}

TEST(RequestManagerTest, DoesNotAdmitRequestsBeforeArrival) {
    RequestManager manager;
    manager.submit(Request(1, 2, 1, 15));
    manager.submit(Request(2, 2, 1, 20));

    EXPECT_FALSE(manager.admit_next(14).has_value());
    EXPECT_TRUE(manager.running_requests().empty());
    EXPECT_EQ(manager.waiting_requests(), (std::vector<RequestId>{1, 2}));

    const auto admitted = manager.admit_next(15);
    ASSERT_TRUE(admitted.has_value());
    EXPECT_EQ(*admitted, 1);
}

TEST(RequestManagerTest, AdmitsEligibleRequestsInFcfsOrder) {
    RequestManager manager;
    manager.submit(Request(1, 2, 1, 5));
    manager.submit(Request(2, 2, 1, 3));
    manager.submit(Request(3, 2, 1, 3));

    EXPECT_EQ(manager.admit_next(5), 2);
    EXPECT_EQ(manager.admit_next(5), 3);
    EXPECT_EQ(manager.admit_next(5), 1);
    EXPECT_FALSE(manager.admit_next(5).has_value());
    EXPECT_TRUE(manager.waiting_requests().empty());
    EXPECT_EQ(manager.running_requests(), (std::vector<RequestId>{2, 3, 1}));
}

TEST(RequestManagerTest, CancellationUpdatesWaitingRunningAndTerminalCollections) {
    RequestManager manager;
    manager.submit(Request(1, 2, 1, 0));
    manager.submit(Request(2, 2, 1, 1));

    manager.cancel(1, 0);
    EXPECT_EQ(manager.get(1).state(), RequestState::CANCELLED);
    EXPECT_EQ(manager.waiting_requests(), (std::vector<RequestId>{2}));
    EXPECT_EQ(manager.terminal_requests(), (std::vector<RequestId>{1}));

    ASSERT_EQ(manager.admit_next(1), 2);
    manager.cancel(2, 2);
    EXPECT_EQ(manager.get(2).state(), RequestState::CANCELLED);
    EXPECT_TRUE(manager.running_requests().empty());
    EXPECT_EQ(manager.terminal_requests(), (std::vector<RequestId>{1, 2}));
}

TEST(RequestManagerTest, CompletionMovesRequestToTerminalCollection) {
    RequestManager manager;
    manager.submit(Request(1, 2, 1, 0));
    ASSERT_EQ(manager.admit_next(0), 1);
    manager.mark_prefill_completed(1);
    manager.record_generated_token(1, 1);
    manager.complete(1, 2);

    EXPECT_EQ(manager.get(1).state(), RequestState::COMPLETED);
    EXPECT_TRUE(manager.running_requests().empty());
    EXPECT_EQ(manager.terminal_requests(), (std::vector<RequestId>{1}));
}

TEST(RequestManagerTest, FailedTerminalTransitionKeepsCollectionsConsistent) {
    RequestManager manager;
    manager.submit(Request(1, 2, 1, 0));
    ASSERT_EQ(manager.admit_next(0), 1);

    EXPECT_THROW(manager.complete(1, 1), std::logic_error);
    EXPECT_EQ(manager.running_requests(), (std::vector<RequestId>{1}));
    EXPECT_TRUE(manager.terminal_requests().empty());
    EXPECT_EQ(manager.get(1).state(), RequestState::RUNNING);
}

TEST(RequestManagerTest, ProvidesReadOnlyRequestAccessAndRejectsUnknownIds) {
    RequestManager manager;
    manager.submit(Request(1, 2, 1, 0));

    EXPECT_EQ(manager.get(1).prompt_token_count(), 2);
    EXPECT_THROW(static_cast<void>(manager.get(2)), std::out_of_range);
    EXPECT_THROW(manager.mark_prefill_completed(2), std::out_of_range);
}

}  // namespace
}  // namespace blockstride
