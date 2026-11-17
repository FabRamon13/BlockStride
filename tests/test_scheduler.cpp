#include "blockstride/scheduler.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace blockstride {
namespace {

void expect_capacity_invariants(const Scheduler& scheduler) {
    std::unordered_set<BlockId> seen_blocks;
    std::size_t active_block_count = 0;

    for (const auto request_id : scheduler.request_manager().running_requests()) {
        const auto blocks = scheduler.block_manager().allocated_blocks(request_id);
        active_block_count += blocks.size();
        for (const auto block_id : blocks) {
            EXPECT_LT(block_id, scheduler.block_manager().total_physical_blocks());
            EXPECT_TRUE(seen_blocks.insert(block_id).second)
                << "block " << block_id << " is assigned to multiple requests";
        }
    }

    EXPECT_EQ(scheduler.block_manager().used_block_count(), active_block_count);
    EXPECT_EQ(scheduler.block_manager().used_block_count() +
                  scheduler.block_manager().free_block_count(),
              scheduler.block_manager().total_physical_blocks());
}

TEST(SchedulerTest, RejectsZeroSchedulingLimits) {
    EXPECT_THROW(Scheduler({0, 4}, 4, 4), std::invalid_argument);
    EXPECT_THROW(Scheduler({1, 0}, 4, 4), std::invalid_argument);
}

TEST(SchedulerTest, RejectsPromptsLargerThanTheStepBudget) {
    Scheduler scheduler({2, 4}, 8, 4);

    EXPECT_EQ(scheduler.submit(Request(1, 5, 1, 0)),
              SubmissionResult::PROMPT_EXCEEDS_STEP_TOKEN_BUDGET);
    EXPECT_TRUE(scheduler.request_manager().waiting_requests().empty());
}

TEST(SchedulerTest, AdmitsEligibleRequestsInFcfsOrder) {
    Scheduler scheduler({2, 2}, 8, 4);
    ASSERT_EQ(scheduler.submit(Request(1, 1, 3, 0)), SubmissionResult::ACCEPTED);
    ASSERT_EQ(scheduler.submit(Request(2, 1, 3, 0)), SubmissionResult::ACCEPTED);
    ASSERT_EQ(scheduler.submit(Request(3, 1, 3, 0)), SubmissionResult::ACCEPTED);

    const auto result = scheduler.step(0);

    EXPECT_EQ(result.admitted_requests, (std::vector<RequestId>{1, 2}));
    EXPECT_EQ(scheduler.request_manager().waiting_requests(), (std::vector<RequestId>{3}));
    EXPECT_EQ(scheduler.request_manager().running_requests(), (std::vector<RequestId>{1, 2}));
}

TEST(SchedulerTest, NeverAdmitsBeforeArrivalAndReturnsTheNextClockTime) {
    Scheduler scheduler({1, 4}, 4, 4);
    ASSERT_EQ(scheduler.submit(Request(1, 1, 1, 5)), SubmissionResult::ACCEPTED);

    const auto early_step = scheduler.step(4);
    EXPECT_EQ(early_step.status, SchedulerStepStatus::WAITING_FOR_ARRIVAL);
    EXPECT_TRUE(early_step.future_arrivals_remain);
    ASSERT_TRUE(early_step.next_arrival_timestamp.has_value());
    EXPECT_EQ(*early_step.next_arrival_timestamp, 5);
    EXPECT_TRUE(early_step.admitted_requests.empty());

    const auto arrival_step = scheduler.step(*early_step.next_arrival_timestamp);
    EXPECT_EQ(arrival_step.admitted_requests, (std::vector<RequestId>{1}));
}

TEST(SchedulerTest, EnforcesTheActiveRequestLimit) {
    Scheduler scheduler({1, 2}, 128, 1);
    ASSERT_EQ(scheduler.submit(Request(1, 1, 100, 0)), SubmissionResult::ACCEPTED);
    ASSERT_EQ(scheduler.submit(Request(2, 1, 100, 0)), SubmissionResult::ACCEPTED);

    const auto result = scheduler.step(0);

    EXPECT_EQ(result.admitted_requests, (std::vector<RequestId>{1}));
    EXPECT_EQ(scheduler.request_manager().running_requests().size(), 1);
    EXPECT_EQ(scheduler.request_manager().waiting_requests(), (std::vector<RequestId>{2}));
}

TEST(SchedulerTest, CountsPrefillAndDecodeAgainstOneTokenBudget) {
    Scheduler scheduler({2, 3}, 16, 4);
    ASSERT_EQ(scheduler.submit(Request(1, 2, 10, 0)), SubmissionResult::ACCEPTED);
    ASSERT_EQ(scheduler.submit(Request(2, 2, 10, 0)), SubmissionResult::ACCEPTED);

    const auto result = scheduler.step(0);

    EXPECT_EQ(result.tokens_processed, 3);
    EXPECT_LE(result.tokens_processed, scheduler.config().max_tokens_per_step);
    EXPECT_EQ(result.prefill_completed_requests, (std::vector<RequestId>{1}));
    EXPECT_EQ(result.generated_tokens, (std::vector<RequestId>{1}));
    EXPECT_FALSE(scheduler.request_manager().get(2).prefill_completed());
}

TEST(SchedulerTest, CompletesPrefillBeforeAllowingDecode) {
    Scheduler scheduler({1, 2}, 4, 4);
    ASSERT_EQ(scheduler.submit(Request(1, 2, 2, 0)), SubmissionResult::ACCEPTED);

    const auto prefill_step = scheduler.step(0);
    EXPECT_EQ(prefill_step.prefill_completed_requests, (std::vector<RequestId>{1}));
    EXPECT_TRUE(prefill_step.generated_tokens.empty());
    EXPECT_TRUE(scheduler.request_manager().get(1).prefill_completed());
    EXPECT_EQ(scheduler.request_manager().get(1).generated_token_count(), 0);

    const auto decode_step = scheduler.step(0);
    EXPECT_EQ(decode_step.generated_tokens, (std::vector<RequestId>{1, 1}));
    EXPECT_EQ(decode_step.completed_requests, (std::vector<RequestId>{1}));
}

TEST(SchedulerTest, AdmitsNewRequestsWhileExistingRequestsRemainRunning) {
    Scheduler scheduler({2, 2}, 32, 4);
    ASSERT_EQ(scheduler.submit(Request(1, 1, 20, 0)), SubmissionResult::ACCEPTED);
    const auto first_step = scheduler.step(0);
    ASSERT_EQ(first_step.admitted_requests, (std::vector<RequestId>{1}));
    ASSERT_EQ(scheduler.request_manager().get(1).state(), RequestState::RUNNING);

    ASSERT_EQ(scheduler.submit(Request(2, 1, 20, 0)), SubmissionResult::ACCEPTED);
    const auto second_step = scheduler.step(0);

    EXPECT_EQ(second_step.admitted_requests, (std::vector<RequestId>{2}));
    EXPECT_EQ(scheduler.request_manager().running_requests(),
              (std::vector<RequestId>{1, 2}));
}

TEST(SchedulerTest, StopsAtGenerationLimitAndReleasesCompletedRequests) {
    Scheduler scheduler({1, 8}, 4, 1);
    ASSERT_EQ(scheduler.submit(Request(1, 1, 2, 0)), SubmissionResult::ACCEPTED);

    const auto result = scheduler.step(0);

    EXPECT_EQ(result.generated_tokens, (std::vector<RequestId>{1, 1}));
    EXPECT_EQ(result.completed_requests, (std::vector<RequestId>{1}));
    EXPECT_EQ(result.tokens_processed, 3);
    EXPECT_EQ(scheduler.request_manager().get(1).generated_token_count(), 2);
    EXPECT_EQ(scheduler.request_manager().get(1).state(), RequestState::COMPLETED);
    EXPECT_EQ(scheduler.block_manager().used_block_count(), 0);
}

TEST(SchedulerTest, GrowsKvAllocationBeforeRecordingTokensThatCrossBlockBoundaries) {
    Scheduler scheduler({1, 3}, 4, 4);
    ASSERT_EQ(scheduler.submit(Request(1, 2, 5, 0)), SubmissionResult::ACCEPTED);

    const auto first_step = scheduler.step(0);
    ASSERT_EQ(first_step.generated_tokens, (std::vector<RequestId>{1}));
    EXPECT_EQ(scheduler.block_manager().allocated_blocks(1).size(), 1);

    const auto second_step = scheduler.step(0);
    EXPECT_EQ(second_step.generated_tokens, (std::vector<RequestId>{1, 1, 1}));
    EXPECT_EQ(scheduler.request_manager().get(1).generated_token_count(), 4);
    EXPECT_EQ(scheduler.block_manager().allocated_blocks(1).size(), 2);

    const auto third_step = scheduler.step(0);
    EXPECT_EQ(third_step.completed_requests, (std::vector<RequestId>{1}));
    EXPECT_EQ(scheduler.block_manager().used_block_count(), 0);
}

TEST(SchedulerTest, CancellationReclaimsRunningBlocksAndReportsBothCancellations) {
    Scheduler scheduler({1, 1}, 4, 4);
    ASSERT_EQ(scheduler.submit(Request(1, 1, 10, 0)), SubmissionResult::ACCEPTED);
    ASSERT_EQ(scheduler.submit(Request(2, 1, 10, 0)), SubmissionResult::ACCEPTED);
    static_cast<void>(scheduler.step(0));
    ASSERT_EQ(scheduler.block_manager().used_block_count(), 1);

    scheduler.cancel(2, 0);
    scheduler.cancel(1, 0);
    EXPECT_EQ(scheduler.block_manager().used_block_count(), 0);
    EXPECT_EQ(scheduler.request_manager().get(1).state(), RequestState::CANCELLED);
    EXPECT_EQ(scheduler.request_manager().get(2).state(), RequestState::CANCELLED);

    const auto result = scheduler.step(0);
    EXPECT_EQ(result.cancelled_requests, (std::vector<RequestId>{2, 1}));
}

TEST(SchedulerTest, TemporaryMemoryPressurePreservesAndLaterAdmitsWaitingRequests) {
    Scheduler scheduler({2, 8}, 1, 8);
    ASSERT_EQ(scheduler.submit(Request(1, 1, 1, 0)), SubmissionResult::ACCEPTED);
    ASSERT_EQ(scheduler.submit(Request(2, 7, 1, 0)), SubmissionResult::ACCEPTED);

    const auto first_step = scheduler.step(0);
    EXPECT_EQ(first_step.memory_blocked_requests, (std::vector<RequestId>{2}));
    EXPECT_EQ(first_step.completed_requests, (std::vector<RequestId>{1}));
    EXPECT_EQ(first_step.admitted_requests, (std::vector<RequestId>{1, 2}));
    EXPECT_EQ(scheduler.request_manager().get(2).state(), RequestState::RUNNING);

    const auto second_step = scheduler.step(0);
    EXPECT_EQ(second_step.completed_requests, (std::vector<RequestId>{2}));
    EXPECT_EQ(scheduler.block_manager().used_block_count(), 0);
}

TEST(SchedulerTest, ReportsPermanentlyUnserviceableRequestsWithoutCancellingThem) {
    Scheduler scheduler({2, 8}, 1, 4);
    ASSERT_EQ(scheduler.submit(Request(1, 5, 1, 0)), SubmissionResult::ACCEPTED);

    const auto result = scheduler.step(0);

    EXPECT_EQ(result.status, SchedulerStepStatus::PERMANENTLY_UNSERVICEABLE);
    EXPECT_EQ(result.permanently_unserviceable_requests, (std::vector<RequestId>{1}));
    EXPECT_EQ(scheduler.request_manager().get(1).state(), RequestState::WAITING);
    EXPECT_EQ(scheduler.block_manager().used_block_count(), 0);
}

TEST(SchedulerTest, DoesNotBypassAnUnserviceableOldestEligibleRequest) {
    Scheduler scheduler({2, 8}, 1, 4);
    ASSERT_EQ(scheduler.submit(Request(1, 5, 1, 0)), SubmissionResult::ACCEPTED);
    ASSERT_EQ(scheduler.submit(Request(2, 1, 1, 0)), SubmissionResult::ACCEPTED);

    const auto result = scheduler.step(0);

    EXPECT_TRUE(result.admitted_requests.empty());
    EXPECT_EQ(result.permanently_unserviceable_requests,
              (std::vector<RequestId>{1}));
    EXPECT_EQ(scheduler.request_manager().waiting_requests(),
              (std::vector<RequestId>{1, 2}));
    EXPECT_EQ(scheduler.request_manager().get(2).state(), RequestState::WAITING);
}

TEST(SchedulerTest, UsesDeterministicRoundRobinDecodeOrder) {
    Scheduler scheduler({2, 8}, 8, 4);
    ASSERT_EQ(scheduler.submit(Request(2, 1, 3, 0)), SubmissionResult::ACCEPTED);
    ASSERT_EQ(scheduler.submit(Request(1, 1, 3, 0)), SubmissionResult::ACCEPTED);

    const auto result = scheduler.step(0);

    EXPECT_EQ(result.admitted_requests, (std::vector<RequestId>{2, 1}));
    EXPECT_EQ(result.generated_tokens,
              (std::vector<RequestId>{2, 1, 2, 1, 2, 1}));
    EXPECT_EQ(result.completed_requests, (std::vector<RequestId>{2, 1}));
}

TEST(SchedulerTest, ReportsEmptyWorkload) {
    Scheduler scheduler({2, 4}, 4, 4);

    const auto result = scheduler.step(0);

    EXPECT_EQ(result.status, SchedulerStepStatus::EMPTY_WORKLOAD);
    EXPECT_FALSE(result.made_progress);
    EXPECT_FALSE(result.future_arrivals_remain);
}

TEST(SchedulerTest, AdvancesByReturningTheNextFutureArrivalTimestamp) {
    Scheduler scheduler({1, 4}, 4, 4);
    ASSERT_EQ(scheduler.submit(Request(1, 1, 1, 10)), SubmissionResult::ACCEPTED);

    const auto waiting_result = scheduler.step(0);
    ASSERT_EQ(waiting_result.status, SchedulerStepStatus::WAITING_FOR_ARRIVAL);
    ASSERT_TRUE(waiting_result.next_arrival_timestamp.has_value());

    const auto arrival_result = scheduler.step(*waiting_result.next_arrival_timestamp);
    EXPECT_EQ(arrival_result.admitted_requests, (std::vector<RequestId>{1}));
    EXPECT_EQ(scheduler.request_manager().get(1).state(), RequestState::COMPLETED);
}

TEST(SchedulerTest, DetectsDeadlockWhenActiveRequestsCannotGrowKvAllocations) {
    Scheduler scheduler({2, 20}, 3, 2);
    ASSERT_EQ(scheduler.submit(Request(1, 2, 4, 0)), SubmissionResult::ACCEPTED);
    ASSERT_EQ(scheduler.submit(Request(2, 2, 4, 0)), SubmissionResult::ACCEPTED);

    static_cast<void>(scheduler.step(0));
    const auto blocked_result = scheduler.step(0);

    EXPECT_EQ(blocked_result.status, SchedulerStepStatus::DEADLOCKED);
    EXPECT_EQ(blocked_result.memory_blocked_requests,
              (std::vector<RequestId>{1, 2}));
    EXPECT_FALSE(blocked_result.made_progress);
    EXPECT_EQ(scheduler.request_manager().get(2).generated_token_count(), 0);
    expect_capacity_invariants(scheduler);
}

TEST(SchedulerTest, CompetingRequestsNeverShareTheFinalFreeBlock) {
    Scheduler scheduler({2, 4}, 3, 2);
    ASSERT_EQ(scheduler.submit(Request(1, 2, 2, 0)), SubmissionResult::ACCEPTED);
    ASSERT_EQ(scheduler.submit(Request(2, 2, 2, 0)), SubmissionResult::ACCEPTED);

    const auto prefill_result = scheduler.step(0);
    ASSERT_EQ(prefill_result.prefill_completed_requests,
              (std::vector<RequestId>{1, 2}));
    ASSERT_EQ(scheduler.block_manager().free_block_count(), 1);

    const auto decode_result = scheduler.step(0);
    EXPECT_EQ(decode_result.memory_blocked_requests, (std::vector<RequestId>{2}));
    EXPECT_EQ(decode_result.generated_tokens, (std::vector<RequestId>{1, 1, 2, 2}));
    EXPECT_EQ(decode_result.completed_requests, (std::vector<RequestId>{1, 2}));
    EXPECT_EQ(scheduler.block_manager().used_block_count(), 0);
    expect_capacity_invariants(scheduler);
}

TEST(SchedulerTest, PreservesCapacityInvariantsAcrossSeveralActiveRequests) {
    Scheduler scheduler({3, 3}, 12, 4);
    ASSERT_EQ(scheduler.submit(Request(1, 2, 8, 0)), SubmissionResult::ACCEPTED);
    ASSERT_EQ(scheduler.submit(Request(2, 3, 8, 0)), SubmissionResult::ACCEPTED);
    ASSERT_EQ(scheduler.submit(Request(3, 1, 8, 0)), SubmissionResult::ACCEPTED);

    const auto result = scheduler.step(0);

    EXPECT_LE(result.tokens_processed, scheduler.config().max_tokens_per_step);
    expect_capacity_invariants(scheduler);
}

}  // namespace
}  // namespace blockstride
