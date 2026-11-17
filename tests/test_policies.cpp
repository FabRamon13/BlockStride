#include "blockstride/simulator.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace blockstride {
namespace {

constexpr std::array<SchedulingPolicy, 3> kPolicies{
    SchedulingPolicy::SEQUENTIAL,
    SchedulingPolicy::STATIC_BATCHING,
    SchedulingPolicy::CONTINUOUS_BATCHING,
};

SimulationOutcome run_policy(SchedulingPolicy policy,
                             const std::vector<WorkloadRequest>& workload,
                             SchedulerConfig scheduler_config = {2, 8},
                             std::size_t blocks = 16, std::size_t tokens_per_block = 4,
                             SimulationConfig costs = {}) {
    scheduler_config.policy = policy;
    ExecutionSimulator simulator(scheduler_config, blocks, tokens_per_block, costs);
    std::vector<Request> requests;
    requests.reserve(workload.size());
    for (const auto& specification : workload) {
        requests.emplace_back(specification.id, specification.prompt_token_count,
                              specification.max_new_tokens,
                              specification.arrival_timestamp);
    }
    static_cast<void>(simulator.submit_workload(std::move(requests)));
    return simulator.run();
}

TEST(SchedulingPolicyTest, PolicyNamesAndDefaultAreExplicit) {
    Scheduler scheduler({2, 4}, 8, 4);

    EXPECT_EQ(scheduler.config().policy, SchedulingPolicy::CONTINUOUS_BATCHING);
    EXPECT_EQ(scheduling_policy_name(SchedulingPolicy::SEQUENTIAL), "sequential");
    EXPECT_EQ(scheduling_policy_name(SchedulingPolicy::STATIC_BATCHING), "static-batching");
    EXPECT_EQ(scheduling_policy_name(SchedulingPolicy::CONTINUOUS_BATCHING),
              "continuous-batching");
}

TEST(SchedulingPolicyTest, SequentialLimitsConcurrencyAndPreservesFcfsTies) {
    const std::vector<WorkloadRequest> workload{
        {2, 1, 1, 0},
        {1, 1, 1, 0},
        {3, 1, 1, 0},
    };
    const auto outcome =
        run_policy(SchedulingPolicy::SEQUENTIAL, workload, {3, 4}, 8, 4, {0, 0, 2, 0});

    ASSERT_EQ(outcome.status, SimulationStatus::COMPLETED);
    ASSERT_EQ(outcome.requests.size(), 3);
    EXPECT_EQ(outcome.requests[0].request_id, 2);
    EXPECT_EQ(outcome.requests[1].request_id, 1);
    EXPECT_EQ(outcome.requests[2].request_id, 3);
    EXPECT_EQ(outcome.requests[0].start_timestamp, 0);
    EXPECT_EQ(outcome.requests[0].completion_timestamp, 2);
    EXPECT_EQ(outcome.requests[1].start_timestamp, 2);
    EXPECT_EQ(outcome.requests[1].completion_timestamp, 4);
    EXPECT_EQ(outcome.requests[2].start_timestamp, 4);
    EXPECT_EQ(outcome.statistics.completed_requests, 3);
}

TEST(SchedulingPolicyTest, StaticBatchFormsAnEligibleCohortAndDefersTheNextOne) {
    const std::vector<WorkloadRequest> workload{
        {1, 1, 1, 0},
        {2, 1, 1, 0},
        {3, 1, 1, 0},
    };
    const auto outcome = run_policy(SchedulingPolicy::STATIC_BATCHING, workload, {2, 8}, 8, 4,
                                    {0, 0, 5, 0});

    ASSERT_EQ(outcome.status, SimulationStatus::COMPLETED);
    ASSERT_EQ(outcome.requests.size(), 3);
    EXPECT_EQ(outcome.requests[0].start_timestamp, 0);
    EXPECT_EQ(outcome.requests[1].start_timestamp, 0);
    EXPECT_EQ(outcome.requests[2].start_timestamp, 5);
    EXPECT_EQ(outcome.requests[2].completion_timestamp, 10);
}

TEST(SchedulingPolicyTest, StaticCohortBarrierWaitsForEveryMember) {
    const std::vector<WorkloadRequest> workload{
        {1, 1, 1, 0},
        {2, 1, 3, 0},
        {3, 1, 1, 0},
    };
    const auto outcome = run_policy(SchedulingPolicy::STATIC_BATCHING, workload, {2, 4}, 8, 4,
                                    {0, 0, 5, 0});

    ASSERT_EQ(outcome.status, SimulationStatus::COMPLETED);
    EXPECT_EQ(outcome.requests[0].completion_timestamp, 5);
    EXPECT_EQ(outcome.requests[1].completion_timestamp, 15);
    EXPECT_EQ(outcome.requests[2].start_timestamp, 15);
}

TEST(SchedulingPolicyTest, StaticCohortDoesNotExceedAvailablePromptMemory) {
    Scheduler scheduler({3, 4, SchedulingPolicy::STATIC_BATCHING}, 1, 2);
    ASSERT_EQ(scheduler.submit(Request(1, 1, 1, 0)), SubmissionResult::ACCEPTED);
    ASSERT_EQ(scheduler.submit(Request(2, 1, 1, 0)), SubmissionResult::ACCEPTED);

    const auto first_step = scheduler.step(0);

    EXPECT_EQ(first_step.admitted_requests, (std::vector<RequestId>{1}));
    EXPECT_EQ(first_step.memory_blocked_requests, (std::vector<RequestId>{2}));
    EXPECT_EQ(scheduler.request_manager().get(2).state(), RequestState::WAITING);
    EXPECT_EQ(scheduler.block_manager().used_block_count(), 0);

    const auto next_cohort = scheduler.step(0);
    EXPECT_EQ(next_cohort.admitted_requests, (std::vector<RequestId>{2}));
}

TEST(SchedulingPolicyTest, ContinuousBatchingAdmitsReplacementWhileCohortMemberRuns) {
    const std::vector<WorkloadRequest> workload{
        {1, 1, 1, 0},
        {2, 1, 4, 0},
        {3, 1, 1, 0},
    };
    const auto outcome = run_policy(SchedulingPolicy::CONTINUOUS_BATCHING, workload, {2, 4}, 16,
                                    4, {0, 0, 3, 0});

    ASSERT_EQ(outcome.status, SimulationStatus::COMPLETED);
    EXPECT_LT(*outcome.requests[2].start_timestamp, *outcome.requests[1].completion_timestamp);
}

TEST(SchedulingPolicyTest, ComparisonRunsUseTheSameImmutableWorkloadAndResources) {
    const std::vector<WorkloadRequest> workload{
        {1, 1, 2, 0},
        {2, 1, 2, 0},
        {3, 1, 2, 4},
    };
    const auto results =
        compare_scheduling_policies(workload, {2, 4}, 16, 4, {1, 1, 2, 1});

    ASSERT_EQ(results.size(), kPolicies.size());
    for (std::size_t index = 0; index < results.size(); ++index) {
        EXPECT_EQ(results[index].policy, kPolicies[index]);
        ASSERT_EQ(results[index].outcome.status, SimulationStatus::COMPLETED);
        EXPECT_EQ(results[index].outcome.statistics.completed_requests, 3);
        EXPECT_EQ(results[index].outcome.statistics.generated_tokens, 6);
        ASSERT_EQ(results[index].outcome.requests.size(), workload.size());
        for (std::size_t request_index = 0; request_index < workload.size(); ++request_index) {
            EXPECT_EQ(results[index].outcome.requests[request_index].request_id,
                      workload[request_index].id);
            EXPECT_EQ(results[index].outcome.requests[request_index].arrival_timestamp,
                      workload[request_index].arrival_timestamp);
        }
    }
    EXPECT_EQ(workload[0].max_new_tokens, 2);
}

TEST(SchedulingPolicyTest, EveryPolicyEnforcesGenerationLimitsAndReclaimsKvBlocks) {
    const std::vector<WorkloadRequest> workload{{7, 2, 3, 0}};
    for (const auto policy : kPolicies) {
        SchedulerConfig config{2, 8, policy};
        ExecutionSimulator simulator(config, 8, 4, {0, 0, 1, 0});
        ASSERT_EQ(simulator.submit(Request(7, 2, 3, 0)), SubmissionResult::ACCEPTED);

        const auto outcome = simulator.run();

        EXPECT_EQ(outcome.status, SimulationStatus::COMPLETED);
        EXPECT_EQ(outcome.requests[0].generated_tokens, 3);
        EXPECT_EQ(outcome.statistics.completed_requests, 1);
        EXPECT_EQ(simulator.scheduler().request_manager().get(7).generated_token_count(), 3);
        EXPECT_EQ(simulator.scheduler().block_manager().used_block_count(), 0);
    }
}

TEST(SchedulingPolicyTest, DeterministicOutputsRepeatForEachPolicy) {
    const std::vector<WorkloadRequest> workload{{1, 2, 2, 0}, {2, 1, 1, 3}};
    for (const auto policy : kPolicies) {
        const auto first = run_policy(policy, workload, {2, 4}, 8, 4, {2, 1, 3, 1});
        const auto second = run_policy(policy, workload, {2, 4}, 8, 4, {2, 1, 3, 1});

        EXPECT_EQ(first.status, second.status);
        EXPECT_EQ(first.final_timestamp, second.final_timestamp);
        ASSERT_EQ(first.requests.size(), second.requests.size());
        for (std::size_t index = 0; index < first.requests.size(); ++index) {
            EXPECT_EQ(first.requests[index].start_timestamp,
                      second.requests[index].start_timestamp);
            EXPECT_EQ(first.requests[index].first_token_timestamp,
                      second.requests[index].first_token_timestamp);
            EXPECT_EQ(first.requests[index].completion_timestamp,
                      second.requests[index].completion_timestamp);
            EXPECT_EQ(first.requests[index].generated_tokens,
                      second.requests[index].generated_tokens);
        }
    }
}

TEST(SchedulingPolicyTest, AllPoliciesHandleEmptyWorkloads) {
    for (const auto policy : kPolicies) {
        const auto outcome = run_policy(policy, {}, {2, 4});
        EXPECT_EQ(outcome.status, SimulationStatus::EMPTY_WORKLOAD);
        EXPECT_EQ(outcome.statistics.scheduler_steps, 0);
    }
}

TEST(SchedulingPolicyTest, AllPoliciesAdvanceToFutureArrivals) {
    const std::vector<WorkloadRequest> workload{{1, 1, 1, 10}};
    for (const auto policy : kPolicies) {
        const auto outcome = run_policy(policy, workload, {2, 2}, 4, 4, {0, 0, 2, 0});

        ASSERT_EQ(outcome.status, SimulationStatus::COMPLETED);
        EXPECT_EQ(outcome.requests[0].start_timestamp, 10);
        EXPECT_EQ(outcome.requests[0].first_token_timestamp, 12);
    }
}

TEST(SchedulingPolicyTest, RequestsArrivingDuringDecodeJoinOnlyAtNextBoundary) {
    const std::vector<WorkloadRequest> workload{
        {1, 1, 1, 0},
        {2, 1, 1, 2},
    };
    for (const auto policy : kPolicies) {
        const auto outcome = run_policy(policy, workload, {2, 2}, 8, 4, {0, 0, 10, 0});

        ASSERT_EQ(outcome.status, SimulationStatus::COMPLETED);
        EXPECT_EQ(outcome.requests[0].first_token_timestamp, 10);
        EXPECT_EQ(outcome.requests[1].start_timestamp, 10);
        EXPECT_EQ(outcome.requests[1].first_token_timestamp, 20);
    }
}

TEST(SchedulingPolicyTest, AllPoliciesReportPermanentlyUnserviceableRequests) {
    const std::vector<WorkloadRequest> workload{{1, 5, 1, 0}};
    for (const auto policy : kPolicies) {
        const auto outcome = run_policy(policy, workload, {2, 8}, 1, 4);

        EXPECT_EQ(outcome.status, SimulationStatus::PERMANENTLY_UNSERVICEABLE);
        EXPECT_EQ(outcome.requests[0].final_state, RequestState::WAITING);
        EXPECT_EQ(outcome.statistics.generated_tokens, 0);
    }
}

TEST(SchedulingPolicyTest, CommonFeasibleWorkloadHasEqualCompletionTotals) {
    const std::vector<WorkloadRequest> workload{{1, 1, 1, 0}, {2, 1, 1, 0}};
    const auto results =
        compare_scheduling_policies(workload, {2, 4}, 8, 4, {2, 0, 5, 0});

    ASSERT_EQ(results.size(), 3);
    for (const auto& result : results) {
        EXPECT_EQ(result.outcome.status, SimulationStatus::COMPLETED);
        EXPECT_EQ(result.outcome.statistics.completed_requests, 2);
        EXPECT_EQ(result.outcome.statistics.generated_tokens, 2);
        EXPECT_EQ(result.outcome.statistics.prefill_execution_time_us, 4);
    }
    EXPECT_EQ(results[1].outcome.requests[0].first_token_timestamp, 9);
    EXPECT_EQ(results[1].outcome.requests[1].first_token_timestamp, 9);
    EXPECT_EQ(results[2].outcome.requests[0].completion_timestamp, 9);
    EXPECT_EQ(results[2].outcome.requests[1].completion_timestamp, 9);
}

TEST(SchedulingPolicyTest, PoliciesCanHaveDifferentFeasibilityUnderSharedCapacity) {
    const std::vector<WorkloadRequest> workload{{1, 1, 5, 0}, {2, 1, 5, 0}};
    const auto results = compare_scheduling_policies(workload, {2, 20}, 3, 2, {});

    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].outcome.status, SimulationStatus::COMPLETED);
    EXPECT_EQ(results[0].outcome.statistics.completed_requests, 2);
    EXPECT_EQ(results[1].outcome.status, SimulationStatus::DEADLOCKED);
    EXPECT_EQ(results[1].outcome.statistics.completed_requests, 0);
    EXPECT_EQ(results[2].outcome.status, SimulationStatus::DEADLOCKED);
    EXPECT_EQ(results[2].outcome.statistics.completed_requests, 0);
}

TEST(SchedulingPolicyTest, StaticCohortCancellationKeepsAdmissionBarrierUntilRemainingMemberEnds) {
    ExecutionSimulator simulator({2, 20, SchedulingPolicy::STATIC_BATCHING}, 3, 2, {});
    static_cast<void>(simulator.submit_workload(
        {Request(1, 1, 5, 0), Request(2, 1, 5, 0), Request(3, 1, 1, 0)}));
    ASSERT_EQ(simulator.run().status, SimulationStatus::DEADLOCKED);

    simulator.cancel(1);
    const auto outcome = simulator.run();

    ASSERT_EQ(outcome.status, SimulationStatus::COMPLETED);
    EXPECT_EQ(outcome.requests[0].final_state, RequestState::CANCELLED);
    EXPECT_EQ(outcome.requests[1].final_state, RequestState::COMPLETED);
    EXPECT_EQ(outcome.requests[2].start_timestamp, outcome.requests[1].completion_timestamp);
    EXPECT_EQ(outcome.statistics.cancelled_requests, 1);
}

TEST(SchedulingPolicyTest, PrefillAndDecodeTimestampsUseOperationEndBoundaries) {
    const std::vector<WorkloadRequest> workload{{1, 1, 2, 0}};
    for (const auto policy : kPolicies) {
        const auto outcome = run_policy(policy, workload, {2, 3}, 8, 4, {4, 2, 5, 1});

        ASSERT_EQ(outcome.status, SimulationStatus::COMPLETED);
        EXPECT_EQ(outcome.requests[0].start_timestamp, 0);
        EXPECT_EQ(outcome.requests[0].first_token_timestamp, 12);
        EXPECT_EQ(outcome.requests[0].completion_timestamp, 18);
        EXPECT_LE(*outcome.requests[0].start_timestamp,
                  *outcome.requests[0].first_token_timestamp);
        EXPECT_LE(*outcome.requests[0].first_token_timestamp,
                  *outcome.requests[0].completion_timestamp);
    }
}

}  // namespace
}  // namespace blockstride
