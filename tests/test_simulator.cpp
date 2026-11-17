#include "blockstride/simulator.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace blockstride {
namespace {

SimulationOutcome run_workload(SchedulerConfig scheduler_config, std::size_t blocks,
                              std::size_t tokens_per_block, SimulationConfig costs,
                              std::vector<Request> requests) {
    ExecutionSimulator simulator(scheduler_config, blocks, tokens_per_block, costs);
    static_cast<void>(simulator.submit_workload(std::move(requests)));
    return simulator.run();
}

TEST(SimulationClockTest, AdvancesDeterministicallyAndRejectsBackwardTime) {
    SimulationClock clock(5);

    clock.advance_by(3);
    EXPECT_EQ(clock.now(), 8);
    clock.advance_to(12);
    EXPECT_EQ(clock.now(), 12);
    EXPECT_THROW(clock.advance_to(11), std::invalid_argument);
}

TEST(SimulationClockTest, ReportsDurationOverflowWithoutChangingTime) {
    SimulationClock clock(std::numeric_limits<SimulatedTimestamp>::max() - 1);

    EXPECT_FALSE(clock.try_advance_by(2));
    EXPECT_EQ(clock.now(), std::numeric_limits<SimulatedTimestamp>::max() - 1);
    EXPECT_THROW(clock.advance_by(2), std::overflow_error);
}

TEST(SimulationCostModelTest, CalculatesPrefillAndDecodeCosts) {
    const SimulationCostModel model({10, 2, 5, 3});

    EXPECT_EQ(model.prefill_cost(4), 18);
    EXPECT_EQ(model.decode_batch_cost(3), 14);
}

TEST(SimulationCostModelTest, ZeroWorkHasNoExecutionCost) {
    const SimulationCostModel model({10, 2, 5, 3});

    EXPECT_EQ(model.prefill_cost(0), 0);
    EXPECT_EQ(model.decode_batch_cost(0), 0);
}

TEST(SimulationCostModelTest, DetectsCostArithmeticOverflow) {
    const SimulationCostModel model(
        {0, std::numeric_limits<std::uint64_t>::max(), 0, 0});

    EXPECT_THROW(static_cast<void>(model.prefill_cost(2)), std::overflow_error);
}

TEST(SimulationConfigTest, RejectsOneTokenCostThatCannotBeRepresented) {
    EXPECT_THROW(
        SimulationCostModel(
            {std::numeric_limits<std::uint64_t>::max(), 1, 0, 0}),
        std::overflow_error);
}

TEST(ExecutionSimulatorTest, ReportsExplicitFirstTokenAndCompletionTimes) {
    ExecutionSimulator simulator({1, 2}, 8, 4, {10, 2, 5, 3});
    ASSERT_EQ(simulator.submit(Request(1, 1, 2, 0)), SubmissionResult::ACCEPTED);

    const auto outcome = simulator.run();

    ASSERT_EQ(outcome.status, SimulationStatus::COMPLETED);
    ASSERT_EQ(outcome.requests.size(), 1);
    EXPECT_EQ(outcome.requests[0].start_timestamp, 0);
    EXPECT_EQ(outcome.requests[0].first_token_timestamp, 20);
    EXPECT_EQ(outcome.requests[0].completion_timestamp, 28);
    EXPECT_EQ(outcome.requests[0].time_to_first_token_us, 20);
    EXPECT_EQ(outcome.requests[0].completion_latency_us, 28);
    EXPECT_EQ(outcome.requests[0].generated_tokens, 2);
    EXPECT_EQ(outcome.statistics.prefill_execution_time_us, 12);
    EXPECT_EQ(outcome.statistics.decode_execution_time_us, 16);
    EXPECT_EQ(outcome.statistics.decode_batches, 2);
    EXPECT_EQ(outcome.final_timestamp, 28);
}

TEST(ExecutionSimulatorTest, JumpsDirectlyToStaggeredFutureArrivals) {
    ExecutionSimulator simulator({1, 2}, 4, 4, {0, 0, 5, 0});
    ASSERT_EQ(simulator.submit_workload({Request(1, 1, 1, 0), Request(2, 1, 1, 10)}),
              (std::vector<SubmissionResult>{SubmissionResult::ACCEPTED,
                                             SubmissionResult::ACCEPTED}));

    const auto outcome = simulator.run();

    ASSERT_EQ(outcome.status, SimulationStatus::COMPLETED);
    EXPECT_EQ(outcome.statistics.scheduler_steps, 2);
    ASSERT_EQ(outcome.requests.size(), 2);
    EXPECT_EQ(outcome.requests[0].start_timestamp, 0);
    EXPECT_EQ(outcome.requests[1].start_timestamp, 10);
    EXPECT_EQ(outcome.requests[1].first_token_timestamp, 15);
    EXPECT_EQ(outcome.final_timestamp, 15);
}

TEST(ExecutionSimulatorTest, ArrivalDuringADecodeBatchWaitsUntilTheNextBoundary) {
    ExecutionSimulator simulator({2, 2}, 8, 4, {0, 0, 10, 0});
    static_cast<void>(
        simulator.submit_workload({Request(1, 1, 1, 0), Request(2, 1, 1, 2)}));

    const auto outcome = simulator.run();

    ASSERT_EQ(outcome.status, SimulationStatus::COMPLETED);
    ASSERT_EQ(outcome.requests.size(), 2);
    EXPECT_EQ(outcome.requests[0].first_token_timestamp, 10);
    EXPECT_EQ(outcome.requests[1].start_timestamp, 10);
    EXPECT_EQ(outcome.requests[1].first_token_timestamp, 20);
}

TEST(ExecutionSimulatorTest, RecordsStartTimeWhenCapacityBecomesAvailable) {
    ExecutionSimulator simulator({1, 4}, 4, 4, {0, 0, 10, 0});
    static_cast<void>(
        simulator.submit_workload({Request(1, 1, 1, 0), Request(2, 1, 1, 0)}));

    const auto outcome = simulator.run();

    ASSERT_EQ(outcome.status, SimulationStatus::COMPLETED);
    EXPECT_EQ(outcome.requests[0].start_timestamp, 0);
    EXPECT_EQ(outcome.requests[1].start_timestamp, 10);
    EXPECT_EQ(outcome.requests[1].first_token_timestamp, 20);
}

TEST(ExecutionSimulatorTest, ChargesDecodeFixedOverheadOncePerBatch) {
    ExecutionSimulator simulator({2, 4}, 8, 4, {0, 0, 7, 0});
    static_cast<void>(
        simulator.submit_workload({Request(1, 1, 1, 0), Request(2, 1, 1, 0)}));

    const auto outcome = simulator.run();

    ASSERT_EQ(outcome.status, SimulationStatus::COMPLETED);
    EXPECT_EQ(outcome.statistics.decode_batches, 1);
    EXPECT_EQ(outcome.statistics.decode_execution_time_us, 7);
    EXPECT_EQ(outcome.requests[0].completion_timestamp, 7);
    EXPECT_EQ(outcome.requests[1].completion_timestamp, 7);
}

TEST(ExecutionSimulatorTest, ContinuesThroughTemporaryMemoryPressureAndReclaimsBlocks) {
    ExecutionSimulator simulator({2, 4}, 3, 1, {0, 0, 0, 0});
    static_cast<void>(
        simulator.submit_workload({Request(1, 1, 1, 0), Request(2, 1, 1, 0)}));

    const auto outcome = simulator.run();

    ASSERT_EQ(outcome.status, SimulationStatus::COMPLETED);
    EXPECT_EQ(outcome.memory_blocked_requests, (std::vector<RequestId>{2}));
    EXPECT_EQ(outcome.statistics.generated_tokens, 2);
    EXPECT_EQ(simulator.scheduler().block_manager().used_block_count(), 0);
}

TEST(ExecutionSimulatorTest, DetectsActiveRequestMemoryDeadlock) {
    ExecutionSimulator simulator({2, 20}, 3, 2, {0, 0, 0, 0});
    static_cast<void>(
        simulator.submit_workload({Request(1, 2, 4, 0), Request(2, 2, 4, 0)}));

    const auto outcome = simulator.run();

    EXPECT_EQ(outcome.status, SimulationStatus::DEADLOCKED);
    EXPECT_FALSE(outcome.outstanding_requests.empty());
    EXPECT_EQ(simulator.scheduler().block_manager().used_block_count(), 3);
}

TEST(ExecutionSimulatorTest, ReportsPermanentlyUnserviceableRequests) {
    const auto outcome =
        run_workload({1, 8}, 1, 4, {0, 0, 0, 0}, {Request(1, 5, 1, 0)});

    EXPECT_EQ(outcome.status, SimulationStatus::PERMANENTLY_UNSERVICEABLE);
    EXPECT_EQ(outcome.permanently_unserviceable_requests,
              (std::vector<RequestId>{1}));
    EXPECT_EQ(outcome.requests[0].final_state, RequestState::WAITING);
}

TEST(ExecutionSimulatorTest, EmptyWorkloadTerminatesWithoutAWorkStep) {
    ExecutionSimulator simulator({1, 1}, 1, 1, {});

    const auto outcome = simulator.run();

    EXPECT_EQ(outcome.status, SimulationStatus::EMPTY_WORKLOAD);
    EXPECT_EQ(outcome.statistics.scheduler_steps, 0);
    EXPECT_EQ(outcome.final_timestamp, 0);
}

TEST(ExecutionSimulatorTest, ReportsTimeOverflowWithoutRecordingUnexecutedTokens) {
    ExecutionSimulator simulator(
        {1, 4}, 4, 4,
        {0, std::numeric_limits<std::uint64_t>::max(), 0, 0});
    ASSERT_EQ(simulator.submit(Request(1, 2, 1, 0)), SubmissionResult::ACCEPTED);

    const auto outcome = simulator.run();

    EXPECT_EQ(outcome.status, SimulationStatus::TIME_OVERFLOW);
    ASSERT_EQ(outcome.requests.size(), 1);
    EXPECT_EQ(outcome.requests[0].generated_tokens, 0);
    EXPECT_FALSE(outcome.requests[0].first_token_timestamp.has_value());
    EXPECT_EQ(outcome.final_timestamp, 0);
    EXPECT_EQ(outcome.statistics.prefill_tokens, 0);
}

TEST(ExecutionSimulatorTest, HandlesClockOverflowAtAnArrivalBoundary) {
    const auto latest_arrival = std::numeric_limits<SimulatedTimestamp>::max() - 1;
    ExecutionSimulator simulator({1, 2}, 4, 4, {0, 0, 2, 0});
    ASSERT_EQ(simulator.submit(Request(1, 1, 1, latest_arrival)),
              SubmissionResult::ACCEPTED);

    const auto outcome = simulator.run();

    EXPECT_EQ(outcome.status, SimulationStatus::TIME_OVERFLOW);
    EXPECT_EQ(outcome.final_timestamp, latest_arrival);
    EXPECT_EQ(outcome.requests[0].generated_tokens, 0);
}

TEST(ExecutionSimulatorTest, RejectsArrivalTimesEarlierThanTheCurrentClock) {
    ExecutionSimulator simulator({1, 2}, 4, 4, {0, 0, 1, 0});
    ASSERT_EQ(simulator.submit(Request(1, 1, 1, 5)), SubmissionResult::ACCEPTED);
    ASSERT_EQ(simulator.run().status, SimulationStatus::COMPLETED);

    EXPECT_THROW(static_cast<void>(simulator.submit(Request(2, 1, 1, 4))),
                 std::invalid_argument);
}

TEST(ExecutionSimulatorTest, ReproducesTheSameWorkloadAndTimestamps) {
    const SimulationConfig costs{3, 2, 4, 1};
    const std::vector<Request> requests{Request(1, 2, 2, 0), Request(2, 1, 1, 6)};

    const auto first = run_workload({2, 4}, 8, 2, costs, requests);
    const auto second = run_workload({2, 4}, 8, 2, costs, requests);

    ASSERT_EQ(first.status, SimulationStatus::COMPLETED);
    ASSERT_EQ(second.status, SimulationStatus::COMPLETED);
    EXPECT_EQ(first.final_timestamp, second.final_timestamp);
    ASSERT_EQ(first.requests.size(), second.requests.size());
    for (std::size_t index = 0; index < first.requests.size(); ++index) {
        EXPECT_EQ(first.requests[index].start_timestamp, second.requests[index].start_timestamp);
        EXPECT_EQ(first.requests[index].first_token_timestamp,
                  second.requests[index].first_token_timestamp);
        EXPECT_EQ(first.requests[index].completion_timestamp,
                  second.requests[index].completion_timestamp);
        EXPECT_EQ(first.requests[index].generated_tokens, second.requests[index].generated_tokens);
    }
}

TEST(ExecutionSimulatorTest, RecordsRejectedPromptsWithoutChargingExecutionCosts) {
    ExecutionSimulator simulator({1, 1}, 4, 4, {7, 3, 5, 2});
    ASSERT_EQ(simulator.submit(Request(1, 2, 1, 0)),
              SubmissionResult::PROMPT_EXCEEDS_STEP_TOKEN_BUDGET);

    const auto outcome = simulator.run();

    EXPECT_EQ(outcome.status, SimulationStatus::EMPTY_WORKLOAD);
    ASSERT_EQ(outcome.rejected_requests.size(), 1);
    EXPECT_EQ(outcome.rejected_requests[0].request_id, 1);
    EXPECT_EQ(outcome.statistics.prefill_execution_time_us, 0);
    EXPECT_EQ(outcome.statistics.decode_execution_time_us, 0);
}

TEST(ExecutionSimulatorTest, CancellationReclaimsBlocksAndRecordsTerminalState) {
    ExecutionSimulator simulator({2, 20}, 3, 2, {0, 0, 0, 0});
    static_cast<void>(
        simulator.submit_workload({Request(1, 1, 5, 0), Request(2, 1, 5, 0)}));
    EXPECT_EQ(simulator.run().status, SimulationStatus::DEADLOCKED);
    ASSERT_EQ(simulator.scheduler().block_manager().used_block_count(), 3);

    simulator.cancel(1);
    const auto outcome = simulator.run();

    EXPECT_EQ(outcome.status, SimulationStatus::COMPLETED);
    EXPECT_EQ(outcome.statistics.cancelled_requests, 1);
    EXPECT_EQ(outcome.requests[0].final_state, RequestState::CANCELLED);
    EXPECT_EQ(simulator.scheduler().block_manager().used_block_count(), 0);
}

TEST(ExecutionSimulatorTest, CancellingFutureWaitingRequestAdvancesNoExecutionCosts) {
    ExecutionSimulator simulator({1, 2}, 2, 1, {10, 2, 5, 3});
    ASSERT_EQ(simulator.submit(Request(1, 1, 1, 10)), SubmissionResult::ACCEPTED);

    simulator.cancel(1);
    const auto outcome = simulator.run();

    EXPECT_EQ(outcome.status, SimulationStatus::COMPLETED);
    EXPECT_EQ(outcome.final_timestamp, 10);
    EXPECT_EQ(outcome.statistics.cancelled_requests, 1);
    EXPECT_EQ(outcome.statistics.prefill_execution_time_us, 0);
    EXPECT_EQ(outcome.requests[0].final_state, RequestState::CANCELLED);
}

}  // namespace
}  // namespace blockstride
