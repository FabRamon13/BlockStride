#include "blockstride/request.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace blockstride {
namespace {

TEST(RequestTest, FollowsValidLifecycleToCompletion) {
    Request request(1, 4, 2, 10);

    EXPECT_EQ(request.state(), RequestState::WAITING);
    EXPECT_FALSE(request.terminal_timestamp().has_value());
    request.start(10);
    EXPECT_EQ(request.state(), RequestState::RUNNING);
    request.mark_prefill_completed();
    request.record_generated_token(12);
    request.record_generated_token(13);
    request.complete(14);

    EXPECT_EQ(request.state(), RequestState::COMPLETED);
    EXPECT_EQ(request.generated_token_count(), 2);
    ASSERT_TRUE(request.completion_timestamp().has_value());
    EXPECT_EQ(*request.completion_timestamp(), 14);
    ASSERT_TRUE(request.terminal_timestamp().has_value());
    EXPECT_EQ(*request.terminal_timestamp(), 14);
}

TEST(RequestTest, RejectsInvalidTransitionsAndGenerationBeforePrefill) {
    Request request(1, 4, 1, 10);

    EXPECT_THROW(request.mark_prefill_completed(), std::logic_error);
    EXPECT_THROW(request.record_generated_token(10), std::logic_error);
    EXPECT_THROW(request.complete(10), std::logic_error);

    request.start(10);
    EXPECT_THROW(request.start(10), std::logic_error);
    EXPECT_THROW(request.record_generated_token(10), std::logic_error);
}

TEST(RequestTest, RequiresNonzeroPromptAndGenerationLimit) {
    EXPECT_THROW(Request(1, 0, 1, 0), std::invalid_argument);
    EXPECT_THROW(Request(1, 1, 0, 0), std::invalid_argument);
}

TEST(RequestTest, EnforcesGenerationLimit) {
    Request request(1, 2, 1, 0);
    request.start(0);
    request.mark_prefill_completed();
    request.record_generated_token(1);

    EXPECT_THROW(request.record_generated_token(2), std::overflow_error);
    EXPECT_EQ(request.generated_token_count(), 1);
}

TEST(RequestTest, AssignsFirstTokenTimestampOnlyOnce) {
    Request request(1, 2, 2, 0);
    request.start(0);
    request.mark_prefill_completed();
    request.record_generated_token(5);
    request.record_generated_token(8);

    ASSERT_TRUE(request.first_token_timestamp().has_value());
    EXPECT_EQ(*request.first_token_timestamp(), 5);
}

TEST(RequestTest, CompletionRequiresPrefillTokenLimitAndValidTimestamp) {
    Request request(1, 2, 1, 10);
    request.start(10);
    EXPECT_THROW(request.complete(10), std::logic_error);
    request.mark_prefill_completed();
    EXPECT_THROW(request.complete(10), std::logic_error);
    request.record_generated_token(12);

    EXPECT_THROW(request.complete(11), std::invalid_argument);
    EXPECT_FALSE(request.completion_timestamp().has_value());
    request.complete(12);

    EXPECT_THROW(request.complete(12), std::logic_error);
    EXPECT_TRUE(request.completion_timestamp().has_value());
}

TEST(RequestTest, RejectsTimestampsBeforeArrival) {
    Request request(1, 2, 1, 10);

    EXPECT_THROW(request.start(9), std::invalid_argument);
    request.start(10);
    request.mark_prefill_completed();
    EXPECT_THROW(request.record_generated_token(9), std::invalid_argument);
    EXPECT_THROW(request.cancel(9), std::invalid_argument);
}

TEST(RequestTest, SupportsCancellationFromWaitingAndRunning) {
    Request waiting(1, 2, 1, 10);
    EXPECT_FALSE(waiting.terminal_timestamp().has_value());
    waiting.cancel(10);
    EXPECT_EQ(waiting.state(), RequestState::CANCELLED);
    ASSERT_TRUE(waiting.terminal_timestamp().has_value());
    EXPECT_EQ(*waiting.terminal_timestamp(), 10);

    Request running(2, 2, 1, 10);
    running.start(10);
    running.mark_prefill_completed();
    running.record_generated_token(11);
    running.cancel(11);
    EXPECT_EQ(running.state(), RequestState::CANCELLED);
    ASSERT_TRUE(running.terminal_timestamp().has_value());
    EXPECT_EQ(*running.terminal_timestamp(), 11);
    EXPECT_FALSE(running.completion_timestamp().has_value());
}

TEST(RequestTest, RunningCancellationCannotPrecedeFirstToken) {
    Request request(1, 2, 2, 10);
    request.start(10);
    request.mark_prefill_completed();
    request.record_generated_token(15);

    EXPECT_THROW(request.cancel(14), std::invalid_argument);
    EXPECT_EQ(request.state(), RequestState::RUNNING);
    EXPECT_FALSE(request.terminal_timestamp().has_value());

    request.cancel(16);
    EXPECT_EQ(request.terminal_timestamp(), 16);
}

TEST(RequestTest, TerminalStatesAreImmutable) {
    Request completed(1, 2, 1, 0);
    completed.start(0);
    completed.mark_prefill_completed();
    completed.record_generated_token(1);
    completed.complete(1);
    EXPECT_THROW(completed.cancel(2), std::logic_error);
    EXPECT_EQ(*completed.terminal_timestamp(), 1);

    Request cancelled(2, 2, 1, 0);
    cancelled.cancel(0);
    EXPECT_THROW(cancelled.start(1), std::logic_error);
    EXPECT_THROW(cancelled.cancel(1), std::logic_error);
    EXPECT_EQ(*cancelled.terminal_timestamp(), 0);
}

}  // namespace
}  // namespace blockstride
