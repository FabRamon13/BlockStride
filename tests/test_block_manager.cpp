#include "blockstride/block_manager.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace blockstride {
namespace {

void expect_invariants(
    const BlockManager& manager,
    const std::unordered_map<RequestId, std::size_t>& expected_block_counts) {
    std::unordered_set<BlockId> allocated_ids;
    std::size_t expected_used_blocks = 0;

    for (const auto& [request_id, expected_count] : expected_block_counts) {
        const auto blocks = manager.allocated_blocks(request_id);
        EXPECT_EQ(blocks.size(), expected_count);
        expected_used_blocks += blocks.size();

        for (const auto block_id : blocks) {
            EXPECT_LT(block_id, manager.total_physical_blocks());
            EXPECT_TRUE(allocated_ids.insert(block_id).second)
                << "block " << block_id << " appears in multiple block tables";
        }
    }

    EXPECT_EQ(manager.used_block_count(), expected_used_blocks);
    EXPECT_EQ(manager.used_block_count() + manager.free_block_count(),
              manager.total_physical_blocks());
}

TEST(BlockManagerTest, RejectsInvalidConfiguration) {
    EXPECT_THROW(BlockManager(0, 16), std::invalid_argument);
    EXPECT_THROW(BlockManager(4, 0), std::invalid_argument);
}

TEST(BlockManagerTest, CalculatesZeroAndExactBlockBoundaries) {
    const BlockManager manager(8, 16);

    EXPECT_EQ(manager.blocks_needed(0), 0);
    EXPECT_EQ(manager.blocks_needed(1), 1);
    EXPECT_EQ(manager.blocks_needed(16), 1);
    EXPECT_EQ(manager.blocks_needed(17), 2);
    EXPECT_EQ(manager.blocks_needed(32), 2);
    EXPECT_EQ(manager.blocks_needed(33), 3);
}

TEST(BlockManagerTest, SupportsInitialAllocationAndPartialBlocks) {
    BlockManager manager(4, 16);

    EXPECT_EQ(manager.allocate_or_grow(1, 1), AllocationResult::SUCCESS);
    EXPECT_EQ(manager.allocated_blocks(1).size(), 1);
    EXPECT_EQ(manager.allocate_or_grow(2, 16), AllocationResult::SUCCESS);
    EXPECT_EQ(manager.allocated_blocks(2).size(), 1);
    EXPECT_EQ(manager.allocate_or_grow(3, 17), AllocationResult::SUCCESS);
    EXPECT_EQ(manager.allocated_blocks(3).size(), 2);
    EXPECT_EQ(manager.used_block_count(), 4);
    EXPECT_EQ(manager.free_block_count(), 0);
}

TEST(BlockManagerTest, GrowsIncrementallyAndNeverShrinksOnSmallerRequests) {
    BlockManager manager(5, 4);

    ASSERT_EQ(manager.allocate_or_grow(1, 1), AllocationResult::SUCCESS);
    const auto initial_blocks = manager.allocated_blocks(1);
    ASSERT_EQ(initial_blocks.size(), 1);

    EXPECT_EQ(manager.allocate_or_grow(1, 5), AllocationResult::SUCCESS);
    auto blocks = manager.allocated_blocks(1);
    ASSERT_EQ(blocks.size(), 2);
    EXPECT_EQ(blocks.front(), initial_blocks.front());

    EXPECT_EQ(manager.allocate_or_grow(1, 9), AllocationResult::SUCCESS);
    blocks = manager.allocated_blocks(1);
    ASSERT_EQ(blocks.size(), 3);
    EXPECT_EQ(blocks[0], initial_blocks[0]);

    EXPECT_EQ(manager.allocate_or_grow(1, 2), AllocationResult::SUCCESS);
    EXPECT_EQ(manager.allocated_blocks(1).size(), 3);
}

TEST(BlockManagerTest, RepeatedCoveredAllocationIsIdempotent) {
    BlockManager manager(3, 8);
    ASSERT_EQ(manager.allocate_or_grow(1, 9), AllocationResult::SUCCESS);
    const auto blocks = manager.allocated_blocks(1);

    EXPECT_EQ(manager.allocate_or_grow(1, 16), AllocationResult::SUCCESS);
    EXPECT_EQ(manager.allocated_blocks(1), blocks);
    EXPECT_EQ(manager.used_block_count(), 2);
    EXPECT_EQ(manager.free_block_count(), 1);
}

TEST(BlockManagerTest, ZeroTokenAllocationUsesNoBlocks) {
    BlockManager manager(2, 8);

    EXPECT_EQ(manager.allocate_or_grow(1, 0), AllocationResult::SUCCESS);
    EXPECT_TRUE(manager.allocated_blocks(1).empty());
    EXPECT_EQ(manager.used_block_count(), 0);
    manager.release(1);
    EXPECT_EQ(manager.free_block_count(), 2);
}

TEST(BlockManagerTest, ReportsExhaustionWithoutChangingExistingAllocations) {
    BlockManager manager(3, 16);
    ASSERT_EQ(manager.allocate_or_grow(1, 16), AllocationResult::SUCCESS);
    ASSERT_EQ(manager.allocate_or_grow(2, 16), AllocationResult::SUCCESS);
    const auto first_request_blocks = manager.allocated_blocks(1);
    const auto second_request_blocks = manager.allocated_blocks(2);
    const auto free_before = manager.free_block_count();

    EXPECT_EQ(manager.allocate_or_grow(1, 48), AllocationResult::CAPACITY_EXHAUSTED);
    EXPECT_EQ(manager.allocated_blocks(1), first_request_blocks);
    EXPECT_EQ(manager.allocated_blocks(2), second_request_blocks);
    EXPECT_EQ(manager.free_block_count(), free_before);
    EXPECT_THROW(static_cast<void>(manager.allocated_blocks(3)), std::out_of_range);
}

TEST(BlockManagerTest, ReleasesAllBlocksAndRejectsDoubleRelease) {
    BlockManager manager(4, 8);
    ASSERT_EQ(manager.allocate_or_grow(1, 17), AllocationResult::SUCCESS);
    ASSERT_EQ(manager.used_block_count(), 3);

    manager.release(1);
    EXPECT_EQ(manager.used_block_count(), 0);
    EXPECT_EQ(manager.free_block_count(), 4);
    EXPECT_THROW(manager.release(1), std::out_of_range);
}

TEST(BlockManagerTest, ReusesReleasedBlocksForAnotherRequest) {
    BlockManager manager(1, 8);
    ASSERT_EQ(manager.allocate_or_grow(1, 1), AllocationResult::SUCCESS);
    const auto released_block = manager.allocated_blocks(1).front();

    manager.release(1);
    ASSERT_EQ(manager.allocate_or_grow(2, 1), AllocationResult::SUCCESS);
    EXPECT_EQ(manager.allocated_blocks(2).front(), released_block);
}

TEST(BlockManagerTest, TracksUniqueOwnershipAcrossRequests) {
    BlockManager manager(8, 4);
    std::unordered_map<RequestId, std::size_t> expected_blocks{
        {1, 2},
        {2, 3},
        {3, 1},
    };

    EXPECT_EQ(manager.allocate_or_grow(1, 5), AllocationResult::SUCCESS);
    EXPECT_EQ(manager.allocate_or_grow(2, 9), AllocationResult::SUCCESS);
    EXPECT_EQ(manager.allocate_or_grow(3, 1), AllocationResult::SUCCESS);
    expect_invariants(manager, expected_blocks);
}

TEST(BlockManagerTest, PeakUtilizationDoesNotDecreaseAfterRelease) {
    BlockManager manager(4, 8);
    EXPECT_EQ(manager.peak_utilization(), 0.0);
    ASSERT_EQ(manager.allocate_or_grow(1, 9), AllocationResult::SUCCESS);
    ASSERT_EQ(manager.allocate_or_grow(2, 9), AllocationResult::SUCCESS);

    EXPECT_EQ(manager.peak_used_block_count(), 4);
    EXPECT_DOUBLE_EQ(manager.peak_utilization(), 1.0);
    manager.release(1);
    EXPECT_EQ(manager.used_block_count(), 2);
    EXPECT_EQ(manager.peak_used_block_count(), 4);
    EXPECT_DOUBLE_EQ(manager.peak_utilization(), 1.0);
}

TEST(BlockManagerTest, RandomizedAllocationAndReleasePreserveInvariants) {
    constexpr std::size_t total_blocks = 48;
    constexpr std::size_t tokens_per_block = 8;
    BlockManager manager(total_blocks, tokens_per_block);
    std::unordered_map<RequestId, std::size_t> expected_blocks;
    std::mt19937 random_engine(0xB10C);
    RequestId next_request_id = 1;

    for (std::size_t iteration = 0; iteration < 1200; ++iteration) {
        const auto action = random_engine() % 3;
        if (expected_blocks.empty() || (action == 0 && expected_blocks.size() < 12)) {
            const auto request_id = next_request_id++;
            const auto token_count = static_cast<std::size_t>(random_engine() % 96);
            const auto blocks_required =
                token_count / tokens_per_block +
                static_cast<std::size_t>(token_count % tokens_per_block != 0);

            if (blocks_required <= manager.free_block_count()) {
                EXPECT_EQ(manager.allocate_or_grow(request_id, token_count),
                          AllocationResult::SUCCESS);
                expected_blocks.emplace(request_id, blocks_required);
            } else {
                EXPECT_EQ(manager.allocate_or_grow(request_id, token_count),
                          AllocationResult::CAPACITY_EXHAUSTED);
                EXPECT_THROW(static_cast<void>(manager.allocated_blocks(request_id)),
                             std::out_of_range);
            }
        } else if (action == 1) {
            std::vector<RequestId> active_ids;
            active_ids.reserve(expected_blocks.size());
            for (const auto& [request_id, block_count] : expected_blocks) {
                static_cast<void>(block_count);
                active_ids.push_back(request_id);
            }
            const auto selected =
                static_cast<std::size_t>(random_engine() % active_ids.size());
            const auto request_id = active_ids[selected];
            const auto previous_count = expected_blocks.at(request_id);
            const auto token_count = static_cast<std::size_t>(random_engine() % 128);
            const auto blocks_required =
                token_count / tokens_per_block +
                static_cast<std::size_t>(token_count % tokens_per_block != 0);

            if (blocks_required <= previous_count ||
                blocks_required - previous_count <= manager.free_block_count()) {
                EXPECT_EQ(manager.allocate_or_grow(request_id, token_count),
                          AllocationResult::SUCCESS);
                expected_blocks[request_id] = std::max(previous_count, blocks_required);
            } else {
                const auto blocks_before = manager.allocated_blocks(request_id);
                EXPECT_EQ(manager.allocate_or_grow(request_id, token_count),
                          AllocationResult::CAPACITY_EXHAUSTED);
                EXPECT_EQ(manager.allocated_blocks(request_id), blocks_before);
            }
        } else {
            std::vector<RequestId> active_ids;
            active_ids.reserve(expected_blocks.size());
            for (const auto& [request_id, block_count] : expected_blocks) {
                static_cast<void>(block_count);
                active_ids.push_back(request_id);
            }
            const auto selected =
                static_cast<std::size_t>(random_engine() % active_ids.size());
            const auto request_id = active_ids[selected];
            manager.release(request_id);
            expected_blocks.erase(request_id);
        }

        expect_invariants(manager, expected_blocks);
    }
}

}  // namespace
}  // namespace blockstride
