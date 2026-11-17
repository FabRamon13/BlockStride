#include "blockstride/block_manager.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace blockstride {

std::size_t BlockManager::validate_configuration(std::size_t total_physical_blocks,
                                                 std::size_t tokens_per_block) {
    if (total_physical_blocks == 0) {
        throw std::invalid_argument("total physical blocks must be greater than zero");
    }
    if (tokens_per_block == 0) {
        throw std::invalid_argument("tokens per block must be greater than zero");
    }
    return total_physical_blocks;
}

BlockManager::BlockManager(std::size_t total_physical_blocks, std::size_t tokens_per_block)
    : total_physical_blocks_(validate_configuration(total_physical_blocks, tokens_per_block)),
      tokens_per_block_(tokens_per_block),
      block_owners_(total_physical_blocks_) {
    free_blocks_.reserve(total_physical_blocks_);
    for (BlockId block_id = 0; block_id < total_physical_blocks_; ++block_id) {
        free_blocks_.push_back(block_id);
    }
}

std::size_t BlockManager::total_physical_blocks() const noexcept {
    return total_physical_blocks_;
}

std::size_t BlockManager::tokens_per_block() const noexcept {
    return tokens_per_block_;
}

std::size_t BlockManager::blocks_needed(std::size_t total_token_count) const noexcept {
    const auto full_blocks = total_token_count / tokens_per_block_;
    const auto has_partial_block = total_token_count % tokens_per_block_ != 0;
    return full_blocks + static_cast<std::size_t>(has_partial_block);
}

AllocationResult BlockManager::allocate_or_grow(RequestId request_id,
                                                std::size_t total_token_count) {
    const auto required_blocks = blocks_needed(total_token_count);
    auto table = block_tables_.find(request_id);
    const auto current_blocks =
        table == block_tables_.end() ? std::size_t{0} : table->second.size();

    if (required_blocks <= current_blocks) {
        if (table == block_tables_.end()) {
            block_tables_.try_emplace(request_id);
        }
        return AllocationResult::SUCCESS;
    }

    const auto additional_blocks = required_blocks - current_blocks;
    if (additional_blocks > free_blocks_.size()) {
        return AllocationResult::CAPACITY_EXHAUSTED;
    }

    for (std::size_t offset = 0; offset < additional_blocks; ++offset) {
        const auto block_id = free_blocks_[free_blocks_.size() - offset - 1];
        if (block_id >= block_owners_.size() || block_owners_[block_id].has_value()) {
            throw std::logic_error("free-block pool contains an owned or invalid block");
        }
    }

    bool inserted_table = false;
    if (table == block_tables_.end()) {
        const auto insertion = block_tables_.try_emplace(request_id);
        table = insertion.first;
        inserted_table = insertion.second;
    }

    try {
        table->second.reserve(required_blocks);
    } catch (...) {
        if (inserted_table) {
            block_tables_.erase(table);
        }
        throw;
    }

    for (std::size_t count = 0; count < additional_blocks; ++count) {
        const auto block_id = free_blocks_.back();
        free_blocks_.pop_back();
        block_owners_[block_id] = request_id;
        table->second.push_back(block_id);
    }

    peak_used_block_count_ = std::max(peak_used_block_count_, used_block_count());
    return AllocationResult::SUCCESS;
}

std::vector<BlockId> BlockManager::allocated_blocks(RequestId request_id) const {
    const auto table = block_tables_.find(request_id);
    if (table == block_tables_.end()) {
        throw std::out_of_range("request has no block table");
    }
    return table->second;
}

void BlockManager::release(RequestId request_id) {
    const auto table = block_tables_.find(request_id);
    if (table == block_tables_.end()) {
        throw std::out_of_range("request has no block table to release");
    }

    for (const auto block_id : table->second) {
        if (block_id >= block_owners_.size() || block_owners_[block_id] != request_id) {
            throw std::logic_error("block table contains a block not owned by its request");
        }
    }

    for (const auto block_id : table->second) {
        block_owners_[block_id].reset();
        free_blocks_.push_back(block_id);
    }
    block_tables_.erase(table);
}

std::size_t BlockManager::used_block_count() const noexcept {
    return total_physical_blocks_ - free_blocks_.size();
}

std::size_t BlockManager::free_block_count() const noexcept {
    return free_blocks_.size();
}

std::size_t BlockManager::peak_used_block_count() const noexcept {
    return peak_used_block_count_;
}

double BlockManager::peak_utilization() const noexcept {
    return static_cast<double>(peak_used_block_count_) /
           static_cast<double>(total_physical_blocks_);
}

}  // namespace blockstride
