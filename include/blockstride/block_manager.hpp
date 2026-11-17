#pragma once

#include "blockstride/request.hpp"

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

namespace blockstride {

using BlockId = std::size_t;

enum class AllocationResult {
    SUCCESS,
    CAPACITY_EXHAUSTED
};

class BlockManager {
public:
    BlockManager(std::size_t total_physical_blocks, std::size_t tokens_per_block);

    [[nodiscard]] std::size_t total_physical_blocks() const noexcept;
    [[nodiscard]] std::size_t tokens_per_block() const noexcept;
    [[nodiscard]] std::size_t blocks_needed(std::size_t total_token_count) const noexcept;

    [[nodiscard]] AllocationResult allocate_or_grow(RequestId request_id,
                                                     std::size_t total_token_count);
    [[nodiscard]] std::vector<BlockId> allocated_blocks(RequestId request_id) const;
    void release(RequestId request_id);

    [[nodiscard]] std::size_t used_block_count() const noexcept;
    [[nodiscard]] std::size_t free_block_count() const noexcept;
    [[nodiscard]] std::size_t peak_used_block_count() const noexcept;
    [[nodiscard]] double peak_utilization() const noexcept;

private:
    static std::size_t validate_configuration(std::size_t total_physical_blocks,
                                              std::size_t tokens_per_block);

    std::size_t total_physical_blocks_;
    std::size_t tokens_per_block_;
    std::vector<BlockId> free_blocks_;
    std::vector<std::optional<RequestId>> block_owners_;
    std::unordered_map<RequestId, std::vector<BlockId>> block_tables_;
    std::size_t peak_used_block_count_ = 0;
};

}  // namespace blockstride
