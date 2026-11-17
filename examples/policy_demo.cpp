#include "blockstride/simulator.hpp"

#include <iostream>
#include <vector>

int main() {
    const std::vector<blockstride::WorkloadRequest> workload{
        {1, 2, 3, 0},
        {2, 2, 3, 0},
        {3, 2, 3, 5},
    };
    const auto results = blockstride::compare_scheduling_policies(
        workload, {2, 8}, 16, 4, {2, 1, 3, 1});

    for (const auto& result : results) {
        std::cout << "policy=" << blockstride::scheduling_policy_name(result.policy)
                  << " completed=" << result.outcome.statistics.completed_requests
                  << " generated_tokens=" << result.outcome.statistics.generated_tokens
                  << " simulated_elapsed_us=" << result.outcome.final_timestamp
                  << " status=" << blockstride::simulation_status_name(result.outcome.status)
                  << '\n';
    }
}
