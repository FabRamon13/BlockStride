#include "blockstride/version.hpp"

#include <gtest/gtest.h>

TEST(VersionTest, ReturnsProjectVersion) {
    EXPECT_EQ(blockstride::version(), "0.1.0-dev");
}
