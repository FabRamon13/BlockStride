#include "blockstride/version.hpp"

#include <iostream>

int main() {
    std::cout << "BlockStride " << blockstride::version() << '\n';
    return 0;
}
