cmake_minimum_required(VERSION 3.20)

project(BlockStride VERSION 0.1.0 LANGUAGES CXX)

option(BLOCKSTRIDE_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "BlockStride requires an out-of-source build directory")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(blockstride_options INTERFACE)
if(MSVC)
    target_compile_options(blockstride_options INTERFACE /W4 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(blockstride_options INTERFACE -Wall -Wextra -Wpedantic)
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

if(BLOCKSTRIDE_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(blockstride_options INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(blockstride_options INTERFACE -fsanitize=address,undefined)
    else()
        message(FATAL_ERROR "AddressSanitizer and UndefinedBehaviorSanitizer are not configured for this compiler")
    endif()
endif()

add_library(blockstride_core STATIC
    src/request.cpp
    src/request_manager.cpp
    src/version.cpp
)
target_include_directories(blockstride_core PUBLIC include)
target_link_libraries(blockstride_core PRIVATE blockstride_options)

add_executable(blockstride_bench benchmarks/main.cpp)
target_link_libraries(blockstride_bench PRIVATE blockstride_core blockstride_options)

include(CTest)
if(BUILD_TESTING)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.15.2
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(blockstride_tests
        tests/test_request.cpp
        tests/test_request_manager.cpp
        tests/test_smoke.cpp
    )
    target_link_libraries(blockstride_tests PRIVATE blockstride_core GTest::gtest_main blockstride_options)

    include(GoogleTest)
    gtest_discover_tests(blockstride_tests)
endif()
