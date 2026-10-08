set(TEST_SOURCES
    test_main.cpp
    test_device_database.cpp
    test_json_serialization.cpp
    test_math_and_parity.cpp
    test_memory_suballocator.cpp
    test_kernel_path.cpp
    test_pixelfill.cpp
)

add_executable(gpubench_tests ${TEST_SOURCES})

target_link_libraries(gpubench_tests PRIVATE
    gpubench_lib
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

if(Vulkan_FOUND)
    target_link_libraries(gpubench_tests PRIVATE Vulkan::Vulkan)
endif()

if(APPLE)
    target_link_libraries(gpubench_tests PRIVATE "-framework IOKit" "-framework CoreFoundation")
endif()

target_include_directories(gpubench_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_src
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_src/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../external
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/vulkan/Include
    ${CMAKE_CURRENT_BINARY_DIR}/../cpp_src
)

target_compile_definitions(gpubench_tests PRIVATE GPUBENCH_VERSION="${PROJECT_FULL_VERSION}")

# Register individual test suites with CTest
add_test(NAME DeviceDatabase COMMAND gpubench_tests --filter=DeviceDatabase)
add_test(NAME JsonSerialization COMMAND gpubench_tests --filter=JsonSerialization)
add_test(NAME MathAndParity COMMAND gpubench_tests --filter=MathAndParity)
add_test(NAME BlockSuballocator COMMAND gpubench_tests --filter=BlockSuballocator)
add_test(NAME KernelPath COMMAND gpubench_tests --filter=KernelPath)
add_test(NAME PixelFill COMMAND gpubench_tests --filter=PixelFill)
add_test(NAME AllTests COMMAND gpubench_tests)

set_tests_properties(
    DeviceDatabase
    JsonSerialization
    MathAndParity
    BlockSuballocator
    KernelPath
    PixelFill
    AllTests
    PROPERTIES WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS gpubench_tests
    COMMENT "Running all automated headless unit tests with CTest..."
)
