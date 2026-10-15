set(TEST_SOURCES
    test_main.cpp
    test_device_database.cpp
    test_json_serialization.cpp
    test_math_and_parity.cpp
    test_memory_suballocator.cpp
    test_kernel_path.cpp
    test_pixelfill.cpp
    test_descriptions_and_thumbnails.cpp
    test_e2e_tiers.cpp
    test_m1_adversarial_challenge.cpp
    test_m1_timing_fidelity.cpp
    test_m2_adversarial_challenge.cpp
    test_m2_challenger2.cpp
    test_m3_adversarial_challenge.cpp
    test_m3_challenger2.cpp
    test_rfc8259_json_stress.cpp
)

add_executable(gpubench_tests ${TEST_SOURCES})

add_dependencies(gpubench_tests gpubench)

target_link_libraries(gpubench_tests PRIVATE
    gpubench_lib
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

if(Vulkan_FOUND)
    target_link_libraries(gpubench_tests PRIVATE Vulkan::Vulkan)
    target_compile_definitions(gpubench_tests PRIVATE HAVE_VULKAN)
endif()

if(APPLE)
    target_link_libraries(gpubench_tests PRIVATE "-framework IOKit" "-framework CoreFoundation")
endif()

target_include_directories(gpubench_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_src
    ${CMAKE_CURRENT_BINARY_DIR}/../cpp_src
)
target_include_directories(gpubench_tests SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_src/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../external
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/vulkan/Include
)

target_compile_definitions(gpubench_tests PRIVATE GPUBENCH_VERSION="${PROJECT_FULL_VERSION}")

# Register individual test suites with CTest
add_test(NAME DeviceDatabase COMMAND gpubench_tests --filter=DeviceDatabase)
add_test(NAME JsonSerialization COMMAND gpubench_tests --filter=JsonSerialization)
add_test(NAME MathAndParity COMMAND gpubench_tests --filter=MathAndParity)
add_test(NAME BlockSuballocator COMMAND gpubench_tests --filter=BlockSuballocator)
add_test(NAME KernelPath COMMAND gpubench_tests --filter=KernelPath)
add_test(NAME PixelFill COMMAND gpubench_tests --filter=PixelFill)
add_test(NAME DescriptionsAndThumbnails COMMAND gpubench_tests --filter=DescriptionsAndThumbnails)
add_test(NAME E2E_Tier1 COMMAND gpubench_tests --filter=E2E_Tier1)
add_test(NAME E2E_Tier2 COMMAND gpubench_tests --filter=E2E_Tier2)
add_test(NAME E2E_Tier3 COMMAND gpubench_tests --filter=E2E_Tier3)
add_test(NAME E2E_Tier4 COMMAND gpubench_tests --filter=E2E_Tier4)
add_test(NAME E2E_All COMMAND gpubench_tests --filter=E2E_)
add_test(NAME M1_AdversarialChallenge COMMAND gpubench_tests --filter=M1_AdversarialChallenge)
add_test(NAME M1_TimingFidelity COMMAND gpubench_tests --filter=M1_TimingFidelity)
add_test(NAME M2_AdversarialChallenge COMMAND gpubench_tests --filter=M2_AdversarialChallenge)
add_test(NAME M2_Challenger2 COMMAND gpubench_tests --filter=M2_Challenger2)
add_test(NAME M3_AdversarialChallenge COMMAND gpubench_tests --filter=M3_AdversarialChallenge)
add_test(NAME M3_Challenger2 COMMAND gpubench_tests --filter=M3_Challenger2)
add_test(NAME RFC8259_JsonStress COMMAND gpubench_tests --filter=RFC8259_JsonStress)
add_test(NAME AllTests COMMAND gpubench_tests)

set_tests_properties(
    DeviceDatabase
    JsonSerialization
    MathAndParity
    BlockSuballocator
    KernelPath
    PixelFill
    DescriptionsAndThumbnails
    E2E_Tier1
    E2E_Tier2
    E2E_Tier3
    E2E_Tier4
    E2E_All
    M1_AdversarialChallenge
    M1_TimingFidelity
    M2_AdversarialChallenge
    M2_Challenger2
    M3_AdversarialChallenge
    M3_Challenger2
    RFC8259_JsonStress
    AllTests
    PROPERTIES WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" RESOURCE_LOCK gpu_execution
)

add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS gpubench_tests gpubench
    COMMENT "Running all automated unit and E2E tests with CTest..."
)
