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
    test_indirect_and_asbuild_validation.cpp
    test_timing_fidelity.cpp
    test_lds_and_bf16_validation.cpp
    test_compute_validation.cpp
    test_memory_and_cache_validation.cpp
    test_graphics_and_rt_validation.cpp
    test_rfc8259_json_stress.cpp
    test_modernization_and_2026.cpp
    test_ray_scheduling_parity.cpp
    test_showcase_validation.cpp
    test_rocm_parity.cpp
    test_gui_logic.cpp
    test_runner_and_utils.cpp
    test_opencl_validation.cpp
    ../cpp_src/gui/TelemetryWorker.cpp
)

add_executable(gpubench_tests ${TEST_SOURCES})

add_dependencies(gpubench_tests gpubench)
if(HIP_FOUND)
    add_dependencies(gpubench_tests hip_kernels)
endif()

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

# Register individual test suites with CTest (exact suite matching + skip exit code 77)
add_test(NAME DeviceDatabase COMMAND gpubench_tests --suite=DeviceDatabase)
add_test(NAME JsonSerialization COMMAND gpubench_tests --suite=JsonSerialization)
add_test(NAME MathAndParity COMMAND gpubench_tests --suite=MathAndParity)
add_test(NAME BlockSuballocator COMMAND gpubench_tests --suite=BlockSuballocator)
add_test(NAME KernelPath COMMAND gpubench_tests --suite=KernelPath)
add_test(NAME PixelFill COMMAND gpubench_tests --suite=PixelFill)
add_test(NAME DescriptionsAndThumbnails COMMAND gpubench_tests --suite=DescriptionsAndThumbnails)
add_test(NAME GuiLogic COMMAND gpubench_tests --suite=GuiLogic)
add_test(NAME RunnerAndUtils COMMAND gpubench_tests --suite=RunnerAndUtils)
add_test(NAME E2E_Tier1 COMMAND gpubench_tests --suite=E2E_Tier1)
add_test(NAME E2E_Tier2 COMMAND gpubench_tests --suite=E2E_Tier2)
add_test(NAME E2E_Tier3 COMMAND gpubench_tests --suite=E2E_Tier3)
add_test(NAME E2E_Tier4 COMMAND gpubench_tests --suite=E2E_Tier4)
add_test(NAME IndirectAndASBuildValidation COMMAND gpubench_tests --suite=IndirectAndASBuildValidation)
add_test(NAME TimingFidelity COMMAND gpubench_tests --suite=TimingFidelity)
add_test(NAME LdsAndBf16Validation COMMAND gpubench_tests --suite=LdsAndBf16Validation)
add_test(NAME ComputeValidation COMMAND gpubench_tests --suite=ComputeValidation)
add_test(NAME MemoryAndCacheValidation COMMAND gpubench_tests --suite=MemoryAndCacheValidation)
add_test(NAME GraphicsAndRayTracingValidation COMMAND gpubench_tests --suite=GraphicsAndRayTracingValidation)
add_test(NAME RFC8259_JsonStress COMMAND gpubench_tests --suite=RFC8259_JsonStress)
add_test(NAME ModernizationAnd2026 COMMAND gpubench_tests --suite=ModernizationAnd2026)
add_test(NAME RaySchedulingParity COMMAND gpubench_tests --suite=RaySchedulingParity)
add_test(NAME ShowcaseValidation COMMAND gpubench_tests --suite=ShowcaseValidation)
add_test(NAME ROCmParityValidation COMMAND gpubench_tests --suite=ROCmParityValidation)
add_test(NAME OpenCLValidation COMMAND gpubench_tests --suite=OpenCLValidation)

set_tests_properties(
    DeviceDatabase
    JsonSerialization
    MathAndParity
    BlockSuballocator
    KernelPath
    PixelFill
    DescriptionsAndThumbnails
    GuiLogic
    RunnerAndUtils
    RFC8259_JsonStress
    PROPERTIES WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" LABELS "cpu" SKIP_RETURN_CODE 77
)

set_tests_properties(
    E2E_Tier1
    E2E_Tier2
    E2E_Tier3
    E2E_Tier4
    IndirectAndASBuildValidation
    TimingFidelity
    LdsAndBf16Validation
    ComputeValidation
    MemoryAndCacheValidation
    GraphicsAndRayTracingValidation
    ModernizationAnd2026
    RaySchedulingParity
    ShowcaseValidation
    ROCmParityValidation
    OpenCLValidation
    PROPERTIES WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" RESOURCE_LOCK gpu_execution LABELS "gpu" SKIP_RETURN_CODE 77
)

add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS gpubench_tests gpubench
    COMMENT "Running all automated unit and E2E tests with CTest..."
)
