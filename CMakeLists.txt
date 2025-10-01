cmake_minimum_required(VERSION 3.16)
project(GPUBench)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to find Vulkan (primary backend)
find_package(Vulkan)
if(Vulkan_FOUND)
    message(STATUS "Vulkan found: ${Vulkan_LIBRARIES}")
    find_program(GLSLC_EXECUTABLE glslc)
    if(NOT GLSLC_EXECUTABLE)
        message(WARNING "glslc not found! Vulkan backend will be disabled.")
        set(Vulkan_FOUND FALSE)
    endif()
else()
    message(STATUS "Vulkan not found, will use OpenCL as fallback")
endif()

# Try to find OpenCL (fallback backend)
find_package(OpenCL)
if(OpenCL_FOUND)
    message(STATUS "OpenCL found: ${OpenCL_LIBRARIES}")
else()
    message(STATUS "OpenCL not found")
endif()

# Try to find HIP for ROCm backend
find_package(HIP)
if(HIP_FOUND)
    message(STATUS "HIP (ROCm) found: ${HIP_LIBRARIES}")
    enable_language(HIP)
else()
    message(STATUS "HIP (ROCm) not found")
endif()

# Check that at least one backend is available
if(NOT Vulkan_FOUND AND NOT OpenCL_FOUND AND NOT HIP_FOUND)
    message(FATAL_ERROR "Neither Vulkan, OpenCL, nor ROCm found! At least one compute backend is required.")
endif()

# Shader/Kernel compilation
if(Vulkan_FOUND)
    file(GLOB SHADERS "shaders/*.comp")
    foreach(SHADER ${SHADERS})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(SPIRV_SHADER "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv")
        add_custom_command(
            OUTPUT ${SPIRV_SHADER}
            COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 --target-spv=spv1.5 ${SHADER} -o ${SPIRV_SHADER}
            DEPENDS ${SHADER}
            COMMENT "Compiling shader: ${SHADER}"
        )
        list(APPEND SPIRV_SHADERS ${SPIRV_SHADER})
    endforeach()
    add_custom_target(shaders ALL DEPENDS ${SPIRV_SHADERS})
endif()

# Copy OpenCL kernels to build directory
if(OpenCL_FOUND)
    file(GLOB KERNELS "kernels/*.cl")
    foreach(KERNEL ${KERNELS})
        get_filename_component(KERNEL_NAME ${KERNEL} NAME)
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/kernels/${KERNEL_NAME}"
            COMMAND ${CMAKE_COMMAND} -E copy ${KERNEL} "${CMAKE_CURRENT_BINARY_DIR}/kernels/${KERNEL_NAME}"
            DEPENDS ${KERNEL}
        )
        list(APPEND COPIED_KERNELS "${CMAKE_CURRENT_BINARY_DIR}/kernels/${KERNEL_NAME}")
    endforeach()
    add_custom_target(kernels ALL DEPENDS ${COPIED_KERNELS})
endif()

# Compile HIP kernels
if(HIP_FOUND)
    file(GLOB HIP_KERNELS "hip_kernels/*.hip")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hip_kernels)
    foreach(HIP_KERNEL ${HIP_KERNELS})
        get_filename_component(KERNEL_NAME ${HIP_KERNEL} NAME_WE)
        set(HIP_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/hip_kernels/${KERNEL_NAME}.o")
        add_custom_command(
            OUTPUT ${HIP_OBJECT}
            COMMAND ${HIP_HIPCC_EXECUTABLE} --genco -c ${HIP_KERNEL} -o ${HIP_OBJECT}
            DEPENDS ${HIP_KERNEL}
        )
        list(APPEND HIP_OBJECTS ${HIP_OBJECT})
    endforeach()
    add_custom_target(hip_kernels ALL DEPENDS ${HIP_OBJECTS})
endif()

# Source files - always include core files
set(SOURCES
    src/main.cpp
    src/core/BenchmarkRunner.cpp
    src/core/ResultFormatter.cpp
    src/benchmarks/Fp32Bench.cpp
    src/benchmarks/Fp64Bench.cpp
    src/benchmarks/Fp16Bench.cpp
    src/benchmarks/Fp8Bench.cpp
    src/benchmarks/Fp4Bench.cpp
    src/benchmarks/Int8Bench.cpp
    src/benchmarks/Int4Bench.cpp
    src/benchmarks/MemBandwidthBench.cpp
    src/benchmarks/CacheBandwidthBench.cpp
    src/benchmarks/CacheLatencyBench.cpp
    src/benchmarks/Fp6Bench.cpp
)

# Add backend-specific sources
if(Vulkan_FOUND)
    list(APPEND SOURCES src/core/VulkanContext.cpp)
endif()

if(OpenCL_FOUND)
    list(APPEND SOURCES src/core/OpenCLContext.cpp)
endif()

if(HIP_FOUND)
    list(APPEND SOURCES src/core/ROCmContext.cpp)
endif()

add_executable(gpubench ${SOURCES})

# Add dependencies on shader/kernel compilation
if(Vulkan_FOUND)
    add_dependencies(gpubench shaders)
endif()
if(OpenCL_FOUND)
    add_dependencies(gpubench kernels)
endif()
if(HIP_FOUND)
    add_dependencies(gpubench hip_kernels)
endif()

# Include directories
target_include_directories(gpubench PUBLIC 
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)

if(Vulkan_FOUND)
    target_include_directories(gpubench PUBLIC ${Vulkan_INCLUDE_DIRS})
endif()

if(OpenCL_FOUND)
    target_include_directories(gpubench PUBLIC ${OpenCL_INCLUDE_DIRS})
endif()

if(HIP_FOUND)
    target_include_directories(gpubench PUBLIC ${HIP_INCLUDE_DIRS})
endif()

# Link libraries
if(Vulkan_FOUND)
    target_link_libraries(gpubench PUBLIC ${Vulkan_LIBRARIES})
endif()

if(OpenCL_FOUND)
    target_link_libraries(gpubench PUBLIC ${OpenCL_LIBRARIES})
endif()

if(HIP_FOUND)
    target_link_libraries(gpubench PRIVATE hip::host)
endif()

# Compiler definitions for conditional compilation
if(Vulkan_FOUND)
    target_compile_definitions(gpubench PRIVATE HAVE_VULKAN)
endif()

if(OpenCL_FOUND)
    target_compile_definitions(gpubench PRIVATE HAVE_OPENCL)
endif()

if(HIP_FOUND)
    target_compile_definitions(gpubench PRIVATE HAVE_ROCM)
endif()
