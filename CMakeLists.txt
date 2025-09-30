cmake_minimum_required(VERSION 3.16)
project(GPUBench)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE glslc)

if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found! Please install the Vulkan SDK and make sure glslc is in your PATH.")
endif()

# Shader compilation
file(GLOB SHADERS "shaders/*.comp")

foreach(SHADER ${SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPIRV_SHADER "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV_SHADER}
        COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 ${SHADER} -o ${SPIRV_SHADER}
        DEPENDS ${SHADER}
    )
    list(APPEND SPIRV_SHADERS ${SPIRV_SHADER})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPIRV_SHADERS})

add_executable(gpubench
    src/main.cpp
    src/core/VulkanContext.cpp
    src/core/BenchmarkRunner.cpp
    src/benchmarks/Fp32Bench.cpp
    src/benchmarks/Fp64Bench.cpp
    src/benchmarks/Fp16Bench.cpp
    src/benchmarks/Fp8Bench.cpp
    src/benchmarks/Int8Bench.cpp
    src/benchmarks/Int4Bench.cpp
)
add_dependencies(gpubench shaders)


target_include_directories(gpubench PUBLIC 
    ${Vulkan_INCLUDE_DIRS}
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)
target_link_libraries(gpubench ${Vulkan_LIBRARIES})
