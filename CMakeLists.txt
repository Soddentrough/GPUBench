cmake_minimum_required(VERSION 3.16)
project(GPUBench VERSION 1.0.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate config header with install paths
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/src/utils/Config.h
)

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
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kernels/vulkan)
    foreach(SHADER ${SHADERS})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(SPIRV_SHADER "${CMAKE_CURRENT_BINARY_DIR}/kernels/vulkan/${SHADER_NAME}.spv")
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
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels/opencl
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/kernels ${CMAKE_CURRENT_BINARY_DIR}/kernels/opencl
        DEPENDS ${KERNELS}
    )
    add_custom_target(kernels ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernels/opencl)
endif()

# Compile HIP kernels
if(HIP_FOUND)
    file(GLOB HIP_KERNELS "hip_kernels/*.hip")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kernels/rocm)
    foreach(HIP_KERNEL ${HIP_KERNELS})
        get_filename_component(KERNEL_NAME ${HIP_KERNEL} NAME_WE)
        set(HIP_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/kernels/rocm/${KERNEL_NAME}.o")
        add_custom_command(
            OUTPUT ${HIP_OBJECT}
            COMMAND ${HIP_HIPCC_EXECUTABLE} --genco -c ${HIP_KERNEL} -o ${HIP_OBJECT} -D__HIP_PLATFORM_AMD__
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
    src/benchmarks/CacheBench.cpp
    src/benchmarks/Fp6Bench.cpp
    src/utils/KernelPath.cpp
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
    ${CMAKE_CURRENT_BINARY_DIR}/src
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

# Copy kernels directory to project root for running from base directory
add_custom_command(TARGET gpubench POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_BINARY_DIR}/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    COMMENT "Copying kernels directory to project root"
)

# Install targets
install(TARGETS gpubench DESTINATION bin)

# Install kernel files for each backend
if(Vulkan_FOUND)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kernels/vulkan
            DESTINATION share/gpubench/kernels
            FILES_MATCHING PATTERN "*.spv")
endif()

if(OpenCL_FOUND)
    file(GLOB OPENCL_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cl")
    install(FILES ${OPENCL_KERNELS}
            DESTINATION share/gpubench/kernels/opencl)
endif()

if(HIP_FOUND)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kernels/rocm
            DESTINATION share/gpubench/kernels
            FILES_MATCHING PATTERN "*.o")
endif()

# CPack configuration for creating packages
set(CPACK_PACKAGE_NAME "GPUBench")
set(CPACK_PACKAGE_VENDOR "GPUBench")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GPU Compute Benchmarking Tool")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "GPUBench")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Platform-specific package settings
if(WIN32)
    # Windows-specific packaging
    set(CPACK_GENERATOR "ZIP;NSIS")
    
    # NSIS (Windows installer) settings
    set(CPACK_NSIS_DISPLAY_NAME "GPUBench ${PROJECT_VERSION}")
    set(CPACK_NSIS_PACKAGE_NAME "GPUBench")
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    set(CPACK_NSIS_MODIFY_PATH ON)
    set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/packaging/windows/icon.ico")
    set(CPACK_NSIS_MUI_UNIICON "${CMAKE_CURRENT_SOURCE_DIR}/packaging/windows/icon.ico")
    set(CPACK_NSIS_HELP_LINK "https://github.com/yourusername/GPUBench")
    set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/yourusername/GPUBench")
    set(CPACK_NSIS_CONTACT "your.email@example.com")
    set(CPACK_NSIS_MENU_LINKS
        "bin/gpubench.exe" "GPUBench"
        "https://github.com/yourusername/GPUBench" "GPUBench Website"
    )
    
    # Include Visual C++ Redistributable if needed
    # set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "ExecWait '$INSTDIR\\\\vcredist_x64.exe /quiet'")
    
elseif(UNIX AND NOT APPLE)
    # Linux packaging
    set(CPACK_GENERATOR "TGZ;DEB;RPM")
    
    # DEB package settings
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "GPUBench Maintainer <your.email@example.com>")
    set(CPACK_DEBIAN_PACKAGE_SECTION "utils")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
    
    # RPM package settings
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
    set(CPACK_RPM_PACKAGE_GROUP "Applications/System")
    
elseif(APPLE)
    # macOS packaging
    set(CPACK_GENERATOR "TGZ;DragNDrop")
    set(CPACK_DMG_VOLUME_NAME "GPUBench")
endif()

# Architecture detection
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CPACK_SYSTEM_NAME "win64")
else()
    set(CPACK_SYSTEM_NAME "win32")
endif()

include(CPack)
