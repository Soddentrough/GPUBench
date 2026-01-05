cmake_minimum_required(VERSION 3.16)
project(GPUBench VERSION 1.0.0)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Versioning
set(BUILD_NUMBER "1" CACHE STRING "Build number for the project")
set(PROJECT_FULL_VERSION "${PROJECT_VERSION}.${BUILD_NUMBER}")


# Set proper debug flags based on compiler
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Generate config header with install paths
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/src/utils/Config.h
)

# Try to find Vulkan (primary backend)
find_package(Vulkan REQUIRED)
if(Vulkan_FOUND)
    message(STATUS "Vulkan found: ${Vulkan_LIBRARIES}")
    # Try to find glslc, checking VULKAN_SDK environment variable if defined
    find_program(GLSLC_EXECUTABLE glslc
        HINTS "$ENV{VULKAN_SDK}/Bin"
    )
    if(NOT GLSLC_EXECUTABLE)
        message(WARNING "glslc not found! Searching for glslc in Vulkan SDK...")
        # Fallback search if HINTS didn't work and GLSLC is still not found
        file(GLOB GLSLC_FALLBACK "C:/VulkanSDK/*/Bin/glslc.exe")
        if(GLSLC_FALLBACK)
            list(GET GLSLC_FALLBACK 0 GLSLC_EXECUTABLE)
            set(GLSLC_EXECUTABLE ${GLSLC_EXECUTABLE} CACHE FILEPATH "glslc executable")
        endif()
    endif()

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
    
    # Define optional shaders that are allowed to fail compilation
    set(OPTIONAL_SHADERS 
        "coop_matrix_fp8" 
        "coop_matrix_int4" 
        "fp8_native" 
        "fp4_native" 
        "int4_native"
    )

    foreach(SHADER ${SHADERS})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(SPIRV_SHADER "${CMAKE_CURRENT_BINARY_DIR}/kernels/vulkan/${SHADER_NAME}.spv")
        
        # Check if this shader is optional
        list(FIND OPTIONAL_SHADERS ${SHADER_NAME} IS_OPTIONAL)
        
        if(IS_OPTIONAL GREATER -1)
            # For optional shaders, we use a command that always succeeds (exit 0)
            # even if glslc fails. We print a warning if it fails.
            add_custom_command(
                OUTPUT ${SPIRV_SHADER}
                COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.3 --target-spv=spv1.6 ${SHADER} -o ${SPIRV_SHADER} || echo "WARNING: Optional shader ${SHADER_NAME} failed to compile"
                DEPENDS ${SHADER}
                COMMENT "Compiling optional shader: ${SHADER}"
            )
        else()
            # For standard shaders, fail the build if compilation fails
            add_custom_command(
                OUTPUT ${SPIRV_SHADER}
                COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.3 --target-spv=spv1.6 ${SHADER} -o ${SPIRV_SHADER}
                DEPENDS ${SHADER}
                COMMENT "Compiling shader: ${SHADER}"
            )
        endif()
        
        list(APPEND SPIRV_SHADERS ${SPIRV_SHADER})
    endforeach()
    add_custom_target(shaders ALL DEPENDS ${SPIRV_SHADERS})
endif()

# Copy OpenCL kernels to build directory
if(OpenCL_FOUND)
    file(GLOB KERNELS "kernels/*.cl")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kernels/opencl)
    foreach(KERNEL_FILE ${KERNELS})
        get_filename_component(KERNEL_NAME ${KERNEL_FILE} NAME)
        set(DESTINATION_FILE "${CMAKE_CURRENT_BINARY_DIR}/kernels/opencl/${KERNEL_NAME}")
        add_custom_command(
            OUTPUT ${DESTINATION_FILE}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${KERNEL_FILE} ${DESTINATION_FILE}
            DEPENDS ${KERNEL_FILE}
        )
        list(APPEND KERNEL_DESTINATIONS ${DESTINATION_FILE})
    endforeach()
    add_custom_target(kernels ALL DEPENDS ${KERNEL_DESTINATIONS})
endif()

# Compile HIP kernels
if(HIP_FOUND)
    file(GLOB HIP_KERNELS "hip_kernels/*.hip")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kernels/rocm)
    
    # Define optional HIP kernels
    set(OPTIONAL_HIP_KERNELS "fp8" "int4")
    
    foreach(HIP_KERNEL ${HIP_KERNELS})
        get_filename_component(KERNEL_NAME ${HIP_KERNEL} NAME_WE)
        set(HIP_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/kernels/rocm/${KERNEL_NAME}.co")
        
        list(FIND OPTIONAL_HIP_KERNELS ${KERNEL_NAME} IS_OPTIONAL)
        
        if(IS_OPTIONAL GREATER -1)
             add_custom_command(
                OUTPUT ${HIP_OBJECT}
                COMMAND ${HIP_HIPCC_EXECUTABLE} --genco --offload-arch=gfx1201 ${HIP_KERNEL} -o ${HIP_OBJECT} -D__HIP_PLATFORM_AMD__ || echo "WARNING: Optional HIP kernel ${KERNEL_NAME} failed to compile"
                DEPENDS ${HIP_KERNEL}
                COMMENT "Compiling optional HIP kernel: ${HIP_KERNEL}"
            )
        else()
             add_custom_command(
                OUTPUT ${HIP_OBJECT}
                COMMAND ${HIP_HIPCC_EXECUTABLE} --genco --offload-arch=gfx1201 ${HIP_KERNEL} -o ${HIP_OBJECT} -D__HIP_PLATFORM_AMD__
                DEPENDS ${HIP_KERNEL}
                COMMENT "Compiling HIP kernel: ${HIP_KERNEL}"
            )
        endif()
       
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
    src/benchmarks/SysMemBandwidthBench.cpp
    src/benchmarks/SysMemLatencyBench.cpp
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
    #target_link_libraries(gpubench PUBLIC ${Vulkan_LIBRARIES})
    target_link_libraries(gpubench PUBLIC Vulkan::Vulkan)
endif()

if(OpenCL_FOUND)
    target_link_libraries(gpubench PUBLIC ${OpenCL_LIBRARIES})
endif()

if(HIP_FOUND)
    target_link_libraries(gpubench PRIVATE hip::host)
endif()

# Compiler definitions for conditional compilation
target_compile_definitions(gpubench PRIVATE GPUBENCH_VERSION="${PROJECT_FULL_VERSION}")

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

# Bundle MinGW DLLs
if(MINGW)
    find_program(MINGW_GCC gcc)
    if(MINGW_GCC)
        get_filename_component(MINGW_BIN_DIR ${MINGW_GCC} DIRECTORY)
        install(FILES
            "${MINGW_BIN_DIR}/libstdc++-6.dll"
            "${MINGW_BIN_DIR}/libgcc_s_seh-1.dll"
            "${MINGW_BIN_DIR}/libwinpthread-1.dll"
            DESTINATION bin
            OPTIONAL
        )
    endif()
endif()

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
            FILES_MATCHING PATTERN "*.co")
endif()

# Install Windows batch script for easy benchmarking
if(WIN32)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/packaging/windows/RunAllBenchmarks.bat"
            DESTINATION .)
endif()

# CPack configuration for creating packages
set(CPACK_PACKAGE_NAME "GPUBench")
set(CPACK_PACKAGE_VENDOR "GPUBench")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GPU Compute Benchmarking Tool")
set(CPACK_PACKAGE_VERSION ${PROJECT_FULL_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_RPM_PACKAGE_RELEASE ${BUILD_NUMBER})
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
        "RunAllBenchmarks.bat" "Run All Benchmarks"
        "bin/gpubench.exe" "GPUBench (CLI)"
        "https://github.com/yourusername/GPUBench" "GPUBench Website"
    )
    
    # Include Visual C++ Redistributable if needed
    # set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "ExecWait '$INSTDIR\\\\vcredist_x64.exe /quiet'")
    
elseif(UNIX AND NOT APPLE)
    # Linux packaging
    set(CPACK_GENERATOR "TGZ;DEB;RPM")
    set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")
    
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
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(CPACK_SYSTEM_NAME "win64")
    else()
        set(CPACK_SYSTEM_NAME "win32")
    endif()
endif()

include(CPack)
