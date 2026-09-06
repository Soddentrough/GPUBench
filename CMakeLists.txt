cmake_minimum_required(VERSION 3.22)
project(GPUBench VERSION 1.5.1)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Versioning
set(BUILD_NUMBER "0")
set(PROJECT_FULL_VERSION "${PROJECT_VERSION}")


# Set proper debug flags based on compiler
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Generate config header with install paths
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/utils/Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/cpp_src/utils/Config.h
)

# Try to find Vulkan (primary backend) — optional so OpenCL/ROCm-only builds work
find_package(Vulkan)
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
    
    # Find hiprtc for runtime compilation
    find_library(HIPRTC_LIBRARY NAMES hiprtc)
    if(HIPRTC_LIBRARY)
        message(STATUS "Found hiprtc: ${HIPRTC_LIBRARY}")
    endif()
else()
    message(STATUS "HIP (ROCm) not found")
endif()

# Find shaderc for runtime Vulkan shader compilation
find_path(SHADERC_INCLUDE_DIR NAMES shaderc/shaderc.hpp)
find_library(SHADERC_LIBRARY NAMES shaderc_shared)
if(SHADERC_INCLUDE_DIR AND SHADERC_LIBRARY)
    set(SHADERC_FOUND TRUE)
    message(STATUS "Found shaderc: ${SHADERC_LIBRARY}")
endif()

# Check that at least one backend is available
if(NOT Vulkan_FOUND AND NOT OpenCL_FOUND AND NOT HIP_FOUND)
    message(FATAL_ERROR "Neither Vulkan, OpenCL, nor ROCm found! At least one compute backend is required.")
endif()

# Shader/Kernel sources
if(Vulkan_FOUND)
    file(GLOB SHADERS "shaders/*.comp" "shaders/*.rgen" "shaders/*.rmiss" "shaders/*.rchit" "shaders/*.rahit" "shaders/*.rint" "shaders/*.vert" "shaders/*.frag")
    # Filter out problematic shaders that use non-standard extensions
    list(REMOVE_ITEM SHADERS 
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/fp8_native.comp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/int4_native.comp"
    )
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kernels/vulkan)
    
    file(GLOB SHADER_HEADERS "shaders/*.glsl" "shaders/*.h")
    foreach(HEADER ${SHADER_HEADERS})
        get_filename_component(HEADER_NAME ${HEADER} NAME)
        set(DEST_HEADER "${CMAKE_CURRENT_BINARY_DIR}/kernels/vulkan/${HEADER_NAME}")
        add_custom_command(
            OUTPUT ${DEST_HEADER}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HEADER} ${DEST_HEADER}
            DEPENDS ${HEADER}
            COMMENT "Copying Vulkan shader header: ${HEADER}"
        )
        list(APPEND VULKAN_SHADERS ${DEST_HEADER})
    endforeach()

    foreach(SHADER ${SHADERS})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(DEST_SHADER "${CMAKE_CURRENT_BINARY_DIR}/kernels/vulkan/${SHADER_NAME}")
        set(SPV_SHADER "${DEST_SHADER}.spv")
        
        # Pre-compile GLSL to SPIR-V
        add_custom_command(
            OUTPUT ${SPV_SHADER}
            COMMAND ${GLSLC_EXECUTABLE} -O --target-env=vulkan1.4 -I${CMAKE_CURRENT_SOURCE_DIR}/shaders ${SHADER} -o ${SPV_SHADER}
            DEPENDS ${SHADER} ${SHADER_HEADERS}
            COMMENT "Compiling Vulkan shader to SPIR-V: ${SHADER}"
        )
        
        # Also copy the source as fallback
        add_custom_command(
            OUTPUT ${DEST_SHADER}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER} ${DEST_SHADER}
            DEPENDS ${SHADER}
            COMMENT "Copying Vulkan shader source: ${SHADER}"
        )
        list(APPEND VULKAN_SHADERS ${DEST_SHADER} ${SPV_SHADER})
    endforeach()

    add_custom_target(shaders ALL DEPENDS ${VULKAN_SHADERS})
endif()

# Copy OpenCL kernels to build directory
if(OpenCL_FOUND)
    file(GLOB KERNELS "kernels/opencl/*.cl")
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

# Copy HIP kernels and compile code objects
if(HIP_FOUND)
    find_program(HIPCC_EXECUTABLE hipcc HINTS /usr/bin /opt/rocm/core-10.0/bin /opt/rocm/bin)
    if(NOT HIPCC_EXECUTABLE)
        set(HIPCC_EXECUTABLE hipcc)
    endif()
    find_path(ROCM_CLANG_PATH clang++ HINTS /usr/lib64/rocm/llvm/bin /opt/rocm/core-10.0/lib/llvm/bin)
    if(ROCM_CLANG_PATH)
        if(EXISTS "/opt/rocm/core-10.0")
            set(HIPCC_ENV ${CMAKE_COMMAND} -E env HIP_CLANG_PATH=${ROCM_CLANG_PATH} ROCM_PATH=/opt/rocm/core-10.0)
        else()
            set(HIPCC_ENV ${CMAKE_COMMAND} -E env HIP_CLANG_PATH=${ROCM_CLANG_PATH})
        endif()
    else()
        set(HIPCC_ENV)
    endif()

    file(GLOB HIP_SOURCES "hip_kernels/*.hip")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kernels/rocm)
    
    foreach(HIP_SOURCE ${HIP_SOURCES})
        get_filename_component(KERNEL_NAME ${HIP_SOURCE} NAME)
        get_filename_component(KERNEL_NAME_WE ${HIP_SOURCE} NAME_WLE)
        set(DEST_HIP_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/kernels/rocm/${KERNEL_NAME}")
        set(DEST_HIP_CO "${CMAKE_CURRENT_BINARY_DIR}/kernels/rocm/${KERNEL_NAME_WE}.co")
        
        # Compile to .co using hipcc
        add_custom_command(
            OUTPUT ${DEST_HIP_CO}
            COMMAND ${HIPCC_ENV} ${HIPCC_EXECUTABLE} -O3 --genco --offload-arch=native ${HIP_SOURCE} -o ${DEST_HIP_CO}
            DEPENDS ${HIP_SOURCE}
            COMMENT "Compiling HIP kernel to code object: ${HIP_SOURCE}"
        )

        add_custom_command(
            OUTPUT ${DEST_HIP_SOURCE}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HIP_SOURCE} ${DEST_HIP_SOURCE}
            DEPENDS ${HIP_SOURCE}
            COMMENT "Copying HIP source: ${HIP_SOURCE}"
        )
       
        list(APPEND HIP_KERNELS_SRC ${DEST_HIP_SOURCE} ${DEST_HIP_CO})
    endforeach()
    add_custom_target(hip_kernels ALL DEPENDS ${HIP_KERNELS_SRC})
endif()

# Source files - always include core files
set(LIB_SOURCES
    cpp_src/core/BenchmarkRunner.cpp
    cpp_src/core/ResultFormatter.cpp
    cpp_src/benchmarks/Fp32Bench.cpp
    cpp_src/benchmarks/Fp64Bench.cpp
    cpp_src/benchmarks/Fp16Bench.cpp
    cpp_src/benchmarks/Bf16Bench.cpp
    cpp_src/benchmarks/Fp8Bench.cpp
    cpp_src/benchmarks/Fp4Bench.cpp
    cpp_src/benchmarks/Int8Bench.cpp
    cpp_src/benchmarks/Int4Bench.cpp
    cpp_src/benchmarks/MemBandwidthBench.cpp
    cpp_src/benchmarks/SysMemBandwidthBench.cpp
    cpp_src/benchmarks/SysMemLatencyBench.cpp
    cpp_src/benchmarks/CacheBench.cpp
    cpp_src/benchmarks/PixelFillRateBench.cpp
    cpp_src/benchmarks/RayTracingBench.cpp
    cpp_src/benchmarks/RayDivergenceBench.cpp
    cpp_src/benchmarks/RayAnyHitBench.cpp
    cpp_src/benchmarks/RayIncoherentBench.cpp
    cpp_src/benchmarks/RayPayloadBench.cpp
    cpp_src/benchmarks/RayASBuildBench.cpp
    cpp_src/benchmarks/RayProceduralBench.cpp
    cpp_src/benchmarks/RayMaterialDivergenceBench.cpp
    cpp_src/benchmarks/RayPathTracingBench.cpp
    cpp_src/benchmarks/RaySchedulingBench.cpp
    cpp_src/scene/GltfScene.cpp
    cpp_src/utils/HardwareTelemetry.cpp
    cpp_src/utils/KernelPath.cpp
    cpp_src/utils/ShaderCache.cpp
    cpp_src/utils/SleepInhibitor.cpp
)

# Add backend-specific sources
if(Vulkan_FOUND)
    list(APPEND LIB_SOURCES cpp_src/core/VulkanContext.cpp)
endif()

if(OpenCL_FOUND)
    list(APPEND LIB_SOURCES cpp_src/core/OpenCLContext.cpp)
endif()

if(HIP_FOUND)
    list(APPEND LIB_SOURCES cpp_src/core/ROCmContext.cpp)
endif()

list(APPEND LIB_SOURCES cpp_src/core/RunnerAPI.cpp)

set(SOURCES cpp_src/main.cpp ${LIB_SOURCES})
if(WIN32)
    if(MINGW)
        set_source_files_properties(packaging/windows/gpubench.rc PROPERTIES
            COMPILE_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/packaging/windows -DEMBED_RC_MANIFEST"
        )
    else()
        set_source_files_properties(packaging/windows/gpubench.rc PROPERTIES
            COMPILE_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/packaging/windows"
        )
    endif()
    list(APPEND SOURCES packaging/windows/gpubench.rc)
    if(MSVC)
        list(APPEND SOURCES packaging/windows/gpubench.manifest)
    endif()
endif()

add_library(gpubench_lib STATIC ${LIB_SOURCES})
set_target_properties(gpubench_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_src
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/external/vulkan/Include
    ${CMAKE_CURRENT_BINARY_DIR}/cpp_src
)

target_include_directories(gpubench_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_src
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/external/vulkan/Include
    ${CMAKE_CURRENT_BINARY_DIR}/cpp_src
)

if(Vulkan_FOUND)
    target_include_directories(gpubench PUBLIC ${Vulkan_INCLUDE_DIRS})
    target_include_directories(gpubench_lib PUBLIC ${Vulkan_INCLUDE_DIRS})
    if(SHADERC_FOUND)
        target_include_directories(gpubench PUBLIC ${SHADERC_INCLUDE_DIR})
        target_include_directories(gpubench_lib PUBLIC ${SHADERC_INCLUDE_DIR})
    endif()
endif()

if(OpenCL_FOUND)
    target_include_directories(gpubench PUBLIC ${OpenCL_INCLUDE_DIRS})
    target_include_directories(gpubench_lib PUBLIC ${OpenCL_INCLUDE_DIRS})
endif()

if(HIP_FOUND)
    target_include_directories(gpubench PUBLIC ${HIP_INCLUDE_DIRS})
    target_include_directories(gpubench_lib PUBLIC ${HIP_INCLUDE_DIRS})
endif()

if(WIN32)
    target_include_directories(gpubench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/packaging/windows)
endif()

# Link libraries
if(Vulkan_FOUND)
    target_link_libraries(gpubench PUBLIC Vulkan::Vulkan)
    target_link_libraries(gpubench_lib PUBLIC Vulkan::Vulkan)
    # Removing shaderc dependency for runtime compilation
    # if(SHADERC_FOUND)
    #     target_link_libraries(gpubench PUBLIC ${SHADERC_LIBRARY})
    #     target_compile_definitions(gpubench PRIVATE HAVE_SHADERC)
    # endif()
endif()

if(OpenCL_FOUND)
    # No longer linking OpenCL directly to avoid RPM dependency
    # target_link_libraries(gpubench PUBLIC ${OpenCL_LIBRARIES})
endif()

if(HIP_FOUND)
    # No longer linking HIP directly to avoid RPM dependency
    # target_link_libraries(gpubench PRIVATE hip::host)
    # if(HIPRTC_LIBRARY)
    #     target_link_libraries(gpubench PRIVATE ${HIPRTC_LIBRARY})
    # endif()
    target_compile_definitions(gpubench PRIVATE HAVE_HIPRTC __HIP_PLATFORM_AMD__)
    target_compile_definitions(gpubench_lib PRIVATE HAVE_HIPRTC __HIP_PLATFORM_AMD__)
endif()

if(APPLE)
    target_link_libraries(gpubench PRIVATE "-framework IOKit" "-framework CoreFoundation")
    target_link_libraries(gpubench_lib PRIVATE "-framework IOKit" "-framework CoreFoundation")
endif()

# Compiler definitions for conditional compilation
target_compile_definitions(gpubench PRIVATE GPUBENCH_VERSION="${PROJECT_FULL_VERSION}")
target_compile_definitions(gpubench_lib PRIVATE GPUBENCH_VERSION="${PROJECT_FULL_VERSION}")

if(Vulkan_FOUND)
    target_compile_definitions(gpubench PRIVATE HAVE_VULKAN)
    target_compile_definitions(gpubench_lib PRIVATE HAVE_VULKAN)
endif()

if(OpenCL_FOUND)
    target_compile_definitions(gpubench PRIVATE HAVE_OPENCL)
    target_compile_definitions(gpubench_lib PRIVATE HAVE_OPENCL)
endif()

if(HIP_FOUND)
    target_compile_definitions(gpubench PRIVATE HAVE_ROCM)
    target_compile_definitions(gpubench_lib PRIVATE HAVE_ROCM)
endif()

# Copy kernels directory to project root for running from base directory
add_custom_command(TARGET gpubench POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_BINARY_DIR}/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    COMMENT "Copying kernels directory to project root"
)

# Copy MinGW runtime DLLs to build directory so binary runs standalone out of the box
if(MINGW)
    find_program(MINGW_GCC gcc)
    if(MINGW_GCC)
        get_filename_component(MINGW_BIN_DIR ${MINGW_GCC} DIRECTORY)
        add_custom_command(TARGET gpubench POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MINGW_BIN_DIR}/libstdc++-6.dll"
            "${CMAKE_CURRENT_BINARY_DIR}/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MINGW_BIN_DIR}/libgcc_s_seh-1.dll"
            "${CMAKE_CURRENT_BINARY_DIR}/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MINGW_BIN_DIR}/libwinpthread-1.dll"
            "${CMAKE_CURRENT_BINARY_DIR}/"
            COMMENT "Copying MinGW runtime DLLs to binary directory"
        )
    endif()
endif()

# Build and package Rust GUI if Cargo is available
# GPUBENCH_REQUIRE_GUI (default ON): require cargo for the Rust GUI build.
# Set -DGPUBENCH_REQUIRE_GUI=OFF to build/package a CLI-only release without cargo.
option(GPUBENCH_REQUIRE_GUI "Require cargo to build the Rust GUI (disable for CLI-only builds)" ON)
option(GPUBENCH_BUILD_RUST_GUI "Build Rust GUI and workspace via Cargo" ON)

# If CMake is being invoked from inside cargo (e.g. gpubench-sys build.rs), disable recursive cargo invocation
if(DEFINED ENV{CARGO_PKG_NAME} OR DEFINED ENV{CARGO_MANIFEST_DIR})
    set(GPUBENCH_BUILD_RUST_GUI OFF)
endif()

find_program(CARGO_EXECUTABLE cargo)
if(CARGO_EXECUTABLE AND GPUBENCH_BUILD_RUST_GUI)
    message(STATUS "Cargo found: ${CARGO_EXECUTABLE}. Configuring Rust GUI build.")
    
    # Custom target to compile Rust workspace
    add_custom_target(gpubench_rust_gui ALL
        COMMAND ${CARGO_EXECUTABLE} build --release --workspace
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building Rust GUI and core library"
    )
    
    # Ensure C++ gpubench target depends on it
    add_dependencies(gpubench gpubench_rust_gui)

    # Install the Rust GUI binary (only when the GUI is actually built)
    if(WIN32)
        install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/target/release/gpubench-gui.exe"
                DESTINATION bin
                OPTIONAL)
    else()
        install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/target/release/gpubench-gui"
                DESTINATION bin
                OPTIONAL)
    endif()
elseif(NOT GPUBENCH_BUILD_RUST_GUI)
    message(STATUS "GPUBENCH_BUILD_RUST_GUI is OFF (or invoked from cargo). Skipping gpubench_rust_gui target.")
    if(WIN32)
        install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/target/release/gpubench-gui.exe"
                DESTINATION bin
                OPTIONAL)
    else()
        install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/target/release/gpubench-gui"
                DESTINATION bin
                OPTIONAL)
    endif()
else()
    if(GPUBENCH_REQUIRE_GUI)
        message(FATAL_ERROR
            "Cargo (Rust toolchain) not found, but GPUBENCH_REQUIRE_GUI is ON. "
            "The Rust GUI (gpubench-gui) is required for the default build. "
            "Install Rust (https://rustup.rs) or configure with "
            "-DGPUBENCH_REQUIRE_GUI=OFF to build and package the CLI only.")
    else()
        message(WARNING "Cargo not found and GPUBENCH_REQUIRE_GUI=OFF: "
            "skipping Rust GUI build and install (CLI-only build).")
    endif()
endif()

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
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kernels/vulkan/
            DESTINATION share/gpubench/kernels/vulkan
            FILES_MATCHING PATTERN "*.comp" PATTERN "*.spv")
endif()

if(OpenCL_FOUND)
    file(GLOB OPENCL_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/opencl/*.cl")
    install(FILES ${OPENCL_KERNELS}
            DESTINATION share/gpubench/kernels/opencl)
endif()

if(HIP_FOUND)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kernels/rocm
            DESTINATION share/gpubench/kernels
            FILES_MATCHING PATTERN "*.hip" PATTERN "*.co")
endif()

# Install Windows batch script for easy benchmarking
if(WIN32)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/packaging/windows/RunAllBenchmarks.bat"
            DESTINATION .)
endif()

# Linux desktop integration files
if(UNIX AND NOT APPLE)
    # Desktop Entry
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/io.github.soddentrough.gpubench.desktop"
            DESTINATION share/applications)

    # AppStream Metainfo
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/io.github.soddentrough.gpubench.metainfo.xml"
            DESTINATION share/metainfo)

    # Hicolor Icons
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/icons/hicolor/"
            DESTINATION share/icons/hicolor)

    # Fallback Pixmap
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/icons/pixmaps/gpubench.png"
            DESTINATION share/pixmaps)

    # Shell Completions
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/completions/gpubench"
                  "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/completions/gpubench-gui"
            DESTINATION share/bash-completion/completions)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/completions/_gpubench"
                  "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/completions/_gpubench-gui"
            DESTINATION share/zsh/site-functions)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/completions/gpubench.fish"
                  "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/completions/gpubench-gui.fish"
            DESTINATION share/fish/vendor_completions.d)

    # Man Page
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/man/gpubench.1"
                  "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/man/gpubench-gui.1"
            DESTINATION share/man/man1)

    # Benchmark 3D Models
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets/models")
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/assets/models/"
                DESTINATION share/gpubench/models
                FILES_MATCHING PATTERN "*.glb")
    endif()

    # Reference Renders & Stitching Script
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/renders")
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/renders/"
                DESTINATION share/gpubench/renders
                FILES_MATCHING PATTERN "*.png" PATTERN "*.json")
    endif()
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/scripts/make_triptych.py"
            DESTINATION share/gpubench/scripts
            PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
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
set(CPACK_DEBIAN_PACKAGE_RELEASE ${BUILD_NUMBER})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "GPUBench")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Platform-specific package settings
if(WIN32)
    # Windows-specific packaging
    set(CPACK_GENERATOR "ZIP;NSIS")
    
    # NSIS (Windows installer) settings
    set(CPACK_NSIS_DISPLAY_NAME "GPUBench ${PROJECT_FULL_VERSION}")
    set(CPACK_NSIS_PACKAGE_NAME "GPUBench")
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    set(CPACK_NSIS_MODIFY_PATH ON)
    set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/packaging/windows/icon.ico")
    set(CPACK_NSIS_MUI_UNIICON "${CMAKE_CURRENT_SOURCE_DIR}/packaging/windows/icon.ico")
    set(CPACK_NSIS_HELP_LINK "https://github.com/Soddentrough/GPUBench")
    set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/Soddentrough/GPUBench")
    set(CPACK_NSIS_CONTACT "https://github.com/Soddentrough/GPUBench/issues")
    set(CPACK_NSIS_MENU_LINKS
        "bin/gpubench-gui.exe" "GPUBench"
        "bin/gpubench.exe" "GPUBench (CLI)"
        "RunAllBenchmarks.bat" "Run All Benchmarks"
        "https://github.com/Soddentrough/GPUBench" "GPUBench Website"
    )
    set(CPACK_NSIS_MUI_FINISHPAGE_RUN "bin\\\\gpubench-gui.exe")
    set(CPACK_NSIS_DEFINES "!define MUI_FINISHPAGE_RUN_TEXT \"Start GPUBench\"")
    
    # Include Visual C++ Redistributable if needed
    # set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "ExecWait '$INSTDIR\\\\vcredist_x64.exe /quiet'")
    
elseif(UNIX AND NOT APPLE)
    # Linux packaging
    set(CPACK_GENERATOR "TGZ;DEB;RPM")
    set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")
    
    # DEB package settings
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Soddentrough <https://github.com/Soddentrough/GPUBench>")
    set(CPACK_DEBIAN_PACKAGE_SECTION "utils")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
    else()
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    # DEB package dependencies (Debian package names; libvulkan1 is the Vulkan loader).
    # OpenCL/ROCm stay OPTIONAL -- they are resolved at runtime via dynamic loading,
    # so they are intentionally not listed here (mirrors the RPM policy below).
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libvulkan1, libc6, libstdc++6")
    
    # RPM package settings
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
    set(CPACK_RPM_PACKAGE_GROUP "Applications/System")
    
    # Disable automatic dependency detection to avoid 2GB ROCm requirement
    set(CPACK_RPM_PACKAGE_AUTOREQ "no")
    
    # Manually specify minimal required dependencies
    # We only REQUIRE Vulkan loader and standard libraries
    # OpenCL/ROCm are OPTIONAL due to dynamic loading
    set(CPACK_RPM_PACKAGE_REQUIRES "vulkan-loader, glibc, libgcc, libstdc++")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tmp")
    set(CPACK_RPM_SPEC_MORE_DEFINE "%define _tmppath ${CMAKE_CURRENT_BINARY_DIR}/tmp\n%define _build_id_links none")

    # Exclude system directories from RPM filelist to prevent ownership conflicts with filesystem package
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST
        /etc /etc/init.d /usr /usr/bin
        /usr/include /usr/lib /usr/libx32 /usr/lib64
        /usr/lib/.build-id
        /usr/share /usr/share/aclocal /usr/share/doc
        /usr/share/applications
        /usr/share/bash-completion
        /usr/share/bash-completion/completions
        /usr/share/fish
        /usr/share/fish/vendor_completions.d
        /usr/share/icons
        /usr/share/icons/hicolor
        /usr/share/icons/hicolor/16x16
        /usr/share/icons/hicolor/16x16/apps
        /usr/share/icons/hicolor/32x32
        /usr/share/icons/hicolor/32x32/apps
        /usr/share/icons/hicolor/48x48
        /usr/share/icons/hicolor/48x48/apps
        /usr/share/icons/hicolor/64x64
        /usr/share/icons/hicolor/64x64/apps
        /usr/share/icons/hicolor/128x128
        /usr/share/icons/hicolor/128x128/apps
        /usr/share/icons/hicolor/256x256
        /usr/share/icons/hicolor/256x256/apps
        /usr/share/man
        /usr/share/man/man1
        /usr/share/metainfo
        /usr/share/pixmaps
        /usr/share/zsh
        /usr/share/zsh/site-functions
    )
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION ${CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST})

    set(CPACK_RPM_DEFAULT_DIR_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    
    # Use sanitized rpmbuild wrapper to protect against environment module syntax errors on Fedora/RHEL
    find_program(SYSTEM_RPMBUILD rpmbuild)
    if(SYSTEM_RPMBUILD)
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/rpmbuild_wrapper.sh"
            "${CMAKE_CURRENT_BINARY_DIR}/rpmbuild_wrapper.sh"
            COPYONLY
        )
        execute_process(COMMAND chmod +x "${CMAKE_CURRENT_BINARY_DIR}/rpmbuild_wrapper.sh")
        set(CPACK_PROJECT_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/CPackProjectConfig.cmake")
        file(WRITE "${CPACK_PROJECT_CONFIG_FILE}"
            "# Sanitize shell environment for rpmbuild\n"
            "set(RPMBUILD_EXECUTABLE \"${CMAKE_CURRENT_BINARY_DIR}/rpmbuild_wrapper.sh\" CACHE FILEPATH \"Sanitized rpmbuild wrapper\" FORCE)\n"
            "set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST \"${CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST}\")\n"
            "set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION \"${CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION}\")\n"
            "set(CPACK_RPM_DEFAULT_DIR_PERMISSIONS \"${CPACK_RPM_DEFAULT_DIR_PERMISSIONS}\")\n"
        )
    endif()

    # Dedicated packaging targets for Linux
    add_custom_target(package_rpm
        COMMAND ${CMAKE_COMMAND} -E env --unset=BASH_FUNC_module%% --unset=BASH_FUNC_ml%% --unset=BASH_FUNC__module_raw%% cpack -G RPM
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Building RPM package with sanitized shell environment"
    )
    add_custom_target(check-rpm
        COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_rpm.py" "${CMAKE_CURRENT_BINARY_DIR}/GPUBench-${PROJECT_VERSION}-Linux.rpm"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        DEPENDS package_rpm
        COMMENT "Running RPM dry-run and structural verification..."
    )
    add_custom_target(package_deb
        COMMAND ${CMAKE_COMMAND} -E env --unset=BASH_FUNC_module%% --unset=BASH_FUNC_ml%% --unset=BASH_FUNC__module_raw%% cpack -G DEB
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Building DEB package with sanitized shell environment"
    )
    add_custom_target(package_tgz
        COMMAND ${CMAKE_COMMAND} -E env --unset=BASH_FUNC_module%% --unset=BASH_FUNC_ml%% --unset=BASH_FUNC__module_raw%% cpack -G TGZ
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Building TGZ package with sanitized shell environment"
    )
    add_custom_target(package_linux
        COMMAND ${CMAKE_COMMAND} -E env --unset=BASH_FUNC_module%% --unset=BASH_FUNC_ml%% --unset=BASH_FUNC__module_raw%% cpack -G "TGZ\\;DEB\\;RPM"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Building all Linux packages (TGZ, DEB, RPM) with sanitized shell environment"
    )
    
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
