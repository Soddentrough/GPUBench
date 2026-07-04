# GPUBench Release Process

This document describes how GPUBench builds and publishes releases for different targets.

## Release Architecture

GPUBench uses a hybrid release process:
1. **Windows & macOS Binaries**: Built and published automatically in the cloud via GitHub Actions.
2. **Linux Packages**: Built locally on the development machine and uploaded manually using the GitHub CLI (`gh`).

---

## 1. Windows & macOS Releases (Automated)

Windows and macOS binaries are handled by the GitHub Actions workflow in `.github/workflows/release.yml`.

- **Trigger**: Pushing a tag matching `v*` (e.g. `v1.1.0.2`).
- **Generated Assets**:
  - Windows: `GPUBench-*-win64.zip` and `GPUBench-*-win64.exe`
  - macOS: `GPUBench-*-Darwin.dmg` and `GPUBench-*-Darwin.tar.gz`
- **Workflow**:
  1. Commit changes and update the `BUILD_NUMBER` in `CMakeLists.txt`.
  2. Tag the commit (e.g., `git tag v1.1.0.2`).
  3. Push the tag to remote (`git push origin v1.1.0.2`).
  4. The GitHub Actions runner will automatically build, package, and publish the Windows and macOS binaries to the release page.

---

## 2. Linux Releases (Local Dev Machine)

Linux binaries (DEB, RPM, Tarball) must be built locally on the development Linux machine (which has the Vulkan SDK, OpenCL development libraries, and `glslc` compiler installed).

### Step 1: Build and Package Locally
Run the compile and packaging steps inside the local `build/` directory.

> [!IMPORTANT]
> Fedora/RedHat systems may have exported shell function definitions (like `ml` or `module` from Lmod) in the environment that cause `rpmbuild` to fail with bad exit statuses. Always package using a clean environment wrapper (`env -i`).

```bash
# Navigate to the build directory
cd build

# Configure and compile release targets
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)

# Generate packages using a clean environment wrapper
env -i HOME=$HOME PATH=$PATH USER=$USER cpack -G DEB
env -i HOME=$HOME PATH=$PATH USER=$USER cpack -G RPM
env -i HOME=$HOME PATH=$PATH USER=$USER cpack -G TGZ
```

### Step 2: Upload to GitHub Release
Upload the generated assets directly to the existing GitHub release page using the GitHub CLI:

```bash
# Upload assets to the corresponding tag
gh release upload v1.1.0.2 \
  build/GPUBench-1.1.0.2-Linux.deb \
  build/GPUBench-1.1.0.2-Linux.rpm \
  build/GPUBench-1.1.0.2-Linux.tar.gz \
  --clobber
```
