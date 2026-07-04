# GPUBench Release Process

This document describes how GPUBench builds and publishes releases for different targets.

## Release Architecture

GPUBench uses a hybrid release process:
1. **Windows Binaries**: Built and published automatically in the cloud via GitHub Actions.
2. **Linux Packages**: Built locally on the development machine and uploaded manually using the GitHub CLI (`gh`).

---

## 1. Windows Releases (Automated)

Windows binaries are handled by the GitHub Actions workflow in `.github/workflows/release.yml`.

- **Trigger**: Pushing a tag matching `v*` (e.g. `v1.1.0.1`).
- **Generated Assets**:
  - `GPUBench-*-win64.zip` (Portable ZIP)
  - `GPUBench-*-win64.exe` (NSIS Installer)
- **Workflow**:
  1. Commit changes and update the `BUILD_NUMBER` in `CMakeLists.txt`.
  2. Tag the commit (e.g., `git tag v1.1.0.1`).
  3. Push the tag to remote (`git push origin v1.1.0.1`).
  4. The GitHub Actions runner will automatically build, package, and publish the Windows binaries to the release page.

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
gh release upload v1.1.0.1 \
  build/GPUBench-1.1.0.1-Linux.deb \
  build/GPUBench-1.1.0.1-Linux.rpm \
  build/GPUBench-1.1.0.1-Linux.tar.gz \
  --clobber
```

---

## 3. macOS Releases (Local macOS Machine)

If you compile for macOS, those packages (DMG, Tarball) are built locally on a macOS machine and uploaded manually using the GitHub CLI.

### Step 1: Build and Package Locally
```bash
# Navigate to the build directory
cd build

# Configure and compile release targets
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(sysctl -n hw.ncpu)

# Generate DMG installer and tarball
cpack -G DragNDrop
cpack -G TGZ
```

### Step 2: Upload to GitHub Release
```bash
# Upload assets to the corresponding tag
gh release upload v1.1.0.1 \
  build/GPUBench-1.1.0.1-Darwin.dmg \
  build/GPUBench-1.1.0.1-Darwin.tar.gz \
  --clobber
```
