# GPUBench Release Process

This document describes how GPUBench builds and publishes releases for different targets.

## Release Architecture

GPUBench uses a fully automated release process via GitHub Actions:
1. **Windows, macOS & Linux Binaries**: Built and published automatically in the cloud via GitHub Actions.
2. **Linux Manual Fallback**: If the CI build is unavailable, Linux packages can still be built locally (see Section 2).

---

## 1. Windows, macOS & Linux Releases (Automated)

Windows, macOS, and Linux binaries are handled by the GitHub Actions workflow in `.github/workflows/release.yml`.

- **Trigger**: Pushing a tag matching `v*` (e.g. `v1.1.0.2`).
- **Generated Assets**:
  - Windows: `GPUBench-*-win64.zip` and `GPUBench-*-win64.exe`
  - macOS: `GPUBench-*-Darwin.dmg` and `GPUBench-*-Darwin.tar.gz`
  - Linux: `GPUBench-*-Linux.deb`, `GPUBench-*-Linux.rpm`, and `GPUBench-*-Linux.tar.gz`
- **Workflow**:
  1. Commit changes and update the `BUILD_NUMBER` in `CMakeLists.txt`.
  2. Tag the commit (e.g., `git tag v1.1.0.2`).
  3. Push the tag to remote (`git push origin v1.1.0.2`).
  4. The GitHub Actions runner will automatically build, package, and publish the Windows, macOS, and Linux binaries to the release page.

---

## 2. Linux Releases (Built in CI)

Linux packages (DEB, RPM, Tarball) are now built automatically in CI by the `build-linux` job in `.github/workflows/release.yml` and uploaded to the release page alongside the Windows/macOS assets. No local packaging is required for a normal release.

### Manual Fallback: Build and Package Locally

If the CI Linux job is unavailable, packages can still be built on a local Linux machine (which needs the Vulkan SDK, OpenCL development libraries, `glslc`, and `rpmbuild` installed).

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
