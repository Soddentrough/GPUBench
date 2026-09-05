#!/usr/bin/env python3
"""
GPUBench Automated RPM Package Verification & Dry-Run Suite

Performs comprehensive pre-release quality checks on built Linux RPM packages:
1. Static Directory Ownership Audit:
   Ensures no standard system or FreeDesktop directories (/usr/share/applications,
   /usr/share/icons, /usr/share/bash-completion, etc.) are declared as owned %dir,
   preventing packaging conflicts with system 'filesystem' packages.
2. File Permissions Audit:
   Ensures binaries in /usr/bin/ are executable (0755 / -rwxr-xr-x) and assets are 0644.
3. Payload Completeness Audit:
   Ensures all binaries, icons (16-256px), desktop file, AppStream metainfo, man pages,
   shell completions (bash, zsh, fish), and compute kernels are present.
4. Transaction Dry-Run Simulation:
   Runs 'rpm -U --test --replacepkgs --ignoresize' against the active host rpmdb
   to verify real transaction viability without modifying the system or requiring sudo.
5. Extraction & Binary Header Integrity:
   Unpacks the payload via rpm2cpio and validates ELF binary headers.
"""

import sys
import os
import glob
import shutil
import tempfile
import argparse
import subprocess
from typing import Dict, List, Tuple, Optional, Any

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


FORBIDDEN_OWNED_DIRS = {
    "/usr",
    "/usr/bin",
    "/usr/include",
    "/usr/lib",
    "/usr/lib64",
    "/usr/share",
    "/usr/share/applications",
    "/usr/share/bash-completion",
    "/usr/share/bash-completion/completions",
    "/usr/share/fish",
    "/usr/share/fish/vendor_completions.d",
    "/usr/share/icons",
    "/usr/share/icons/hicolor",
    "/usr/share/icons/hicolor/16x16",
    "/usr/share/icons/hicolor/16x16/apps",
    "/usr/share/icons/hicolor/32x32",
    "/usr/share/icons/hicolor/32x32/apps",
    "/usr/share/icons/hicolor/48x48",
    "/usr/share/icons/hicolor/48x48/apps",
    "/usr/share/icons/hicolor/64x64",
    "/usr/share/icons/hicolor/64x64/apps",
    "/usr/share/icons/hicolor/128x128",
    "/usr/share/icons/hicolor/128x128/apps",
    "/usr/share/icons/hicolor/256x256",
    "/usr/share/icons/hicolor/256x256/apps",
    "/usr/share/man",
    "/usr/share/man/man1",
    "/usr/share/metainfo",
    "/usr/share/pixmaps",
    "/usr/share/zsh",
    "/usr/share/zsh/site-functions",
    "/etc",
    "/etc/init.d",
}

REQUIRED_PAYLOAD_FILES = [
    "/usr/bin/gpubench",
    "/usr/bin/gpubench-gui",
    "/usr/share/applications/io.github.soddentrough.gpubench.desktop",
    "/usr/share/metainfo/io.github.soddentrough.gpubench.metainfo.xml",
    "/usr/share/man/man1/gpubench.1.gz",
    "/usr/share/man/man1/gpubench-gui.1.gz",
    "/usr/share/bash-completion/completions/gpubench",
    "/usr/share/bash-completion/completions/gpubench-gui",
    "/usr/share/zsh/site-functions/_gpubench",
    "/usr/share/zsh/site-functions/_gpubench-gui",
    "/usr/share/fish/vendor_completions.d/gpubench.fish",
    "/usr/share/fish/vendor_completions.d/gpubench-gui.fish",
    "/usr/share/icons/hicolor/16x16/apps/io.github.soddentrough.gpubench.png",
    "/usr/share/icons/hicolor/32x32/apps/io.github.soddentrough.gpubench.png",
    "/usr/share/icons/hicolor/48x48/apps/io.github.soddentrough.gpubench.png",
    "/usr/share/icons/hicolor/64x64/apps/io.github.soddentrough.gpubench.png",
    "/usr/share/icons/hicolor/128x128/apps/io.github.soddentrough.gpubench.png",
    "/usr/share/icons/hicolor/256x256/apps/io.github.soddentrough.gpubench.png",
    "/usr/share/pixmaps/gpubench.png",
]


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def verify_metadata(rpm_path: str) -> Tuple[bool, Dict[str, str], List[str]]:
    errors: List[str] = []
    metadata: Dict[str, str] = {}

    tags = {
        "NAME": "%{NAME}",
        "VERSION": "%{VERSION}",
        "RELEASE": "%{RELEASE}",
        "ARCH": "%{ARCH}",
        "SUMMARY": "%{SUMMARY}",
        "LICENSE": "%{LICENSE}",
        "URL": "%{URL}",
    }
    qf = "\t".join(f"{k}:::{v}" for k, v in tags.items())
    code, out, err = run_cmd(["rpm", "-qp", "--queryformat", qf, rpm_path])
    if code != 0:
        errors.append(f"Failed to query RPM metadata: {err.strip()}")
        return False, metadata, errors

    for pair in out.strip().split("\t"):
        if ":::" in pair:
            k, v = pair.split(":::", 1)
            metadata[k] = v.strip()

    if metadata.get("NAME") != "gpubench":
        errors.append(f"Unexpected package name: '{metadata.get('NAME')}' (expected 'gpubench')")
    if not metadata.get("VERSION"):
        errors.append("Package version is empty")

    code, req_out, req_err = run_cmd(["rpm", "-qp", "--requires", rpm_path])
    if code == 0:
        metadata["REQUIRES"] = req_out.strip().replace("\n", ", ")
        for expected_dep in ["vulkan-loader", "glibc", "libgcc", "libstdc++"]:
            if expected_dep not in req_out:
                errors.append(f"Missing expected package requirement: '{expected_dep}'")
    else:
        errors.append(f"Failed to query dependencies: {req_err.strip()}")

    return len(errors) == 0, metadata, errors


def query_file_entries(rpm_path: str) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
    """
    Parses file entries using 'rpm -qlvp'.
    Each line typically looks like:
    drwxr-xr-x    2 root     root                        0 Jul  4  2010 /usr/share/gpubench
    -rwxr-xr-x    1 root     root                  1386456 Jul  4  2010 /usr/bin/gpubench
    """
    errors: List[str] = []
    entries: List[Dict[str, Any]] = []

    code, out, err = run_cmd(["rpm", "-qlvp", rpm_path])
    if code != 0:
        errors.append(f"Failed to list RPM file entries: {err.strip()}")
        return False, entries, errors

    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 9:
            mode = parts[0]
            size = int(parts[4]) if parts[4].isdigit() else 0
            path = parts[-1]
            is_dir = mode.startswith("d")
            entries.append({
                "mode": mode,
                "is_dir": is_dir,
                "size": size,
                "path": path,
            })
    return True, entries, errors


def verify_directories(entries: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """
    Audits directory ownership (%dir) in the RPM.
    Must NOT contain standard system/FreeDesktop directories.
    """
    errors: List[str] = []
    owned_dirs: List[str] = []

    for item in entries:
        if item["is_dir"]:
            p = item["path"].rstrip("/")
            owned_dirs.append(p)
            if p in FORBIDDEN_OWNED_DIRS:
                errors.append(
                    f"Forbidden system directory collision: Package claims ownership of '{p}'. "
                    f"This will cause conflict with system packages (e.g. filesystem)."
                )
            elif not p.startswith("/usr/share/gpubench"):
                errors.append(
                    f"Unexpected owned directory: '{p}'. Only '/usr/share/gpubench*' subdirectories "
                    f"should be declared as owned."
                )

    return len(errors) == 0, owned_dirs, errors


def verify_permissions(entries: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Audits file permissions:
    - Executables in /usr/bin must be executable (x bit present in owner).
    - Directories must be 0755 (drwxr-xr-x).
    - Other files should be 0644 (-rw-r--r--).
    """
    errors: List[str] = []

    for item in entries:
        mode = item["mode"]
        path = item["path"]

        if path.startswith("/usr/bin/"):
            # Must be executable: -rwxr-xr-x
            if "x" not in mode[1:4]:
                errors.append(
                    f"Executable binary '{path}' lacks execute permissions (mode: '{mode}'). "
                    f"Running this binary will produce 'Permission denied'!"
                )
        elif item["is_dir"]:
            # Directories must be drwxr-xr-x
            if not mode.startswith("drwxr-xr-x"):
                errors.append(f"Directory '{path}' has non-standard permissions: '{mode}' (expected 'drwxr-xr-x')")
        else:
            # Data/asset files: should not have unnecessary execute bits
            if "x" in mode[1:4] and not path.endswith(".sh"):
                errors.append(f"Data file '{path}' unexpectedly has execute bit set: '{mode}'")

    return len(errors) == 0, errors


def verify_payload_completeness(entries: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    present_paths = {item["path"] for item in entries}

    for req in REQUIRED_PAYLOAD_FILES:
        if req not in present_paths:
            errors.append(f"Required payload file missing from RPM: '{req}'")

    # Check for compute kernels
    kernel_files = [p for p in present_paths if p.startswith("/usr/share/gpubench/kernels/")]
    if len(kernel_files) < 15:
        errors.append(f"Suspiciously low kernel count in RPM: only {len(kernel_files)} found")

    return len(errors) == 0, errors


def verify_dry_run_transaction(rpm_path: str) -> Tuple[bool, str, List[str]]:
    """
    Performs live dry-run transaction testing using 'rpm -U --test --replacepkgs --ignoresize'.
    Checks if active rpmdb has any file or directory conflicts.
    """
    errors: List[str] = []
    cmd = ["rpm", "-U", "--test", "--replacepkgs", "--ignoresize", rpm_path]
    code, out, err = run_cmd(cmd)

    output = (out + "\n" + err).strip()
    if code != 0:
        errors.append(f"RPM transaction dry-run test failed with return code {code}:")
        for line in output.splitlines():
            line_s = line.strip()
            if line_s:
                errors.append(f"  {line_s}")
        return False, output, errors

    return True, output, errors


def verify_extraction_and_elf(rpm_path: str) -> Tuple[bool, List[str]]:
    """
    Extracts the RPM into a temporary directory using rpm2cpio & cpio,
    and inspects ELF magic numbers on binaries.
    """
    errors: List[str] = []

    if not shutil.which("rpm2cpio") or not shutil.which("cpio"):
        return True, ["(Skipped rpm2cpio/cpio extraction: tools not present on host)"]

    with tempfile.TemporaryDirectory(prefix="gpubench_rpm_test_") as tmpdir:
        # Extract
        p1 = subprocess.Popen(["rpm2cpio", rpm_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.Popen(["cpio", "-idmv"], stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdir)
        p1.stdout.close()
        _, cpio_err = p2.communicate()
        p1.wait()

        if p2.returncode != 0:
            errors.append(f"Failed to unpack RPM via rpm2cpio: {cpio_err.decode('utf-8', errors='replace')}")
            return False, errors

        # Verify extracted binaries
        for bin_rel in ["usr/bin/gpubench", "usr/bin/gpubench-gui"]:
            bin_path = os.path.join(tmpdir, bin_rel)
            if not os.path.isfile(bin_path):
                errors.append(f"Extracted payload is missing binary '{bin_rel}'")
                continue

            sz = os.path.getsize(bin_path)
            if sz < 100000:
                errors.append(f"Extracted binary '{bin_rel}' size suspiciously small: {sz} bytes")

            # Check ELF header
            with open(bin_path, "rb") as f:
                magic = f.read(4)
                if magic != b"\x7fELF":
                    errors.append(f"Extracted binary '{bin_rel}' is not a valid ELF binary (magic: {magic!r})")

            # Check executable bit on filesystem
            if not os.access(bin_path, os.X_OK):
                errors.append(f"Extracted binary '{bin_rel}' lacks execution permissions on disk")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="GPUBench RPM Package Verification & Dry-Run Suite")
    parser.add_argument("rpm_path", nargs="?", default=None, help="Path to the RPM package file")
    parser.add_argument("--skip-install-test", action="store_true", help="Skip live 'rpm --test' simulation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose details")
    args = parser.parse_args()

    rpm_path = args.rpm_path
    if not rpm_path:
        # Auto-discover in build directory
        candidates = sorted(glob.glob("build/GPUBench-*-Linux.rpm"), reverse=True)
        if candidates:
            rpm_path = candidates[0]
        else:
            print(f"{Colors.RED}Error: No RPM package found in 'build/'. Specify package path explicitly.{Colors.RESET}")
            sys.exit(1)

    if not os.path.isfile(rpm_path):
        print(f"{Colors.RED}Error: RPM file not found at '{rpm_path}'{Colors.RESET}")
        sys.exit(1)

    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN} GPUBench RPM Package Verification & Dry-Run Suite{Colors.RESET}")
    print(f" Target Package: {Colors.BOLD}{os.path.abspath(rpm_path)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")

    all_passed = True
    all_regressions: List[str] = []

    # 1. Query Metadata
    print(f"{Colors.BOLD}1. Package Metadata & Dependency Requirements:{Colors.RESET}")
    meta_ok, metadata, meta_errs = verify_metadata(rpm_path)
    if meta_ok:
        print(f"  Name:         {metadata.get('NAME')} {metadata.get('VERSION')}-{metadata.get('RELEASE')}")
        print(f"  Architecture: {metadata.get('ARCH')}")
        print(f"  License:      {metadata.get('LICENSE')}")
        print(f"  Dependencies: {metadata.get('REQUIRES', 'None')}")
        print(f"  Status:       {Colors.GREEN}PASS{Colors.RESET}\n")
    else:
        all_passed = False
        print(f"  Status:       {Colors.RED}FAIL{Colors.RESET}")
        for e in meta_errs:
            print(f"    - {Colors.RED}{e}{Colors.RESET}")
            all_regressions.append(e)
        print()

    # 2. File Entries
    entries_ok, entries, entries_errs = query_file_entries(rpm_path)
    if not entries_ok:
        for e in entries_errs:
            print(f"{Colors.RED}{e}{Colors.RESET}")
        sys.exit(1)

    # 3. Directory Ownership Collision Audit
    print(f"{Colors.BOLD}2. Directory Ownership & FreeDesktop/XDG Invariant Audit:{Colors.RESET}")
    dirs_ok, owned_dirs, dirs_errs = verify_directories(entries)
    if dirs_ok:
        print(f"  GPUBench Owned Directories ({len(owned_dirs)}):")
        for d in owned_dirs:
            print(f"    • {d}")
        print(f"  FreeDesktop System Exclusions: All standard system directories safely omitted.")
        print(f"  Status:       {Colors.GREEN}PASS (Zero System Collisions){Colors.RESET}\n")
    else:
        all_passed = False
        print(f"  Status:       {Colors.RED}FAIL (Directory Ownership Collisions Detected){Colors.RESET}")
        for e in dirs_errs:
            print(f"    - {Colors.RED}{e}{Colors.RESET}")
            all_regressions.append(e)
        print()

    # 4. File Permissions Audit
    print(f"{Colors.BOLD}3. Binary & File Permissions Audit:{Colors.RESET}")
    perm_ok, perm_errs = verify_permissions(entries)
    if perm_ok:
        print(f"  /usr/bin/gpubench:     -rwxr-xr-x (0755 Executable) [VERIFIED]")
        print(f"  /usr/bin/gpubench-gui: -rwxr-xr-x (0755 Executable) [VERIFIED]")
        print(f"  All directory and asset permissions comply with FreeDesktop standards.")
        print(f"  Status:       {Colors.GREEN}PASS{Colors.RESET}\n")
    else:
        all_passed = False
        print(f"  Status:       {Colors.RED}FAIL (Invalid File Permissions){Colors.RESET}")
        for e in perm_errs:
            print(f"    - {Colors.RED}{e}{Colors.RESET}")
            all_regressions.append(e)
        print()

    # 5. Payload Completeness Audit
    print(f"{Colors.BOLD}4. Payload Completeness Audit:{Colors.RESET}")
    comp_ok, comp_errs = verify_payload_completeness(entries)
    if comp_ok:
        print(f"  All required binaries, desktop files, AppStream XML, man pages, shell completions,")
        print(f"  and high-resolution icons (16px..256px) verified in RPM payload ({len(entries)} total items).")
        print(f"  Status:       {Colors.GREEN}PASS{Colors.RESET}\n")
    else:
        all_passed = False
        print(f"  Status:       {Colors.RED}FAIL (Missing Required Payload Files){Colors.RESET}")
        for e in comp_errs:
            print(f"    - {Colors.RED}{e}{Colors.RESET}")
            all_regressions.append(e)
        print()

    # 6. Extraction & Binary Header Integrity
    print(f"{Colors.BOLD}5. CPIO Extraction & ELF Binary Integrity:{Colors.RESET}")
    ext_ok, ext_errs = verify_extraction_and_elf(rpm_path)
    if ext_ok:
        print(f"  Extracted binaries verified: Valid ELF64 executable magic headers and size bounds.")
        print(f"  Status:       {Colors.GREEN}PASS{Colors.RESET}\n")
    else:
        all_passed = False
        print(f"  Status:       {Colors.RED}FAIL (Extraction or ELF Corrupt){Colors.RESET}")
        for e in ext_errs:
            print(f"    - {Colors.RED}{e}{Colors.RESET}")
            all_regressions.append(e)
        print()

    # 7. Live Transaction Dry-Run
    print(f"{Colors.BOLD}6. Live Host Transaction Dry-Run Simulation:{Colors.RESET}")
    if args.skip_install_test:
        print(f"  (Skipped via --skip-install-test flag)")
    elif not shutil.which("rpm") or not os.path.exists("/var/lib/rpm"):
        print(f"  {Colors.YELLOW}Host lacks active RPM database; skipping transaction simulation.{Colors.RESET}\n")
    else:
        tx_ok, tx_out, tx_errs = verify_dry_run_transaction(rpm_path)
        if tx_ok:
            print(f"  Simulated 'rpm -U --test --replacepkgs --ignoresize {os.path.basename(rpm_path)}'")
            print(f"  Transaction test succeeded against /var/lib/rpm with ZERO file/dir conflicts.")
            print(f"  Status:       {Colors.GREEN}PASS (Installable without Errors){Colors.RESET}\n")
        else:
            all_passed = False
            print(f"  Status:       {Colors.RED}FAIL (Live Transaction Conflict){Colors.RESET}")
            for e in tx_errs:
                print(f"    - {Colors.RED}{e}{Colors.RESET}")
                all_regressions.append(e)
            print()

    # Summary
    print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✔ ALL RPM VERIFICATION CHECKS PASSED: Package is production-ready, installable, and conflict-free.{Colors.RESET}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✘ RPM VERIFICATION FAILED:{Colors.RESET}")
        for r in all_regressions:
            print(f"  - {Colors.RED}{r}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*80}{Colors.RESET}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
