"""Exercise the real CMake library selector without CANN, Docker or an NPU."""
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "cmake/Sam3CannLibraries.cmake"
CMAKE = os.environ.get("SAM3_TEST_CMAKE") or shutil.which("cmake")


@unittest.skipUnless(CMAKE, "CMake required for library selection tests")
class CannSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root / "cann toolkit with spaces"
        self.home.mkdir()

    def library(self, folder, name):
        result = self.home / folder / ("lib" + name + ".so")
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_bytes(b"fixture, not a real shared library")
        return result.as_posix()

    def select(self, arch="aarch64", extra="", success=True):
        script = self.root / "select.cmake"
        output = self.root / "selected.txt"
        script.write_text(
            'cmake_minimum_required(VERSION 3.19)\n'
            'set(CMAKE_FIND_LIBRARY_PREFIXES "lib")\n'
            'set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")\n'
            f'set(ASCEND_HOME "{self.home.as_posix()}")\n'
            f'set(CANN_HOST_ARCH "{arch}")\n'
            + extra + f'\ninclude("{MODULE.as_posix()}")\n'
            f'file(WRITE "{output.as_posix()}" "${{CANN_LIB_DIR}}\\n${{CANN_LIBS}}")\n',
            encoding="utf-8")
        process = subprocess.run([CMAKE, "-P", str(script)], capture_output=True, text=True)
        if success:
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            return output.read_text().splitlines()
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        return process.stdout + process.stderr

    def test_devlib_preferred_over_runtime_and_generic(self):
        self.library("lib64", "ascendcl")
        self.library("lib64", "runtime")
        self.library("devlib", "ascendcl")
        self.library("devlib/linux/x86_64", "ascendcl")
        expected = self.library("devlib/linux/aarch64", "ascendcl")
        runtime = self.library("devlib/linux/aarch64", "acl_rt")
        directory, libraries = self.select()
        self.assertEqual(directory, (self.home / "devlib/linux/aarch64").as_posix())
        self.assertEqual(libraries, expected + ";" + runtime)

    def test_x86_64_does_not_select_aarch64(self):
        self.library("devlib/linux/aarch64", "ascendcl")
        expected = self.library("devlib/linux/x86_64", "ascendcl")
        self.assertEqual(self.select("x86_64")[1], expected)

    def test_old_runtime_cache_is_migrated(self):
        old = self.library("lib64", "ascendcl")
        runtime = self.library("lib64", "runtime")
        expected = self.library("devlib/aarch64", "ascendcl")
        extra = (f'set(CANN_ASCENDCL_LIB "{old}" CACHE FILEPATH "old")\n'
                 f'set(CANN_ACLRT_LIB "{runtime}" CACHE FILEPATH "old")\n'
                 f'set(CANN_LIB_DIR "{self.home.as_posix()}/lib64" CACHE PATH "old")\n'
                 f'set(CANN_ASCENDCL_LIB "{old}")\n')
        self.assertEqual(self.select(extra=extra)[1], expected)

    def test_legacy_monolithic_stub_needs_no_runtime(self):
        expected = self.library("runtime/lib64/stub", "ascendcl")
        self.library("lib64", "runtime")
        self.assertEqual(self.select()[1], expected)

    def test_arm64_legacy_arch_root(self):
        expected = self.library("arm64-linux/devlib", "ascendcl")
        self.assertEqual(self.select()[1], expected)

    def test_missing_devlib_does_not_fall_back_to_real_or_simulator(self):
        self.library("lib64", "ascendcl")
        self.library("lib64", "runtime")
        self.library("tools/simulator/Ascend310P3/camodel", "ascendcl")
        self.assertIn("Cannot find CANN development", self.select(success=False))

    def test_missing_native_arch_does_not_select_other_arch(self):
        self.library("devlib/linux/x86_64", "ascendcl")
        self.assertIn("Cannot find CANN development", self.select(success=False))

    def test_split_runtime_is_not_taken_from_another_directory(self):
        expected = self.library("devlib/linux/aarch64", "ascendcl")
        self.library("devlib/linux/x86_64", "acl_rt")
        self.library("devlib", "acl_rt")
        self.assertEqual(self.select()[1], expected)

    def test_unexpected_arch_is_rejected(self):
        self.assertIn("supported CANN_HOST_ARCH", self.select("riscv64", success=False))


class CannBuildIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(CMAKE and (os.environ.get("SAM3_TEST_ZIG") or
                                  (os.name != "nt" and shutil.which("cc"))),
                         "CMake and a Linux C compiler (or SAM3_TEST_ZIG) required")
    def test_link_executable_and_module_without_development_rpath(self):
        # Real CMake/compiler/linker, but tiny mock libraries, not real CANN.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, build = root / "source", root / "build with spaces"
            source.mkdir()
            (source / "ascendcl.c").write_text("int aclInit(void) { return 0; }\n")
            (source / "acl_rt.c").write_text("int aclrtGetDeviceCount(void) { return 1; }\n")
            (source / "probe.c").write_text(
                "extern int aclInit(void); extern int aclrtGetDeviceCount(void);\n"
                "int main(void) { return aclInit() + aclrtGetDeviceCount() - 1; }\n")
            (source / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.19)\nproject(CannLinkTest C)\n'
                'set(ASCEND_HOME "${CMAKE_BINARY_DIR}/toolkit")\n'
                'set(CANN_HOST_ARCH "x86_64")\n'
                'set(dev "${ASCEND_HOME}/devlib/linux/x86_64")\n'
                'file(MAKE_DIRECTORY "${dev}")\n'
                'file(WRITE "${dev}/libascendcl.so" "")\n'
                'file(WRITE "${dev}/libacl_rt.so" "")\n'
                'add_library(ascendcl SHARED ascendcl.c)\n'
                'add_library(acl_rt SHARED acl_rt.c)\n'
                'set_target_properties(ascendcl acl_rt PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${dev}")\n'
                f'include("{MODULE.as_posix()}")\n'
                'add_executable(probe probe.c)\n'
                'add_library(probe_module MODULE probe.c)\n'
                'foreach(target probe probe_module)\n'
                '  add_dependencies(${target} ascendcl acl_rt)\n'
                '  target_link_libraries(${target} PRIVATE ${CANN_LIBS})\n'
                '  sam3_configure_cann_target(${target})\n'
                '  get_target_property(options ${target} LINK_OPTIONS)\n'
                '  file(WRITE "${CMAKE_BINARY_DIR}/${target}.link-options" "${options}")\n'
                # Zig 0.13's frontend rejects GNU rpath-link. Check generation
                # separately and omit only this linker-only flag in mock builds.
                '  if(TEST_ZIG)\n'
                '    set_target_properties(${target} PROPERTIES LINK_OPTIONS "")\n'
                '  endif()\n'
                'endforeach()\n', encoding="utf-8")
            command = [CMAKE, "-S", str(source), "-B", str(build), "-G", "Ninja"]
            if os.environ.get("SAM3_TEST_ZIG"):
                command += ["-DCMAKE_SYSTEM_NAME=Linux", "-DCMAKE_SYSTEM_PROCESSOR=x86_64",
                            "-DTEST_ZIG=ON",
                            "-DCMAKE_C_COMPILER=" + Path(os.environ["SAM3_TEST_ZIG"]).as_posix()
                            + ";cc;-target;x86_64-linux-musl"]
            if os.environ.get("SAM3_TEST_NINJA"):
                command += ["-DCMAKE_MAKE_PROGRAM=" + Path(os.environ["SAM3_TEST_NINJA"]).as_posix()]
            for cmd in (command, [CMAKE, "--build", str(build)]):
                result = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for path in (build / "probe", build / "libprobe_module.so"):
                needed, rpaths = elf_dynamic_paths(path.read_bytes())
                self.assertIn("libascendcl.so", needed)
                self.assertIn("libacl_rt.so", needed)
                self.assertNotIn("libruntime.so", needed)
                self.assertFalse(rpaths, (path, rpaths))
            for target in ("probe", "probe_module"):
                self.assertIn("-Wl,-rpath-link,", (build / (target + ".link-options")).read_text())

    def test_all_runtime_targets_disable_stub_rpath(self):
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        for target in ("ascendsam3_demo", "ascendsam3_bench", "ascendsam3_vision_bench", "ascendsam3_py"):
            self.assertIn(f"sam3_configure_cann_target({target})", cmake)
        module = MODULE.read_text(encoding="utf-8")
        self.assertIn("SKIP_BUILD_RPATH TRUE", module)
        self.assertIn('INSTALL_RPATH ""', module)
        self.assertIn("INSTALL_RPATH_USE_LINK_PATH FALSE", module)
        self.assertNotIn("--allow-shlib-undefined", module)

    def test_selector_is_copied_only_to_business_stage(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertGreater(dockerfile.index("COPY cmake /app/cmake"),
                           dockerfile.index("FROM dependencies AS builder"))
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertGreater(cmake.index('include("${CMAKE_SOURCE_DIR}/cmake/Sam3CannLibraries.cmake")'),
                           cmake.index("if(SAM3_DEPENDENCIES_ONLY)"))
        self.assertNotIn("link_directories(", cmake)

    def test_runtime_environment_does_not_add_development_dirs(self):
        runtime = (ROOT / "Dockerfile").read_text(encoding="utf-8").split("AS runtime", 1)[1]
        self.assertNotIn("/devlib", runtime)
        self.assertNotIn("/stub", runtime)
        self.assertIn("/usr/local/Ascend/driver/lib64", runtime)


def elf_dynamic_paths(data):
    """Read DT_NEEDED/RPATH/RUNPATH from the tiny Linux ELF64 test artifacts."""
    if data[:6] != b"\x7fELF\x02\x01":
        raise AssertionError("Expected little-endian ELF64")
    phoff = struct.unpack_from("<Q", data, 32)[0]
    size, count = struct.unpack_from("<HH", data, 54)
    segments = [struct.unpack_from("<IIQQQQQQ", data, phoff + size*i) for i in range(count)]
    dynamic = next(p for p in segments if p[0] == 2)
    entries = []
    for offset in range(dynamic[2], dynamic[2] + dynamic[5], 16):
        entry = struct.unpack_from("<qQ", data, offset)
        if entry[0] == 0:
            break
        entries.append(entry)
    strings_address = next(value for tag, value in entries if tag == 5)
    segment = next(p for p in segments if p[0] == 1 and p[3] <= strings_address < p[3] + p[5])
    strings = segment[2] + strings_address - segment[3]
    def string(index):
        start = strings + index
        return data[start:data.index(b"\0", start)].decode()
    return ([string(value) for tag, value in entries if tag == 1],
            [string(value) for tag, value in entries if tag in (15, 29)])


if __name__ == "__main__":
    unittest.main()
