"""Vision conversion/config regression tests; never invoke real Docker/ATC."""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_lf(path, text):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


class VisionConfigTest(unittest.TestCase):
    def test_aipp_normalization_formula(self):
        config = (ROOT / "models/config/vision.cfg").read_text(encoding="utf-8")
        fields = dict(re.findall(r"^\s*(mean_chn_\d|min_chn_\d|var_reci_chn_\d)\s*:\s*([\d.]+)", config, re.M))
        for channel in range(3):
            mean = float(fields[f"mean_chn_{channel}"])
            minimum = float(fields[f"min_chn_{channel}"])
            reciprocal = float(fields[f"var_reci_chn_{channel}"])
            self.assertEqual(mean, 127)
            self.assertEqual(minimum, 0.5)
            for pixel in range(256):
                self.assertAlmostEqual((pixel - mean - minimum) * reciprocal, pixel / 127.5 - 1, places=12)
        self.assertRegex(config, r"rbuv_swap_switch\s*:\s*true")

    def test_both_compose_variants_select_vision_from_env(self):
        expected = 'VISION_MODEL: "${VISION_MODEL:-models/om-models/vision-encoder.om}"'
        self.assertEqual((ROOT / "docker-compose.yml").read_text(encoding="utf-8").count(expected), 1)
        self.assertEqual((ROOT / "docker-compose.dual.yml").read_text(encoding="utf-8").count(expected), 2)


class VisionConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.name == "nt":
            cls.bash = str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe")
            if not Path(cls.bash).is_file():
                raise unittest.SkipTest("Git Bash unavailable")
        else:
            cls.bash = shutil.which("bash")
            if not cls.bash:
                raise unittest.SkipTest("Bash unavailable")

    def run_conversion(self, options, existing=False):
        with tempfile.TemporaryDirectory(prefix="sam3-convert-test-") as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            # Normalize line endings like a Linux checkout, not a Windows CRLF checkout.
            write_lf(root / "scripts/convert_models.sh",
                     (ROOT / "scripts/convert_models.sh").read_text(encoding="utf-8"))
            models = root / "models/onnx-models"
            models.mkdir(parents=True)
            # Deliberately omit Text/Decoder: Vision-only conversion must not require them.
            (models / "vision-encoder.onnx").touch()
            if existing:
                (root / "models/om-models").mkdir()
                (root / "models/om-models/vision-encoder-trt.om").touch()
            fake_bin = root / "fake-bin"
            fake_bin.mkdir()
            docker = fake_bin / "docker"
            write_lf(docker, '#!/bin/bash\nprintf "FAKE_DOCKER_ARG=%s\\n" "$@"\n')
            docker.chmod(0o755)
            env = dict(os.environ)
            for key in ("ONLY_MODEL", "VISION_OUTPUT_NAME", "FORCE", "CANN_IMAGE", "SOC_VERSION"):
                env.pop(key, None)
            env.update(options)
            return subprocess.run(
                [self.bash, "--noprofile", "--norc", "-c",
                 'export PATH="$PWD/fake-bin:$PATH"; exec bash ./scripts/convert_models.sh'],
                cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=20)

    def test_only_vision_new_filename(self):
        result = self.run_conversion({"ONLY_MODEL": "vision-encoder", "VISION_OUTPUT_NAME": "vision-encoder-trt"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("FAKE_DOCKER_ARG=atc"), 1)
        self.assertIn("FAKE_DOCKER_ARG=--model=/app/models/onnx-models/vision-encoder.onnx", result.stdout)
        self.assertIn("FAKE_DOCKER_ARG=--output=/app/models/om-models/vision-encoder-trt", result.stdout)
        self.assertIn("FAKE_DOCKER_ARG=--insert_op_conf=/app/models/config/vision.cfg", result.stdout)

    def test_existing_output_not_overwritten(self):
        result = self.run_conversion({"ONLY_MODEL": "vision-encoder", "VISION_OUTPUT_NAME": "vision-encoder-trt"}, existing=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("FAKE_DOCKER_ARG=atc", result.stdout)

    def test_force_rebuilds_only_vision(self):
        result = self.run_conversion({"ONLY_MODEL": "vision-encoder", "VISION_OUTPUT_NAME": "vision-encoder-trt", "FORCE": "1"}, existing=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("FAKE_DOCKER_ARG=atc"), 1)

    def test_invalid_selection_and_path_rejected(self):
        for options in ({"ONLY_MODEL": "wrong"},
                        {"ONLY_MODEL": "vision-encoder", "VISION_OUTPUT_NAME": "../outside"},
                        {"ONLY_MODEL": "vision-encoder", "VISION_OUTPUT_NAME": "vision-encoder-trt.om"}):
            result = self.run_conversion(options)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("FAKE_DOCKER_ARG=atc", result.stdout)


if __name__ == "__main__":
    unittest.main()
