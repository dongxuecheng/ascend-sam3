"""Offline Vision workflow tests; no Docker daemon, CANN or NPU required."""
import argparse
import contextlib
import ctypes
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_vision as bench
import tune_vision as tune
import vision_workflow as workflow


class WorkflowTest(unittest.TestCase):
    def test_env_literal_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text('ASCEND_PHYSICAL_DEVICE_ID=2\nCANN_IMAGE="repo:tag" # comment\nEMPTY=\nLITERAL="$(do-not-run)"\n', encoding="utf-8")
            with patch.dict(os.environ, {"ASCEND_PHYSICAL_DEVICE_ID": "3"}, clear=True):
                config = workflow.settings(root)
            self.assertEqual(config["ASCEND_PHYSICAL_DEVICE_ID"], "3")
            self.assertEqual(config["CANN_IMAGE"], "repo:tag")
            self.assertEqual(config["EMPTY"], "")
            self.assertEqual(config["LITERAL"], "$(do-not-run)")

    def test_paths_reject_escape_accept_spaces_and_container_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(workflow.workspace_path("/app/models/a.om", root), root / "models/a.om")
            self.assertEqual(workflow.container_path(root / "space dir/a.om", root), "/app/space dir/a.om")
            for value in ("../escape", "bad\nname"):
                with self.assertRaises(ValueError):
                    workflow.workspace_path(value, root)

    def test_idle_guard_targets_device_not_other_devices(self):
        containers = [{"Name": "/llm", "HostConfig": {"Devices": [{"PathOnHost": "/dev/davinci0"}]}},
                      {"Name": "/sam3", "HostConfig": {"Devices": [{"PathOnHost": "/dev/davinci2"}]}}]
        with patch.object(workflow.subprocess, "check_output", side_effect=["one two", json.dumps(containers)]):
            workflow.ensure_idle(3)
        with patch.object(workflow.subprocess, "check_output", side_effect=["one two", json.dumps(containers)]):
            with self.assertRaisesRegex(RuntimeError, "sam3"):
                workflow.ensure_idle(2)

    def test_privileged_container_cannot_prove_idle(self):
        with patch.object(workflow.subprocess, "check_output", side_effect=["id", json.dumps([
                {"Name": "/privileged", "HostConfig": {"Privileged": True}}])]):
            with self.assertRaises(RuntimeError):
                workflow.ensure_idle(2)

    def test_privileged_peer_confirmation_never_bypasses_same_device_mapping(self):
        peer = [{"Name": "/llm", "HostConfig": {"Privileged": True}}]
        with patch.object(workflow.subprocess, "check_output", side_effect=["id", json.dumps(peer)]), \
             contextlib.redirect_stdout(io.StringIO()):
            workflow.ensure_idle(2, allow_privileged=True)
        peer[0]["HostConfig"]["Devices"] = [{"PathOnHost": "/dev/davinci2"}]
        with patch.object(workflow.subprocess, "check_output", side_effect=["id", json.dumps(peer)]):
            with self.assertRaises(RuntimeError):
                workflow.ensure_idle(2, allow_privileged=True)


class BenchmarkTest(unittest.TestCase):
    def test_manifest_full_image_and_actual_integer_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "test-images"
            images.mkdir()
            (images / "person space.jpg").touch()
            (images / "not-an-image.txt").touch()
            crop = root / "crops.json"
            crop.write_text(json.dumps([{"image": "test-images/person space.jpg", "roi": [952, 297, 314, 314]}]))
            cases = bench.discover_cases(images, crop, root=root)
            self.assertEqual(len(cases), 2)
            self.assertIsNone(cases[0]["roi"])
            self.assertEqual(cases[1]["roi"], [952, 297, 314, 314])
            for invalid in ([0, 0, 0, 10], [False, 0, 10, 10], [0, 0, 10.5, 10], [0, 0, 2147483648, 10]):
                crop.write_text(json.dumps([{"image": "test-images/person space.jpg", "roi": invalid}]))
                with self.assertRaises(ValueError):
                    bench.discover_cases(images, crop, root=root)

    def test_summary_uses_raw_samples_not_means_of_quantiles(self):
        full = {"roi": None, "samples": [[1, 2, 100, 103], [2, 2, 200, 204]]}
        crop = {"roi": [0, 0, 20, 20], "samples": [[3, 2, 300, 305]]}
        data = bench.aggregate([{"cases": [full, crop]}])
        self.assertEqual(data["all"]["inference_ms"]["mean"], 200)
        self.assertEqual(data["all"]["inference_ms"]["p95"], 290)
        self.assertEqual(data["full"]["inference_ms"]["samples"], 2)
        candidate = bench.aggregate([{"cases": [{"roi": None, "samples": [[1, 2, 80, 83], [2, 2, 160, 164]]},
                                                {"roi": [0, 0, 20, 20], "samples": [[3, 2, 240, 245]]}]}])
        comparison = bench.speed_comparison(data, candidate)
        self.assertAlmostEqual(comparison["all"]["inference_ms"]["latency_reduction_percent"], 20)
        with self.assertRaises(ValueError):
            bench.summary([float("nan")])

    def args(self):
        return argparse.Namespace(baseline="models/base.om", candidate="models/candidate.om", images="test-images",
            crops=None, max_images=None, device=2, image="fake", warmup=1, iterations=2, repeats=2,
            atol=.001, rtol=.001, output_dir="benchmark-results", keep_features=False, profile=False,
            local=True, binary="fake-bench")

    def run_fake(self, passed=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models").mkdir()
            for name in ("base.om", "candidate.om"):
                (root / "models" / name).write_bytes(name.encode())
            (root / "fake-bench").touch()
            (root / "test-images").mkdir()
            (root / "test-images/one.jpg").write_bytes(b"image")
            calls = []
            def fake_run(command, log, accepted):
                calls.append(command)
                output = Path(command[command.index("--json-output")+1])
                label = output.stem.split("-")[0]
                compare = "--reference-dir" in command
                has_dump = "--features-dir" in command
                features = [{"index": i, "passed": passed if compare else True} for i in range(3)]
                data = {"schema_version": 1, "soc": "Ascend310P3", "comparison_enabled": compare,
                        "features_passed": passed if compare else True, "cases": [{"image": str(root / "test-images/one.jpg"),
                            "roi": None, "samples": [[1, 2, 100 if label == "baseline" else 80, 103]]*2,
                            "features": features}]}
                workflow.write_json(output, data)
                if has_dump:
                    Path(command[command.index("--features-dir")+1], "0-fpn0.f32").write_bytes(b"reference")
                if compare:
                    self.assertTrue(Path(command[command.index("--reference-dir")+1], "0-fpn0.f32").exists())
                return 0 if not compare or passed else 2
            with patch.object(bench, "workspace_path", side_effect=lambda p, *_: workflow.workspace_path(p, root)), \
                 patch.object(bench, "logged_run", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                status = bench.run(self.args())
            reports = list((root / "benchmark-results").glob("vision-*/summary.json"))
            result = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual([Path(c[c.index("--json-output")+1]).stem for c in calls],
                             ["baseline-1", "candidate-1", "candidate-2", "baseline-2"])
            self.assertEqual(sum("--reference-dir" in c for c in calls), 1)
            self.assertEqual(sum("--features-dir" in c for c in calls), 1)
            self.assertFalse(list(reports[0].parent.glob("feature-reference-*")))
            self.assertAlmostEqual(result["speed_comparison"]["all"]["inference_ms"]["speedup"], 1.25)
            self.assertEqual(result["feature_comparison"]["passed"], passed)
            return status

    def test_process_order_and_feature_comparison(self):
        self.assertEqual(self.run_fake(), 0)

    def test_failed_comparison_retains_report_returns_two(self):
        self.assertEqual(self.run_fake(False), 2)


class TuningTest(unittest.TestCase):
    def test_preflight_verifies_chip_persists_bank_then_executes(self):
        acl = MagicMock()
        for name in ("aclInit", "aclFinalize", "aclrtSetDevice", "aclrtResetDevice"):
            getattr(acl, name).return_value = 0
        def count(pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint32)).contents.value = 1
            return 0
        acl.aclrtGetDeviceCount.side_effect = count
        acl.aclrtGetSocName.return_value = b"Ascend310P3"
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            argv = ["preflight", "Ascend310P3", str(folder), str(folder / "bank"), "aoe", "--device=0"]
            help_output = argparse.Namespace(stdout="--device --job_type --output --insert_op_conf --input_shape", stderr="")
            with patch.object(sys, "argv", argv), patch.object(ctypes, "CDLL", return_value=acl), \
                 patch.object(tune.subprocess, "run", return_value=help_output), \
                 patch.object(os, "chdir"), patch.object(os, "execvp") as execute, \
                 patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "2"}), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(tune.PREFLIGHT, "preflight", "exec"), {})
                self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", os.environ)
                self.assertEqual(os.environ["TUNE_BANK_PATH"], str(folder / "bank"))
            self.assertTrue((folder / "bank").is_dir())
            self.assertEqual(json.loads((folder / "preflight.json").read_text())["soc"], "Ascend310P3")
            execute.assert_called_once_with("aoe", ["aoe", "--device=0"])
            acl.aclrtResetDevice.assert_called_once_with(0)
            acl.aclFinalize.assert_called_once()

    def test_preflight_chip_mismatch_does_not_execute_aoe(self):
        acl = MagicMock()
        for name in ("aclInit", "aclFinalize", "aclrtSetDevice", "aclrtResetDevice"):
            getattr(acl, name).return_value = 0
        def count(pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint32)).contents.value = 1
            return 0
        acl.aclrtGetDeviceCount.side_effect = count
        acl.aclrtGetSocName.return_value = b"Ascend310P1"
        with patch.object(sys, "argv", ["preflight", "Ascend310P3", "/unused", "/unused/bank", "aoe"]), \
             patch.object(ctypes, "CDLL", return_value=acl), patch.object(os, "execvp") as execute, \
             patch.dict(os.environ):
            with self.assertRaisesRegex(RuntimeError, "Chip mismatch"):
                exec(compile(tune.PREFLIGHT, "preflight", "exec"), {})
        execute.assert_not_called()
        acl.aclFinalize.assert_called_once()

    def test_complete_mock_tuning_publishes_candidate_not_active_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models/om-models").mkdir(parents=True)
            (root / "models/vision.onnx").write_bytes(b"onnx")
            (root / "models/vision.cfg").write_bytes(b"current-aipp")
            active = root / "models/om-models/active.om"
            active.write_bytes(b"deployed-model")
            args = argparse.Namespace(onnx="models/vision.onnx", aipp="models/vision.cfg", device=2, image="cann",
                output_name="candidate", active_model="models/om-models/active.om", force=False,
                output_dir="benchmark-results", soc="Ascend310P3")
            def command(_args, folder, _onnx, _aipp):
                return [str(folder)]
            def fake_run(command, log):
                folder = Path(command[0])
                (folder / "candidate.om").write_bytes(b"tuned-model")
                workflow.write_json(folder / "preflight.json", {"soc": "Ascend310P3"})
                (folder / "bank").mkdir()
                (folder / "bank/op-bank.txt").write_text("bank")
                log.write_text("AOE completed")
                return 0
            with patch.object(tune, "workspace_path", side_effect=lambda p: workflow.workspace_path(p, root)), \
                 patch.object(tune, "ensure_idle"), patch.object(tune, "image_identity", return_value={"id": "image-id"}), \
                 patch.object(tune, "make_command", side_effect=command), \
                 patch.object(tune, "logged_run", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(tune.run(args), 0)
            self.assertEqual(active.read_bytes(), b"deployed-model")
            self.assertEqual((root / "models/om-models/candidate.om").read_bytes(), b"tuned-model")
            info = json.loads((root / "models/om-models/candidate.build-info.json").read_text())
            self.assertEqual(info["status"], "completed_unvalidated")
            self.assertFalse(info["auto_promoted"])
            self.assertEqual(info["bank_files"], ["bank/op-bank.txt"])

    def test_only_vision_device_zero_and_latest_aipp(self):
        args = argparse.Namespace(device=2, image="cann", soc="Ascend310P3", output_name="vision-trt-tuned")
        with patch.object(tune, "docker_args", return_value=["docker", "run"]):
            command = tune.make_command(args, ROOT / "benchmark-results/aoe", ROOT / "models/onnx-models/vision-encoder.onnx",
                                        ROOT / "models/config/vision.cfg")
        aoe = command[command.index("aoe"):]
        self.assertIn("--device=0", aoe)
        self.assertIn("--job_type=2", aoe)
        self.assertIn("--insert_op_conf=/app/models/config/vision.cfg", aoe)
        self.assertFalse(any("--soc_version" in item for item in aoe))
        self.assertFalse(any("decoder" in item for item in aoe))
        compile(tune.PREFLIGHT, "preflight", "exec")

    def test_missing_or_ambiguous_outputs_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                tune.resolve_output(root, "vision")
            (root / "vision_linux_aarch64.om").write_bytes(b"candidate")
            self.assertEqual(tune.resolve_output(root, "vision").name, "vision_linux_aarch64.om")
            (root / "vision_other.om").write_bytes(b"candidate")
            with self.assertRaises(RuntimeError):
                tune.resolve_output(root, "vision")

    def test_publish_is_opt_in_and_no_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "result.om", root / "candidate.om"
            source.write_bytes(b"candidate")
            target.write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                tune.publish(source, target, False)
            self.assertEqual(target.read_bytes(), b"old")
            tune.publish(source, target, True)
            self.assertEqual(target.read_bytes(), b"candidate")
            self.assertFalse(list(root.glob("*.partial")))

    def test_current_model_protected_even_with_force(self):
        args = argparse.Namespace(onnx="models/vision.onnx", aipp="models/config/vision.cfg",
            output_name="active", active_model="models/om-models/active.om", force=True)
        with self.assertRaisesRegex(ValueError, "VISION_MODEL"):
            tune.run(args)

    def test_image_build_includes_native_benchmark(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("--target ascendsam3_py ascendsam3_vision_bench", dockerfile)
        self.assertIn("/app/bin/ascendsam3_vision_bench", dockerfile)


if __name__ == "__main__":
    unittest.main()
