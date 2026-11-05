"""Check accelerator detection against what the tools really print."""

import unittest
from unittest import mock

from scribe_eval import hardware


class HardwareTests(unittest.TestCase):
    def test_rocm_lists_gpus_and_skips_the_cpu_agent(self):
        output = """
          Marketing Name:          AMD RYZEN AI MAX PRO 390 w/ Radeon 8050S
          Device Type:             CPU
          Marketing Name:          AMD Radeon(TM) 8050S Graphics
          Device Type:             GPU
        """
        with mock.patch.object(hardware, "_command", return_value=output):
            found = hardware._rocm()
        names = [entry["name"] for entry in found]
        self.assertEqual(names, ["AMD Radeon(TM) 8050S Graphics"])

    def test_apple_silicon_reports_gpu_and_neural_engine(self):
        output = '{"SPDisplaysDataType": [{"sppci_model": "Apple M3 Max"}]}'
        with (
            mock.patch.object(hardware.platform, "system", return_value="Darwin"),
            mock.patch.object(hardware.platform, "machine", return_value="arm64"),
            mock.patch.object(hardware, "_command", return_value=output),
        ):
            found = hardware._apple()
        self.assertEqual([entry["type"] for entry in found], ["gpu", "npu"])

    def test_cpu_name_hints_fill_only_what_tools_missed(self):
        cpu = "AMD RYZEN AI MAX PRO 390 w/ Radeon 8050S"
        kinds = [entry["type"] for entry in hardware._cpu_hints(cpu, [])]
        self.assertEqual(kinds, ["gpu", "npu"])
        gpu = [{"type": "gpu", "name": "Radeon", "source": "rocminfo"}]
        kinds = [entry["type"] for entry in hardware._cpu_hints(cpu, gpu)]
        self.assertEqual(kinds, ["npu"])
