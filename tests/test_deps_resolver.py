import tempfile
from pathlib import Path
import unittest

from src.model_kernel_analyzer.deps_resolver import (
    parse_dep_specs_from_text,
    resolve_or_fallback_specs,
)


class TestDepsResolver(unittest.TestCase):
    def test_parse_versions_from_requirements_text(self):
        text = """
torch==2.3.1
transformers>=4.40.0
"""
        parsed = parse_dep_specs_from_text(text)
        self.assertIn("torch", parsed)
        self.assertEqual(parsed["torch"].pip_spec, "torch==2.3.1")
        # 这里 transformers 没有 ==，会走 >= 的规范
        self.assertIn("transformers", parsed)
        self.assertTrue(parsed["transformers"].pip_spec.startswith("transformers>="))

    def test_priority_requirements_over_readme(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "requirements.txt").write_text(
                "torch==2.0.0\ntransformers==4.45.0\n",
                encoding="utf-8",
            )
            (root / "README.md").write_text(
                "pip install torch==1.13.1 transformers==4.31.0",
                encoding="utf-8",
            )
            resolved = resolve_or_fallback_specs(root)
            self.assertEqual(resolved["torch"].pip_spec, "torch==2.0.0")
            self.assertEqual(
                resolved["transformers"].pip_spec, "transformers==4.45.0"
            )


if __name__ == "__main__":
    unittest.main()

