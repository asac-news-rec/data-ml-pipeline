"""serving/openapi.json이 코드와 같은지 확인한다."""
import json
import unittest
from pathlib import Path

try:
    from serving.app import app
except ImportError as exc:
    raise unittest.SkipTest(f"serving 묶음이 없다: {exc}")

SPEC = Path(__file__).resolve().parents[1] / "serving/openapi.json"


class OpenApi(unittest.TestCase):
    def test_spec_matches_code(self):
        self.assertEqual(json.loads(SPEC.read_text(encoding="utf-8")), app.openapi())

    def test_only_was_api_is_documented(self):
        paths = app.openapi()["paths"]
        self.assertEqual(list(paths), ["/v1/recommend"])
        self.assertEqual(set(paths["/v1/recommend"]["post"]["responses"]), {"200", "400", "503"})


if __name__ == "__main__":
    unittest.main()
