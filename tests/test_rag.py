import tempfile
import unittest
from pathlib import Path

from src.rag import supported_data_files


class RagTests(unittest.TestCase):
    def test_supported_data_files_filters_expected_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "a.pdf").write_text("pdf", encoding="utf-8")
            (data_dir / "b.txt").write_text("txt", encoding="utf-8")
            (data_dir / "c.md").write_text("md", encoding="utf-8")

            files = supported_data_files(data_dir)
            names = [f.name for f in files]

            self.assertEqual(names, ["a.pdf", "b.txt"])

    def test_supported_data_files_raises_for_missing_folder(self):
        with self.assertRaises(FileNotFoundError):
            supported_data_files(Path("folder-that-does-not-exist"))


if __name__ == "__main__":
    unittest.main()
