import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ingestion import transcribe_to_data_folder


class FakeWhisperModel:
    def transcribe(self, audio_path, fp16=False):
        return {"text": f"Transcript for {Path(audio_path).name}"}


class IngestionTests(unittest.TestCase):
    def test_transcribe_writes_text_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_file = tmp_path / "meeting.wav"
            audio_file.write_bytes(b"fake-audio")
            output_dir = tmp_path / "data"

            with patch("whisper.load_model", return_value=FakeWhisperModel()), patch("src.ingestion.shutil.which", return_value="ffmpeg"):
                output_file = transcribe_to_data_folder(str(audio_file), model_size="base", data_dir=str(output_dir))

            self.assertTrue(output_file.exists())
            self.assertEqual(output_file.name, "meeting.txt")
            self.assertIn("Transcript for meeting.wav", output_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
