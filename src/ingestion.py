import argparse
import shutil
from pathlib import Path


def transcribe_to_data_folder(audio_path: str, model_size: str = "base", data_dir: str = "data") -> Path:
    """Transcribe an audio file and store the transcript under the data directory."""
    import whisper

    source = Path(audio_path)
    if not source.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")

    destination_dir = Path(data_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / f"{source.stem}.txt"

    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required by Whisper but was not found on PATH. "
            "Install it with `winget install Gyan.FFmpeg` and restart your terminal."
        )

    print(f"Transcribing: {source}")
    result = model.transcribe(str(source), fp16=False)

    output_path.write_text(result["text"], encoding="utf-8")
    print(f"Transcript saved to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe an audio file into the data folder.")
    parser.add_argument("audio_path", help="Path to the input audio file.")
    parser.add_argument("--model-size", default="base", help="Whisper model: tiny, base, small, medium, large.")
    parser.add_argument("--data-dir", default="data", help="Directory where transcript .txt files are stored.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcribe_to_data_folder(args.audio_path, model_size=args.model_size, data_dir=args.data_dir)


if __name__ == "__main__":
    main()