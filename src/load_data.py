from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_transcripts():
    transcripts = []

    for file_path in sorted(DATA_DIR.glob("Transcript_*.txt")):
        text = file_path.read_text(encoding="utf-8")

        transcripts.append({
            "filename": file_path.name,
            "text": text
        })

    return transcripts


def load_interview_guide():
    guide_path = DATA_DIR / "Interview_Guide.txt"
    return guide_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    transcripts = load_transcripts()
    guide = load_interview_guide()

    print(f"Transcripts loaded: {len(transcripts)}")
    print(f"Interview guide loaded: {len(guide)} characters")

    for transcript in transcripts:
        print(
            f"- {transcript['filename']}: "
            f"{len(transcript['text'])} characters"
        )