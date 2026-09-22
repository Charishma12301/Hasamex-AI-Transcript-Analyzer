import re

from src.load_data import load_transcripts


def extract_metadata(text, filename):
    if "France" in filename:
        country = "France"
        expert = "Dr. Jean Martin"
        role = "Head of Urology"

    elif "Germany" in filename:
        country = "Germany"
        expert = "Anna Keller"
        role = "Former Hospital Procurement Director"

    else:
        country = "UK"
        expert = "Dr. Emily Carter"
        role = "Consultant Urologist"

    return {
        "country": country,
        "expert": expert,
        "role": role
    }


def create_chunks(text, filename):
    metadata = extract_metadata(text, filename)

    parts = re.split(
        r"(?=(?:^|\n)\d{2}:\d{2})",
        text.strip()
    )

    chunks = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        timestamp_match = re.match(
            r"(\d{2}:\d{2})",
            part
        )

        if not timestamp_match:
            continue

        start_time = timestamp_match.group(1)

        content = re.sub(
            r"^\d{2}:\d{2}\s*",
            "",
            part
        ).strip()

        # Keep only expert responses.
        if content.startswith("Interviewer:"):
            continue

        chunks.append({
            **metadata,
            "filename": filename,
            "start_time": start_time,
            "text": content
        })

    return chunks


def build_all_chunks():
    all_chunks = []

    for transcript in load_transcripts():
        chunks = create_chunks(
            transcript["text"],
            transcript["filename"]
        )

        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    chunks = build_all_chunks()

    print(f"Total expert chunks: {len(chunks)}")

    for chunk in chunks:
        print(
            f"[{chunk['country']}] "
            f"{chunk['start_time']} - "
            f"{chunk['text'][:80]}..."
        )