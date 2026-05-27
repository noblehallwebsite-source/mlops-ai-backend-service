def chunk_text(
    text: str,
    chunk_size: int = 2,
    overlap: int = 1
):

    # Split into sentences
    sentences = text.split(".")

    # Remove empty entries
    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    chunks = []

    step = chunk_size - overlap

    for i in range(
        0,
        len(sentences),
        step
    ):

        chunk_sentences = sentences[
            i:i + chunk_size
        ]

        chunk = ". ".join(
            chunk_sentences
        )

        if chunk:
            chunks.append(chunk)

    return chunks