def chunk_text(
    text: str,
    chunk_size: int = 300
):

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[i:i + chunk_size]

        chunks.append(chunk)

    return chunks