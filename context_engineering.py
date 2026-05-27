def deduplicate_results(results):
    """
    Remove duplicate or near-identical chunks.
    """

    seen = set()
    unique_results = []

    for item in results:

        text = item["text"].strip()

        if text not in seen:
            seen.add(text)
            unique_results.append(item)

    return unique_results


def limit_context(results, max_chunks=3):
    """
    Restrict how many chunks are sent to the LLM.
    """

    return results[:max_chunks]


def build_context(results):
    """
    Convert retrieved chunks into clean LLM context.
    """

    context_sections = []

    for idx, item in enumerate(results, start=1):

        section = f"""
Context Source {idx}:
{item['text']}
"""

        context_sections.append(section)

    return "\n".join(context_sections)