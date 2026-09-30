"""Local Gradio interface for the WHO RAG assistant."""

from openai import APIError
from requests import RequestException

from src.config import INDEX_PATH, TOP_K
from src.rag_pipeline import answer_question

MAX_QUESTION_LENGTH = 1_000


def _format_sources(sources) -> str:
    if not sources:
        return "_No sources cited._"

    entries = []
    for source in sources:
        title = source.get("title") or "Untitled"
        url = source.get("source") or ""
        page = source.get("page_number")
        page_text = f" — PDF page {page}" if page is not None else ""
        heading = f"**[{source['id']}] {title}{page_text}**"
        entries.append(f"{heading}  \n{url}" if url else heading)
    return "\n\n".join(entries)


def ask(question: str) -> tuple[str, str]:
    """Return an answer and formatted citations for one user question."""
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""
    if len(question) > MAX_QUESTION_LENGTH:
        return f"Please limit the question to {MAX_QUESTION_LENGTH} characters.", ""

    try:
        result = answer_question(question, index_path=INDEX_PATH, top_k=TOP_K)
    except FileNotFoundError:
        return "The local index is missing. Run `python main.py build-index` first.", ""
    except APIError:
        return (
            "Gemini could not complete the request. Check the API key and try again.",
            "",
        )
    except RequestException:
        return "A network request failed. Check the connection and try again.", ""
    except (OSError, ValueError) as exc:
        return f"Unable to answer the question: {exc}", ""

    return result["answer"], _format_sources(result["sources"])


def build_demo():
    """Create the browser interface without starting its server."""
    import gradio as gr

    with gr.Blocks(title="WHO Clinical RAG Assistant") as demo:
        gr.Markdown(
            """
            # WHO Clinical RAG Assistant

            Ask a question about the indexed WHO publications and fact sheets.

            *For research and educational use. This tool does not provide medical advice.*
            """
        )
        question = gr.Textbox(
            label="Question",
            placeholder="How many new leprosy cases were reported globally in 2025?",
            lines=2,
            max_lines=5,
        )
        submit = gr.Button("Ask", variant="primary")
        gr.Markdown("## Answer")
        answer = gr.Markdown()
        gr.Markdown("## Sources")
        sources = gr.Markdown()

        submit.click(ask, inputs=question, outputs=[answer, sources])
        question.submit(ask, inputs=question, outputs=[answer, sources])

    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1")
