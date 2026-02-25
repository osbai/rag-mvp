from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.chunker import chunk_text
from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.rag import rag_query
from src.store import add_documents, collection_stats

app = typer.Typer(help="RAG MVP — ingest documents and query them with Claude.")
console = Console()


def _load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            console.print("[red]pypdf is required for PDF files: pip install pypdf[/red]")
            raise typer.Exit(1)
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest (.txt or .pdf)"),
    chunk_size: int = typer.Option(CHUNK_SIZE, help="Max characters per chunk"),
    overlap: int = typer.Option(CHUNK_OVERLAP, help="Overlap between chunks"),
) -> None:
    """Ingest a file (or all .txt/.pdf files in a directory) into the vector store."""
    files: list[Path] = []
    if path.is_dir():
        files = [f for f in path.rglob("*") if f.suffix.lower() in {".txt", ".pdf"}]
    elif path.is_file():
        files = [path]
    else:
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    if not files:
        console.print("[yellow]No .txt or .pdf files found.[/yellow]")
        raise typer.Exit(0)

    total_chunks = 0
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        for file in files:
            task = progress.add_task(f"Ingesting [cyan]{file.name}[/cyan]…")
            text = _load_text(file)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            n = add_documents(chunks, source=file.name)
            total_chunks += n
            progress.update(task, completed=True, description=f"[green]✓[/green] {file.name} — {n} chunks")

    stats = collection_stats()
    console.print(
        f"\n[bold green]Done.[/bold green] Added {total_chunks} chunks from {len(files)} file(s). "
        f"Collection now has [bold]{stats['count']}[/bold] total chunks."
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, help="Number of chunks to retrieve"),
) -> None:
    """Ask a question — retrieves relevant context and streams Claude's answer."""
    console.print(f"\n[bold cyan]Q:[/bold cyan] {question}\n")
    console.print("[bold cyan]A:[/bold cyan] ", end="")
    for token in rag_query(question, top_k=top_k):
        console.print(token, end="", highlight=False)
    console.print("\n")


@app.command()
def stats() -> None:
    """Show vector store statistics."""
    info = collection_stats()
    console.print(f"Collection [bold]{info['name']}[/bold]: {info['count']} chunks")


if __name__ == "__main__":
    app()
