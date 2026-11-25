# SpaceNova Research AI MVP

Python 3.9 + Flask + Bootstrap research assistant with:
- Ollama local inference or OpenAI-compatible cloud inference.
- PDF extraction with PyMuPDF.
- RAG using Ollama embeddings (`nomic-embed-text`) and NumPy cosine similarity.
- Source filename/page references.
- Candidate fine-tuning dataset generation to JSONL.

## Windows setup

```powershell
py -3.9 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
ollama pull llama3.2
ollama pull nomic-embed-text
python run.py
```

Open http://127.0.0.1:5000

Ollama must be running. If needed, run `ollama serve` in another terminal.

## Cloud inference
Set `INFERENCE_PROVIDER=openai_compatible`, `CLOUD_BASE_URL`, `CLOUD_API_KEY`,
and `CLOUD_MODEL` in `.env`. The endpoint must implement `/chat/completions`.
Embeddings remain local via Ollama.

## Workflow
1. Upload a few PDFs in Knowledge Base.
2. Click Index PDFs. This extracts page text, chunks it and embeds each chunk.
3. Ask questions in Research Chat. Retrieved passages are supplied to the selected LLM.
4. Generate candidate QA/instruction examples in Dataset Factory and download JSONL.

## Notes
- Start with 1–3 PDFs; indexing 500 can take a long time.
- MVP indexing/generation runs synchronously in web requests. A production version
  needs a persistent background queue and resumable worker.
- NumPy vector search is used to avoid Windows FAISS installation friction.
- Automated generated examples are candidates, not verified scientific truth.
  Human-review them before fine-tuning.
- Fine-tuning happens separately in Google Colab with Transformers + PEFT/QLoRA.
