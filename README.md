# Exam Tutor

Local RAG study partner built for my friend Nishant. Paste URLs or upload notes, then ask questions or get quizzed. Everything runs on-device with open-weight models via Ollama. No API key, no internet needed after setup, notes never leave the laptop.

## Setup

```bash
# 1. install Ollama: https://ollama.com
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# 2. run
pip install -r requirements.txt
streamlit run app.py
```

## Use

1. Sidebar: paste URLs (web pages or PDFs) or upload files, click **Ingest**.
2. **Ask** mode: question gets answered only from the ingested notes, with sources shown.
3. **Quiz me** mode: enter a topic, get an exam-style question, answer it, get graded.

Sample sources to try (data structures):

```
https://en.wikipedia.org/wiki/Binary_search_tree
https://en.wikipedia.org/wiki/AVL_tree
https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
```

## How it works

- Chunk text (900 chars, 150 overlap)
- Embed chunks with `nomic-embed-text` (Ollama)
- Cosine similarity over numpy matrix, top 4 chunks
- `qwen2.5:3b` answers using only retrieved chunks
- Swap model: edit `CHAT_MODEL` in `app.py`

## Stack

Ollama, Qwen2.5, nomic-embed-text, Streamlit, trafilatura, pypdf, numpy.
