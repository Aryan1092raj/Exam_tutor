"""Exam Tutor: local RAG study partner. Ollama + numpy + Streamlit. No cloud, no API key."""
import io
import json
from pathlib import Path

import numpy as np
import ollama
import requests
import streamlit as st
import trafilatura
from pypdf import PdfReader

CHAT_MODEL = "qwen2.5:3b"        # swap to qwen2.5:7b / mistral if laptop can handle it
EMBED_MODEL = "nomic-embed-text"
STORE = Path("store.json")
TOP_K = 4

SYSTEM = (
    "You are a patient exam tutor. Answer ONLY from the provided notes. "
    "If notes don't cover it, say so. Be concise, use small examples."
)


# ---------- storage ----------
def load_store():
    return json.loads(STORE.read_text()) if STORE.exists() else []


def save_store(items):
    STORE.write_text(json.dumps(items))


# ---------- ingest ----------
def chunk(text, size=900, overlap=150):
    text = " ".join(text.split())
    return [text[i:i + size] for i in range(0, len(text), size - overlap) if text[i:i + size].strip()]


def read_url(url):
    if url.lower().endswith(".pdf"):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(r.content)).pages)
    html = trafilatura.fetch_url(url)
    return trafilatura.extract(html) or ""


def read_upload(f):
    if f.name.lower().endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(f).pages)
    return f.read().decode("utf-8", errors="ignore")


def add_source(name, text):
    chunks = chunk(text)
    if not chunks:
        return 0
    embs = []
    for i in range(0, len(chunks), 32):
        embs += ollama.embed(model=EMBED_MODEL, input=chunks[i:i + 32])["embeddings"]
    items = [i for i in load_store() if i["src"] != name]
    items += [{"src": name, "text": c, "emb": e} for c, e in zip(chunks, embs)]
    save_store(items)
    return len(chunks)


# ---------- retrieve + generate ----------
def retrieve(query, k=TOP_K):
    items = load_store()
    if not items:
        return []
    q = np.array(ollama.embed(model=EMBED_MODEL, input=query)["embeddings"][0])
    M = np.array([i["emb"] for i in items])
    sims = M @ q / (np.linalg.norm(M, axis=1) * np.linalg.norm(q) + 1e-9)
    return [items[i] for i in np.argsort(-sims)[:k]]


def ctx_block(hits):
    return "\n\n".join(f"[{h['src']}]\n{h['text']}" for h in hits)


def stream(prompt):
    for part in ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        stream=True,
    ):
        yield part["message"]["content"]


# ---------- UI ----------
st.set_page_config(page_title="Exam Tutor", page_icon="📚")
st.title("📚 Exam Tutor")
st.caption("Runs 100% on your laptop. Notes never leave it.")

with st.sidebar:
    st.header("Add notes")
    urls = st.text_area("URLs (one per line, pages or PDFs)")
    files = st.file_uploader("...or files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Ingest"):
        with st.spinner("Reading + embedding..."):
            for u in filter(None, (x.strip() for x in urls.splitlines())):
                try:
                    st.write(f"{u}: {add_source(u, read_url(u))} chunks")
                except Exception as e:
                    st.error(f"{u}: {e}")
            for f in files:
                st.write(f"{f.name}: {add_source(f.name, read_upload(f))} chunks")
    srcs = sorted({i['src'] for i in load_store()})
    st.write(f"**{len(srcs)} sources loaded**")
    for s in srcs:
        st.caption(s)

mode = st.radio("Mode", ["Ask", "Quiz me"], horizontal=True)

if mode == "Ask":
    q = st.chat_input("Ask about your notes...")
    if q:
        st.chat_message("user").write(q)
        hits = retrieve(q)
        if not hits:
            st.warning("No notes yet. Add some in sidebar.")
        else:
            prompt = f"Notes:\n{ctx_block(hits)}\n\nQuestion: {q}"
            st.chat_message("assistant").write_stream(stream(prompt))
            with st.expander("Sources used"):
                for h in hits:
                    st.caption(h["src"])
                    st.text(h["text"][:300] + "...")
else:
    topic = st.text_input("Topic to be quizzed on")
    if st.button("New question") and topic:
        hits = retrieve(topic)
        if not hits:
            st.warning("No notes yet.")
        else:
            st.session_state.hits = hits
            p = f"Notes:\n{ctx_block(hits)}\n\nWrite ONE exam-style question on '{topic}'. Question only, no answer."
            st.session_state.question = "".join(stream(p))
    if st.session_state.get("question"):
        st.info(st.session_state.question)
        ans = st.text_area("Your answer")
        if st.button("Grade me") and ans:
            p = (
                f"Notes:\n{ctx_block(st.session_state.hits)}\n\n"
                f"Question: {st.session_state.question}\nStudent answer: {ans}\n\n"
                "Grade out of 10. Say what is right, what is missing, then give the model answer."
            )
            st.write_stream(stream(p))
