import os, re, json, hashlib, requests
import numpy as np
import fitz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD = os.path.abspath(os.getenv("UPLOAD_DIR", os.path.join(ROOT, "storage", "pdfs")))
DATA = os.path.abspath(os.getenv("DATA_DIR", os.path.join(ROOT, "storage", "data")))
CHUNKS = os.path.join(DATA, "chunks.jsonl")
VECTORS = os.path.join(DATA, "vectors.npy")
DATASET = os.path.join(DATA, "dataset.jsonl")
os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

def read_jsonl(path):
    if not os.path.exists(path): return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                if line.strip(): rows.append(json.loads(line))
            except Exception: pass
    return rows

def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

def chunks(text, size=None, overlap=None):
    size = size or int(os.getenv("CHUNK_SIZE", "1400"))
    overlap = overlap or int(os.getenv("CHUNK_OVERLAP", "200"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    result, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        part = text[start:end].strip()
        if part: result.append(part)
        if end == len(text): break
        start = max(start + 1, end - overlap)
    return result

def extract_pdf(path, filename):
    doc = fitz.open(path)
    result = []
    try:
        for pno in range(len(doc)):
            text = doc.load_page(pno).get_text("text", sort=True).strip()
            if len(text) < 40: continue
            for ci, part in enumerate(chunks(text)):
                uid = hashlib.sha256((filename + str(pno+1) + str(ci) + part[:100]).encode()).hexdigest()[:20]
                result.append({"id": uid, "source_file": filename, "page": pno+1, "text": part})
    finally: doc.close()
    return result

def embed(text):
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    r = requests.post(base + "/api/embeddings", json={"model": model, "prompt": text}, timeout=180)
    r.raise_for_status()
    v = r.json().get("embedding")
    if not v: raise RuntimeError("No embedding returned. Run: ollama pull nomic-embed-text")
    return np.asarray(v, dtype=np.float32)

def index_pdfs():
    files = sorted(x for x in os.listdir(UPLOAD) if x.lower().endswith(".pdf"))
    old = read_jsonl(CHUNKS)
    known = set(x["id"] for x in old)
    new_chunks, new_vecs = [], []
    for name in files:
        for chunk in extract_pdf(os.path.join(UPLOAD, name), name):
            if chunk["id"] in known: continue
            new_chunks.append(chunk)
            new_vecs.append(embed(chunk["text"]))
            known.add(chunk["id"])
    if new_chunks:
        matrix = np.vstack(new_vecs)
        if os.path.exists(VECTORS): matrix = np.vstack([np.load(VECTORS), matrix])
        write_jsonl(CHUNKS, old + new_chunks)
        np.save(VECTORS, matrix)
    return {"files": len(files), "chunks_created": len(new_chunks), "total_chunks": len(old)+len(new_chunks)}

def retrieve(query):
    docs = read_jsonl(CHUNKS)
    if not docs or not os.path.exists(VECTORS): return []
    matrix = np.load(VECTORS)
    if len(matrix) != len(docs): raise RuntimeError("Vector/chunk index mismatch. Rebuild index.")
    q = embed(query)
    sims = matrix.dot(q) / ((np.linalg.norm(matrix, axis=1)+1e-12) * (np.linalg.norm(q)+1e-12))
    k = int(os.getenv("TOP_K", "5"))
    ids = np.argsort(sims)[::-1][:k]
    return [dict(docs[int(i)], score=float(sims[int(i)])) for i in ids if sims[int(i)] > 0.05]

def chat(messages, context):
    system = ("You are SpaceNova Research AI, a specialized space science and aerospace engineering assistant. "
              "Use retrieved context when available. Cite factual claims as [Source: filename, page N]. "
              "If the sources do not support an answer, say so. Never invent technical specifications, "
              "equations, mission facts or citations.")
    if context: system += "\n\nRETRIEVED SOURCES:\n" + context
    msgs = [{"role":"system","content":system}] + messages[-12:]
    provider = os.getenv("INFERENCE_PROVIDER", "ollama").lower()
    if provider == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
        r = requests.post(base+"/api/chat", json={"model":model,"messages":msgs,"stream":False,
                          "options":{"temperature":0.2}}, timeout=240)
        r.raise_for_status()
        return r.json()["message"]["content"]
    if provider == "openai_compatible":
        base = os.getenv("CLOUD_BASE_URL", "").rstrip("/")
        key, model = os.getenv("CLOUD_API_KEY", ""), os.getenv("CLOUD_MODEL", "")
        if not base or not key or not model: raise RuntimeError("Configure CLOUD_BASE_URL, CLOUD_API_KEY and CLOUD_MODEL in .env")
        r = requests.post(base+"/chat/completions", headers={"Authorization":"Bearer "+key},
                          json={"model":model,"messages":msgs,"temperature":0.2}, timeout=240)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError("INFERENCE_PROVIDER must be ollama or openai_compatible")

def teacher_json(prompt):
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    r = requests.post(base+"/api/chat", json={"model":model,"stream":False,"format":"json",
        "messages":[{"role":"system","content":"Return valid JSON only. Generate grounded training examples."},
                    {"role":"user","content":prompt}],
        "options":{"temperature":0.2}}, timeout=240)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])

def make_dataset(n_chunks=5):
    docs = read_jsonl(CHUNKS)
    if not docs: raise RuntimeError("Upload and index PDFs first.")
    old = read_jsonl(DATASET)
    seen = set(x["messages"][1]["content"].lower() for x in old if len(x.get("messages",[])) > 1)
    fresh = []
    for d in docs[:max(1, min(int(n_chunks), 30))]:
        prompt = ("Create up to 2 distinct examples grounded ONLY in this passage. Return JSON "
          '{"examples":[{"task_type":"qa or instruction","category":"Space Science, Rocket Engineering, '
          'Satellite Engineering, Orbital Mechanics, Astrophysics, Spacecraft Systems, Mission Analysis, '
          'or Technical Reasoning","question":"...","answer":"..."}]}. No invented facts. '
          "Answers must be useful and supported by passage.\nSOURCE: "+d["source_file"]+
          " page "+str(d["page"])+"\nPASSAGE:\n"+d["text"])
        data = teacher_json(prompt)
        for ex in data.get("examples", []):
            q, a = str(ex.get("question","")).strip(), str(ex.get("answer","")).strip()
            if len(q) < 12 or len(a) < 60 or q.lower() in seen: continue
            seen.add(q.lower())
            fresh.append({"messages":[
                {"role":"system","content":"You are SpaceNova Research AI, a specialized space science and engineering assistant."},
                {"role":"user","content":q},{"role":"assistant","content":a}],
                "metadata":{"task_type":ex.get("task_type","qa"),"category":ex.get("category","Space Engineering"),
                "source_file":d["source_file"],"source_pages":[d["page"]],"chunk_id":d["id"]}})
    write_jsonl(DATASET, old+fresh)
    return fresh
