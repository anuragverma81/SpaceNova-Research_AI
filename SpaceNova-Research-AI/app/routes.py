import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
from . import core

bp = Blueprint("main", __name__)

@bp.route("/")
def home():
    pdfs = [f for f in os.listdir(core.UPLOAD) if f.lower().endswith(".pdf")]
    return render_template("dashboard.html", pdf_count=len(pdfs),
        chunks=len(core.read_jsonl(core.CHUNKS)), examples=len(core.read_jsonl(core.DATASET)),
        provider=os.getenv("INFERENCE_PROVIDER","ollama"))

@bp.route("/chat")
def chat_page(): return render_template("chat.html")

@bp.route("/knowledge")
def knowledge():
    pdfs = sorted(f for f in os.listdir(core.UPLOAD) if f.lower().endswith(".pdf"))
    return render_template("knowledge.html", pdfs=pdfs)

@bp.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    count = 0
    for f in files:
        if not f.filename: continue
        if not f.filename.lower().endswith(".pdf"):
            flash("Skipped non-PDF: "+f.filename, "warning"); continue
        name = secure_filename(os.path.basename(f.filename)) or "document.pdf"
        target = os.path.join(core.UPLOAD, name)
        if os.path.exists(target):
            base, ext = os.path.splitext(name); i=2
            while os.path.exists(os.path.join(core.UPLOAD, base+"_"+str(i)+ext)): i+=1
            name = base+"_"+str(i)+ext; target=os.path.join(core.UPLOAD,name)
        f.save(target); count += 1
    flash("Uploaded "+str(count)+" PDF(s).", "success")
    return redirect(url_for("main.knowledge"))

@bp.route("/index", methods=["POST"])
def index():
    try:
        result=core.index_pdfs()
        flash("Indexed "+str(result["chunks_created"])+" new chunks from "+str(result["files"])+" PDFs.", "success")
    except Exception as e: flash("Indexing failed: "+str(e), "danger")
    return redirect(url_for("main.knowledge"))

@bp.route("/api/chat", methods=["POST"])
def api_chat():
    data=request.get_json(force=True)
    question=str(data.get("message","")).strip()
    if not question: return jsonify({"error":"Enter a question."}),400
    try:
        found=core.retrieve(question)
        context="\n\n---\n\n".join("[Source: {0}, page {1}]\n{2}".format(x["source_file"],x["page"],x["text"]) for x in found)
        history=[x for x in data.get("history",[])[-10:] if x.get("role") in ("user","assistant")]
        history.append({"role":"user","content":question})
        answer=core.chat(history,context)
        sources=[{"file":x["source_file"],"page":x["page"],"score":round(x["score"],3)} for x in found]
        return jsonify({"answer":answer,"sources":sources})
    except Exception as e: return jsonify({"error":str(e)}),500

@bp.route("/dataset")
def dataset():
    rows=core.read_jsonl(core.DATASET)
    return render_template("dataset.html", rows=rows[-100:][::-1], total=len(rows))

@bp.route("/dataset/generate", methods=["POST"])
def generate():
    try:
        rows=core.make_dataset(request.form.get("chunks",5))
        flash("Generated "+str(len(rows))+" candidate examples.", "success")
    except Exception as e: flash("Generation failed: "+str(e), "danger")
    return redirect(url_for("main.dataset"))

@bp.route("/dataset/download")
def download():
    if not os.path.exists(core.DATASET): return "No dataset generated yet.",404
    return send_file(core.DATASET,as_attachment=True,download_name="spacenova_dataset.jsonl")

@bp.route("/health")
def health():
    return jsonify({"status":"ok","provider":os.getenv("INFERENCE_PROVIDER","ollama")})
