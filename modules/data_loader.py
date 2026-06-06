from pathlib import Path

def load_sample_documents(data_dir: str):
    docs = []
    for i, path in enumerate(sorted(Path(data_dir).glob('*.txt')), start=1):
        docs.append({"doc_id": f"D{i}", "title": path.stem, "text": path.read_text(encoding='utf-8')})
    return docs

def load_documents_from_uploads(uploaded_files):
    docs = []
    if not uploaded_files:
        return docs
    for i, file in enumerate(uploaded_files, start=1):
        docs.append({"doc_id": f"U{i}", "title": file.name.rsplit('.', 1)[0], "text": file.read().decode('utf-8', errors='ignore')})
    return docs