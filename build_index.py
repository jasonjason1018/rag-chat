import os, glob
import chromadb
from sentence_transformers import SentenceTransformer

DOC_DIR = "./docs"
DB_DIR = "./chroma_db"
COLLECTION = "tw_stock_docs"

# 這個模型負責把文字轉成向量（語意座標）
embed_model = SentenceTransformer("intfloat/multilingual-e5-small")

client = chromadb.PersistentClient(path=DB_DIR)
col = client.get_or_create_collection(COLLECTION)

def chunk_text(text: str, chunk_size=800, overlap=100):
    """把長文切成可用的小段，避免太長不好搜尋/塞prompt"""
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - overlap
    return chunks

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# 掃描 docs 內所有 txt/md
doc_paths = glob.glob(os.path.join(DOC_DIR, "**/*.txt"), recursive=True) + \
            glob.glob(os.path.join(DOC_DIR, "**/*.md"), recursive=True)

ids, docs, metas = [], [], []
for p in doc_paths:
    text = read_text(p)
    for j, ch in enumerate(chunk_text(text)):
        ids.append(f"{os.path.basename(p)}::{j}")
        docs.append(ch)
        metas.append({"source": p, "chunk": j})

# E5 embedding 有個慣例：passage: / query: 前綴，效果會好一些
embeddings = embed_model.encode(
    [f"passage: {d}" for d in docs],
    normalize_embeddings=True
)

col.add(
    ids=ids,
    documents=docs,
    metadatas=metas,
    embeddings=embeddings.tolist()
)

print(f"✅ Indexed {len(docs)} chunks into {DB_DIR} / {COLLECTION}")