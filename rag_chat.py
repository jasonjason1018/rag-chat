import chromadb
from sentence_transformers import SentenceTransformer
import ollama

DB_DIR = "./chroma_db"
COLLECTION = "tw_stock_docs"

embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
client = chromadb.PersistentClient(path=DB_DIR)
col = client.get_collection(COLLECTION)

def retrieve(query: str, k=4):
    q_emb = embed_model.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()
    res = col.query(query_embeddings=[q_emb], n_results=k)

    chunks = []
    for doc, meta, _id in zip(res["documents"][0], res["metadatas"][0], res["ids"][0]):
        chunks.append({"id": _id, "text": doc, "source": meta.get("source")})
    return chunks

def ask_llm(query: str):
    chunks = retrieve(query, k=4)

    context = "\n\n".join(
        [f"[{i+1}] {c['source']}\n{c['text']}" for i, c in enumerate(chunks)]
    )

    system = (
        """
        你是 AI 助理，必須根據提供的資料回答。

        規則：
        1. 優先使用提供的資料。
        2. 若資料中有明確答案，請直接回答並標註來源。
        3. 若資料完全沒有相關內容，才回覆「資料不足」。
        4. 若資料有部分線索，請根據資料合理整理答案（不可亂編）。
        5. 回答簡潔，句尾標註來源，例如：[1]。
        """
    )

    user = f"問題：{query}\n\n資料：\n{context}"

    resp = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.2}
    )
    print("\n" + resp["message"]["content"])

if __name__ == "__main__":
    while True:
        q = input("\n> ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        ask_llm(q)
