# rag-chat

本專案是一個使用 **Python + Ollama + ChromaDB** 建立的本地端 **Retrieval-Augmented Generation (RAG)** 文件問答系統。

系統會先將文件切分為多個片段並轉換為向量，存入向量資料庫；當使用者提出問題時，系統會先從向量資料庫中檢索相關內容，再將檢索到的資料與 Prompt 一起送入本地 LLM 生成回答，以提高回答的正確性並降低幻覺（Hallucination）。

---

# 系統架構

整體流程如下：

``` 
文件 (docs)
│
▼
Chunk 切分
│
▼
Embedding (multilingual-e5-small)
│
▼
Chroma 向量資料庫
│
▼
Query embedding
│
▼
Top-K 相似度檢索
│
▼
Prompt + 檢索內容
│
▼
Ollama LLM (qwen2.5:3b)
│
▼
生成回答
```

---

# 專案特色

- 文件 **Chunk 切分 (chunk + overlap)**
- 使用 **SentenceTransformers** 生成 embedding
- 使用 **ChromaDB** 儲存向量資料
- 支援 **Top-K 相似度檢索**
- 使用 **Ollama 本地模型** 進行回答
- Prompt 設計限制模型 **依據資料回答**

---

# 技術棧

- Python
- Ollama
- ChromaDB
- SentenceTransformers
- LLM：`qwen2.5:3b`
- Embedding Model：`intfloat/multilingual-e5-small`

---

# 專案結構

``` 
rag-chat
│
├─ docs/ # 文件資料來源
│
├─ chroma_db/ # Chroma 向量資料庫
│
├─ build_index.py # 建立向量索引
│
├─ rag_chat.py # RAG 問答程式
│
└─ README.md
```

---

# 安裝方式

建議使用 Python 3.10+

## 1. 安裝 Python 套件

```bash
pip install chromadb sentence-transformers ollama
```
## 2. 安裝 Ollama

下載並安裝：

https://ollama.com/

拉取模型：
```bash
ollama pull qwen2.5:3b
```

# 使用方式
## 1. 準備文件

將 .txt 或 .md 文件放入：
```
docs/
```

## 2. 建立向量資料庫

執行：

python build_index.py

程式會：

讀取 docs 內的文件

將文件切分為多個 chunk

使用 embedding model 轉換為向量

存入 ChromaDB

生成：

chroma_db/

## 3. 啟動 RAG 問答

執行：

python rag_chat.py

輸入問題，例如：

問題: 什麼是 RAG？

系統流程：

將問題轉為 embedding

從向量資料庫檢索相關 chunk

組合 Prompt + 檢索內容

使用 Ollama 生成回答

---

## 回答策略

系統 Prompt 設計包含以下規則：

必須根據提供的文件內容回答

回答時引用資料段落

若資料不足則回覆：

資料不足

此策略可降低 LLM 幻覺問題。

## Chunk 策略

目前使用簡單的文字切分方式：

chunk_size = 800

overlap = 100

Overlap 可避免語意被切斷，提高檢索品質。

## 未來優化方向

Web UI

Hybrid Search

Reranker

Token-based chunking

多輪對話記憶

## 專案目的

本專案主要用於學習與實作 RAG 系統的基本架構，包含：

文件切分

向量化

向量資料庫

檢索

LLM 生成

並展示如何在 本地環境使用 LLM 建立文件問答系統。