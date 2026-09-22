"""把 docs/ 目录下的 .md 文件按标题结构切片、向量化，存进 PostgreSQL 库。"""
import os
import re
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

DOCS_DIR = "docs/internal_kb"
CONNECTION = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
COLLECTION_NAME = "internal_docs"

def split_by_markdown_headers(text: str) -> list[str]:
    """按 ## 二级标题切分，每个 chunk 对应文档里的一个独立小节。"""
    # 按照换行后紧跟 ## 的地方进行切分
    sections = re.split(r'\n(?=##\s)', text)
    # 去除首尾空白，并过滤掉空段落
    return [s.strip() for s in sections if s.strip()]

def main():
    print("正在下载/加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    docs = []
    print(f"正在读取 {DOCS_DIR} 下的文档...")
    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".md"):
            continue
            
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
        # 替换原来的 CHUNK_SIZE 硬切逻辑，使用新函数
        chunks = split_by_markdown_headers(text)
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata={"source": filename}))

    print(f"开始连接 Postgres 并将 {len(docs)} 个分块存入 pgvector (正在清空旧数据)...")
    
    vectorstore = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION,
        use_jsonb=True,
        create_extension=True,
        pre_delete_collection=True,  # 关键新增：重建前先清空之前按粗粒度切的旧数据
    )

    print(f"✅ 入库完成，共 {len(docs)} 个 chunk，来自 {DOCS_DIR} 目录")

if __name__ == "__main__":
    main()