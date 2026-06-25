import os
import fitz  # pymupdf
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def chunk_text(text: str, size: int = 500, overlap: int = 50):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        chunks.append(text[start:start + size])
        start += (size - overlap)
    return chunks

def main():
    # 1. 明确当前脚本所在目录：Agent_Distill/legal_rag/
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 【核心修改】data 文件夹在项目根目录，需要通过 ".." 返回上一级目录找到
    DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
    
    # 3. chroma_db 保持在 legal_rag/ 目录下，与 server.py 的读取路径完美对齐
    DB_DIR = os.path.join(BASE_DIR, "chroma_db")

    print(f"📂 脚本所在目录: {BASE_DIR}")
    print(f"📂 PDF 检索目录: {DATA_DIR}")
    print(f"📂 向量库保存目录: {DB_DIR}")
    print("-" * 40)
    
    print("⏳ 开始读取 PDF 文件...")
    pdf_files = ["labor_law.pdf", "minfa.pdf"]
    docs = []
    
    for file_name in pdf_files:
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(file_path):
            # 使用 pymupdf 替代原 PyPDFLoader
            pdf_doc = fitz.open(file_path)
            full_text = ""
            for page in pdf_doc:
                full_text += page.get_text("text") + "\n"
            
            # 进行自定义文本分块
            chunks = chunk_text(full_text, size=500, overlap=50)
            
            # 封装为兼容对象
            for i, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk, 
                    metadata={"source": file_name, "chunk_id": i}
                ))
            print(f"  -> ✅ 成功解析并分块: {file_name} (共 {len(chunks)} 块)")
            pdf_doc.close()
        else:
            print(f"  -> ⚠️ 未找到文件: {file_path}")

    if not docs:
        print("❌ 无有效文件，退出执行。")
        return

    print("\n🧠 正在加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese") 

    print(f"💾 正在写入 Chroma 向量库...")
    vectorstore = Chroma.from_documents(
        documents=docs,  
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print("🎉 向量库构建完成！")

if __name__ == "__main__":
    main()