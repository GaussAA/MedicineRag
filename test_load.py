"""
直接测试文档加载
"""
import sys
import os
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
sys.path.insert(0, '.')

try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb

    print("Step 1: Loading embedding model...")
    embed_model = OllamaEmbedding(model_name='bge-m3:latest', base_url='http://localhost:11434')
    print("OK")
    
    print("Step 2: Reading documents...")
    reader = SimpleDirectoryReader(input_files=['data/documents/医疗知识问答.txt'])
    docs = reader.load_data()
    print(f"Loaded {len(docs)} docs")
    
    print("Step 3: Creating Chroma collection...")
    db = chromadb.PersistentClient(path='data/chroma_db')
    collection = db.get_or_create_collection('medical_knowledge')
    print(f"Collection created, count = {collection.count()}")
    
    print("Step 4: Creating index...")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_documents(
        docs, 
        vector_store=vector_store, 
        embed_model=embed_model
    )
    print("Index created!")
    
    print("Step 5: Verifying...")
    print(f"Final collection count = {collection.count()}")
    print("SUCCESS!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
