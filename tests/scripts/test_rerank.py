"""Test reranking functionality"""
from rag.core.engine import RAGEngine

def test_rerank():
    engine = RAGEngine()
    
    # Test query
    query = 'What are the diagnostic criteria for hypertension?'
    
    # Retrieve documents (with reranking)
    results = engine.retrieve(query, top_k=3)
    
    print('=' * 60)
    print('Retrieval Results (with reranking)')
    print('=' * 60)
    
    for i, doc in enumerate(results):
        print(f'\n--- Result {i+1} ---')
        content = doc["text"][:200] if len(doc["text"]) > 200 else doc["text"]
        print(f'Content: {content}...')
        print(f'Source: {doc.get("source", "Unknown")}')
        print(f'Relevance Score: {doc.get("score", "N/A")}')
    
    print('\n' + '=' * 60)
    print('Reranking test completed')
    print('=' * 60)

if __name__ == '__main__':
    test_rerank()