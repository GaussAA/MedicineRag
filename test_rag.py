"""
测试脚本：测试RAG问答功能
"""
import os
# 禁用代理
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,ollama'
os.environ['no_proxy'] = 'localhost,127.0.0.1,ollama'

import sys
sys.path.insert(0, '.')

from rag.engine import RAGEngine
from backend.services.qa_service import QAService, QARequest
from backend.services.security_service import SecurityService

def main():
    print("=" * 60)
    print("医疗问答系统测试")
    print("=" * 60)
    
    # 1. 初始化RAG引擎
    print("\n[1] 初始化RAG引擎...")
    rag = RAGEngine()
    print(f"    当前知识库文档数: {rag.get_document_count()}")
    
    # 2. 初始化服务
    print("\n[2] 初始化服务...")
    security_service = SecurityService()
    qa_service = QAService(rag, security_service)
    
    # 3. 测试问答
    test_questions = [
        "高血压的诊断标准是什么？",
        "糖尿病有哪些并发症？",
        "感冒了应该吃什么药？",
    ]
    
    print("\n" + "=" * 60)
    print("开始测试问答")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n【问题{i}】{question}")
        print("-" * 50)
        
        try:
            request = QARequest(question=question)
            response = qa_service.ask(request)
            
            print(f"  回答: {response.answer[:300]}...")
            print(f"  参考来源数: {len(response.sources)}")
            for src in response.sources[:2]:
                print(f"    - {src.get('source', '未知')}")
            print(f"  免责声明: {response.disclaimer[:50]}...")
        except Exception as e:
            print(f"  错误: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
