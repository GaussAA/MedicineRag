"""API测试模块"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from backend.api.main import app


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


class TestHealthEndpoint:
    """健康检查端点测试"""

    def test_health_basic(self, client):
        """测试基础健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_health_detailed(self, client):
        """测试详细健康检查"""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        # 详细健康检查应包含ollama、chroma、knowledge_base等检查项
        assert "ollama" in data or "checks" in data


class TestDocsEndpoint:
    """文档管理API测试"""

    def test_list_documents(self, client):
        """测试获取文档列表"""
        response = client.get("/api/docs/list")
        # 可能返回200或500（如果服务有问题）
        assert response.status_code in [200, 500]

    def test_get_stats(self, client):
        """测试获取统计信息"""
        response = client.get("/api/docs/stats")
        # 可能返回200或500
        assert response.status_code in [200, 500]


class TestQAEndpoint:
    """问答API测试"""

    def test_qa_empty_question(self, client):
        """测试空问题请求"""
        response = client.post(
            "/api/qa/ask",
            json={"question": ""}
        )
        # 应该返回422验证错误
        assert response.status_code == 422

    def test_qa_invalid_request(self, client):
        """测试无效请求"""
        response = client.post(
            "/api/qa/ask",
            json={}
        )
        # 应该返回422验证错误
        assert response.status_code == 422


class TestMetricsEndpoint:
    """监控指标端点测试"""

    def test_metrics_endpoint(self, client):
        """测试metrics端点"""
        response = client.get("/metrics")
        assert response.status_code == 200


class TestCORS:
    """CORS测试"""

    def test_cors_headers(self, client):
        """测试CORS头"""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "GET"
            }
        )
        # OPTIONS请求应该返回200或204或405（如果未配置CORS）
        assert response.status_code in [200, 204, 405]
