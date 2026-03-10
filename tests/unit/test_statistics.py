"""测试统计模块"""

import pytest
import time
import tempfile
import os
from pathlib import Path


class TestQAStats:
    """测试问答统计类"""

    def test_stats_initialization(self):
        """测试统计初始化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from backend.statistics import QAStats
            stats_file = os.path.join(tmpdir, "test_stats.json")
            stats = QAStats(stats_file=stats_file)
            
            assert stats._stats["total_questions"] == 0
            assert stats._stats["successful_answers"] == 0
            assert stats._stats["failed_answers"] == 0

    def test_record_question(self):
        """测试记录问答"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from backend.statistics import QAStats
            stats_file = os.path.join(tmpdir, "test_stats.json")
            stats = QAStats(stats_file=stats_file)
            
            stats.record_question(
                question="测试问题",
                question_type="general",
                success=True,
                has_result=True,
                response_time_ms=100.0,
                retrieval_time_ms=50.0,
                llm_time_ms=50.0
            )
            
            assert stats._stats["total_questions"] == 1
            assert stats._stats["successful_answers"] == 1

    def test_record_question_failure(self):
        """测试记录失败问答"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from backend.statistics import QAStats
            stats_file = os.path.join(tmpdir, "test_stats.json")
            stats = QAStats(stats_file=stats_file)
            
            stats.record_question(
                question="测试问题",
                question_type="general",
                success=False,
                has_result=False,
                response_time_ms=100.0,
                retrieval_time_ms=50.0,
                llm_time_ms=50.0
            )
            
            assert stats._stats["total_questions"] == 1
            assert stats._stats["failed_answers"] == 1
            assert stats._stats["no_result_answers"] == 1

    def test_record_sensitive_blocked(self):
        """测试记录敏感词拦截"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from backend.statistics import QAStats
            stats_file = os.path.join(tmpdir, "test_stats.json")
            stats = QAStats(stats_file=stats_file)
            
            stats.record_question(
                question="测试敏感词",
                question_type=None,
                success=False,
                has_result=False,
                response_time_ms=0,
                retrieval_time_ms=0,
                llm_time_ms=0,
                is_sensitive=True
            )
            
            assert stats._stats["sensitive_blocked"] == 1

    def test_record_emergency_warning(self):
        """测试记录紧急警告"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from backend.statistics import QAStats
            stats_file = os.path.join(tmpdir, "test_stats.json")
            stats = QAStats(stats_file=stats_file)
            
            stats.record_question(
                question="胸痛",
                question_type=None,
                success=True,
                has_result=True,
                response_time_ms=100.0,
                retrieval_time_ms=50.0,
                llm_time_ms=50.0,
                is_emergency=True
            )
            
            assert stats._stats["emergency_warnings"] == 1

    def test_get_summary(self):
        """测试获取统计摘要"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from backend.statistics import QAStats
            stats_file = os.path.join(tmpdir, "test_stats.json")
            stats = QAStats(stats_file=stats_file)
            
            stats.record_question(
                question="测试问题",
                question_type="general",
                success=True,
                has_result=True,
                response_time_ms=100.0,
                retrieval_time_ms=50.0,
                llm_time_ms=50.0
            )
            
            summary = stats.get_summary()
            
            assert summary["total_questions"] == 1
            assert summary["successful_answers"] == 1
            assert "success_rate" in summary

    def test_cache_stats(self):
        """测试缓存统计"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from backend.statistics import QAStats
            stats_file = os.path.join(tmpdir, "test_stats.json")
            stats = QAStats(stats_file=stats_file)
            
            stats.record_cache_stats(hits=80, misses=20)
            
            summary = stats.get_summary()
            assert summary["cache_hits"] == 80
            assert summary["cache_misses"] == 20
            assert summary["cache_hit_rate"] == "80.0%"

    def test_question_type_distribution(self):
        """测试问题类型分布"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from backend.statistics import QAStats
            stats_file = os.path.join(tmpdir, "test_stats.json")
            stats = QAStats(stats_file=stats_file)
            
            stats.record_question(
                question="问题1",
                question_type="disease",
                success=True,
                has_result=True,
                response_time_ms=100.0,
                retrieval_time_ms=50.0,
                llm_time_ms=50.0
            )
            
            stats.record_question(
                question="问题2",
                question_type="disease",
                success=True,
                has_result=True,
                response_time_ms=100.0,
                retrieval_time_ms=50.0,
                llm_time_ms=50.0
            )
            
            stats.record_question(
                question="问题3",
                question_type="medicine",
                success=True,
                has_result=True,
                response_time_ms=100.0,
                retrieval_time_ms=50.0,
                llm_time_ms=50.0
            )
            
            distribution = stats.get_question_type_distribution()
            assert distribution["disease"] == 2
            assert distribution["medicine"] == 1

    def test_shutdown(self):
        """测试关闭统计模块"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from backend.statistics import QAStats
            stats_file = os.path.join(tmpdir, "test_stats.json")
            stats = QAStats(stats_file=stats_file)
            
            stats.record_question(
                question="测试",
                question_type="general",
                success=True,
                has_result=True,
                response_time_ms=100.0,
                retrieval_time_ms=50.0,
                llm_time_ms=50.0
            )
            
            # 关闭不应抛出异常
            stats.shutdown()
            
            # 验证文件已保存
            assert os.path.exists(stats_file)
