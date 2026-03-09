"""问答统计模块 - 收集和分析系统运行指标（异步优化版）"""

import json
import time
import threading
import atexit
import os
import tempfile
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)


class QAStats:
    """问答统计收集器 - 异步批量写入版"""

    def __init__(self, stats_file: str = "data/qa_stats.json"):
        self.stats_file = Path(stats_file)
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)

        # 线程安全
        self._lock = threading.Lock()
        self._dirty = False

        # 批量刷新配置
        self._flush_interval = 30  # 30秒刷新一次
        self._max_records_before_flush = 100  # 累积100条记录强制刷新
        self._pending_records = 0  # 待刷新记录数

        # 启动定时刷新线程
        self._flush_thread = None
        self._running = True
        self._stats = self._load_stats()

        # 启动后台刷新线程
        self._start_flush_thread()

        # 注册进程退出回调，确保数据不丢失
        atexit.register(self._atexit_callback)

    def _atexit_callback(self):
        """进程退出时的回调，确保数据保存"""
        try:
            self._running = False
            # 不在atexit中调用flush，因为此时日志系统可能已经关闭
            # 直接尝试同步保存
            try:
                self._save_stats()
            except Exception:
                pass  # 忽略保存失败
        except Exception:
            pass  # 忽略所有异常

    def _load_stats(self) -> Dict[str, Any]:
        """加载统计数据"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    # 确保question_types是dict
                    if 'question_types' in stats and isinstance(stats['question_types'], dict):
                        stats['question_types'] = defaultdict(int, stats['question_types'])
                    return stats
            except Exception as e:
                logger.warning(f"加载统计文件失败: {e}")
        
        return {
            "total_questions": 0,
            "successful_answers": 0,
            "failed_answers": 0,
            "no_result_answers": 0,
            "question_types": defaultdict(int),
            "sensitive_blocked": 0,
            "emergency_warnings": 0,
            "avg_response_time_ms": 0,
            "avg_retrieval_time_ms": 0,
            "avg_llm_time_ms": 0,
            "recent_questions": [],
            "unanswered_questions": [],
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _save_stats(self):
        """保存统计数据到文件"""
        try:
            # 确保目录存在
            stats_dir = self.stats_file.parent
            if not stats_dir.exists():
                stats_dir.mkdir(parents=True, exist_ok=True)
            
            # 准备统计数据
            stats_to_save = self._stats.copy()
            stats_to_save["question_types"] = dict(stats_to_save.get("question_types", {}))

            # 原子写入：先写临时文件，再重命名
            try:
                temp_fd, temp_path = tempfile.mkstemp(
                    dir=str(stats_dir),
                    prefix='.tmp_stats_',
                    suffix='.json'
                )
            except FileNotFoundError:
                # 目录已被删除，使用系统临时目录
                temp_fd, temp_path = tempfile.mkstemp(
                    prefix='.tmp_stats_',
                    suffix='.json'
                )
            
            try:
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                    json.dump(stats_to_save, f, ensure_ascii=False, indent=2)

                # 重命名原子操作，确保文件完整性
                os.replace(temp_path, self.stats_file)
            except Exception:
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise
        except Exception:
            # atexit回调中静默失败，不使用logger
            pass

    def _start_flush_thread(self):
        """启动后台刷新线程"""
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        logger.info("统计刷新线程已启动")

    def _flush_loop(self):
        """后台刷新循环"""
        while self._running:
            time.sleep(self._flush_interval)
            if self._dirty:
                with self._lock:
                    if self._dirty:
                        self._save_stats()
                        self._dirty = False
                        self._pending_records = 0

    def _schedule_flush(self):
        """安排一次刷新（非阻塞）"""
        self._dirty = True
        self._pending_records += 1
        
        # 累积足够多记录时强制刷新
        if self._pending_records >= self._max_records_before_flush:
            with self._lock:
                if self._dirty:
                    self._save_stats()
                    self._dirty = False
                    self._pending_records = 0

    def flush(self):
        """手动刷新（用于应用关闭时）"""
        with self._lock:
            if self._dirty:
                self._save_stats()
                self._dirty = False
                self._pending_records = 0

    def shutdown(self):
        """关闭统计模块"""
        self._running = False
        self.flush()
        logger.info("统计模块已关闭")

    def record_question(
        self,
        question: str,
        question_type: Optional[str],
        success: bool,
        has_result: bool,
        response_time_ms: float,
        retrieval_time_ms: float,
        llm_time_ms: float,
        is_sensitive: bool = False,
        is_emergency: bool = False,
    ):
        """记录一次问答（异步批量写入）"""
        with self._lock:
            self._stats["total_questions"] += 1
            
            if success:
                self._stats["successful_answers"] += 1
            else:
                self._stats["failed_answers"] += 1

            if not has_result:
                self._stats["no_result_answers"] += 1
                # 记录未回答的问题
                if len(self._stats["unanswered_questions"]) < 100:
                    self._stats["unanswered_questions"].append({
                        "question": question,
                        "timestamp": datetime.now().isoformat()
                    })

            if question_type:
                if question_type not in self._stats["question_types"]:
                    self._stats["question_types"][question_type] = 0
                self._stats["question_types"][question_type] += 1

            if is_sensitive:
                self._stats["sensitive_blocked"] += 1

            if is_emergency:
                self._stats["emergency_warnings"] += 1

            # 更新平均响应时间（使用移动平均）
            total = self._stats["total_questions"]
            old_avg = self._stats["avg_response_time_ms"]
            self._stats["avg_response_time_ms"] = (old_avg * (total - 1) + response_time_ms) / total

            old_retrieval = self._stats["avg_retrieval_time_ms"]
            self._stats["avg_retrieval_time_ms"] = (old_retrieval * (total - 1) + retrieval_time_ms) / total

            old_llm = self._stats["avg_llm_time_ms"]
            self._stats["avg_llm_time_ms"] = (old_llm * (total - 1) + llm_time_ms) / total

            # 记录最近问题
            recent_q = {
                "question": question[:100],
                "type": question_type,
                "success": success,
                "has_result": has_result,
                "response_time_ms": response_time_ms,
                "timestamp": datetime.now().isoformat()
            }
            self._stats["recent_questions"].insert(0, recent_q)
            # 只保留最近50个
            self._stats["recent_questions"] = self._stats["recent_questions"][:50]

            # 转换为普通dict
            if isinstance(self._stats["question_types"], defaultdict):
                self._stats["question_types"] = dict(self._stats["question_types"])
        
        # 异步刷新（非阻塞）
        self._schedule_flush()

    def record_cache_stats(self, hits: int, misses: int):
        """记录缓存统计（异步批量写入）"""
        with self._lock:
            self._stats["cache_hits"] += hits
            self._stats["cache_misses"] += misses
        
        self._schedule_flush()

    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        stats = self._stats.copy()
        stats["question_types"] = dict(stats.get("question_types", {}))
        
        # 计算成功率
        total = stats["total_questions"]
        if total > 0:
            stats["success_rate"] = f"{stats['successful_answers'] / total * 100:.1f}%"
        else:
            stats["success_rate"] = "0%"

        # 计算缓存命中率
        total_cache = stats["cache_hits"] + stats["cache_misses"]
        if total_cache > 0:
            stats["cache_hit_rate"] = f"{stats['cache_hits'] / total_cache * 100:.1f}%"
        else:
            stats["cache_hit_rate"] = "0%"

        return stats

    def get_recent_questions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近问题"""
        return self._stats["recent_questions"][:limit]

    def get_unanswered_questions(self) -> List[Dict[str, Any]]:
        """获取未回答的问题（知识库缺口）"""
        return self._stats.get("unanswered_questions", [])

    def get_question_type_distribution(self) -> Dict[str, int]:
        """获取问题类型分布"""
        return dict(self._stats.get("question_types", {}))

    def clear_stats(self):
        """清空统计"""
        self._stats = self._load_stats()
        self._stats.clear()
        self._save_stats()


# 全局统计实例
_stats_instance: Optional[QAStats] = None


def get_stats_instance() -> QAStats:
    """获取全局统计实例"""
    global _stats_instance
    if _stats_instance is None:
        _stats_instance = QAStats()
    return _stats_instance
