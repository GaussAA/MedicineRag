"""系统统计页面 - 可观测性面板"""

import streamlit as st
from datetime import datetime

from app.api_client import get_api_client


def main():
    st.set_page_config(
        page_title="系统统计",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 系统运行统计")
    st.markdown("---")

    # 获取API客户端
    api_client = get_api_client()
    
    try:
        # 获取统计信息
        qa_stats = api_client.get_qa_stats()
        summary = qa_stats.get("summary", {})
    except Exception as e:
        st.error(f"获取统计信息失败: {e}")
        summary = {}
        qa_stats = {"question_type_distribution": {}, "recent_questions": [], "unanswered_questions": []}

    # 核心指标行
    st.subheader("📈 核心指标")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总问题数", summary.get("total_questions", 0))
    with col2:
        st.metric("成功率", summary.get("success_rate", "0%"))
    with col3:
        st.metric("无结果数", summary.get("no_result_answers", 0))
    with col4:
        st.metric("敏感拦截", summary.get("sensitive_blocked", 0))

    st.markdown("---")

    # 性能指标
    st.subheader("⚡ 性能指标")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均响应时间", f"{summary.get('avg_response_time_ms', 0):.0f} ms")
    with col2:
        st.metric("平均检索时间", f"{summary.get('avg_retrieval_time_ms', 0):.0f} ms")
    with col3:
        st.metric("平均LLM时间", f"{summary.get('avg_llm_time_ms', 0):.0f} ms")
    with col4:
        st.metric("缓存命中率", summary.get("cache_hit_rate", "0%"))

    st.markdown("---")

    # 问题类型分布
    st.subheader("📂 问题类型分布")
    
    type_dist = qa_stats.get("question_type_distribution", {})
    if type_dist:
        # 转换类型名称
        type_names = {
            "symptom": "症状相关",
            "disease": "疾病相关",
            "medication": "用药相关",
            "examination": "检查相关"
        }
        display_dist = {type_names.get(k, k): v for k, v in type_dist.items()}
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**问题类型统计：**")
            for qtype, count in sorted(display_dist.items(), key=lambda x: -x[1]):
                st.write(f"- {qtype}: {count} 个")
        with col2:
            st.bar_chart(display_dist)
    else:
        st.info("暂无问题类型数据")

    st.markdown("---")

    # 最近问题
    st.subheader("💬 最近问题")
    
    recent = qa_stats.get("recent_questions", [])
    if recent:
        for i, q in enumerate(recent, 1):
            status_emoji = "✅" if q.get("success") else "❌"
            time_str = q.get("timestamp", "")[:19] if q.get("timestamp") else ""
            st.write(f"{i}. {status_emoji} {q.get('question', '')}")
            st.caption(f"   类型: {q.get('type', '通用')} | 响应: {q.get('response_time_ms', 0):.0f}ms | {time_str}")
    else:
        st.info("暂无最近问题记录")

    st.markdown("---")

    # 知识库缺口
    st.subheader("🔍 知识库缺口（未回答问题）")
    
    unanswered = qa_stats.get("unanswered_questions", [])
    if unanswered:
        st.warning(f"以下问题未能从知识库中找到答案，共 {len(unanswered)} 个：")
        for i, q in enumerate(unanswered[:20], 1):  # 只显示前20个
            time_str = q.get("timestamp", "")[:10] if q.get("timestamp") else ""
            st.write(f"{i}. {q.get('question', '')} ({time_str})")
        if len(unanswered) > 20:
            st.caption(f"还有 {len(unanswered) - 20} 个未显示...")
    else:
        st.success("所有问题都已成功回答！")

    st.markdown("---")

    # 紧急情况统计
    st.subheader("🚨 紧急情况统计")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("紧急症状警告", summary.get("emergency_warnings", 0))
    with col2:
        st.metric("敏感内容拦截", summary.get("sensitive_blocked", 0))

    # 清理缓存按钮
    st.markdown("---")
    if st.button("🗑️ 清空统计数据", type="secondary"):
        try:
            result = api_client.clear_stats()
            if result.get("status") == "success":
                st.success("统计数据已清空！")
                st.rerun()
        except Exception as e:
            st.error(f"清空失败: {e}")


if __name__ == "__main__":
    main()
