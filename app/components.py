'''Streamlit UI组件模块

提供可复用的UI组件，包括：
- 来源显示组件
- 消息渲染组件
- 统计卡片组件
- 确认对话框组件
'''

import streamlit as st
from typing import List, Dict, Any, Optional


def show_sources(sources: List[Dict[str, Any]], expanded: bool = False) -> None:
    """显示参考来源

    Args:
        sources: 来源列表
        expanded: 是否默认展开
    """
    if not sources:
        return

    with st.expander("📄 参考来源", expanded=expanded):
        for idx, src in enumerate(sources, 1):
            # 相似度显示
            score_text = ""
            if src.get("score"):
                score_text = f" (相似度: {src['score']:.1f}%)"

            # 来源名称
            source_name = src.get('source', '未知来源')
            st.markdown(f"**{idx}. {source_name}**{score_text}")

            # 内容摘要
            content = src.get('content', '')
            if len(content) > 200:
                content = content[:200] + "..."
            if content:
                st.markdown(f"_{content}_")


def show_confidence_indicator(confidence: float) -> None:
    """显示置信度指示器

    Args:
        confidence: 置信度值 (0-1)
    """
    if confidence <= 0:
        return

    # 颜色映射
    if confidence >= 0.7:
        color_emoji = "🟢"
        color_text = "green"
    elif confidence >= 0.5:
        color_emoji = "🟡"
        color_text = "orange"
    else:
        color_emoji = "🔴"
        color_text = "red"

    st.caption(f"{color_emoji} 置信度: {confidence:.1%}")


def show_followup_suggestions(suggestions: List[str]) -> None:
    """显示追问建议

    Args:
        suggestions: 建议问题列表
    """
    if not suggestions:
        return

    with st.expander("💭 可能感兴趣的问题"):
        for i, suggestion in enumerate(suggestions):
            if st.button(f"💭 {suggestion}", key=f"followup_{i}"):
                st.session_state.pending_prompt = suggestion
                st.rerun()


def show_knowledge_gaps(gaps: List[str]) -> None:
    """显示知识缺口提示

    Args:
        gaps: 知识缺口列表
    """
    if not gaps:
        return

    with st.expander("📚 知识缺口提示"):
        for gap in gaps:
            st.markdown(f"- {gap}")


def show_disclaimer() -> None:
    """显示免责声明"""
    st.warning("""
    ⚠️ **免责声明**：本系统回答仅供参考，不能替代医生诊断。如有严重症状，请立即就医。
    紧急情况请拨打120急救电话。
    """)


def show_emergency_warning() -> None:
    """显示紧急警告"""
    st.error("""
    🚨 **紧急提示**：根据您的描述，可能存在紧急医疗情况。
    请立即拨打120急救电话或前往最近的医院就诊！
    """)


def show_stats_card(title: str, value: Any, icon: str = "📊", help_text: str = "") -> None:
    """显示统计卡片

    Args:
        title: 卡片标题
        value: 卡片值
        icon: 图标
        help_text: 帮助文本
    """
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"### {icon}")
    with col2:
        st.metric(label=title, value=value, help=help_text)


def confirm_action(
    title: str,
    message: str,
    confirm_label: str = "确认",
    cancel_label: str = "取消",
    key: str = "confirm_dialog"
) -> bool:
    """显示确认对话框

    Args:
        title: 对话框标题
        message: 对话框消息
        confirm_label: 确认按钮标签
        cancel_label: 取消按钮标签
        key: 唯一键

    Returns:
        True表示确认，False表示取消
    """
    if key not in st.session_state:
        st.session_state[key] = False

    if not st.session_state[key]:
        with st.container():
            st.markdown(f"**{title}**")
            st.markdown(message)

            col1, col2 = st.columns(2)
            if col1.button(confirm_label, key=f"{key}_confirm", type="primary"):
                st.session_state[key] = True
                st.rerun()
            if col2.button(cancel_label, key=f"{key}_cancel"):
                st.session_state[key] = False
                return False

    return st.session_state[key]


def reset_confirm_state(key: str = "confirm_dialog") -> None:
    """重置确认对话框状态

    Args:
        key: 唯一键
    """
    if key in st.session_state:
        st.session_state[key] = False


def show_loading_spinner(message: str = "处理中...") -> Any:
    """显示加载中 spinner

    Args:
        message: 加载消息

    Returns:
        spinner上下文管理器
    """
    return st.spinner(message)


def show_success_message(message: str) -> None:
    """显示成功消息

    Args:
        message: 成功消息
    """
    st.success(message)


def show_error_message(message: str) -> None:
    """显示错误消息

    Args:
        message: 错误消息
    """
    st.error(message)


def show_info_message(message: str) -> None:
    """显示信息消息

    Args:
        message: 信息消息
    """
    st.info(message)


def render_chat_message(
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    confidence: Optional[float] = None,
    expanded_sources: bool = False
) -> None:
    """渲染聊天消息（包含来源和置信度）

    Args:
        role: 消息角色 (user/assistant)
        content: 消息内容
        sources: 参考来源列表
        confidence: 置信度
        expanded_sources: 来源是否默认展开
    """
    with st.chat_message(role):
        st.markdown(content)

        # 显示置信度
        if confidence is not None:
            show_confidence_indicator(confidence)

        # 显示来源
        if sources:
            show_sources(sources, expanded=expanded_sources)


def show_empty_state(
    icon: str = "📭",
    title: str = "暂无数据",
    message: str = "请先执行相关操作"
) -> None:
    """显示空状态

    Args:
        icon: 图标
        title: 标题
        message: 消息
    """
    st.info(f"{icon} **{title}**\n\n{message}")


def show_knowledge_base_status(stats: Optional[Dict[str, Any]]) -> None:
    """显示知识库状态

    Args:
        stats: 统计数据字典
    """
    if not stats or stats.get('document_count', 0) == 0:
        show_empty_state(
            icon="📭",
            title="知识库为空",
            message="请先在侧边栏上传医疗文档！"
        )
    elif stats:
        st.markdown(f"""
        **知识库状态：**
        - 文档数量：{stats.get('document_count', 0)}
        - 索引块数：{stats.get('indexed_chunks', 0)}
        - 总大小：{stats.get('total_size', '0 KB')}
        """)
