"""医疗知识问答系统 - 主页面"""

import streamlit as st

from app.api_client import get_api_client


# 页面配置
st.set_page_config(
    page_title="医疗知识问答系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "history" not in st.session_state:
        st.session_state.history = []

    if "agreed_to_terms" not in st.session_state:
        st.session_state.agreed_to_terms = False
    
    if "api_client" not in st.session_state:
        st.session_state.api_client = get_api_client()
    
    # Bug修复：建议按钮功能
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    
    # 性能优化：stats缓存（30秒过期）
    if "_stats_cache" not in st.session_state:
        st.session_state._stats_cache = {"data": None, "timestamp": 0}
    if "_stats_cache_timeout" not in st.session_state:
        st.session_state._stats_cache_timeout = 30  # 30秒


def show_sidebar():
    """显示侧边栏"""
    import time
    api_client = st.session_state.api_client
    
    with st.sidebar:
        st.title("📚 知识库管理")

        # 性能优化：使用缓存的stats，避免重复请求
        current_time = time.time()
        cache = st.session_state._stats_cache
        cache_timeout = st.session_state._stats_cache_timeout
        
        if (cache["data"] is None or 
            current_time - cache["timestamp"] > cache_timeout):
            # 缓存过期，重新获取
            try:
                cache["data"] = api_client.get_stats()
                cache["timestamp"] = current_time
            except:
                cache["data"] = None
        
        stats = cache["data"]
        if stats:
            st.markdown(f"""
            **知识库状态：**
            - 文档数量：{stats.get('document_count', 0)}
            - 索引块数：{stats.get('indexed_chunks', 0)}
            - 总大小：{stats.get('total_size', '0 KB')}
            """)
        else:
            st.markdown("**知识库状态：** 无法获取")

        st.divider()

        # 上传文档
        st.subheader("上传文档")
        uploaded_file = st.file_uploader(
            "选择医疗文档",
            type=["pdf", "docx", "txt", "html", "md"]
        )

        if uploaded_file:
            if st.button("上传并构建索引", type="primary"):
                with st.spinner("正在处理文档..."):
                    try:
                        result = api_client.upload_document_from_uploaded(uploaded_file, uploaded_file.name)
                        if result.get("status") == "success":
                            st.success(result.get("message", "上传成功"))
                            # 性能优化：清除stats缓存
                            st.session_state._stats_cache = {"data": None, "timestamp": 0}
                            st.rerun()
                        else:
                            st.error(result.get("message", "上传失败"))
                    except Exception as e:
                        st.error(f"上传失败: {e}")

        st.divider()

        # 知识库操作
        st.subheader("知识库操作")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("重建索引"):
                with st.spinner("正在重建索引..."):
                    try:
                        result = api_client.rebuild_index()
                        if result.get("status") == "success":
                            st.success(result.get("message", "重建成功"))
                            # 性能优化：清除stats缓存
                            st.session_state._stats_cache = {"data": None, "timestamp": 0}
                            st.rerun()
                        else:
                            st.error(result.get("message", "重建失败"))
                    except Exception as e:
                        st.error(f"重建失败: {e}")

        with col2:
            if st.button("清空知识库"):
                with st.spinner("正在清空..."):
                    try:
                        result = api_client.clear_knowledge_base()
                        if result.get("status") == "success":
                            st.success(result.get("message", "清空成功"))
                            st.rerun()
                        else:
                            st.error(result.get("message", "清空失败"))
                    except Exception as e:
                        st.error(f"清空失败: {e}")

        st.divider()

        # 导航
        st.subheader("导航")
        st.page_link("main.py", label="💬 问答咨询", icon="💬")
        st.page_link("pages/knowledge.py", label="📚 知识库管理", icon="📚")
        st.page_link("pages/analytics.py", label="📊 系统统计", icon="📊")

        st.divider()

        # 清空对话历史
        if st.button("清空对话历史"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()


def show_disclaimer():
    """显示免责声明"""
    st.warning("""
    ⚠️ **免责声明**：本系统回答仅供参考，不能替代医生诊断。如有严重症状，请立即就医。
    紧急情况请拨打120急救电话。
    """)


def main():
    """主函数"""
    try:
        # 初始化
        init_session_state()
        api_client = st.session_state.api_client

        # 检查是否首次使用
        if not st.session_state.agreed_to_terms:
            show_terms_agreement()
            return

        # 页面标题
        st.title("🏥 医疗知识问答系统")
        st.markdown("### 基于RAG技术的医疗健康咨询")

        # 侧边栏
        show_sidebar()

        # 知识库状态提示（使用缓存）
        stats = st.session_state._stats_cache.get("data")
        if stats and stats.get('document_count', 0) == 0:
            st.info("📭 知识库为空，请先在侧边栏上传医疗文档！")
        elif not stats:
            st.info("📭 知识库为空，请先在侧边栏上传医疗文档！")

        # 显示对话历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # 显示参考来源
                if "sources" in message and message["sources"]:
                    with st.expander("📄 参考来源"):
                        for idx, src in enumerate(message["sources"], 1):
                            score_text = f" (相似度: {src['score']:.1f}%)" if src.get("score") else ""
                            st.markdown(f"**{idx}. {src['source']}**{score_text}")
                            st.markdown(f"_{src['content']}_")

        # 免责声明
        show_disclaimer()

        # 如果有对话历史，显示快速追问建议
        if st.session_state.history:
            st.markdown("### 💬 继续提问")
            last_question = st.session_state.history[-1]['question'] if st.session_state.history else ""
            
            # 生成可能的追问建议
            suggestions = []
            if "高血压" in last_question:
                suggestions = ["高血压需要注意什么饮食？", "高血压患者能运动吗？", "高血压吃什么药好？"]
            elif "糖尿病" in last_question:
                suggestions = ["糖尿病的饮食禁忌有哪些？", "如何预防糖尿病并发症？", "糖尿病需要做哪些检查？"]
            elif "感冒" in last_question:
                suggestions = ["感冒了需要吃什么药？", "感冒和流感有什么区别？", "如何预防感冒？"]
            
            # 如果有建议，显示按钮
            if suggestions:
                cols = st.columns(len(suggestions))
                for i, suggestion in enumerate(suggestions):
                    if cols[i].button(f"💭 {suggestion}", key=f"suggestion_{i}"):
                        # Bug修复：保存到session_state而不是直接rerun
                        st.session_state.pending_prompt = suggestion
                        st.rerun()
            
            st.divider()

        # Bug修复：检查是否有待发送的建议问题
        if st.session_state.pending_prompt:
            prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = None  # 清除待发送状态
        else:
            prompt = None
        
        # 聊天输入（限制500字符）
        input_placeholder = "请输入您的医疗问题... (最多500字)"
        if prompt := st.chat_input(input_placeholder, max_chars=500):
            # 用户消息
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # 生成回答（流式输出）
            with st.chat_message("assistant"):
                # 先显示一个空的容器用于流式输出
                response_container = st.empty()
                full_response = ""

                # 使用流式API
                with st.spinner("正在思考..."):
                    sources = []  # Bug修复：先初始化，避免异常时未定义
                    try:
                        # 获取流式响应（传递对话历史）
                        stream_generator = api_client.ask_stream(
                            question=prompt,
                            history=st.session_state.history
                        )

                        # 流式显示
                        for chunk in stream_generator:
                            # 检查是否是sources数据（特殊标记）
                            if chunk.startswith("__SOURCE__: "):
                                import json
                                source_data = json.loads(chunk[12:])
                                sources.append(source_data)
                            elif chunk.strip():
                                # 非source数据直接显示
                                full_response += chunk
                                response_container.markdown(full_response)
                            
                        # 流式完成后显示参考来源
                        if sources:
                            with st.expander("📄 参考来源"):
                                for idx, src in enumerate(sources, 1):
                                    score_text = f" (相似度: {src.get('score', 0):.1f}%)" if src.get("score") else ""
                                    st.markdown(f"**{idx}. {src.get('source', '未知来源')}**{score_text}")
                                    content = src.get('content', '')
                                    if len(content) > 200:
                                        content = content[:200] + "..."
                                    st.markdown(f"_{content}_")
                                    
                    except Exception as e:
                        st.error(f"请求失败: {e}")
                        full_response = "抱歉，处理您的请求时发生错误。"
                        
                # 显示免责声明
                st.error("⚠️ 以上回答仅供参考，不能替代医生诊断。如有严重症状，请立即就医。")

            # 保存到历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })

            # 保存到对话历史（用于多轮对话）
            st.session_state.history.append({
                "question": prompt,
                "answer": full_response
            })

    except Exception as e:
        """全局错误边界"""
        import traceback
        st.error(f"⚠️ 系统发生错误：{str(e)[:100]}")
        st.info("请刷新页面重试，如果问题持续存在，请联系管理员。")
        
        if st.button("🔄 刷新页面"):
            st.rerun()


def show_terms_agreement():
    """显示用户协议同意弹窗"""
    st.markdown("""
    # 欢迎使用医疗知识问答系统

    在使用本系统之前，请仔细阅读以下条款：
    """)

    st.markdown("""
    ### 使用条款

    1. **仅供参考**：本系统提供的回答仅供参考，不能替代专业医生的诊断和治疗。

    2. **紧急情况**：如遇紧急医疗情况，请立即拨打120急救电话或前往医院就诊。

    3. **信息准确性**：本系统的回答基于上传的医疗文档，内容的准确性取决于原始文档的质量。

    4. **隐私保护**：请勿在提问中输入您的个人身份信息（姓名、身份证号等）。

    5. **知识库限制**：系统回答仅基于知识库中的文档，可能不涵盖所有医疗情况。
    """)

    st.checkbox(
        "我已阅读并同意上述条款",
        key="agree_checkbox"
    )

    if st.button("开始使用", type="primary"):
        if st.session_state.agree_checkbox:
            st.session_state.agreed_to_terms = True
            st.rerun()
        else:
            st.error("请先阅读并同意使用条款")


if __name__ == "__main__":
    main()