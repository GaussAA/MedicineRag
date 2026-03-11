"""医疗知识问答系统 - 主页面"""

import json
import logging
import streamlit as st

from app.api_client import get_api_client
from app.components import show_disclaimer, show_sources, show_confidence_indicator
from app.constants import STATS_CACHE_TIMEOUT, MAX_INPUT_CHARS
from backend.config import config

# 配置日志
logger = logging.getLogger(__name__)


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
        st.session_state._stats_cache_timeout = STATS_CACHE_TIMEOUT  # 从常量读取
    
    # Agent 模式相关状态
    if "agent_mode" not in st.session_state:
        st.session_state.agent_mode = False  # 默认使用普通 RAG 模式
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())


def show_sidebar():
    """显示侧边栏"""
    import time
    api_client = st.session_state.api_client
    
    with st.sidebar:
        st.title("📚 知识库管理")
        
        # Agent 模式选择
        st.subheader("🤖 问答模式")
        agent_mode = st.toggle(
            "启用 Agent 模式",
            value=st.session_state.agent_mode,
            help="Agent 模式支持多步骤推理、主动追问和知识缺口识别"
        )
        
        if agent_mode != st.session_state.agent_mode:
            st.session_state.agent_mode = agent_mode
            st.rerun()
        
        if st.session_state.agent_mode:
            st.success("🤖 Agent 模式已启用")
            st.caption("支持：智能推理 | 主动追问 | 知识缺口识别")
            
            # Agent 高级配置
            with st.expander("⚙️ Agent 配置"):
                enable_followup = st.checkbox("启用主动追问", value=True)
                enable_knowledge_gap = st.checkbox("启用知识缺口识别", value=True)
        else:
            st.info("💬 使用标准 RAG 模式")

        st.divider()

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
            except Exception as e:
                logger.warning(f"获取统计信息失败: {e}")
                cache["data"] = None
        
        stats = cache["data"]
        if stats:
            st.markdown(f"""
            **知识库状态：**
            - 文档数量：{stats.get('document_count', 0)}
            - 索引块数：{stats.get('indexed_chunks', 0)}
            - 总大小：{stats.get('total_size', '0 KB')}
            """)
            # 添加手动刷新按钮
            if st.button("🔄 刷新状态", use_container_width=True):
                # 清除缓存
                st.session_state._stats_cache = {"data": None, "timestamp": 0}
                st.rerun()
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
            # 添加二次确认机制
            if "show_clear_confirm" not in st.session_state:
                st.session_state.show_clear_confirm = False
            
            if st.session_state.show_clear_confirm:
                st.warning("⚠️ 确定要清空知识库吗？此操作不可恢复！")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅ 确认清空", type="primary"):
                        with st.spinner("正在清空..."):
                            try:
                                result = api_client.clear_knowledge_base()
                                if result.get("status") == "success":
                                    st.success(result.get("message", "清空成功"))
                                    # 清除stats缓存
                                    st.session_state._stats_cache = {"data": None, "timestamp": 0}
                                    st.session_state.show_clear_confirm = False
                                    st.rerun()
                                else:
                                    st.error(result.get("message", "清空失败"))
                            except Exception as e:
                                st.error(f"清空失败: {e}")
                with col_no:
                    if st.button("❌ 取消"):
                        st.session_state.show_clear_confirm = False
                        st.rerun()
            else:
                if st.button("🗑️ 清空知识库"):
                    st.session_state.show_clear_confirm = True
                    st.rerun()

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


def main():
    """主函数"""
    try:
        # 初始化
        init_session_state()
        api_client = st.session_state.api_client
        
        # API服务健康检查
        try:
            health = api_client.health_check()
            if health.get("status") != "healthy":
                st.warning("⚠️ API服务状态异常，部分功能可能不可用")
        except Exception as e:
            st.error(f"❌ 无法连接到API服务，请确保后端服务正在运行！\n\n**错误**: {str(e)[:100]}\n\n**解决方法**: 请在终端运行 `python scripts/start_all.py` 或手动启动后端服务")
            st.stop()

        # 检查是否首次使用
        if not st.session_state.agreed_to_terms:
            show_terms_agreement()
            return

        # 页面标题
        st.title("🏥 医疗知识问答系统")
        
        # 显示当前模式
        if st.session_state.agent_mode:
            st.markdown("### 🤖 Agent 模式")
            st.caption("智能推理 | 主动追问 | 知识缺口识别")
        else:
            st.markdown("### 💬 标准 RAG 模式")
        
        st.markdown("---")

        # 侧边栏
        show_sidebar()

        # 知识库状态提示（使用缓存）
        stats = st.session_state._stats_cache.get("data")
        if not stats or stats.get('document_count', 0) == 0:
            st.info("📭 知识库为空，请先在侧边栏上传医疗文档！")

        # 显示对话历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # 显示参考来源
                if "sources" in message and message["sources"]:
                    show_sources(message["sources"])

        # 免责声明
        show_disclaimer()

        # 如果有对话历史，显示快速追问建议
        if st.session_state.history:
            st.markdown("### 💬 继续提问")
            last_question = st.session_state.history[-1]['question'] if st.session_state.history else ""
            
            # 生成可能的追问建议（基于关键词匹配）
            suggestions = []
            question_lower = last_question.lower()
            
            # 定义更丰富的关键词-建议映射
            keyword_suggestions = {
                "高血压": ["高血压需要注意什么饮食？", "高血压患者能运动吗？", "高血压吃什么药好？"],
                "糖尿病": ["糖尿病的饮食禁忌有哪些？", "如何预防糖尿病并发症？", "糖尿病需要做哪些检查？"],
                "感冒": ["感冒了需要吃什么药？", "感冒和流感有什么区别？", "如何预防感冒？"],
                "心脏病": ["心脏病的早期症状有哪些？", "如何预防心脏病？", "心脏病饮食需要注意什么？"],
                "癌症": ["如何早期发现癌症？", "癌症预防措施有哪些？", "哪些习惯容易导致癌症？"],
                "肺炎": ["肺炎有哪些症状？", "如何预防肺炎？", "肺炎需要住院治疗吗？"],
                "胃": ["胃炎有哪些症状？", "如何养胃？", "胃痛需要做什么检查？"],
                "肝": ["如何保护肝脏？", "肝炎有哪些传播途径？", "肝功能异常怎么办？"],
                "肾": ["肾病有哪些早期症状？", "如何保护肾脏？", "肾功能检查有哪些？"],
                "肺": ["如何保护肺部健康？", "吸烟对肺部的影响有哪些？", "肺部检查有哪些？"],
                "头痛": ["头痛有哪些原因？", "如何缓解头痛？", "什么样的头痛需要就医？"],
                "发烧": ["发烧了怎么办？", "发烧需要吃退烧药吗？", "什么样的发烧需要重视？"],
                "咳嗽": ["咳嗽一直不好怎么办？", "干咳和湿咳有什么区别？", "什么情况的咳嗽需要就医？"],
            }
            
            # 匹配关键词并生成建议
            for keyword, keyword_suggestions_list in keyword_suggestions.items():
                if keyword in last_question or keyword in question_lower:
                    suggestions = keyword_suggestions_list
                    break
            
            # 如果没有匹配的关键词，显示通用建议
            if not suggestions:
                suggestions = [
                    "这个病需要注意什么？",
                    "如何治疗这个疾病？",
                    "吃什么药效果好？"
                ]
            
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
        input_placeholder = "请输入您的医疗问题... (最多{}字符)".format(MAX_INPUT_CHARS)
        if prompt := st.chat_message("user").chat_input(input_placeholder, max_chars=MAX_INPUT_CHARS):
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
                    followup_questions = []
                    knowledge_gaps = []
                    confidence = 0.0
                    disclaimer_text = ""
                    
                    try:
                        if st.session_state.agent_mode:
                            # Agent 模式
                            stream_generator = api_client.ask_agent_stream(
                                question=prompt,
                                history=st.session_state.history,
                                session_id=st.session_state.session_id,
                                enable_followup=True,
                                enable_knowledge_gap=True
                            )
                        else:
                            # 标准 RAG 模式
                            stream_generator = api_client.ask_stream(
                                question=prompt,
                                history=st.session_state.history
                            )

                        # 流式显示
                        for chunk in stream_generator:
                            # 检查是否是特殊数据（Agent 模式）
                            if chunk.startswith("__SOURCE__: "):
                                source_data = json.loads(chunk[12:])
                                sources.append(source_data)
                            elif chunk.startswith("__FOLLOWUP__: "):
                                followup_questions = json.loads(chunk[13:])
                            elif chunk.startswith("__KNOWLEDGE_GAPS__: "):
                                knowledge_gaps = json.loads(chunk[18:])
                            elif chunk.startswith("__CONFIDENCE__: "):
                                confidence = float(chunk[16:])
                            elif chunk.startswith("__STEPS__: "):
                                # Agent 推理步骤，仅记录不显示
                                pass
                            elif chunk.startswith("__DISCLAIMER__: "):
                                disclaimer_text = json.loads(chunk[15:])
                            elif chunk.strip():
                                # 非特殊数据直接显示
                                full_response += chunk
                                response_container.markdown(full_response)
                        
                        # 流式完成后显示额外信息
                        if st.session_state.agent_mode:
                            # 显示追问建议
                            if followup_questions:
                                with st.expander("💭 可能感兴趣的问题"):
                                    for q in followup_questions:
                                        st.markdown(f"- {q}")
                            
                            # 显示知识缺口
                            if knowledge_gaps:
                                with st.expander("📚 知识缺口提示"):
                                    for gap in knowledge_gaps:
                                        st.markdown(f"- {gap}")
                            
                            # 显示置信度（使用配置阈值）
                            if confidence > 0:
                                conf_color = "🟢" if confidence >= config.CONFIDENCE_HIGH else ("🟡" if confidence >= config.CONFIDENCE_MEDIUM else "🔴")
                                st.caption(f"{conf_color} 置信度: {confidence:.1%}")
                        
                        # 显示免责声明
                        if disclaimer_text:
                            st.caption(f"⚠️ {disclaimer_text}")
                        
                        # 显示参考来源
                        if sources:
                            show_sources(sources)
                                    
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
        st.error(f"⚠️ 系统发生错误：{str(e)[:200]}\n\n如问题持续存在，请刷新页面重试。")
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