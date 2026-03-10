"""知识库管理页面"""

import streamlit as st

from app.api_client import get_api_client


# 页面配置
st.set_page_config(
    page_title="知识库管理 - 医疗知识问答系统",
    page_icon="📚",
    layout="wide"
)


def main():
    """主函数"""
    st.title("📚 知识库管理")
    st.markdown("---")

    # 获取API客户端
    api_client = get_api_client()

    # 侧边栏导航
    with st.sidebar:
        st.header("导航")
        st.page_link("main.py", label="💬 问答咨询", icon="💬")
        st.page_link("pages/knowledge.py", label="📚 知识库管理", icon="📚", disabled=True)

    # 显示统计信息
    st.subheader("📊 知识库统计")
    try:
        stats = api_client.get_stats()
    except Exception as e:
        st.error(f"获取统计信息失败: {e}")
        stats = {'document_count': 0, 'indexed_chunks': 0, 'total_size': '0 KB'}

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("文档数量", stats.get('document_count', 0))
    with col2:
        st.metric("索引块数", stats.get('indexed_chunks', 0))
    with col3:
        st.metric("总大小", stats.get('total_size', '0 KB'))

    st.markdown("---")

    # 上传文档
    st.subheader("📤 上传文档")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "选择医疗文档",
            type=["pdf", "docx", "txt", "html", "md"],
            help=f"支持的格式：PDF、Word、TXT、HTML、Markdown | 最大文件大小：50MB | 不允许重复上传"
        )

    with col2:
        st.write("")
        st.write("")
        if uploaded_file and st.button("上传并处理", type="primary"):
            with st.spinner("正在处理文档..."):
                try:
                    result = api_client.upload_document_from_uploaded(uploaded_file, uploaded_file.name)
                    if result["status"] == "success":
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result["message"])
                except Exception as e:
                    st.error(f"上传失败: {e}")

    # 显示支持的格式
    with st.expander("支持的格式说明"):
        st.markdown(f"""
        ### 上传限制
        - **文件大小**：单个文件最大 50MB
        - **重复检测**：基于内容哈希检测，不允许上传相同内容的文件
        
        ### 支持的格式
        - **PDF**: 医学指南、诊疗手册等
        - **Word**: 文档资料
        - **TXT**: 纯文本格式的医疗知识
        - **HTML**: 网页格式的医学资料
        - **Markdown**: Markdown格式的医学笔记
        """)

    st.markdown("---")

    # 文档列表
    st.subheader("📁 已上传文档")

    try:
        documents = api_client.list_documents()
    except Exception as e:
        st.error(f"获取文档列表失败: {e}")
        documents = []

    if not documents:
        st.info("暂无上传的文档，请先上传医疗文档。")
    else:
        # 显示文档表格
        for doc in documents:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"📄 **{doc.get('name', doc.get('file_name', 'Unknown'))}**")
                with col2:
                    st.write(f"类型: {doc.get('type', 'N/A')}")
                with col3:
                    st.write(f"大小: {doc.get('size', 'N/A')}")
                with col4:
                    doc_id = doc.get('doc_id') or doc.get('name') or doc.get('file_name')
                    if doc_id and st.button("删除", key=f"del_{doc_id}"):
                        try:
                            result = api_client.delete_document(doc_id)
                            if result["status"] == "success":
                                st.success("删除成功")
                                st.rerun()
                            else:
                                st.error(result["message"])
                        except Exception as e:
                            st.error(f"删除失败: {e}")
                st.divider()

    st.markdown("---")

    # 知识库操作
    st.subheader("🔧 知识库操作")

    col1, col2, col3 = st.columns(3)

    with col1:
        # 使用session state跟踪重建状态
        if 'rebuild_active' not in st.session_state:
            st.session_state.rebuild_active = False
        if 'rebuild_status' not in st.session_state:
            st.session_state.rebuild_status = None
        
        if st.button("🔄 重建索引", use_container_width=True):
            with st.spinner("正在启动重建任务..."):
                try:
                    result = api_client.rebuild_index()
                    if result.get("status") in ["started", "success"]:
                        st.session_state.rebuild_active = True
                        st.session_state.rebuild_status = result
                        st.rerun()
                    else:
                        st.error(result.get("message", "重建失败"))
                except Exception as e:
                    st.error(f"重建索引失败: {e}")
        
        # 如果有正在进行的重建任务，显示状态
        if st.session_state.rebuild_active:
            try:
                status = api_client.get_rebuild_status()
                if status.get("status") == "running":
                    progress = status.get("progress", 0)
                    total = status.get("total", 0)
                    message = status.get("message", "处理中...")
                    st.info(f"{message} ({progress}/{total})")
                    # 性能优化：使用手动刷新按钮替代阻塞式sleep
                    col_refresh1, col_refresh2 = st.columns([3, 1])
                    with col_refresh1:
                        st.caption("点击刷新查看最新状态...")
                    with col_refresh2:
                        if st.button("🔄 刷新", key="refresh_status"):
                            st.rerun()
                elif status.get("status") == "completed":
                    st.success(status.get("message", "重建完成"))
                    st.session_state.rebuild_active = False
                    st.session_state.rebuild_status = None
                    st.rerun()
                elif status.get("status") == "failed":
                    st.error(f"重建失败: {status.get('error', '未知错误')}")
                    st.session_state.rebuild_active = False
                    st.session_state.rebuild_status = None
                elif status.get("status") == "idle":
                    st.session_state.rebuild_active = False
            except Exception as e:
                st.warning(f"获取状态失败: {e}")
                st.session_state.rebuild_active = False

    with col2:
        # 用户体验：添加确认对话框
        if "show_clear_confirm" not in st.session_state:
            st.session_state.show_clear_confirm = False
        
        if st.session_state.show_clear_confirm:
            st.warning("⚠️ 确定要清空知识库吗？此操作不可恢复！")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ 确认清空", type="primary", use_container_width=True):
                    with st.spinner("正在清空..."):
                        try:
                            result = api_client.clear_knowledge_base()
                            if result["status"] == "success":
                                st.success(result["message"])
                                st.session_state.show_clear_confirm = False
                                st.rerun()
                            else:
                                st.error(result["message"])
                        except Exception as e:
                            st.error(f"清空失败: {e}")
            with col_no:
                if st.button("❌ 取消", use_container_width=True):
                    st.session_state.show_clear_confirm = False
                    st.rerun()
        else:
            if st.button("🗑️ 清空知识库", use_container_width=True):
                st.session_state.show_clear_confirm = True
                st.rerun()

    with col3:
        if st.button("🔁 刷新页面", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # 使用说明
    with st.expander("📖 使用说明"):
        st.markdown("""
        ### 知识库管理说明

        1. **上传文档**：点击"选择医疗文档"按钮，选择要上传的文件。支持PDF、Word、TXT、HTML、Markdown等格式。

        2. **文档处理**：上传后，系统会自动解析文档内容，进行文本分块，并向量化存储到知识库中。

        3. **重建索引**：如果知识库出现问题，可以点击"重建索引"来重新构建整个知识库。

        4. **清空知识库**：会删除所有已上传的文档和索引，请谨慎操作。

        ### 建议上传的文档类型

        - 常见疾病诊疗指南
        - 药物说明书
        - 医学科普文章
        - 健康养生知识
        - 检查报告解读指南
        """)


if __name__ == "__main__":
    main()
