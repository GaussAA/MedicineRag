"""知识库管理页面"""

import streamlit as st

from backend.config import config
from backend.services.doc_service import DocService
from backend.services.security_service import SecurityService
from rag.engine import RAGEngine


# 页面配置
st.set_page_config(
    page_title="知识库管理 - 医疗知识问答系统",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def init_services():
    """初始化服务（单例）"""
    rag_engine = RAGEngine()
    security_service = SecurityService()
    doc_service = DocService(rag_engine)
    return doc_service, rag_engine


def main():
    """主函数"""
    st.title("📚 知识库管理")
    st.markdown("---")

    # 初始化服务
    doc_service, rag_engine = init_services()

    # 侧边栏导航
    with st.sidebar:
        st.header("导航")
        st.page_link("main.py", label="💬 问答咨询", icon="💬")
        st.page_link("pages/knowledge.py", label="📚 知识库管理", icon="📚", disabled=True)

    # 显示统计信息
    st.subheader("📊 知识库统计")
    stats = doc_service.get_stats()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("文档数量", stats['document_count'])
    with col2:
        st.metric("索引块数", stats['indexed_chunks'])
    with col3:
        st.metric("总大小", stats['total_size'])

    st.markdown("---")

    # 上传文档
    st.subheader("📤 上传文档")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "选择医疗文档",
            type=["pdf", "docx", "txt", "html", "md"],
            help="支持的格式：PDF、Word、TXT、HTML、Markdown"
        )

    with col2:
        st.write("")
        st.write("")
        if uploaded_file and st.button("上传并处理", type="primary"):
            with st.spinner("正在处理文档..."):
                result = doc_service.upload_document(uploaded_file, uploaded_file.name)
                if result["status"] == "success":
                    st.success(result["message"])
                    st.rerun()
                else:
                    st.error(result["message"])

    # 显示支持的格式
    with st.expander("支持的格式说明"):
        st.markdown("""
        - **PDF**: 医学指南、诊疗手册等
        - **Word**: 文档资料
        - **TXT**: 纯文本格式的医疗知识
        - **HTML**: 网页格式的医学资料
        - **Markdown**: Markdown格式的医学笔记
        """)

    st.markdown("---")

    # 文档列表
    st.subheader("📁 已上传文档")

    documents = doc_service.list_documents()

    if not documents:
        st.info("暂无上传的文档，请先上传医疗文档。")
    else:
        # 显示文档表格
        for doc in documents:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"📄 **{doc['name']}**")
                with col2:
                    st.write(f"类型: {doc['type']}")
                with col3:
                    st.write(f"大小: {doc['size']}")
                with col4:
                    if st.button("删除", key=f"del_{doc['name']}"):
                        result = doc_service.delete_document(doc['name'])
                        if result["status"] == "success":
                            st.success("删除成功")
                            st.rerun()
                        else:
                            st.error(result["message"])
                st.divider()

    st.markdown("---")

    # 知识库操作
    st.subheader("🔧 知识库操作")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 重建索引", use_container_width=True):
            with st.spinner("正在重建索引..."):
                result = doc_service.rebuild_index()
                if result["status"] == "success":
                    st.success(result["message"])
                    st.rerun()
                else:
                    st.error(result["message"])

    with col2:
        if st.button("🗑️ 清空知识库", use_container_width=True):
            with st.spinner("正在清空..."):
                result = doc_service.clear_knowledge_base()
                if result["status"] == "success":
                    st.success(result["message"])
                    st.rerun()
                else:
                    st.error(result["message"])

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
