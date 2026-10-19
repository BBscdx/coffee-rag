import streamlit as st
import rag

st.set_page_config(page_title="咖啡产业知识问答助手", page_icon="☕")
st.title("☕ 咖啡产业知识问答助手")
st.caption("先检索、再作答：回答必须引用资料，数字必须来自知识库")

st.info("""
📌 **知识库说明**：
- 当前知识库仅收录 **2023-2026 年** 的咖啡产业资料，共计 **11 篇**（涵盖官方答复、产业报告、海关数据、调研纪要、新闻报道）。
- 每篇资料均明确标注了**来源与日期**（发布/统计时点 2023-2026）。
""")

question = st.text_input("输入你的问题", placeholder="例如：2023 年云南省咖啡种植面积多少？")
top_k = st.slider("检索条数", 1, 8, 5)

if st.button("提问", type="primary"):
    if not question.strip():
        st.warning("先输入一个问题")
        st.stop()

    with st.spinner("检索知识库..."):
        results = rag.search(question, top_k)

    if not results:
        st.error("未检索到相关知识块，换个问法试试。")
        st.stop()

    st.subheader("🔍 检索到的知识块")
    for sim, idx in results:
        with st.expander(f"[{sim:.2f}] {rag.chunks[idx]['title']} · {rag.chunks[idx]['doc_name']}"):
            st.write(rag.chunks[idx]['body'][:600])
    st.subheader("🤖 AI 回答")
    with st.spinner("DeepSeek 生成中..."):
        answer, status = rag.ask(question, top_k)
    if status == "ok":
        st.write(answer)
    elif status == "no_match":
        st.info("知识库中没有相关资料，已拒答——不编造是设计使然。换个问法试试。")
    else:   # api_error
        st.error("⚠️ 回答生成失败：DeepSeek API 余额用尽或网络问题"
                "（演示期作者的 API 未续费时会出现此提示）。这不是系统故障——"
                "上方的检索留痕仍然有效，知识库里的原始内容可以正常查看。")