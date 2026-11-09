import streamlit as st
import asyncio
from main import rag_query

# Page configuration
st.set_page_config(
    page_title="RAG Query Interface",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 RAG Query Interface")
st.markdown("Ask questions about your documents with optional web grounding")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Collection input
    collection = st.text_input(
        "Collection Name",
        value="karan",
        help="Name of the document collection to query"
    )
    
    # Web grounding toggle
    use_grounding = st.checkbox(
        "Enable Web Grounding",
        value=False,
        help="Use web search to supplement answers with real-time information"
    )
    
    st.divider()
    st.info("💡 Using smart defaults for optimal results")

# Main query interface
query = st.text_area(
    "Enter your question:",
    placeholder="How many members oppose a 50 bps cut like Miran?",
    height=100,
    key="query_input"
)

# Submit button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    submit_button = st.button("🔍 Search", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

# Process query
if submit_button:
    if not query.strip():
        st.error("⚠️ Please enter a question")
    elif not collection.strip():
        st.error("⚠️ Please enter a collection name")
    else:
        with st.spinner("🔎 Searching and generating answer..."):
            try:
                # Call the rag_query function directly from main.py
                result = asyncio.run(rag_query(
                    query=query,
                    collection=collection,
                    top_k=None,  # Use smart defaults
                    retrieve_k=None,
                    rerank_k=None,
                    use_reranking=None,
                    temperature=None,
                    use_grounding=use_grounding
                ))
                
                # Display answer
                st.success("✅ Answer Generated")
                st.markdown("### 📝 Answer")
                st.markdown(result.get("answer", "No answer generated"))
                
                # Display sources
                st.divider()
                
                # Sources
                st.markdown("### 📚 Sources")
                sources = result.get("sources", [])
                
                if sources:
                    for source in sources:
                        with st.expander(
                            f"Source {source['source_number']}: {source['file']} (Chunk {source['chunk_index']}) - "
                            f"Score: {source['similarity_score']:.3f}"
                        ):
                            st.text(source.get('text_preview', 'No preview available'))
                else:
                    st.info("No sources found")
                
                # Grounding info (if available)
                if "grounding" in result:
                    grounding = result["grounding"]
                    st.divider()
                    st.markdown("### 🌐 Web Sources")
                    
                    chunks = grounding.get("grounding_chunks", [])
                    if chunks:
                        for i, chunk in enumerate(chunks, 1):
                            if chunk.get('uri'):
                                st.markdown(f"{i}. [{chunk.get('title', 'Link')}]({chunk['uri']})")
                    else:
                        st.info("No web sources were used for this query")
                    
                    queries = grounding.get("web_search_queries", [])
                    if queries:
                        st.markdown("**Search Queries Used:**")
                        for q in queries:
                            st.text(f"• {q}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("📋 Error Details"):
                    st.code(traceback.format_exc())
