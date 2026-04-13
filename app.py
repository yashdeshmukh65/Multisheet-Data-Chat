import streamlit as st
import pandas as pd
import os
from data_loader import load_excel_to_sqlite, get_db_schema
from sql_executor import execute_sql
from visualizer import generate_visualization
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from memory import Memory
from agents import AgentPipeline

st.set_page_config(page_title="Excel Data Chat", layout="wide")

st.title("📊 Multi-Sheet Excel Data Chat System")

# Initialize Session State
if "memory" not in st.session_state:
    st.session_state.memory = Memory()
if "db_schema" not in st.session_state:
    st.session_state.db_schema = ""

# Sidebar settings
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    
    if uploaded_file and st.button("Load File"):
        with st.spinner("Loading file and building database..."):
            schema = load_excel_to_sqlite(uploaded_file)
            st.session_state.db_schema = schema
            st.success("File loaded successfully!")
            with st.expander("Database Schema"):
                st.text(schema)

# Main Chat Interface
if not st.session_state.db_schema:
    st.info("Please upload an Excel file to begin.")
else:
    # Display chat history
    for msg in st.session_state.memory.history:
        with st.chat_message("user"):
            st.markdown(msg["user_query"])
        with st.chat_message("assistant"):
            st.markdown(f"**SQL Executed:**\n```sql\n{msg['sql_query']}\n```")
            st.markdown(msg["ai_response"])
            
    # Input box
    if prompt := st.chat_input("Ask a question about your data..."):
        # 1. Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            # Load Azure configurations from .env
            api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            api_version = "2023-05-15"
            deployment_model = "gpt-4"
            
            agent = AgentPipeline(api_key, endpoint, api_version, deployment_model)
            context = st.session_state.memory.get_context_string()
            
            with st.spinner("Generating SQL query..."):
                sql_query, chart_suggestion = agent.run_query(st.session_state.db_schema, prompt, context)
            
            if not sql_query:
                st.error("Could not generate a valid SQL query.")
            else:
                st.markdown(f"**Generated SQL:**\n```sql\n{sql_query}\n```")
                
                with st.spinner("Executing query..."):
                    success, df = execute_sql(sql_query)
                    
                if not success:
                    st.error(f"Execution failed: {df}")
                else:
                    st.dataframe(df)
                    
                    # Visualization
                    if len(df) > 0 and chart_suggestion and chart_suggestion != 'none':
                        with st.spinner("Generating visualization..."):
                            fig = generate_visualization(df, chart_suggestion)
                            if fig:
                                st.pyplot(fig)
                                
                    # Explanation
                    with st.spinner("Generating explanation..."):
                        # Only pass first few rows if large
                        results_str = df.head(10).to_string() 
                        explanation = agent.generate_explanation(prompt, sql_query, results_str)
                        st.markdown(explanation)
                        
                    # Add to memory
                    st.session_state.memory.add_interaction(prompt, sql_query, explanation)
