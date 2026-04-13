from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
import json

class AgentPipeline:
    def __init__(self, api_key, endpoint, api_version, model="gpt-4"):
        self.llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_deployment=model,
            temperature=0
        )
        
    def run_query(self, database_schema, user_question, context):
        # Step 1: SQL Generation and Visualization Suggestion
        prompt = PromptTemplate.from_template("""
        You are an expert Data Analyst and SQL Generator.
        
        {context}
        
        Database Schema:
        {schema}
        
        User Question: {question}
        
        Generate a SQL query to answer the user's question. Use ONLY the provided tables and columns.
        Do NOT hallucinate schema. Use JOIN if needed. 
        Use sqlite syntax.
        
        Also, suggest a visualization type for the result based on these rules:
        - Time-based analysis -> 'line'
        - Comparison across categories -> 'bar'
        - Distribution/Percentage -> 'pie'
        - Just retrieving facts/numbers -> 'none'
        
        Return your answer ONLY as a valid JSON object with EXACTLY two keys:
        - "sql": the SQL query string
        - "chart_suggestion": the suggested chart type
        
        Example output format:
        {{
            "sql": "SELECT * FROM sales",
            "chart_suggestion": "none"
        }}
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "schema": database_schema,
                "question": user_question,
                "context": context
            })
            
            # clean json if formatted with markdown
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
                
            parsed = json.loads(response.strip())
            return parsed.get("sql", ""), parsed.get("chart_suggestion", "none")
            
        except Exception as e:
            print(f"Error in SQL generation: {e}")
            return "", "none"
            
    def generate_explanation(self, user_question, sql_query, datatable_string):
        prompt = PromptTemplate.from_template("""
        You are a helpful data assistant.
        
        User Question: {question}
        
        SQL Executed: 
        {sql}
        
        Query Results:
        {results}
        
        Explain the results to the user in a short, easy-to-understand response.
        Do NOT mention the SQL query itself in your explanation, just give the insights based on the Query Results.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "question": user_question,
            "sql": sql_query,
            "results": datatable_string
        })
