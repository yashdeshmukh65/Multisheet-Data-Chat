import sqlite3
import pandas as pd
import re

def execute_sql(query, db_path):
    """
    Executes a SQL query against the SQLite database and returns a Pandas DataFrame.
    Returns (success_boolean, result_data)
    """
    try:
        # Pre-process the query string to remove markdown formatting if the LLM adds it
        query = query.strip()
        if query.startswith("```sql"):
            query = query[6:]
        if query.endswith("```"):
            query = query[:-3]
            
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return True, df
    except Exception as e:
        return False, str(e)
