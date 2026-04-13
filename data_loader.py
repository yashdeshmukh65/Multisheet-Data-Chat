import pandas as pd
import sqlite3
import os

DB_PATH = "chat_app.db"

def load_excel_to_sqlite(file_path_or_bytes):
    """
    Reads an Excel file, converts each sheet to a sanitized SQLite table, 
    and returns a string describing the database schema.
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Read all sheets
    xls = pd.read_excel(file_path_or_bytes, sheet_name=None, engine='openpyxl')
    schema_info = []
    
    for sheet_name, df in xls.items():
        # Sanitize table names (no spaces, no minus signs)
        table_name = str(sheet_name).replace(" ", "_").replace("-", "_")
        
        # Sanitize column names
        df.columns = [str(c).replace(" ", "_").replace("-", "_") for c in df.columns]
        
        # Save to sqlite
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        
        # Build schema info
        columns = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
        schema_info.append(f"Table: {table_name}\nColumns: {columns}")

    conn.close()
    return "\n\n".join(schema_info)

def get_db_schema():
    """
    Retrieves the current schema for all tables in the SQLite database.
    """
    if not os.path.exists(DB_PATH):
        return "No database loaded yet."
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_info = []
    for table_name in tables:
        table_name = table_name[0]
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1", conn)
        columns = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
        schema_info.append(f"Table: {table_name}\nColumns: {columns}")
        
    conn.close()
    return "\n\n".join(schema_info)
