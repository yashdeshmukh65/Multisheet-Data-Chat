import pandas as pd
import sqlite3
import os
import re

def load_excel_to_sqlite(file_path_or_bytes, db_path):
    """
    Reads an Excel file, converts each sheet to a sanitized SQLite table, 
    and returns a string describing the database schema.
    """
    conn = sqlite3.connect(db_path)
    
    # Read all sheets (removed hardcoded openpyxl engine to allow auto-detection)
    xls = pd.read_excel(file_path_or_bytes, sheet_name=None)
    schema_info = []
    
    for sheet_name, df in xls.items():
        # Sanitize table names (only alphanumeric and underscores)
        table_name = re.sub(r'\W+', '_', str(sheet_name)).strip('_')
        if not table_name: table_name = "sheet_idx"
        
        # Sanitize column names (only alphanumeric and underscores)
        df.columns = [re.sub(r'\W+', '_', str(c)).strip('_') for c in df.columns]
        
        # Save to sqlite
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        
        # Build schema info
        columns = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
        schema_info.append(f"Table: {table_name}\nColumns: {columns}")

    conn.close()
    return schema_info

def get_db_schema(db_path):
    """
    Retrieves the current schema for all tables in the SQLite database.
    """
    if not os.path.exists(db_path):
        return "No database loaded yet."
        
    conn = sqlite3.connect(db_path)
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
    return schema_info
