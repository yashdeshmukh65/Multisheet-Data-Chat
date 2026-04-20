import matplotlib.pyplot as plt
import pandas as pd

def generate_visualization(df, chart_suggestion):
    """
    Generates a matplotlib figure based on the dataframe and chart_suggestion.
    chart_suggestion is usually "line", "bar", "pie", or "none".
    """
    if df.empty or len(df.columns) < 2:
        return None
        
    chart_type = chart_suggestion.lower()
    if 'none' in chart_type:
        return None
        
    fig, ax = plt.subplots(figsize=(8, 5))
    
    try:
        x_col = df.columns[0]
        # if only one column available for Y, use it. Else try to use the second column.
        y_cols = df.columns[1:]
        
        if 'line' in chart_type:
            for y_col in y_cols:
                ax.plot(df[x_col], df[y_col], marker='o', label=y_col)
            ax.set_xlabel(x_col)
            ax.set_ylabel("Values")
            ax.set_title(f"Line Chart Grouped by {x_col}")
            plt.xticks(rotation=45)
            if len(y_cols) > 1:
                ax.legend()
                
        elif 'pie' in chart_type:
            y_col = y_cols[0] # Pie charts typically only take one value column
            # Use .abs() to prevent ValueError crashes if LLM suggests a pie chart for data with negative values
            ax.pie(df[y_col].abs(), labels=df[x_col], autopct='%1.1f%%', startangle=90)
            ax.set_title(f"Distribution by {x_col}")
            
        else: # Default to Bar Chart
            for y_col in y_cols:
                ax.bar(df[x_col], df[y_col], label=y_col)
            ax.set_xlabel(x_col)
            ax.set_ylabel("Values")
            ax.set_title(f"Bar Chart Grouped by {x_col}")
            plt.xticks(rotation=45)
            if len(y_cols) > 1:
                ax.legend()
                
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Visualization error: {e}")
        return None
