class Memory:
    def __init__(self):
        self.history = []
        
    def add_interaction(self, user_query, sql_query, ai_response):
        self.history.append({
            "user_query": user_query,
            "sql_query": sql_query,
            "ai_response": ai_response
        })
        
    def get_context_string(self):
        if not self.history:
            return "No previous context."
            
        context_str = "Conversation History:\n"
        for i, turn in enumerate(self.history[-3:]): # keep last 3 turns
            context_str += f"Turn {i+1}:\n"
            context_str += f"User: {turn['user_query']}\n"
            context_str += f"SQL Executed: {turn['sql_query']}\n"
            context_str += f"System: {turn['ai_response']}\n\n"
        return context_str
