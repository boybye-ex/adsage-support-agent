import sqlite3
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq 
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph

# Load variables from .env
load_dotenv() 

# --- 0) Database Setup ---
DB_FILE = "ecommerce.db"

def init_db():
    """Initialize a local SQLite database with some dummy data if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            status TEXT
        )
    """)
    # Insert some default records if the table is empty
    cursor.execute("SELECT count(*) FROM orders")
    if cursor.fetchone()[0] == 0:
        sample_orders = [
            ("A12345", "Processing - Not shipped yet."),
            ("B98765", "Shipped - Out for delivery."),
            ("C11111", "Delivered yesterday.")
        ]
        cursor.executemany("INSERT INTO orders VALUES (?, ?)", sample_orders)
        conn.commit()
    conn.close()

# Run DB initialization on startup
init_db()

# --- 1) Define our business tools ---

@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order that hasn't shipped."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Check if order exists first
    cursor.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return f"Order {order_id} not found."
        
    if "Shipped" in row[0] or "Delivered" in row[0]:
        conn.close()
        return f"Cannot cancel order {order_id} because it has already shipped or been delivered."
        
    # Update the status to Cancelled
    cursor.execute("UPDATE orders SET status = 'Cancelled' WHERE order_id = ?", (order_id,))
    conn.commit()
    conn.close()
    
    return f"Order {order_id} has been successfully cancelled in the database."

@tool
def check_order_status(order_id: str) -> str:
    """Check the current shipping status of an order."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return f"Status for {order_id}: {row[0]}"
    return f"Status unknown. Order {order_id} not found."

@tool
def escalate_to_human(summary: str) -> str:
    """Escalate the conversation to a human agent. Use this if the user is angry, asks for a human, or has an issue you can't solve."""
    return f"Ticket successfully created for human support team. (Summary: {summary})"

tools_dict = {
    "cancel_order": cancel_order,
    "check_order_status": check_order_status,
    "escalate_to_human": escalate_to_human
}

# --- 2) The agent "brain" ---
def call_model(state):
    msgs = state["messages"]
    order = state.get("order", {"order_id": "UNKNOWN"})
    
    prompt = (
        f"You are a helpful ecommerce support agent.\n"
        f"The user's current ORDER ID is: {order['order_id']}. DO NOT ask the user for their order ID, use this one automatically.\n"
        f"RULES:\n"
        f"1. If they ask to cancel, call cancel_order(order_id). DO NOT use this tool for returns.\n"
        f"2. If they ask for shipping status, call check_order_status(order_id).\n"
        f"3. If they are angry, ask for a real person, ask to RETURN an item, or ask something you cannot do, call escalate_to_human(summary).\n"
        f"4. After calling a tool, you MUST reply to the user with the exact outcome of the tool."
    )
    
    full_messages = [SystemMessage(content=prompt)] + msgs
    
    # Upgrade to the 70 Billion parameter model for better tool reliability!
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    llm_with_tools = llm.bind_tools([cancel_order, check_order_status, escalate_to_human])
    
    first = llm_with_tools.invoke(full_messages)
    new_messages = [first]
    
    if getattr(first, "tool_calls", None):
        # UPGRADE: Loop through ALL tool calls requested by the LLM
        for tc in first.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            
            print(f"\n[System: Running backend tool '{tool_name}'...]")
            selected_tool = tools_dict[tool_name]
            result = selected_tool.invoke(tool_args) 
            
            # Append each tool result sequentially
            new_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            
        # 2nd pass: Let the LLM read all tool results and respond
        second = llm_with_tools.invoke(full_messages + new_messages)
        new_messages.append(second)
        
    return {"messages": msgs + new_messages}

# --- 3) Wire it all up ---
def construct_graph():
    g = StateGraph(dict) 
    g.add_node("assistant", call_model)
    g.set_entry_point("assistant")
    return g.compile()

# --- Testing our Agent (Interactive Chat) ---
if __name__ == "__main__":
    graph = construct_graph()
    
    # We will use A12345 so we can successfully test a cancellation!
    example_order = {"order_id": "A12345"}
    
    print("Welcome to Customer Support! (Type 'quit' to exit)")
    print("---------------------------------------------------")
    
    chat_history = []
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Ending chat. Goodbye!")
            break
            
        chat_history.append(HumanMessage(content=user_input))
        result = graph.invoke({"order": example_order, "messages": chat_history})
        chat_history = result["messages"]
        
        final_response = chat_history[-1].content
        print(f"Agent: {final_response}")