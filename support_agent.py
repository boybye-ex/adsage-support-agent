import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph

# Load variables from .env
load_dotenv() 

# --- 1) Define our business tools ---

@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order that hasn't shipped."""
    return f"Order {order_id} has been cancelled."

@tool
def check_order_status(order_id: str) -> str:
    """Check the current shipping status of an order."""
    # A tiny fake database for testing
    simulated_db = {
        "A12345": "Processing - Not shipped yet.",
        "B98765": "Shipped - Out for delivery.",
        "C11111": "Delivered yesterday."
    }
    return simulated_db.get(order_id, "Status unknown. Order not found.")

# Create a dictionary so we can easily look up the right tool to run later
tools_dict = {
    "cancel_order": cancel_order,
    "check_order_status": check_order_status
}

# --- 2) The agent "brain" ---
def call_model(state):
    msgs = state["messages"]
    order = state.get("order", {"order_id": "UNKNOWN"})
    
    # Updated Prompt: Now tells the agent about checking statuses!
    prompt = (
        f"You are a helpful ecommerce support agent.\n"
        f"The user's current ORDER ID is: {order['order_id']}\n"
        f"If the customer asks to cancel, call cancel_order(order_id).\n"
        f"If the customer asks for the status or where their package is, call check_order_status(order_id).\n"
        f"Otherwise, just respond normally and politely."
    )
    
    full_messages = [SystemMessage(content=prompt)] + msgs
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    # Update: Bind BOTH tools to the LLM
    llm_with_tools = llm.bind_tools([cancel_order, check_order_status])
    
    first = llm_with_tools.invoke(full_messages)
    out = [first]
    
    if getattr(first, "tool_calls", None):
        # The LLM decided to call a tool!
        tc = first.tool_calls[0]
        tool_name = tc["name"]
        tool_args = tc["args"]
        
        # Look up the tool in our dictionary and execute it
        print(f"\n[System: Running backend tool '{tool_name}'...]")
        selected_tool = tools_dict[tool_name]
        result = selected_tool.invoke(tool_args) 
        
        out.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        
        # 2nd pass: Let the LLM read the tool result and respond to the user
        second = llm_with_tools.invoke(full_messages + out)
        out.append(second)
        
    return {"messages": out}

# --- 3) Wire it all up ---
def construct_graph():
    g = StateGraph(dict) 
    g.add_node("assistant", call_model)
    g.set_entry_point("assistant")
    return g.compile()

# --- Testing our Agent (Interactive Chat) ---
if __name__ == "__main__":
    graph = construct_graph()
    
    # Let's test with order B98765, which is already "Shipped"
    example_order = {"order_id": "B98765"}
    
    print("Welcome to Customer Support! (Type 'quit' to exit)")
    print("---------------------------------------------------")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Ending chat. Goodbye!")
            break
            
        convo = [HumanMessage(content=user_input)]
        result = graph.invoke({"order": example_order, "messages": convo})
        
        final_response = result["messages"][-1].content
        print(f"Agent: {final_response}")