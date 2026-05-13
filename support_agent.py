import os
from langchain.tools import tool
# CHANGE 1: Import ChatGroq instead of ChatOpenAI
from langchain_groq import ChatGroq 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph
from dotenv import load_dotenv

# This loads the variables from .env into your environment
load_dotenv()

# 1) Define our business tool
@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order that hasn't shipped."""
    return f"Order {order_id} has been cancelled."

# 2) The agent "brain"
def call_model(state):
    msgs = state["messages"]
    order = state.get("order", {"order_id": "UNKNOWN"})
    
    prompt = (
        f"You are an ecommerce support agent.\n"
        f"ORDER ID: {order['order_id']}\n"
        f"If the customer asks to cancel, call cancel_order(order_id)\n"
        f"and then send a simple confirmation.\n"
        f"Otherwise, just respond normally."
    )
    
    full_messages = [SystemMessage(content=prompt)] + msgs
    
    # CHANGE 2: Initialize Groq using the Llama 3 8B model (which is fast and free)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    llm_with_tools = llm.bind_tools([cancel_order])
    
    first = llm_with_tools.invoke(full_messages)
    out = [first]
    
    if getattr(first, "tool_calls", None):
        tc = first.tool_calls[0]
        result = cancel_order.invoke(tc["args"]) 
        out.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        
        second = llm_with_tools.invoke(full_messages + out)
        out.append(second)
        
    return {"messages": out}

# 3) Wire it all up
def construct_graph():
    g = StateGraph(dict) 
    g.add_node("assistant", call_model)
    g.set_entry_point("assistant")
    return g.compile()

# --- Testing our Agent (Interactive Chat) ---
if __name__ == "__main__":
    graph = construct_graph()
    
    # Define our test context
    example_order = {"order_id": "A12345"}
    
    print("Welcome to Customer Support! (Type 'quit' to exit)")
    print("---------------------------------------------------")
    
    while True:
        # 1. Get input from the user
        user_input = input("\nYou: ")
        
        # 2. Check if the user wants to quit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Ending chat. Goodbye!")
            break
            
        # 3. Send the message to the agent
        convo = [HumanMessage(content=user_input)]
        result = graph.invoke({"order": example_order, "messages": convo})
        
        # 4. Print only the final response from the agent
        final_response = result["messages"][-1].content
        print(f"Agent: {final_response}")