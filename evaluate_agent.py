from langchain_core.messages import HumanMessage
from support_agent import construct_graph, init_db
import sqlite3

def run_evaluation():
    # Ensure the database is fresh for testing
    init_db()
    
    # Compile the graph
    graph = construct_graph()
    
    # Define our test scenarios
    # Format: (User Message, Expected Tool to be Called)
    scenarios = [
        ("Can you tell me where my package is?", "check_order_status"),
        ("I changed my mind, please cancel my order.", "cancel_order"),
        ("Let me speak to a manager right now, this is terrible!", "escalate_to_human"),
        ("I want to return my order.", "escalate_to_human"), # Agent doesn't have a return tool, should escalate
        ("Hello, how are you today?", None) # Casual chat, shouldn't trigger a tool
    ]
    
    print("Starting Automated Agent Evaluation...\n")
    print("="*50)
    
    passed_tests = 0
    total_tests = len(scenarios)
    
    for i, (user_input, expected_tool) in enumerate(scenarios, 1):
        print(f"Test {i}: {user_input}")
        
        # Fresh context for each test
        example_order = {"order_id": "A12345"}
        convo = [HumanMessage(content=user_input)]
        
        # Invoke the agent
        result = graph.invoke({"order": example_order, "messages": convo})
        
        # We need to look at the AI's first response to see what tool it tried to call
        # In our graph, result["messages"] looks like: [Human, AI, Tool, AI]
        # We want to check the first AI message, which is index 1
        first_ai_message = result["messages"][1]
        
        # Determine which tool was actually called (if any)
        actual_tool = None
        if hasattr(first_ai_message, "tool_calls") and len(first_ai_message.tool_calls) > 0:
            actual_tool = first_ai_message.tool_calls[0]["name"]
            
        # Evaluate
        if actual_tool == expected_tool:
            print(f"✅ PASS: Expected '{expected_tool}', Got '{actual_tool}'")
            passed_tests += 1
        else:
            print(f"❌ FAIL: Expected '{expected_tool}', Got '{actual_tool}'")
            
        print("-" * 50)
        
    print(f"\nFinal Score: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100}%)")

if __name__ == "__main__":
    run_evaluation()