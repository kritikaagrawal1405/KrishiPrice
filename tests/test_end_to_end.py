import os
import sys

# Ensure the project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.orchestration.graph import MultiAgentOrchestrator

def run_tests():
    print("Initializing Orchestrator...")
    orchestrator = MultiAgentOrchestrator()
    
    queries = [
        "What is the price of Potato in Tulsipur market tomorrow?",
        "Will there be an alert for Potato in Tulsipur next week?"
    ]
    
    for query in queries:
        print(f"\n--- Testing Query: '{query}' ---")
        state = orchestrator.process_query(query)
        
        print("Intent:", state.intent)
        print("Entities:", state.entities.dict() if state.entities else None)
        
        if state.errors:
            print("Errors encountered:", state.errors)
        
        print("Final Response:", state.final_response)
        
        if state.alert_decision:
            print("Alert Generated:", state.alert_decision.alert)
            if state.alert_decision.alert:
                print("Alert Message:", state.alert_decision.message)

if __name__ == "__main__":
    run_tests()
