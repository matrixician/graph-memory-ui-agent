import os
import time
from dotenv import load_dotenv

# Import our custom modules
from memory_graph import GraphMemory
from web_scraper import LookupEngine
from vision_engine import VisionEngine

load_dotenv()


print("[System] Booting Cognitive-UI Orchestrator...")

# Init Neo4j 
memory = GraphMemory()

# Qwen2.5-VL 
vision = VisionEngine()


lookup = LookupEngine(vlm_client=None) 


def execute_action(action_data):
    """
    Translates the JSON output from the VLM into physical computer actions.
    In a full build, this hooks into PyAutoGUI, Playwright, or X11.
    """
    action = action_data.get('action_type', 'unknown')
    target = action_data.get('target_box_id', action_data.get('target', 'unknown'))
    
    print(f"\n[Execution] ----------------------------------")
    print(f"[Execution] Firing Action:  {action.upper()}")
    print(f"[Execution] Target Element: Box #{target}")
    print(f"[Execution] Reasoning:      {action_data.get('reasoning', 'Retrieved from memory')}")
    print(f"[Execution] ----------------------------------\n")
    
    # Simulate the time it takes for the UI to load after a click
    time.sleep(2) 

def run_agent(goal, software_name):
    """
    The main state-machine loop: Observe -> Recall -> Learn (if needed) -> Act -> Memorize
    """
    print(f"\n=== NEW TASK: {goal} ===")
    
    # STEP 1: Observe Current Screen
    # In reality, this triggers a screenshot and Set-of-Mark bounding box script.
    current_image = "current_screen_som.jpg" 
    
    # STEP 2: Identify State
    state_a_desc = vision.get_screen_description(current_image)
    print(f"[State] Current Screen identified as: {state_a_desc}")
    
    # STEP 3: Recall (Check Graph Memory)
    print("[Memory] Checking neural graph for existing pathways...")
    known_action = memory.get_known_action(state_a_desc, goal)
    
    if known_action:
        print("[Memory] Pathway found! Executing via muscle memory.")
        execute_action(known_action)
    else:
        print("[Memory] Unknown state/goal combination. Initiating learning protocol.")
        
        # STEP 4: Learn (Web Scraping / Documentation Lookup)
        learned_context = lookup.search_how_to(goal, software_name)
        
        # STEP 5: Analyze & Ground (Translate text instructions to visual UI targets)
        action_data = vision.analyze_screen(current_image, goal, context=learned_context)
        
        # Check for parse errors from the local VLM
        if action_data.get("action_type") == "error":
            print("[System Error] Agent failed to understand the screen or parse the output. Aborting.")
            return

        # STEP 6: Execute newly learned action
        execute_action(action_data)
        
        # STEP 7: Observe Resulting State
        # Trigger another screenshot to see where the action took us
        new_image = "next_screen_som.jpg"
        state_b_desc = vision.get_screen_description(new_image)
        print(f"[State] Resulting Screen identified as: {state_b_desc}")
        
        # STEP 8: Consolidate Memory (Write to GraphRAG)
        print("[Memory] Consolidating new pathway to graph database...")
        memory.save_memory(state_a_desc, action_data, goal, state_b_desc)

    print("=== TASK CYCLE COMPLETE ===\n")


if __name__ == "__main__":
    # Define the task parameters
    USER_GOAL = "Export user data as CSV"
    SOFTWARE = "Salesforce CRM"
    
    try:
        # Run the agent pipeline
        run_agent(USER_GOAL, SOFTWARE)
    except KeyboardInterrupt:
        print("\n[System] Agent forcefully stopped by user.")
    except Exception as e:
        print(f"\n[System Error] Critical failure: {e}")
    finally:
        # Always ensure the database connection is closed safely
        print("[System] Shutting down memory connections...")
        memory.close()
        print("[System] Offline.")