from locust import HttpUser, task, between
import random

class ChatUser(HttpUser):
    wait_time = between(2, 5)  # Wait 2-5 seconds between requests (realistic typing)
    
    questions = [
        "What is economics?",
        "Tell me about microeconomics",
        "What is anatomy?",
        "Explain machine learning",
        "What is political science?",
        "How does supply and demand work?",
        "What are the parts of a cell?",
    ]
    
    @task
    def chat(self):
        # Simulate a chat message via Streamlit's websocket
        # Note: Streamlit uses websockets, so HTTP testing is limited
        # This tests if the server can handle concurrent connections
        self.client.get("/")
    
    @task(3)  # 3x more likely to run this
    def health_check(self):
        self.client.get("/_stcore/health")