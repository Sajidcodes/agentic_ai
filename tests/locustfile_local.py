from locust import HttpUser, task, between
import random

class LocalChatUser(HttpUser):
    wait_time = between(2, 5)
    
    # Test questions across your 11 PDFs
    questions = [
        # Economics
        "What is economics?",
        "Explain supply and demand",
        "What is microeconomics vs macroeconomics?",
        # Anatomy
        "What is human anatomy?",
        "Describe the muscular system",
        "What are the major organs?",
        # Political Science
        "What is political science?",
        "Explain democracy",
        # Business
        "What is entrepreneurship?",
        "Explain market competition",
        # Sociology
        "What is sociology?",
        "Explain social stratification",
        # Machine Learning (d2l, cs229, mml)
        "What is machine learning?",
        "Explain neural networks",
        "What is gradient descent?",
        # Data Science
        "What is data science?",
        "Explain statistical inference",
        # Linear Algebra (LADR)
        "What is a vector space?",
        "Explain linear transformations",
    ]
    
    @task
    def load_page(self):
        """Test page load"""
        self.client.get("/")
    
    @task
    def health_check(self):
        """Test Streamlit health endpoint"""
        self.client.get("/_stcore/health")