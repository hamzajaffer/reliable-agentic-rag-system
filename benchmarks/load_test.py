"""
Load Test — uses Locust for API load testing.
Run: locust -f benchmarks/load_test.py --host http://localhost:8000
"""

from locust import HttpUser, task, between


class RAGUser(HttpUser):
    """Simulates a user querying the RAG system."""
    
    wait_time = between(1, 3)
    
    @task(3)
    def query_rag(self):
        """Test the RAG query endpoint."""
        self.client.post(
            "/query",
            json={"query": "What is the main function?"}
        )
    
    @task(2)
    def query_agent(self):
        """Test the agent endpoint."""
        self.client.post(
            "/agent",
            json={"query": "Explain the authentication flow"}
        )
    
    @task(1)
    def health_check(self):
        """Test health endpoint."""
        self.client.get("/health")
