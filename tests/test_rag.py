from rag import generate_response

def test_generate_response():
    class MockCompletions:
        def create(self, **kwargs):
            return type("Response", (), {"choices": [type("Choice", (), {"message": "Mock response"})]})()

    class MockChat:
        def __init__(self):
            self.completions = MockCompletions()

    class MockClient:
        def __init__(self):
            self.chat = MockChat()

    mock_client = MockClient()
    result = generate_response("What is AI?", ["AI is artificial intelligence."], mock_client)
    assert "Mock response" in str(result)