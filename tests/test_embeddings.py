from embeddings import get_openai_embedding

def test_embedding_structure():
    class MockEmbeddings:
        def create(self, **kwargs):
            return type("Response", (), {"data": [type("Embed", (), {"embedding": [0.1, 0.2, 0.3]})]})()

    class MockClient:
        def __init__(self):
            self.embeddings = MockEmbeddings()

    mock_client = MockClient()
    embedding = get_openai_embedding("test text", mock_client)
    assert isinstance(embedding, list)
    assert len(embedding) == 3