from data_loader import split_text

def test_split_text():
    text = "Hello world! " * 100
    chunks = split_text(text, chunk_size=50, chunk_overlap=10)
    
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)