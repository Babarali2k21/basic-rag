from data_loader import load_documents_from_directory
import os

def test_load_documents_from_directory(tmp_path):
    file_path = tmp_path / "test_doc.txt"
    file_path.write_text("This is a test document.")
    
    docs = load_documents_from_directory(tmp_path)
    
    assert len(docs) == 1
    assert docs[0]["text"] == "This is a test document."