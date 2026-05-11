import pytest
import numpy as np
from textual_inversion.data.dataset import pad_embedding

def test_pad_embedding():
    embedding = [1, 2, 3]
    end_token = 49407
    max_len = 10
    
    padded = pad_embedding(embedding, end_token, max_len)
    
    assert len(padded) == max_len
    assert padded[:3] == [1, 2, 3]
    assert padded[3:] == [end_token] * 7
