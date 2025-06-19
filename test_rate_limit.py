
import pytest
from fastapi.testclient import TestClient
from app.middleware.rate_limit import app

client = TestClient(app)

@pytest.mark.parametrize("endpoint,limit", [
    ("/general", 10),  # 10 requests per minute
    ("/restricted", 2)  # 2 requests per minute
])
def test_rate_limiting(endpoint, limit):
    for i in range(limit):
        response = client.get(endpoint)
        assert response.status_code == 200, f"Request {i+1} failed unexpectedly"

    # Exceed the limit
    response = client.get(endpoint)
    assert response.status_code == 429, "Rate limit not enforced"
    assert response.json() == {"error": "Rate limit exceeded"}