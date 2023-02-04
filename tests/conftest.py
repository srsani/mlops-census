import pytest
from fastapi.testclient import TestClient

from src.main_api import app

import sys
sys.path.append('../..')


@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client
