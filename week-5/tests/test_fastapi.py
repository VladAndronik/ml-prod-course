from pathlib import Path

from fastapi.testclient import TestClient

from serving.fast_api import app

client = TestClient(app)


def test_health():
    response = client.get('/check')
    assert response.status_code == 200
    assert response.json() == 'Okay'


def test_predict():
    test_image_path = Path(__file__).parent / 'test.jpeg'
    file = [('image_file', open(test_image_path, 'rb'))]
    response = client.post('/predict', files=file)
    assert response.status_code == 200
    assert isinstance(response.json()['output'], int)
