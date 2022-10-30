from locust import HttpUser, task
from pathlib import Path


class UPCUser(HttpUser):
    @task(weight=1)
    def random_request(self):
        test_image_path = Path(__file__).parent.parent / 'tests/test.jpeg'
        file = [('image_file', open(test_image_path, 'rb'))]
        self.client.post(
            url="http://0.0.0.0:8080/predict",
            files=file,
            auth=None,
        )


if __name__ == '__main__':
    import requests
    test_image_path = Path(__file__).parent.parent / 'tests/test.jpeg'
    f = [('image_file', open(test_image_path, 'rb'))] * 2
    resp = requests.post(url='http://0.0.0.0:8080/predict', files=f)
    print(resp.status_code)
