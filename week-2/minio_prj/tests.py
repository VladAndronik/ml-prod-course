import hashlib
import unittest
from pathlib import Path

from .client import MinioClient


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class MinioTests(unittest.TestCase):

    def setUp(self):
        self.client = MinioClient()
        root = Path(__file__).parent
        self.test_file = root / 'test_file.txt'
        self.bucket_name = 'test'
        self.test_download_file = root / 'temp.txt'
        print(self.test_file, self.test_download_file)

        if not self.client.bucket_exists(self.bucket_name):
            self.client.create_bucket(self.bucket_name)
        assert self.client.bucket_exists(self.bucket_name), "Bucket was not created"

    def test_file_upload(self):
        self.client.upload_file(self.test_file, self.bucket_name)
        objects = self.client.get_objects(self.bucket_name)
        objects = [obj.object_name for obj in objects]
        self.assertIn(self.test_file.name, objects)

    def test_file_downlaod(self):
        print(self.bucket_name, self.test_file.name, self.test_download_file)
        self.client.download_file(self.bucket_name, self.test_file.name, str(self.test_download_file))
        hash_true = md5(str(self.test_file))
        hash_down = md5(str(self.test_download_file))

        self.assertTrue(hash_true, hash_down)

    # def test_delete_obj(self):
    #     self.client.delete_object(self.bucket_name, self.test_file.name)
    #     objects = self.client.get_objects(self.bucket_name)
    #     objects = [obj.object_name for obj in objects]
    #     self.assertNotIn(self.test_file.name, objects)


if __name__ == '__main__':
    unittest.main()
