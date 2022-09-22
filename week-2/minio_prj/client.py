import minio

MINIO_USERNAME = 'minio'
MINIO_PASSWORD = 'minio123'
URL_API = '0.0.0.0:9000'


class MinioClient:

    def __init__(self):
        self.client = minio.Minio(URL_API, MINIO_USERNAME, MINIO_PASSWORD, secure=False)

    def create_bucket(self, bucket_name):
        self.client.make_bucket(bucket_name)

    def upload_file(self, path, bucket_name):
        self.client.fput_object(bucket_name, path.name, path)

    def download_file(self, bucket_name, object_name, path):
        self.client.fget_object(bucket_name, object_name, path)

    def get_objects(self, bucket_name):
        return self.client.list_objects(bucket_name)

    def delete_object(self, bucket_name, object_name):
        self.client.remove_object(bucket_name, object_name)

    def get_buckets(self):
        buckets = self.client.list_buckets()
        return buckets

    def bucket_exists(self, bucket_name):
        return self.client.bucket_exists(bucket_name)
