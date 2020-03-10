
import boto3

from .record import Record
from .const import (
    S3_KEY_ID,
    S3_KEY,
    S3_BUCKET,
    S3_ENDPOINT,
    S3_REGION
)


class S3(Record):
    __attributes__ = ['key_id', 'key', 'bucket', 'endpoint', 'region']

    def __init__(self, key_id=S3_KEY_ID, key=S3_KEY, bucket=S3_BUCKET,
                 endpoint=S3_ENDPOINT, region=S3_REGION):
        self.key_id = key_id
        self.key = key
        self.bucket = bucket
        self.endpoint = endpoint
        self.region = region

        self.client = boto3.client(
            's3',
            aws_access_key_id=key_id,
            aws_secret_access_key=key,
            region_name=region,
            endpoint_url=endpoint,
        )

    def upload(self, path, key):
        self.client.upload_file(path, self.bucket, key)

    def download(self, key, path):
        self.client.download_file(self.bucket, key, path)
