
import boto3

from .const import (
    S3_ENDPOINT,
    S3_REGION
)


def s3_client(key_id, key, endpoint=S3_ENDPOINT, region=S3_REGION):
    return boto3.client(
        's3',
        aws_access_key_id=key_id,
        aws_secret_access_key=key,
        region_name=region,
        endpoint_url=endpoint,
    )
