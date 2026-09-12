"""Optional private S3/MinIO transfer with SHA-256 verification."""
import hashlib
import os
from pathlib import Path


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def client():
    import boto3
    return boto3.client('s3', endpoint_url=os.environ.get('HOME_RAY_S3_ENDPOINT'),
                       region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'))


def upload(path, bucket, prefix='artifacts', s3=None):
    path = Path(path)
    digest = sha256(path)
    key = prefix.strip('/') + '/' + digest + '/' + path.name
    (s3 or client()).upload_file(str(path), bucket, key,
                                ExtraArgs={'Metadata': {'sha256': digest}})
    return {'bucket': bucket, 'key': key, 'sha256': digest}


def download(reference, path, s3=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    handle, name = tempfile.mkstemp(dir=path.parent, prefix='.artifact-')
    os.close(handle)
    temp = Path(name)
    try:
        (s3 or client()).download_file(reference['bucket'], reference['key'], str(temp))
        if sha256(temp) != reference['sha256']:
            raise ValueError('Artifact SHA-256 mismatch')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path
