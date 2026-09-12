import tempfile
import unittest
from pathlib import Path
from home_ray.storage import upload, download

class MemoryS3:
    def upload_file(self, path, bucket, key, ExtraArgs):
        self.data = Path(path).read_bytes()
    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.data)

class StorageTests(unittest.TestCase):
    def test_verified_roundtrip_and_corruption_preserves_destination(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'source';source.write_bytes(b'checkpoint')
            target=Path(root)/'restored';s3=MemoryS3()
            ref=upload(source,'private-bucket',s3=s3)
            download(ref,target,s3=s3)
            self.assertEqual(target.read_bytes(),source.read_bytes())
            s3.data=b'corrupted'
            with self.assertRaises(ValueError):download(ref,target,s3=s3)
            self.assertEqual(target.read_bytes(),source.read_bytes())
