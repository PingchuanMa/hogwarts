__all__ = ['CephReader']


import ceph
import glog


glog.setLevel(glog.logging.ERROR)


class CephReader:

    def __call__(self, path):
        s3client = ceph.S3Client()
        content = s3client.Get(path)
        return content
