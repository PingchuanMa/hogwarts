__all__ = ['LMDBReader']


import lmdb


class LMDBReader:

    def __init__(self, lmdb_path, coding='utf8'):
        self.lmdb_path = lmdb_path
        self.coding = coding

    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()

    def __call__(self, path):
        if not hasattr(self, 'env'):
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            if not self.env:
                raise Exception('cannot open lmdb from %s' % (self.lmdb_path))

        with self.env.begin(write=False) as txn:
            value = txn.get(path.encode(self.coding))
        return value
