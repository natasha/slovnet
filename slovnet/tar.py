
import tarfile
from io import BytesIO

from .record import Record


class Tar(Record):
    __attributes__ = ['path']

    mode = 'r'

    def __enter__(self):
        self.tar = tarfile.open(self.path, self.mode)
        return self

    def __exit__(self, *args):
        self.tar.close()

    def open(self, name):
        member = self.tar.getmember(name)
        return self.tar.extractfile(member)

    def read(self, name):
        return self.open(name).read()

    def list(self, prefix=None):
        for member in self.tar:
            name = member.name
            if not prefix or name.startswith(prefix):
                yield name


class DumpTar(Tar):
    mode = 'w'

    def write(self, bytes, name):
        file = BytesIO(bytes)
        info = tarfile.TarInfo(name)
        info.size = len(bytes)
        self.tar.addfile(tarinfo=info, fileobj=file)
