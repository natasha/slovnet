
import tarfile
from io import BytesIO

from .record import Record


class Tar(Record):
    __attributes__ = ['path']

    mode = 'r'

    def __enter__(self):
        """
        Enter the tar file.

        Args:
            self: (todo): write your description
        """
        self.tar = tarfile.open(self.path, self.mode)
        return self

    def __exit__(self, *args):
        """
        Called when the connection is closed.

        Args:
            self: (todo): write your description
        """
        self.tar.close()

    def open(self, name):
        """
        Open a tar archive.

        Args:
            self: (todo): write your description
            name: (str): write your description
        """
        member = self.tar.getmember(name)
        return self.tar.extractfile(member)

    def read(self, name):
        """
        Reads a file.

        Args:
            self: (todo): write your description
            name: (str): write your description
        """
        return self.open(name).read()

    def list(self, prefix=None):
        """
        Return a list of the given prefix.

        Args:
            self: (todo): write your description
            prefix: (str): write your description
        """
        for member in self.tar:
            name = member.name
            if not prefix or name.startswith(prefix):
                yield name


class DumpTar(Tar):
    mode = 'w'

    def write(self, bytes, name):
        """
        Write a tar file.

        Args:
            self: (todo): write your description
            bytes: (todo): write your description
            name: (str): write your description
        """
        file = BytesIO(bytes)
        info = tarfile.TarInfo(name)
        info.size = len(bytes)
        self.tar.addfile(tarinfo=info, fileobj=file)
