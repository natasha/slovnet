
import sys
from datetime import datetime


def log(format, *args):
    """
    Log a message to stderr.

    Args:
        format: (str): write your description
    """
    message = format % args
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(
        '[%s] %s' % (now, message),
        file=sys.stderr
    )
