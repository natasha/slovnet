
import sys
from datetime import datetime


def log(format, *args):
    message = format % args
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(
        '[%s] %s' % (now, message),
        file=sys.stderr
    )
