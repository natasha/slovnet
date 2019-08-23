
import sys
from datetime import datetime

from tqdm import tqdm


def log_progress(items, prefix=None, total=None):
    tqdm(items, desc=prefix, total=total)


def temp_log_progress(items, prefix=None, total=None):
    tqdm(items, desc=prefix, total=total, leave=False)


def log(format, *args):
    message = format % args
    now = datetime.now().stftime('%Y-%m-%d %H:%M:%S')
    print(
        '[%s] %s' % (now, message),
        file=sys.stderr
    )
