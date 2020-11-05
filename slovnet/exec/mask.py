

def split_masked(input, mask):
    """
    Split the given mask into masked mask.

    Args:
        input: (array): write your description
        mask: (array): write your description
    """
    sizes = mask.sum(-1)
    for index, size in enumerate(sizes):
        yield input[index, :size]


def fill_masked(input, mask, fill=0):
    """
    Fill nan.

    Args:
        input: (array): write your description
        mask: (array): write your description
        fill: (str): write your description
    """
    return fill * mask + input * ~mask
