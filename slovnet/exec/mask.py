

def split_masked(input, mask):
    sizes = mask.sum(-1)
    for index, size in enumerate(sizes):
        yield input[index, :size]


def fill_masked(input, mask, fill=0):
    return fill * mask + input * ~mask
