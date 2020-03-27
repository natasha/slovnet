
import torch


def every(step, period):
    return step > 0 and step % period == 0
