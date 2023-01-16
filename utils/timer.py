# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import time


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.

    def tic(self):
        # Record initial timestamp
        self.start_time = time.time()
        self.calls = 0

    def toc(self):
        # Time elapsed, Average time
        self.calls += 1
        self.diff = time.time() - self.start_time

        return self.diff, (self.diff / self.calls)


# Formatter
def sec2minhrs(sec):
    if sec < 60:
        return 0, 0, round(sec)
    
    m = int(sec / 60)
    s = sec - m * 60
    
    if m < 60:
        return 0, m, round(s)
    
    h = int(m / 60)
    m = m - h * 60

    return h, m, round(s)