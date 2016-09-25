#!/usr/bin/env python
"""
AUTHOR
    N. Mauchle <nicolas@nicolasmauchle.ch>

LICENSE
    MIT

VERSION
    1.0

"""

import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'data/ex1data2.txt')
    data = pd.read_csv(path, header=None, names=['Sizes', 'Bedrooms', 'Price'])
    print(data.head())

    # Feature normilize
    data = (data - data.mean() / data.std())
    print(data.head())
