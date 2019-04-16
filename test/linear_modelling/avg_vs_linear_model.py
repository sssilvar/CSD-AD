#!/bin/env python3
import numpy as np

if __name__ == '__main__':
    # Generate several matrices: 25/25
    set_a = np.random.normal(size=[50, 50, 50, 25])
    set_b = np.random.normal(size=[50, 50, 50, 25])

    # Calculate average
    avg_a = set_a.mean(axis=3)
    avg_b = set_b.mean(axis=3)

    # Linear model
    Y = []
    for i in range(set_a.shape[3]):
        print(set_a[:, :, :, i].ravel().shape)
