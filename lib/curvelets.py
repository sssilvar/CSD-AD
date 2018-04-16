import numpy as np
import matplotlib.pyplot as plt


def curvelet_plot(scales, angles, values):
    # Create colors
    gr = plt.cm.Greys

    # Set up plot
    fig, ax = plt.subplots()
    ax.axis('equal')

    for scale in range(0, scales):
        val = values[str(scale)]

        sub_bands = []
        if scale == 0:
            sub_bands.append(gr(float(val[0])))
            angles_scale = [1]
        elif scale == 1:
            angles_scale = range(0, angles)
        elif scale % 2 == 0:
            angles_scale = range(0, int(scale * angles))
        elif scale % 2 != 0:
            angles_scale = range(0, int((scale - 1) * angles))
        else:
            angles_scale = []
            sub_bands = group_size = None
            raise ValueError('There is no angles inside the scale')

        for angle in angles_scale:
            sub_bands.append(gr(float(val[angle])))
        group_size = list(np.ones(len(angles_scale)))

        # First Ring (outside)
        mypie, _ = ax.pie(group_size, radius=(scale + 1) * 1 / scales, colors=sub_bands)
        plt.setp(mypie, width=1 / scales, edgecolor='white')
        plt.margins(0, 0)


def clarray_to_mean_dict(A, f, scales, n_angles):
    curve_data = {}
    for scale in range(0, scales):
        scale_data = []

        if scale == 0:
            angles = [0]
        elif scale == 1:
            angles = range(0, n_angles)
        elif scale % 2 == 0:
            angles = range(0, int(scale * n_angles))
        elif scale % 2 != 0:
            angles = range(0, int((scale - 1) * n_angles))
        else:
            angles = []
            raise ValueError('There is no angles inside the scale')

        # Go over all the angles in the scale
        for angle in angles:
            ix_min, ix_max = A.index(scale, angle)
            scale_data.append(np.mean(f[ix_min:ix_max]))
        scale_data = np.abs(scale_data)  # / np.max(np.abs(scale_data))
        curve_data[str(scale)] = map(str, scale_data)

    # Return data converted to dictionary
    return curve_data
