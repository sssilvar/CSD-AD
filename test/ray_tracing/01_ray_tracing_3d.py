import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

# Set root folder
root = os.path.join(os.getcwd(), '..', '..')


def sector_mask(shape, centre, min_radius, max_radius, theta_range, phi_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    """

    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cx, cy, cz = centre
    t_min, t_max = np.deg2rad(theta_range)
    p_min, p_max = np.deg2rad(phi_range)

    # ensure stop angle > start angle
    if t_max < t_min:
        t_max += 2 * np.pi
    if p_max < p_min:
        p_max += 2 * np.pi

    # convert cartesian --> polar coordinates
    r2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    theta = np.arctan2(y - cy, x - cx) - t_min
    phi = np.arctan2(np.sqrt((x - cx) ** 2 + (y - cy) ** 2), (z - cz) ** 2)

    # wrap angles between 0 and 2*pi
    theta %= (2 * np.pi)
    phi %= (2 * np.pi)

    # circular mask
    sphere_mask = np.logical_and(r2 >= min_radius ** 2, r2 <= max_radius ** 2)

    # angular mask
    anglemask = np.logical_and(theta <= (t_max - t_min), phi <= (p_max - p_min))

    return sphere_mask * anglemask


if __name__ == '__main__':
    # Define shape parameters
    d = 10
    ax_min, ax_max = (-d, d)

    # Define the coordinate system
    x, y, z = np.ogrid[ax_min:ax_max, ax_min:ax_max, ax_min:ax_max]

    mgz = nb.load(os.path.join(root, 'test', 'test_data', '941_S_1363.mgz'))
    img = mgz.get_data()
    img_slice = img[:, :, 128]

    mask = sector_mask(img.shape, (128, 128, 128),
                       min_radius=30, max_radius=40, theta_range=(0, 90), phi_range=(-90, 90))

    # Apply mask
    img_masked = img * mask
    print(np.sum(img_masked))

    # Slide
    i_slice = 128

    plt.figure()
    plt.subplot(121)
    plt.imshow(img[:, :, i_slice], cmap='gray')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(img_masked[:, :, i_slice], cmap='gray')
    plt.axis('off')

    img_2d = np.zeros([360, 180])
    for i, theta in enumerate(range(0, 360)):
        for j, phi in enumerate(range(-90, 90)):
            print('I: %d / J: %d' % (i, j))
            # Calculate mask
            mask = sector_mask(img.shape, (128, 128, 128),
                               min_radius=30, max_radius=40, theta_range=(i, i+1), phi_range=(j, j+1))

            # Project over the
            img_2d[i, j] = np.mean(img * mask)

    # Plot image
    plt.figure()
    plt.imshow(img_2d, cmap='gray')
    plt.axis('off')

    plt.show()
