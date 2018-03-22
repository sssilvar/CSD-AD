import os

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.offline import iplot, plot
from plotly import figure_factory as FF
from skimage import measure


def make_mesh(image, threshold=-300, step_size=1):
    print("Transposing surface")
    p = image.transpose(2, 1, 0)

    print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces


def plotly_3d(verts, faces):
    x, y, z = zip(*verts)

    print("Drawing")

    # Make the colormap single color since the axes are positional not intensity.
    # colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap = ['rgb(226, 226, 202)', 'rgb(226, 226, 202)']

    fig = FF.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=faces,
                            backgroundcolor='rgb(64, 64, 64)',
                            title="Interactive Visualization")
    plot(fig)


def plt_3d(verts, faces):
    print("Drawing")
    x, y, z = zip(*verts)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    plt.show()


if __name__ == '__main__':
    # Choose the file to be played with
    filename = 'test_data/941_S_1363.mgz'

    # Correct filename
    filename = os.path.join(os.getcwd(), filename)
    print('[  OK  ] File to be processed is located in: %s' % filename)

    # Load MRI file
    mri = nb.load(filename)
    img = mri.get_data()

    # Generate a sphere
    r = 40
    cx, cy, cz = (0, 0, 0)

    ax_min, ax_max = (-128, 128)
    y, x, z = np.ogrid[ax_min:ax_max, ax_min:ax_max, ax_min:ax_max]
    eqn = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    mask = np.bitwise_and(eqn < (r + 2. / 256 * r) ** 2, eqn > (r - 2. / 256 * r) ** 2)
    img_sph = img * mask
    img_ball = img * (eqn <= r ** 2)

    slide = 128 + 0

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img_sph[:, :, slide], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_ball[:, :, slide], cmap='gray')
    plt.axis('off')

    v, f = make_mesh(img_sph, threshold=60, step_size=2)
    plotly_3d(v, f)
