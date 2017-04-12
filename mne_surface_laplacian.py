import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import lpmn
from numpy.linalg import inv
from mne import pick_types


def surface_laplacian(inst, x, y, z, inst_type, m='auto', leg_order='auto',
                      smoothing=1e-5):
    """Compute Surface Laplacian using Perrin 1989 Spherical Splines Method

    Parameters
    ----------
    inst : instance of Epochs or Raw or Evoked


    Returns
    -------
    lap_inst : instance of Epochs or Raw or Evoked
        The modified instance with surface laplacian transformed EEG data.

    """

    inst = inst.copy()

    # Get indices of EEG data
    eeg_ix = pick_types(inst.info, eeg=True, meg=False)

    # Extract data depending on the type of MNE object
    # Goal is to get EEG data into # chs x (# time points * # epochs) matrix
    #if not isinstance(inst, (BaseEpochs, BaseRaw, Evoked)):
    #    raise ValueError('inst must be an instance of Epochs, Raw, or'
    #                     'Evoked. Got type {0}'.format(type(inst)))
    #if isinstance(inst, BaseRaw):
    #    data = inst[eeg_ix, :]
    #    num_chs, num_time_points = data.shape
    #elif isinstance(inst, BaseEpochs):
    #    data = inst.get_data()[:, eeg_ix, :]
    #    num_epochs, num_chs, num_time_points = data.shape
    #    data = np.reshape(data, (num_chs, num_epochs * num_time_points))
    #else:  # Evoked
    if inst_type == 'evoked':
    	data = inst.data[eeg_ix, :]
    elif inst_type == 'epochs':
        data = inst._data[:, eeg_ix, :]
        num_epochs, num_chs, num_time_points = data.shape
        data = np.concatenate([data[i, :, :] for i in range(num_epochs)], axis=-1)

    # Compute the G & H matrices
    G, H, _ = _compute_GH(x, y, z)

    # Compute the surface laplacian transform
    lap = _compute_perrin_surf_laplacian(data, G, H, smoothing)

    # Insert the laplacian transformed data back into the MNE object
    #if isinstance(inst, BaseRaw):
    #    inst[eeg_ix, :] = lap
    #elif isinstance(inst, BaseEpochs):
    #    lap = np.reshape(lap, (num_epochs, num_chs, num_time_points))
    #    inst.get_data()[:, eeg_ix, :] = lap
    #else:  # Evoked
    if inst_type == 'evoked':
    	inst.data[eeg_ix, :] = lap
    elif inst_type == 'epochs':
	lap = np.split(lap, num_epochs, axis=-1)
	lap = np.concatenate([d[np.newaxis, :, :] for d in lap], axis=0) 
        inst._data[:, eeg_ix, :] = lap

    return inst


def _transform_unit_sphere(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    max_r = np.max(r)
    x /= max_r
    y /= max_r
    z /= max_r
    return x, y, z


@np.vectorize
def _vector_legendre(n, x):
    return lpmn(0, n, x)[0][-1, -1]


def _compute_GH(x, y, z):

    num_electrodes = len(x)
    if num_electrodes > 100:
        m = 3
        leg_order = 40
    else:
        m = 4
        leg_order = 20

    x, y, z = _transform_unit_sphere(x, y, z)
    G = np.zeros((num_electrodes, num_electrodes))
    H = np.zeros((num_electrodes, num_electrodes))
    cos_dist = np.zeros((num_electrodes, num_electrodes))

    # Compute the Cosine Distance Between Electrodes
    for i in range(num_electrodes):
        for j in range(i + 1, num_electrodes):
            cos_dist[i, j] = 1 - (((x[i] - x[j])**2 + (y[i] - y[j])**2 +
                                   (z[i] - z[j])**2) / 2.)

    cos_dist = cos_dist + cos_dist.T + np.identity(num_electrodes)

    # Compute the Legendre Polynomials
    leg_poly = np.zeros((leg_order, num_electrodes, num_electrodes))
    for n in range(1, leg_order + 1):
        leg_poly[n - 1, :, :] = _vector_legendre(n, cos_dist)

    for i in range(num_electrodes):
        for j in range(i, num_electrodes):

            g = 0
            h = 0

            for n in range(1, leg_order + 1):
                g += ((2 * n + 1) * leg_poly[n - 1, i, j]) / ((n * (n + 1)) ** m)
                h -= ((2 * n + 1) * leg_poly[n - 1, i, j]) / ((n * (n + 1)) ** (m - 1))

            G[i, j] = g / (4 * np.pi)
            H[i, j] = -h / (4 * np.pi)

    G += G.T
    H += H.T

    # ??
    G -= np.identity(num_electrodes) * G[0, 0] / 2.
    H -= np.identity(num_electrodes) * H[0, 0] / 2.

    return G, H, cos_dist


def _compute_perrin_surf_laplacian(data, G, H, smoothing):

    num_electrodes, num_time_points = data.shape

    Gs = G + np.identity(num_electrodes) * smoothing

    Gi = inv(Gs)

    d = data.T.dot(Gi)

    tmp = (np.sum(d, axis=1) / np.sum(Gi))
    C = d - tmp[:, np.newaxis].dot(np.sum(Gi, axis=1)[np.newaxis, :])

    surf_lap = C.dot(H.T).T

    return surf_lap
