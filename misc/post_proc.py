import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA


PI = float(np.pi)


def fuv2img(fuv, coorW=1024, floorW=1024, floorH=512):
    '''
    Project 1d signal in uv space to 2d floor plane image
    '''
    floor_plane_x, floor_plane_y = np.meshgrid(range(floorW), range(floorH))
    floor_plane_x, floor_plane_y = -(floor_plane_y - floorH / 2), floor_plane_x - floorW / 2
    floor_plane_coridx = (np.arctan2(floor_plane_y, floor_plane_x) / (2 * PI) + 0.5) * coorW - 0.5
    floor_plane = map_coordinates(fuv, floor_plane_coridx.reshape(1, -1), order=1, mode='wrap')
    floor_plane = floor_plane.reshape(floorH, floorW)
    return floor_plane


def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * PI


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * PI


def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    coor: N x 2, index of array in (col, row) format
    '''
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = c * np.sin(u) + floorW / 2 - 0.5
    y = -c * np.cos(u) + floorH / 2 - 0.5
    return np.hstack([x[:, None], y[:, None]])


def np_x_u_solve_y(x, u, floorW=1024, floorH=512):
    c = (x - floorW / 2 + 0.5) / np.sin(u)
    return -c * np.cos(u) + floorH / 2 - 0.5


def np_y_u_solve_x(y, u, floorW=1024, floorH=512):
    c = -(y - floorH / 2 + 0.5) / np.cos(u)
    return c * np.sin(u) + floorW / 2 - 0.5


def np_xy2coor(xy, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    xy: N x 2
    '''
    x = xy[:, 0] - floorW / 2 + 0.5
    y = xy[:, 1] - floorH / 2 + 0.5

    u = np.arctan2(x, -y)
    v = np.arctan(z / np.sqrt(x**2 + y**2))

    coorx = (u / (2 * PI) + 0.5) * coorW - 0.5
    coory = (-v / PI + 0.5) * coorH - 0.5

    return np.hstack([coorx[:, None], coory[:, None]])


def mean_percentile(vec, p1=25, p2=75):
    vmin = np.percentile(vec, p1)
    vmax = np.percentile(vec, p2)
    return vec[(vmin <= vec) & (vec <= vmax)].mean()


def vote(vec, tol):
    vec = np.sort(vec)
    n = np.arange(len(vec))[::-1]
    n = n[:, None] - n[None, :] + 1.0
    l = squareform(pdist(vec[:, None], 'minkowski', p=1) + 1e-9)

    invalid = (n < len(vec) * 0.4) | (l > tol)
    if (~invalid).sum() == 0 or len(vec) < 5:
        best_fit = vec.mean()
        p_score = 0
    else:
        l[invalid] = 1e5
        n[invalid] = -1
        score = n
        max_idx = score.argmax()
        max_row = max_idx // len(vec)
        max_col = max_idx % len(vec)
        assert max_col > max_row
        best_fit = vec[max_row:max_col+1].mean()
        p_score = (max_col - max_row + 1) / len(vec)

    l1_score = np.abs(vec - best_fit).mean()

    return best_fit, p_score, l1_score


def get_z1(coory0, coory1, z0=50, coorH=512):
    v0 = np_coory2v(coory0, coorH)
    v1 = np_coory2v(coory1, coorH)
    c0 = z0 / np.tan(v0)
    z1 = c0 * np.tan(v1)
    return z1


def np_refine_by_fix_z(coory0, coory1, z0=50, coorH=512):
    '''
    Refine coory1 by coory0
    coory0 are assumed on given plane z
    '''
    v0 = np_coory2v(coory0, coorH)
    v1 = np_coory2v(coory1, coorH)

    c0 = z0 / np.tan(v0)
    z1 = c0 * np.tan(v1)
    z1_mean = mean_percentile(z1)
    v1_refine = np.arctan2(z1_mean, c0)
    coory1_refine = (-v1_refine / PI + 0.5) * coorH - 0.5

    return coory1_refine, z1_mean


def infer_coory(coory0, h, z0=50, coorH=512):
    v0 = np_coory2v(coory0, coorH)
    c0 = z0 / np.tan(v0)
    z1 = z0 + h
    v1 = np.arctan2(z1, c0)
    return (-v1 / PI + 0.5) * coorH - 0.5


def get_gpid(coorx, coorW):
    gpid = np.zeros(coorW)
    gpid[np.round(coorx).astype(int)] = 1
    gpid = np.cumsum(gpid).astype(int)
    gpid[gpid == gpid[-1]] = 0
    return gpid


def get_gpid_idx(gpid, j):
    idx = np.where(gpid == j)[0]
    if idx[0] == 0 and idx[-1] != len(idx) - 1:
        _shift = -np.where(idx != np.arange(len(idx)))[0][0]
        idx = np.roll(idx, _shift)
    return idx


def gpid_two_split(xy, tpid_a, tpid_b):
    m = np.arange(len(xy)) + 1
    cum_a = np.cumsum(xy[:, tpid_a])
    cum_b = np.cumsum(xy[::-1, tpid_b])
    l1_a = cum_a / m - cum_a / (m * m)
    l1_b = cum_b / m - cum_b / (m * m)
    l1_b = l1_b[::-1]

    score = l1_a[:-1] + l1_b[1:]
    best_split = score.argmax() + 1

    va = xy[:best_split, tpid_a].mean()
    vb = xy[best_split:, tpid_b].mean()

    return va, vb


def _get_rot_rad(px, py):
    if px < 0:
        px, py = -px, -py
    rad = np.arctan2(py, px) * 180 / np.pi
    if rad > 45:
        return 90 - rad
    if rad < -45:
        return -90 - rad
    return -rad


def get_rot_rad(init_coorx, coory, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512, tol=5):
    gpid = get_gpid(init_coorx, coorW)
    coor = np.hstack([np.arange(coorW)[:, None], coory[:, None]])
    xy = np_coor2xy(coor, z, coorW, coorH, floorW, floorH)
    xy_cor = []

    rot_rad_suggestions = []
    for j in range(len(init_coorx)):
        pca = PCA(n_components=1)
        pca.fit(xy[gpid == j])
        rot_rad_suggestions.append(_get_rot_rad(*pca.components_[0]))
    rot_rad_suggestions = np.sort(rot_rad_suggestions + [1e9])

    rot_rad = np.mean(rot_rad_suggestions[:-1])
    best_rot_rad_sz = -1
    last_j = 0
    for j in range(1, len(rot_rad_suggestions)):
        if rot_rad_suggestions[j] - rot_rad_suggestions[j-1] > tol:
            last_j = j
        elif j - last_j > best_rot_rad_sz:
            rot_rad = rot_rad_suggestions[last_j:j+1].mean()
            best_rot_rad_sz = j - last_j

    dx = int(round(rot_rad * 1024 / 360))
    return dx, rot_rad


def init_cuboid(init_coorx, coory, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512, tol=3):
    gpid = get_gpid(init_coorx, coorW)
    coor = np.hstack([np.arange(coorW)[:, None], coory[:, None]])
    xy = np_coor2xy(coor, z, coorW, coorH, floorW, floorH)

    # For each part seperated by wall-wall peak, voting for a wall
    xy_cor = []
    for j in range(4):
        now_x = xy[gpid == j, 0]
        now_y = xy[gpid == j, 1]
        new_x, x_score, x_l1 = vote(now_x, tol)
        new_y, y_score, y_l1 = vote(now_y, tol)
        if (x_score, -x_l1) > (y_score, -y_l1):
            xy_cor.append({'type': 0, 'val': new_x, 'score': x_score})
        else:
            xy_cor.append({'type': 1, 'val': new_y, 'score': y_score})

    # Sanity fallback
    scores = [0, 0]
    for j in range(4):
        if xy_cor[j]['type'] == 0:
            scores[j % 2] += xy_cor[j]['score']
        else:
            scores[j % 2] -= xy_cor[j]['score']
    if scores[0] > scores[1]:
        xy_cor[0]['type'] = 0
        xy_cor[1]['type'] = 1
        xy_cor[2]['type'] = 0
        xy_cor[3]['type'] = 1
    else:
        xy_cor[0]['type'] = 1
        xy_cor[1]['type'] = 0
        xy_cor[2]['type'] = 1
        xy_cor[3]['type'] = 0

    # Ceiling view to normal view
    cor = []
    for j in range(len(xy_cor)):
        next_j = (j + 1) % len(xy_cor)
        if xy_cor[j]['type'] == 1:
            cor.append((xy_cor[next_j]['val'], xy_cor[j]['val']))
        else:
            cor.append((xy_cor[j]['val'], xy_cor[next_j]['val']))
    cor = np_xy2coor(np.array(cor), z, coorW, coorH, floorW, floorH)
    cor = cor[np.argsort(cor[:, 0])]

    return cor, xy_cor
