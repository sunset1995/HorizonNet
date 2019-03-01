'''
This script is helper function for preprocessing.
Most of the code are converted from LayoutNet official's matlab code.
All functions, naming rule and data flow follow official for easier
converting and comparing.
Code is not optimized for python or numpy yet.

Author: Cheng Sun
Email : chengsun@gapp.nthu.edu.tw
'''

import sys
import numpy as np
from scipy.ndimage import map_coordinates
import cv2


def computeUVN(n, in_, planeID):
    '''
    compute v given u and normal.
    '''
    if planeID == 2:
        n = np.array([n[1], n[2], n[0]])
    elif planeID == 3:
        n = np.array([n[2], n[0], n[1]])
    bc = n[0] * np.sin(in_) + n[1] * np.cos(in_)
    bs = n[2]
    out = np.arctan(-bc / (bs + 1e-9))
    return out


def computeUVN_vec(n, in_, planeID):
    '''
    vectorization version of computeUVN
    @n         N x 3
    @in_      MN x 1
    @planeID   N
    '''
    n = n.copy()
    if (planeID == 2).sum():
        n[planeID == 2] = np.roll(n[planeID == 2], 2, axis=1)
    if (planeID == 3).sum():
        n[planeID == 3] = np.roll(n[planeID == 3], 1, axis=1)
    n = np.repeat(n, in_.shape[0] // n.shape[0], axis=0)
    assert n.shape[0] == in_.shape[0]
    bc = n[:, [0]] * np.sin(in_) + n[:, [1]] * np.cos(in_)
    bs = n[:, [2]]
    out = np.arctan(-bc / (bs + 1e-9))
    return out


def xyz2uvN(xyz, planeID=1):
    ID1 = (int(planeID) - 1 + 0) % 3
    ID2 = (int(planeID) - 1 + 1) % 3
    ID3 = (int(planeID) - 1 + 2) % 3
    normXY = np.sqrt(xyz[:, [ID1]] ** 2 + xyz[:, [ID2]] ** 2)
    normXY[normXY < 0.000001] = 0.000001
    normXYZ = np.sqrt(xyz[:, [ID1]] ** 2 + xyz[:, [ID2]] ** 2 + xyz[:, [ID3]] ** 2)
    v = np.arcsin(xyz[:, [ID3]] / normXYZ)
    u = np.arcsin(xyz[:, [ID1]] / normXY)
    valid = (xyz[:, [ID2]] < 0) & (u >= 0)
    u[valid] = np.pi - u[valid]
    valid = (xyz[:, [ID2]] < 0) & (u <= 0)
    u[valid] = -np.pi - u[valid]
    uv = np.hstack([u, v])
    uv[np.isnan(uv[:, 0]), 0] = 0
    return uv


def uv2xyzN(uv, planeID=1):
    ID1 = (int(planeID) - 1 + 0) % 3
    ID2 = (int(planeID) - 1 + 1) % 3
    ID3 = (int(planeID) - 1 + 2) % 3
    xyz = np.zeros((uv.shape[0], 3))
    xyz[:, ID1] = np.cos(uv[:, 1]) * np.sin(uv[:, 0])
    xyz[:, ID2] = np.cos(uv[:, 1]) * np.cos(uv[:, 0])
    xyz[:, ID3] = np.sin(uv[:, 1])
    return xyz


def uv2xyzN_vec(uv, planeID):
    '''
    vectorization version of uv2xyzN
    @uv       N x 2
    @planeID  N
    '''
    assert (planeID.astype(int) != planeID).sum() == 0
    planeID = planeID.astype(int)
    ID1 = (planeID - 1 + 0) % 3
    ID2 = (planeID - 1 + 1) % 3
    ID3 = (planeID - 1 + 2) % 3
    ID = np.arange(len(uv))
    xyz = np.zeros((len(uv), 3))
    xyz[ID, ID1] = np.cos(uv[:, 1]) * np.sin(uv[:, 0])
    xyz[ID, ID2] = np.cos(uv[:, 1]) * np.cos(uv[:, 0])
    xyz[ID, ID3] = np.sin(uv[:, 1])
    return xyz


def warpImageFast(im, XXdense, YYdense):
    minX = max(1., np.floor(XXdense.min()) - 1)
    minY = max(1., np.floor(YYdense.min()) - 1)

    maxX = min(im.shape[1], np.ceil(XXdense.max()) + 1)
    maxY = min(im.shape[0], np.ceil(YYdense.max()) + 1)

    im = im[int(round(minY-1)):int(round(maxY)),
            int(round(minX-1)):int(round(maxX))]

    assert XXdense.shape == YYdense.shape
    out_shape = XXdense.shape
    coordinates = [
        (YYdense - minY).reshape(-1),
        (XXdense - minX).reshape(-1),
    ]
    im_warp = np.stack([
        map_coordinates(im[..., c], coordinates, order=1).reshape(out_shape)
        for c in range(im.shape[-1])],
        axis=-1)

    return im_warp


def rotatePanorama(img, vp=None, R=None):
    '''
    Rotate panorama
        if R is given, vp (vanishing point) will be overlooked
        otherwise R is computed from vp
    '''
    sphereH, sphereW, C = img.shape

    # new uv coordinates
    TX, TY = np.meshgrid(range(1, sphereW + 1), range(1, sphereH + 1))
    TX = TX.reshape(-1, 1, order='F')
    TY = TY.reshape(-1, 1, order='F')
    ANGx = (TX - sphereW/2 - 0.5) / sphereW * np.pi * 2
    ANGy = -(TY - sphereH/2 - 0.5) / sphereH * np.pi
    uvNew = np.hstack([ANGx, ANGy])
    xyzNew = uv2xyzN(uvNew, 1)

    # rotation matrix
    if R is None:
        R = np.linalg.inv(vp.T)

    xyzOld = np.linalg.solve(R, xyzNew.T).T
    uvOld = xyz2uvN(xyzOld, 1)

    Px = (uvOld[:, 0] + np.pi) / (2*np.pi) * sphereW + 0.5
    Py = (-uvOld[:, 1] + np.pi/2) / np.pi * sphereH + 0.5

    Px = Px.reshape(sphereH, sphereW, order='F')
    Py = Py.reshape(sphereH, sphereW, order='F')

    # boundary
    imgNew = np.zeros((sphereH+2, sphereW+2, C), np.float64)
    imgNew[1:-1, 1:-1, :] = img
    imgNew[1:-1, 0, :] = img[:, -1, :]
    imgNew[1:-1, -1, :] = img[:, 0, :]
    imgNew[0, 1:sphereW//2+1, :] = img[0, sphereW-1:sphereW//2-1:-1, :]
    imgNew[0, sphereW//2+1:-1, :] = img[0, sphereW//2-1::-1, :]
    imgNew[-1, 1:sphereW//2+1, :] = img[-1, sphereW-1:sphereW//2-1:-1, :]
    imgNew[-1, sphereW//2+1:-1, :] = img[0, sphereW//2-1::-1, :]
    imgNew[0, 0, :] = img[0, 0, :]
    imgNew[-1, -1, :] = img[-1, -1, :]
    imgNew[0, -1, :] = img[0, -1, :]
    imgNew[-1, 0, :] = img[-1, 0, :]

    rotImg = warpImageFast(imgNew, Px+1, Py+1)

    return rotImg


def imgLookAt(im, CENTERx, CENTERy, new_imgH, fov):
    sphereH = im.shape[0]
    sphereW = im.shape[1]
    warped_im = np.zeros((new_imgH, new_imgH, 3))
    TX, TY = np.meshgrid(range(1, new_imgH + 1), range(1, new_imgH + 1))
    TX = TX.reshape(-1, 1, order='F')
    TY = TY.reshape(-1, 1, order='F')
    TX = TX - 0.5 - new_imgH/2
    TY = TY - 0.5 - new_imgH/2
    r = new_imgH / 2 / np.tan(fov/2)

    # convert to 3D
    R = np.sqrt(TY ** 2 + r ** 2)
    ANGy = np.arctan(- TY / r)
    ANGy = ANGy + CENTERy

    X = np.sin(ANGy) * R
    Y = -np.cos(ANGy) * R
    Z = TX

    INDn = np.nonzero(np.abs(ANGy) > np.pi/2)

    # project back to sphere
    ANGx = np.arctan(Z / -Y)
    RZY = np.sqrt(Z ** 2 + Y ** 2)
    ANGy = np.arctan(X / RZY)

    ANGx[INDn] = ANGx[INDn] + np.pi
    ANGx = ANGx + CENTERx

    INDy = np.nonzero(ANGy < -np.pi/2)
    ANGy[INDy] = -np.pi - ANGy[INDy]
    ANGx[INDy] = ANGx[INDy] + np.pi

    INDx = np.nonzero(ANGx <= -np.pi);   ANGx[INDx] = ANGx[INDx] + 2 * np.pi
    INDx = np.nonzero(ANGx >   np.pi);   ANGx[INDx] = ANGx[INDx] - 2 * np.pi
    INDx = np.nonzero(ANGx >   np.pi);   ANGx[INDx] = ANGx[INDx] - 2 * np.pi
    INDx = np.nonzero(ANGx >   np.pi);   ANGx[INDx] = ANGx[INDx] - 2 * np.pi

    Px = (ANGx + np.pi) / (2*np.pi) * sphereW + 0.5
    Py = ((-ANGy) + np.pi/2) / np.pi * sphereH + 0.5

    INDxx = np.nonzero(Px < 1)
    Px[INDxx] = Px[INDxx] + sphereW
    im = np.concatenate([im, im[:, :2]], 1)

    Px = Px.reshape(new_imgH, new_imgH, order='F')
    Py = Py.reshape(new_imgH, new_imgH, order='F')

    warped_im = warpImageFast(im, Px, Py)

    return warped_im


def separatePano(panoImg, fov, x, y, imgSize=320):
    '''cut a panorama image into several separate views'''
    assert x.shape == y.shape
    if not isinstance(fov, np.ndarray):
        fov = fov * np.ones_like(x)

    sepScene = [
        {
            'img': imgLookAt(panoImg.copy(), xi, yi, imgSize, fovi),
            'vx': xi,
            'vy': yi,
            'fov': fovi,
            'sz': imgSize,
        }
        for xi, yi, fovi in zip(x, y, fov)
    ]

    return sepScene


def lsdWrap(img, LSD=None, **kwargs):
    '''
    Opencv implementation of
    Rafael Grompone von Gioi, Jérémie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
    LSD: a Line Segment Detector, Image Processing On Line, vol. 2012.
    [Rafael12] http://www.ipol.im/pub/art/2012/gjmr-lsd/?utm_source=doi
    @img
        input image
    @LSD
        Constructing by cv2.createLineSegmentDetector
        https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#linesegmentdetector
        if LSD is given, kwargs will be ignored
    @kwargs
        is used to construct LSD
        work only if @LSD is not given
    '''
    if LSD is None:
        LSD = cv2.createLineSegmentDetector(**kwargs)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    lines, width, prec, nfa = LSD.detect(img)
    if lines is None:
        return np.zeros_like(img), np.array([])
    edgeMap = LSD.drawSegments(np.zeros_like(img), lines)[..., -1]
    lines = np.squeeze(lines, 1)
    edgeList = np.concatenate([lines, width, prec, nfa], 1)
    return edgeMap, edgeList


def edgeFromImg2Pano(edge):
    edgeList = edge['edgeLst']
    if len(edgeList) == 0:
        return np.array([])

    vx = edge['vx']
    vy = edge['vy']
    fov = edge['fov']
    imH, imW = edge['img'].shape

    R = (imW/2) / np.tan(fov/2)

    # im is the tangent plane, contacting with ball at [x0 y0 z0]
    x0 = R * np.cos(vy) * np.sin(vx)
    y0 = R * np.cos(vy) * np.cos(vx)
    z0 = R * np.sin(vy)
    vecposX = np.array([np.cos(vx), -np.sin(vx), 0])
    vecposY = np.cross(np.array([x0, y0, z0]), vecposX)
    vecposY = vecposY / np.sqrt(vecposY @ vecposY.T)
    vecposX = vecposX.reshape(1, -1)
    vecposY = vecposY.reshape(1, -1)
    Xc = (0 + imW-1) / 2
    Yc = (0 + imH-1) / 2

    vecx1 = edgeList[:, [0]] - Xc
    vecy1 = edgeList[:, [1]] - Yc
    vecx2 = edgeList[:, [2]] - Xc
    vecy2 = edgeList[:, [3]] - Yc

    vec1 = np.tile(vecx1, [1, 3]) * vecposX + np.tile(vecy1, [1, 3]) * vecposY
    vec2 = np.tile(vecx2, [1, 3]) * vecposX + np.tile(vecy2, [1, 3]) * vecposY
    coord1 = [[x0, y0, z0]] + vec1
    coord2 = [[x0, y0, z0]] + vec2

    normal = np.cross(coord1, coord2, axis=1)
    normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)

    panoList = np.hstack([normal, coord1, coord2, edgeList[:, [-1]]])

    return panoList


def _intersection(range1, range2):
    if range1[1] < range1[0]:
        range11 = [range1[0], 1]
        range12 = [0, range1[1]]
    else:
        range11 = range1
        range12 = [0, 0]

    if range2[1] < range2[0]:
        range21 = [range2[0], 1]
        range22 = [0, range2[1]]
    else:
        range21 = range2
        range22 = [0, 0]

    b = max(range11[0], range21[0]) < min(range11[1], range21[1])
    if b:
        return b
    b2 = max(range12[0], range22[0]) < min(range12[1], range22[1])
    b = b or b2
    return b


def _insideRange(pt, range):
    if range[1] > range[0]:
        b = pt >= range[0] and pt <= range[1]
    else:
        b1 = pt >= range[0] and pt <= 1
        b2 = pt >= 0 and pt <= range[1]
        b = b1 or b2
    return b


def combineEdgesN(edges):
    '''
    Combine some small line segments, should be very conservative
    OUTPUT
        lines: combined line segments
        ori_lines: original line segments
        line format [nx ny nz projectPlaneID umin umax LSfov score]
    '''
    arcList = []
    for edge in edges:
        panoLst = edge['panoLst']
        if len(panoLst) == 0:
            continue
        arcList.append(panoLst)
    arcList = np.vstack(arcList)

    # ori lines
    numLine = len(arcList)
    ori_lines = np.zeros((numLine, 8))
    areaXY = np.abs(arcList[:, 2])
    areaYZ = np.abs(arcList[:, 0])
    areaZX = np.abs(arcList[:, 1])
    planeIDs = np.argmax(np.stack([areaXY, areaYZ, areaZX], -1), 1) + 1  # XY YZ ZX

    for i in range(numLine):
        ori_lines[i, :3] = arcList[i, :3]
        ori_lines[i, 3] = planeIDs[i]
        coord1 = arcList[i, 3:6]
        coord2 = arcList[i, 6:9]
        uv = xyz2uvN(np.stack([coord1, coord2]), planeIDs[i])
        umax = uv[:, 0].max() + np.pi
        umin = uv[:, 0].min() + np.pi
        if umax - umin > np.pi:
            ori_lines[i, 4:6] = np.array([umax, umin]) / 2 / np.pi
        else:
            ori_lines[i, 4:6] = np.array([umin, umax]) / 2 / np.pi
        ori_lines[i, 6] = np.arccos((
            np.dot(coord1, coord2) / (np.linalg.norm(coord1) * np.linalg.norm(coord2))
            ).clip(-1, 1))
        ori_lines[i, 7] = arcList[i, 9]

    # additive combination
    lines = ori_lines.copy()
    for _ in range(3):
        numLine = len(lines)
        valid_line = np.ones(numLine, bool)
        for i in range(numLine):
            if not valid_line[i]:
                continue
            dotProd = (lines[:, :3] * lines[[i], :3]).sum(1)
            valid_curr = np.logical_and((np.abs(dotProd) > np.cos(np.pi / 180)), valid_line)
            valid_curr[i] = False
            for j in np.nonzero(valid_curr)[0]:
                range1 = lines[i, 4:6]
                range2 = lines[j, 4:6]
                valid_rag = _intersection(range1, range2)
                if not valid_rag:
                    continue

                # combine
                I = np.argmax(np.abs(lines[i, :3]))
                if lines[i, I] * lines[j, I] > 0:
                    nc = lines[i, :3] * lines[i, 6] + lines[j, :3] * lines[j, 6]
                else:
                    nc = lines[i, :3] * lines[i, 6] - lines[j, :3] * lines[j, 6]
                nc = nc / np.linalg.norm(nc)

                if _insideRange(range1[0], range2):
                    nrmin = range2[0]
                else:
                    nrmin = range1[0]

                if _insideRange(range1[1], range2):
                    nrmax = range2[1]
                else:
                    nrmax = range1[1]

                u = np.array([[nrmin], [nrmax]]) * 2 * np.pi - np.pi
                v = computeUVN(nc, u, lines[i, 3])
                xyz = uv2xyzN(np.hstack([u, v]), lines[i, 3])
                l = np.arccos(np.dot(xyz[0, :], xyz[1, :]).clip(-1, 1))
                scr = (lines[i,6]*lines[i,7] + lines[j,6]*lines[j,7]) / (lines[i,6]+lines[j,6])

                lines[i] = [*nc, lines[i, 3], nrmin, nrmax, l, scr]
                valid_line[j] = False

        lines = lines[valid_line]

    return lines, ori_lines


def icosahedron2sphere(level):
    # this function use a icosahedron to sample uniformly on a sphere
    a = 2 / (1 + np.sqrt(5))
    M = np.array([
        0, a, -1, a, 1, 0, -a, 1, 0,
        0, a, 1, -a, 1, 0, a, 1, 0,
        0, a, 1, 0, -a, 1, -1, 0, a,
        0, a, 1, 1, 0, a, 0, -a, 1,
        0, a, -1, 0, -a, -1, 1, 0, -a,
        0, a, -1, -1, 0, -a, 0, -a, -1,
        0, -a, 1, a, -1, 0, -a, -1, 0,
        0, -a, -1, -a, -1, 0, a, -1, 0,
        -a, 1, 0, -1, 0, a, -1, 0, -a,
        -a, -1, 0, -1, 0, -a, -1, 0, a,
        a, 1, 0, 1, 0, -a, 1, 0, a,
        a, -1, 0, 1, 0, a, 1, 0, -a,
        0, a, 1, -1, 0, a, -a, 1, 0,
        0, a, 1, a, 1, 0, 1, 0, a,
        0, a, -1, -a, 1, 0, -1, 0, -a,
        0, a, -1, 1, 0, -a, a, 1, 0,
        0, -a, -1, -1, 0, -a, -a, -1, 0,
        0, -a, -1, a, -1, 0, 1, 0, -a,
        0, -a, 1, -a, -1, 0, -1, 0, a,
        0, -a, 1, 1, 0, a, a, -1, 0])

    coor = M.T.reshape(3, 60, order='F').T
    coor, idx = np.unique(coor, return_inverse=True, axis=0)
    tri = idx.reshape(3, 20, order='F').T

    # extrude
    coor = list(coor / np.tile(np.linalg.norm(coor, axis=1, keepdims=True), (1, 3)))

    for _ in range(level):
        triN = []
        for t in range(len(tri)):
            n = len(coor)
            coor.append((coor[tri[t, 0]] + coor[tri[t, 1]]) / 2)
            coor.append((coor[tri[t, 1]] + coor[tri[t, 2]]) / 2)
            coor.append((coor[tri[t, 2]] + coor[tri[t, 0]]) / 2)

            triN.append([n, tri[t, 0], n+2])
            triN.append([n, tri[t, 1], n+1])
            triN.append([n+1, tri[t, 2], n+2])
            triN.append([n, n+1, n+2])
        tri = np.array(triN)

        # uniquefy
        coor, idx = np.unique(coor, return_inverse=True, axis=0)
        tri = idx[tri]

        # extrude
        coor = list(coor / np.tile(np.sqrt(np.sum(coor * coor, 1, keepdims=True)), (1, 3)))

    return np.array(coor), np.array(tri)


def curveFitting(inputXYZ, weight):
    '''
    @inputXYZ: N x 3
    @weight  : N x 1
    '''
    l = np.linalg.norm(inputXYZ, axis=1, keepdims=True)
    inputXYZ = inputXYZ / l
    weightXYZ = inputXYZ * weight
    XX = np.sum(weightXYZ[:, 0] ** 2)
    YY = np.sum(weightXYZ[:, 1] ** 2)
    ZZ = np.sum(weightXYZ[:, 2] ** 2)
    XY = np.sum(weightXYZ[:, 0] * weightXYZ[:, 1])
    YZ = np.sum(weightXYZ[:, 1] * weightXYZ[:, 2])
    ZX = np.sum(weightXYZ[:, 2] * weightXYZ[:, 0])

    A = np.array([
        [XX, XY, ZX],
        [XY, YY, YZ],
        [ZX, YZ, ZZ]])
    U, S, Vh = np.linalg.svd(A)
    outputNM = Vh[-1, :]
    outputNM = outputNM / np.linalg.norm(outputNM)

    return outputNM


def sphereHoughVote(segNormal, segLength, segScores, binRadius, orthTolerance, candiSet, force_unempty=True):
    # initial guess
    numLinesg = len(segNormal)

    voteBinPoints = candiSet.copy()
    voteBinPoints = voteBinPoints[~(voteBinPoints[:,2] < 0)]
    reversValid = (segNormal[:, 2] < 0).reshape(-1)
    segNormal[reversValid] = -segNormal[reversValid]

    voteBinUV = xyz2uvN(voteBinPoints)
    numVoteBin = len(voteBinPoints)
    voteBinValues = np.zeros(numVoteBin)
    for i in range(numLinesg):
        tempNorm = segNormal[[i]]
        tempDots = (voteBinPoints * tempNorm).sum(1)

        valid = np.abs(tempDots) < np.cos((90 - binRadius) * np.pi / 180)

        voteBinValues[valid] = voteBinValues[valid] + segScores[i] * segLength[i]

    checkIDs1 = np.nonzero(voteBinUV[:, [1]] > np.pi / 3)[0]
    voteMax = 0
    checkID1Max = 0
    checkID2Max = 0
    checkID3Max = 0

    for j in range(len(checkIDs1)):
        checkID1 = checkIDs1[j]
        vote1 = voteBinValues[checkID1]
        if voteBinValues[checkID1] == 0 and force_unempty:
            continue
        checkNormal = voteBinPoints[[checkID1]]
        dotProduct = (voteBinPoints * checkNormal).sum(1)
        checkIDs2 = np.nonzero(np.abs(dotProduct) < np.cos((90 - orthTolerance) * np.pi / 180))[0]

        for i in range(len(checkIDs2)):
            checkID2 = checkIDs2[i]
            if voteBinValues[checkID2] == 0 and force_unempty:
                continue
            vote2 = vote1 + voteBinValues[checkID2]
            cpv = np.cross(voteBinPoints[checkID1], voteBinPoints[checkID2]).reshape(1, 3)
            cpn = np.linalg.norm(cpv)
            dotProduct = (voteBinPoints * cpv).sum(1) / cpn
            checkIDs3 = np.nonzero(np.abs(dotProduct) > np.cos(orthTolerance * np.pi / 180))[0]

            for k in range(len(checkIDs3)):
                checkID3 = checkIDs3[k]
                if voteBinValues[checkID3] == 0 and force_unempty:
                    continue
                vote3 = vote2 + voteBinValues[checkID3]
                if vote3 > voteMax:
                    lastStepCost = vote3 - voteMax
                    if voteMax != 0:
                        tmp = (voteBinPoints[[checkID1Max, checkID2Max, checkID3Max]] * \
                               voteBinPoints[[checkID1, checkID2, checkID3]]).sum(1)
                        lastStepAngle = np.arccos(tmp.clip(-1, 1))
                    else:
                        lastStepAngle = np.zeros(3)

                    checkID1Max = checkID1
                    checkID2Max = checkID2
                    checkID3Max = checkID3

                    voteMax = vote3

    if checkID1Max == 0:
        print('[WARN] sphereHoughVote: no orthogonal voting exist', file=sys.stderr)
        return None, 0, 0
    initXYZ = voteBinPoints[[checkID1Max, checkID2Max, checkID3Max]]

    # refine
    refiXYZ = np.zeros((3, 3))
    dotprod = (segNormal * initXYZ[[0]]).sum(1)
    valid = np.abs(dotprod) < np.cos((90 - binRadius) * np.pi / 180)
    validNm = segNormal[valid]
    validWt = segLength[valid] * segScores[valid]
    validWt = validWt / validWt.max()
    refiNM = curveFitting(validNm, validWt)
    refiXYZ[0] = refiNM.copy()

    dotprod = (segNormal * initXYZ[[1]]).sum(1)
    valid = np.abs(dotprod) < np.cos((90 - binRadius) * np.pi / 180)
    validNm = segNormal[valid]
    validWt = segLength[valid] * segScores[valid]
    validWt = validWt / validWt.max()
    validNm = np.vstack([validNm, refiXYZ[[0]]])
    validWt = np.vstack([validWt, validWt.sum(0, keepdims=1) * 0.1])
    refiNM = curveFitting(validNm, validWt)
    refiXYZ[1] = refiNM.copy()

    refiNM = np.cross(refiXYZ[0], refiXYZ[1])
    refiXYZ[2] = refiNM / np.linalg.norm(refiNM)

    return refiXYZ, lastStepCost, lastStepAngle


def findMainDirectionEMA(lines):
    '''compute vp from set of lines'''

    # initial guess
    segNormal = lines[:, :3]
    segLength = lines[:, [6]]
    segScores = np.ones((len(lines), 1))

    shortSegValid = (segLength < 5 * np.pi / 180).reshape(-1)
    segNormal = segNormal[~shortSegValid, :]
    segLength = segLength[~shortSegValid]
    segScores = segScores[~shortSegValid]

    numLinesg = len(segNormal)
    candiSet, tri = icosahedron2sphere(3)
    ang = np.arccos((candiSet[tri[0,0]] * candiSet[tri[0,1]]).sum().clip(-1, 1)) / np.pi * 180
    binRadius = ang / 2
    initXYZ, score, angle = sphereHoughVote(segNormal, segLength, segScores, 2*binRadius, 2, candiSet)

    if initXYZ is None:
        print('[WARN] findMainDirectionEMA: initial failed', file=sys.stderr)
        return None, score, angle

    # iterative refine
    iter_max = 3
    candiSet, tri = icosahedron2sphere(5)
    numCandi = len(candiSet)
    angD = np.arccos((candiSet[tri[0, 0]] * candiSet[tri[0, 1]]).sum().clip(-1, 1)) / np.pi * 180
    binRadiusD = angD / 2
    curXYZ = initXYZ.copy()
    tol = np.linspace(4*binRadius, 4*binRadiusD, iter_max)  # shrink down ls and candi
    for it in range(iter_max):
        dot1 = np.abs((segNormal * curXYZ[[0]]).sum(1))
        dot2 = np.abs((segNormal * curXYZ[[1]]).sum(1))
        dot3 = np.abs((segNormal * curXYZ[[2]]).sum(1))
        valid1 = dot1 < np.cos((90 - tol[it]) * np.pi / 180)
        valid2 = dot2 < np.cos((90 - tol[it]) * np.pi / 180)
        valid3 = dot3 < np.cos((90 - tol[it]) * np.pi / 180)
        valid = valid1 | valid2 | valid3

        if np.sum(valid) == 0:
            print('[WARN] findMainDirectionEMA: zero line segments for voting', file=sys.stderr)
            break

        subSegNormal = segNormal[valid]
        subSegLength = segLength[valid]
        subSegScores = segScores[valid]

        dot1 = np.abs((candiSet * curXYZ[[0]]).sum(1))
        dot2 = np.abs((candiSet * curXYZ[[1]]).sum(1))
        dot3 = np.abs((candiSet * curXYZ[[2]]).sum(1))
        valid1 = dot1 > np.cos(tol[it] * np.pi / 180)
        valid2 = dot2 > np.cos(tol[it] * np.pi / 180)
        valid3 = dot3 > np.cos(tol[it] * np.pi / 180)
        valid = valid1 | valid2 | valid3

        if np.sum(valid) == 0:
            print('[WARN] findMainDirectionEMA: zero line segments for voting', file=sys.stderr)
            break

        subCandiSet = candiSet[valid]

        tcurXYZ, _, _ = sphereHoughVote(subSegNormal, subSegLength, subSegScores, 2*binRadiusD, 2, subCandiSet)

        if tcurXYZ is None:
            print('[WARN] findMainDirectionEMA: no answer found', file=sys.stderr)
            break
        curXYZ = tcurXYZ.copy()

    mainDirect = curXYZ.copy()
    mainDirect[0] = mainDirect[0] * np.sign(mainDirect[0,2])
    mainDirect[1] = mainDirect[1] * np.sign(mainDirect[1,2])
    mainDirect[2] = mainDirect[2] * np.sign(mainDirect[2,2])

    uv = xyz2uvN(mainDirect)
    I1 = np.argmax(uv[:,1])
    J = np.setdiff1d(np.arange(3), I1)
    I2 = np.argmin(np.abs(np.sin(uv[J,0])))
    I2 = J[I2]
    I3 = np.setdiff1d(np.arange(3), np.hstack([I1, I2]))
    mainDirect = np.vstack([mainDirect[I1], mainDirect[I2], mainDirect[I3]])

    mainDirect[0] = mainDirect[0] * np.sign(mainDirect[0,2])
    mainDirect[1] = mainDirect[1] * np.sign(mainDirect[1,1])
    mainDirect[2] = mainDirect[2] * np.sign(mainDirect[2,0])

    mainDirect = np.vstack([mainDirect, -mainDirect])

    return mainDirect, score, angle


def multi_linspace(start, stop, num):
    div = (num - 1)
    y = np.arange(0, num, dtype=np.float64)
    steps = (stop - start) / div
    return steps.reshape(-1, 1) * y + start.reshape(-1, 1)


def assignVanishingType(lines, vp, tol, area=10):
    numLine = len(lines)
    numVP = len(vp)
    typeCost = np.zeros((numLine, numVP))
    # perpendicular
    for vid in range(numVP):
        cosint = (lines[:, :3] * vp[[vid]]).sum(1)
        typeCost[:, vid] = np.arcsin(np.abs(cosint).clip(-1, 1))

    # infinity
    u = np.stack([lines[:, 4], lines[:, 5]], -1)
    u = u.reshape(-1, 1) * 2 * np.pi - np.pi
    v = computeUVN_vec(lines[:, :3], u, lines[:, 3])
    xyz = uv2xyzN_vec(np.hstack([u, v]), np.repeat(lines[:, 3], 2))
    xyz = multi_linspace(xyz[0::2].reshape(-1), xyz[1::2].reshape(-1), 100)
    xyz = np.vstack([blk.T for blk in np.split(xyz, numLine)])
    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    for vid in range(numVP):
        ang = np.arccos(np.abs((xyz * vp[[vid]]).sum(1)).clip(-1, 1))
        notok = (ang < area * np.pi / 180).reshape(numLine, 100).sum(1) != 0
        typeCost[notok, vid] = 100

    I = typeCost.min(1)
    tp = typeCost.argmin(1)
    tp[I > tol] = numVP + 1

    return tp, typeCost


def refitLineSegmentB(lines, vp, vpweight=0.1):
    '''
    Refit direction of line segments
    INPUT:
        lines: original line segments
        vp: vannishing point
        vpweight: if set to 0, lines will not change; if set to inf, lines will
                  be forced to pass vp
    '''
    numSample = 100
    numLine = len(lines)
    xyz = np.zeros((numSample+1, 3))
    wei = np.ones((numSample+1, 1))
    wei[numSample] = vpweight * numSample
    lines_ali = lines.copy()
    for i in range(numLine):
        n = lines[i, :3]
        sid = lines[i, 4] * 2 * np.pi
        eid = lines[i, 5] * 2 * np.pi
        if eid < sid:
            x = np.linspace(sid, eid + 2 * np.pi, numSample) % (2 * np.pi)
        else:
            x = np.linspace(sid, eid, numSample)
        u = -np.pi + x.reshape(-1, 1)
        v = computeUVN(n, u, lines[i, 3])
        xyz[:numSample] = uv2xyzN(np.hstack([u, v]), lines[i, 3])
        xyz[numSample] = vp
        outputNM = curveFitting(xyz, wei)
        lines_ali[i, :3] = outputNM

    return lines_ali


def paintParameterLine(parameterLine, width, height):
    lines = parameterLine.copy()
    panoEdgeC = np.zeros((height, width))

    num_sample = max(height, width)
    for i in range(len(lines)):
        n = lines[i, :3]
        sid = lines[i, 4] * 2 * np.pi
        eid = lines[i, 5] * 2 * np.pi
        if eid < sid:
            x = np.linspace(sid, eid + 2 * np.pi, num_sample)
            x = x % (2 * np.pi)
        else:
            x = np.linspace(sid, eid, num_sample)
        u = -np.pi + x.reshape(-1, 1)
        v = computeUVN(n, u, lines[i, 3])
        xyz = uv2xyzN(np.hstack([u, v]), lines[i, 3])
        uv = xyz2uvN(xyz, 1)
        m = np.minimum(np.floor((uv[:,0] + np.pi) / (2 * np.pi) * width) + 1,
            width).astype(np.int32)
        n = np.minimum(np.floor(((np.pi / 2) - uv[:, 1]) / np.pi * height) + 1,
            height).astype(np.int32)
        panoEdgeC[n-1, m-1] = i

    return panoEdgeC


def panoEdgeDetection(img, viewSize=320, qError=0.7, refineIter=3):
    '''
    line detection on panorama
       INPUT:
           img: image waiting for detection, double type, range 0~1
           viewSize: image size of croped views
           qError: set smaller if more line segment wanted
       OUTPUT:
           oLines: detected line segments
           vp: vanishing point
           views: separate views of panorama
           edges: original detection of line segments in separate views
           panoEdge: image for visualize line segments
    '''
    cutSize = viewSize
    fov = np.pi / 3
    xh = np.arange(-np.pi, np.pi*5/6, np.pi/6)
    yh = np.zeros(xh.shape[0])
    xp = np.array([-3/3, -2/3, -1/3, 0/3,  1/3, 2/3, -3/3, -2/3, -1/3,  0/3,  1/3,  2/3]) * np.pi
    yp = np.array([ 1/4,  1/4,  1/4, 1/4,  1/4, 1/4, -1/4, -1/4, -1/4, -1/4, -1/4, -1/4]) * np.pi
    x = np.concatenate([xh, xp, [0, 0]])
    y = np.concatenate([yh, yp, [np.pi/2., -np.pi/2]])

    sepScene = separatePano(img.copy(), fov, x, y, cutSize)
    edge = []
    LSD = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_ADV, _quant=qError)
    for i, scene in enumerate(sepScene):
        edgeMap, edgeList = lsdWrap(scene['img'], LSD)
        edge.append({
            'img': edgeMap,
            'edgeLst': edgeList,
            'vx': scene['vx'],
            'vy': scene['vy'],
            'fov': scene['fov'],
        })
        edge[-1]['panoLst'] = edgeFromImg2Pano(edge[-1])
    lines, olines = combineEdgesN(edge)

    clines = lines.copy()
    for _ in range(refineIter):
        mainDirect, score, angle = findMainDirectionEMA(clines)

        tp, typeCost = assignVanishingType(lines, mainDirect[:3], 0.1, 10)
        lines1 = lines[tp==0]
        lines2 = lines[tp==1]
        lines3 = lines[tp==2]

        lines1rB = refitLineSegmentB(lines1, mainDirect[0], 0)
        lines2rB = refitLineSegmentB(lines2, mainDirect[1], 0)
        lines3rB = refitLineSegmentB(lines3, mainDirect[2], 0)

        clines = np.vstack([lines1rB, lines2rB, lines3rB])

    panoEdge1r = paintParameterLine(lines1rB, img.shape[1], img.shape[0])
    panoEdge2r = paintParameterLine(lines2rB, img.shape[1], img.shape[0])
    panoEdge3r = paintParameterLine(lines3rB, img.shape[1], img.shape[0])
    panoEdger = np.stack([panoEdge1r, panoEdge2r, panoEdge3r], -1)

    # output
    olines = clines
    vp = mainDirect
    views = sepScene
    edges = edge
    panoEdge = panoEdger

    return olines, vp, views, edges, panoEdge, score, angle


if __name__ == '__main__':

    # disable OpenCV3's non thread safe OpenCL option
    cv2.ocl.setUseOpenCL(False)

    import os
    import argparse
    import PIL
    from PIL import Image
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', required=True)
    parser.add_argument('--o_prefix', required=True)
    parser.add_argument('--qError', default=0.7, type=float)
    parser.add_argument('--refineIter', default=3, type=int)
    args = parser.parse_args()

    # Read image
    img_ori = np.array(Image.open(args.i).resize((1024, 512)))

    # Vanishing point estimation & Line segments detection
    s_time = time.time()
    olines, vp, views, edges, panoEdge, score, angle = panoEdgeDetection(img_ori,
                                                                         qError=args.qError,
                                                                         refineIter=args.refineIter)
    print('Elapsed time: %.2f' % (time.time() - s_time))
    panoEdge = (panoEdge > 0)

    print('Vanishing point:')
    for v in vp[2::-1]:
        print('%.6f %.6f %.6f' % tuple(v))

    # Visualization
    edg = rotatePanorama(panoEdge.astype(np.float64), vp[2::-1])
    img = rotatePanorama(img_ori / 255.0, vp[2::-1])
    one = img.copy() * 0.5
    one[(edg > 0.5).sum(-1) > 0] = 0
    one[edg[..., 0] > 0.5, 0] = 1
    one[edg[..., 1] > 0.5, 1] = 1
    one[edg[..., 2] > 0.5, 2] = 1
    Image.fromarray((edg * 255).astype(np.uint8)).save('%s_edg.png' % args.o_prefix)
    Image.fromarray((img * 255).astype(np.uint8)).save('%s_img.png' % args.o_prefix)
    Image.fromarray((one * 255).astype(np.uint8)).save('%s_one.png' % args.o_prefix)
