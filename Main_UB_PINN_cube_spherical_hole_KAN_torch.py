# ==============================================================
# 3D PINN-based upper-bound limit analysis
# One-octant cube with spherical hole
# HEX8 Gauss integration + cubic B-spline KAN velocity field
# Loads px, py, pz on x=a, y=a, z=a
# PyTorch version of the MATLAB code
# ==============================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from kan_layers import build_kan

torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================================================
# PARAMETERS
# ==============================================================

class Problem:
    pass

sigma0 = 1.0

prob = Problem()
prob.a = 1.0      # one-octant size, full cube is 2a x 2a x 2a
prob.R = 0.2      # spherical hole radius

prob.px = 1.0
prob.py = 0.0
prob.pz = 0.0

prob.nx = 20
prob.ny = prob.nx
prob.nz = prob.nx

prob.numGauss = 2

betaInc = 1.0

nAdam  = 5000
nLBFGS = 550
lr     = 1e-3
gridSchedule = {1500: 9, 3500: 13}

# ==============================================================
# BUILD PROBLEM
# ==============================================================

def buildProblem(prob):
    coords, elem = formnode_pla_3D_multiblock(
        prob.nx, prob.ny, prob.nz, prob.R, prob.a
    )

    Xg, Wg = domainGaussH8(coords, elem, prob.numGauss)

    Xx, Wx = faceGaussH8(coords, elem, "xmax", prob.a, prob.numGauss)
    Xy, Wy = faceGaussH8(coords, elem, "ymax", prob.a, prob.numGauss)
    Xz, Wz = faceGaussH8(coords, elem, "zmax", prob.a, prob.numGauss)

    prob.coords = coords
    prob.elem   = elem

    prob.XgDL = torch.tensor(Xg, device=device)
    prob.WgDL = torch.tensor(Wg, device=device).reshape(-1, 1)

    prob.XxDL = torch.tensor(Xx, device=device)
    prob.WxDL = torch.tensor(Wx, device=device).reshape(-1, 1)

    prob.XyDL = torch.tensor(Xy, device=device)
    prob.WyDL = torch.tensor(Wy, device=device).reshape(-1, 1)

    prob.XzDL = torch.tensor(Xz, device=device)
    prob.WzDL = torch.tensor(Wz, device=device).reshape(-1, 1)

    return prob

# ==============================================================
# NETWORK
# ==============================================================

class KANNet(nn.Module):
    def __init__(self, inDim, outDim, width, depth):
        super().__init__()
        self.net = build_kan(
            inDim, outDim, width, depth, grid_size=5,
            input_scale=prob.a, device=device, dtype=torch.get_default_dtype()
        )

    def forward(self, x):
        return self.net(x)

def buildNet(inDim, outDim, width, depth):
    return KANNet(inDim, outDim, width, depth).to(device)

# ==============================================================
# LOSS
# ==============================================================

def modelLossInfo(net, prob, sigma0, betaInc):
    loss, info = computeLoss(net, prob, sigma0, betaInc)
    return loss, info

def modelLossLBFGS(net, prob, sigma0, betaInc):
    loss, _ = computeLoss(net, prob, sigma0, betaInc)
    return loss

def computeLoss(net, prob, sigma0, betaInc):
    Wraw = externalWork(net, prob)
    alpha = 1.0 / (torch.abs(Wraw) + 1e-12)

    X = prob.XgDL.clone().detach().requires_grad_(True)
    W = prob.WgDL

    uvwRaw = net(X)

    d = hardBC(X, uvwRaw)
    d = alpha * d

    eps = strainRate3D(X, d)

    D = dissipationDensity3D(eps, sigma0)
    Wint = torch.sum(D * W)

    # incompressibility: exx + eyy + ezz = 0
    divp = eps[:, 0:1] + eps[:, 1:2] + eps[:, 2:3]
    Linc = torch.sum((divp ** 2) * W)

    loss = Wint + betaInc * Linc

    info = {
        "Wint": Wint.detach(),
        "Linc": Linc.detach(),
        "Wext": Wraw.detach(),
        "alpha": alpha.detach(),
    }

    return loss, info

# ==============================================================
# HARD BC FOR ONE-OCTANT MODEL
# ==============================================================

def hardBC(X, uvwRaw):
    x = X[:, 0:1]
    y = X[:, 1:2]
    z = X[:, 2:3]

    u0 = uvwRaw[:, 0:1]
    v0 = uvwRaw[:, 1:2]
    w0 = uvwRaw[:, 2:3]

    # symmetry:
    # x=0 -> u=0
    # y=0 -> v=0
    # z=0 -> w=0
    u = x * u0
    v = y * v0
    w = z * w0

    d = torch.cat((u, v, w), dim=1)

    return d

# ==============================================================
# EXTERNAL WORK
# ==============================================================

def externalWork(net, prob):
    # face x=a, traction [px,0,0]
    Xx = prob.XxDL
    Wx = prob.WxDL

    uvwx = net(Xx)
    dx = hardBC(Xx, uvwx)
    ux = dx[:, 0:1]

    WextX = torch.sum(prob.px * ux * Wx)

    # face y=a, traction [0,py,0]
    Xy = prob.XyDL
    Wy = prob.WyDL

    uvwy = net(Xy)
    dy = hardBC(Xy, uvwy)
    vy = dy[:, 1:2]

    WextY = torch.sum(prob.py * vy * Wy)

    # face z=a, traction [0,0,pz]
    Xz = prob.XzDL
    Wz = prob.WzDL

    uvwz = net(Xz)
    dz = hardBC(Xz, uvwz)
    wz = dz[:, 2:3]

    WextZ = torch.sum(prob.pz * wz * Wz)

    Wext = WextX + WextY + WextZ

    return Wext

# ==============================================================
# 3D STRAIN RATE
# ==============================================================

def grad_component(scalar_field, X):
    grad = torch.autograd.grad(
        scalar_field.sum(),
        X,
        create_graph=True,
        retain_graph=True,
        allow_unused=False
    )[0]
    return grad

def strainRate3D(X, d):
    u = d[:, 0:1]
    v = d[:, 1:2]
    w = d[:, 2:3]

    du = grad_component(u, X)
    dv = grad_component(v, X)
    dw = grad_component(w, X)

    du_dx = du[:, 0:1]
    du_dy = du[:, 1:2]
    du_dz = du[:, 2:3]

    dv_dx = dv[:, 0:1]
    dv_dy = dv[:, 1:2]
    dv_dz = dv[:, 2:3]

    dw_dx = dw[:, 0:1]
    dw_dy = dw[:, 1:2]
    dw_dz = dw[:, 2:3]

    exx = du_dx
    eyy = dv_dy
    ezz = dw_dz

    gxy = du_dy + dv_dx
    gyz = dv_dz + dw_dy
    gxz = du_dz + dw_dx

    eps = torch.cat((exx, eyy, ezz, gxy, gyz, gxz), dim=1)

    return eps

# ==============================================================
# 3D VON MISES PLASTIC DISSIPATION
# ==============================================================

def dissipationDensity3D(eps, sigma0):
    # eps = [exx; eyy; ezz; gxy; gyz; gxz]
    # engineering shear strains are used.

    exx = eps[:, 0:1]
    eyy = eps[:, 1:2]
    ezz = eps[:, 2:3]
    gxy = eps[:, 3:4]
    gyz = eps[:, 4:5]
    gxz = eps[:, 5:6]

    trE = exx + eyy + ezz

    exxd = exx - trE / 3.0
    eyyd = eyy - trE / 3.0
    ezzd = ezz - trE / 3.0

    quad = (2.0 / 3.0) * (
        exxd ** 2 + eyyd ** 2 + ezzd ** 2
        + 0.5 * gxy ** 2 + 0.5 * gyz ** 2 + 0.5 * gxz ** 2
    )

    D = sigma0 * torch.sqrt(torch.clamp(quad, min=1e-18))

    return D

# ==============================================================
# 3-BLOCK GEOMETRY-FITTED H8 MESH
# ==============================================================

def formnode_pla_3D_multiblock(nx, ny, nz, R, a):
    coords = np.zeros((0, 3), dtype=np.float64)
    elem   = np.zeros((0, 8), dtype=np.int64)

    nodeX, elemX = makeBlock(nx, ny, nz, R, a, "x")
    elemX = elemX + coords.shape[0]
    coords = np.vstack((coords, nodeX))
    elem   = np.vstack((elem, elemX))

    nodeY, elemY = makeBlock(nx, ny, nz, R, a, "y")
    elemY = elemY + coords.shape[0]
    coords = np.vstack((coords, nodeY))
    elem   = np.vstack((elem, elemY))

    nodeZ, elemZ = makeBlock(nx, ny, nz, R, a, "z")
    elemZ = elemZ + coords.shape[0]
    coords = np.vstack((coords, nodeZ))
    elem   = np.vstack((elem, elemZ))

    coords, elem = mergeDuplicateNodes(coords, elem, 1e-10)

    return coords, elem

def makeBlock(nx, ny, nz, R, a, face):
    uList = np.linspace(0.0, 1.0, nx + 1)
    vList = np.linspace(0.0, 1.0, ny + 1)

    # k=1: outer cube face, k=nz+1: inner spherical hole
    sList = np.linspace(0.0, 1.0, nz + 1) ** 1.4

    node = np.zeros(((nx + 1) * (ny + 1) * (nz + 1), 3), dtype=np.float64)

    def idx(i, j, k):
        return i + (nx + 1) * j + (nx + 1) * (ny + 1) * k

    for i in range(nx + 1):
        u = uList[i]

        for j in range(ny + 1):
            v = vList[j]

            if face == "x":
                d = np.array([1.0, u, v], dtype=np.float64)
                pOuter = np.array([a, a * u, a * v], dtype=np.float64)

            elif face == "y":
                d = np.array([u, 1.0, v], dtype=np.float64)
                pOuter = np.array([a * u, a, a * v], dtype=np.float64)

            elif face == "z":
                d = np.array([u, v, 1.0], dtype=np.float64)
                pOuter = np.array([a * u, a * v, a], dtype=np.float64)

            else:
                raise ValueError("Unknown face.")

            d = d / np.linalg.norm(d)
            pInner = R * d

            for k in range(nz + 1):
                s = sList[k]

                p = (1.0 - s) * pOuter + s * pInner

                node[idx(i, j, k), :] = p

    elem = buildBlockH8(nx, ny, nz)

    return node, elem

def buildBlockH8(nx, ny, nz):
    elem = np.zeros((nx * ny * nz, 8), dtype=np.int64)
    e = 0

    def idx(i, j, k):
        return i + (nx + 1) * j + (nx + 1) * (ny + 1) * k

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n1 = idx(i,     j,     k)
                n2 = idx(i + 1, j,     k)
                n3 = idx(i + 1, j + 1, k)
                n4 = idx(i,     j + 1, k)

                n5 = idx(i,     j,     k + 1)
                n6 = idx(i + 1, j,     k + 1)
                n7 = idx(i + 1, j + 1, k + 1)
                n8 = idx(i,     j + 1, k + 1)

                elem[e, :] = np.array([n1, n2, n3, n4, n5, n6, n7, n8], dtype=np.int64)
                e += 1

    return elem

def mergeDuplicateNodes(node, elem, tol):
    key = np.round(node / tol) * tol
    _, ia, ic = np.unique(key, axis=0, return_index=True, return_inverse=True)

    # MATLAB unique(...,'stable') behavior
    order = np.argsort(ia)
    ia_stable = ia[order]

    old_to_new = np.empty_like(order)
    old_to_new[order] = np.arange(len(order))

    nodeNew = node[ia_stable, :]
    elemNew = old_to_new[ic[elem]]

    return nodeNew, elemNew

# ==============================================================
# H8 SHAPE FUNCTIONS
# ==============================================================

def shapeH8(xi, eta, zeta):
    N = 1.0 / 8.0 * np.array([
        (1-xi)*(1-eta)*(1-zeta),
        (1+xi)*(1-eta)*(1-zeta),
        (1+xi)*(1+eta)*(1-zeta),
        (1-xi)*(1+eta)*(1-zeta),
        (1-xi)*(1-eta)*(1+zeta),
        (1+xi)*(1-eta)*(1+zeta),
        (1+xi)*(1+eta)*(1+zeta),
        (1-xi)*(1+eta)*(1+zeta),
    ], dtype=np.float64)

    dNdxi = 1.0 / 8.0 * np.array([
        -(1-eta)*(1-zeta),
         (1-eta)*(1-zeta),
         (1+eta)*(1-zeta),
        -(1+eta)*(1-zeta),
        -(1-eta)*(1+zeta),
         (1-eta)*(1+zeta),
         (1+eta)*(1+zeta),
        -(1+eta)*(1+zeta),
    ], dtype=np.float64)

    dNdeta = 1.0 / 8.0 * np.array([
        -(1-xi)*(1-zeta),
        -(1+xi)*(1-zeta),
         (1+xi)*(1-zeta),
         (1-xi)*(1-zeta),
        -(1-xi)*(1+zeta),
        -(1+xi)*(1+zeta),
         (1+xi)*(1+zeta),
         (1-xi)*(1+zeta),
    ], dtype=np.float64)

    dNdzeta = 1.0 / 8.0 * np.array([
        -(1-xi)*(1-eta),
        -(1+xi)*(1-eta),
        -(1+xi)*(1+eta),
        -(1-xi)*(1+eta),
         (1-xi)*(1-eta),
         (1+xi)*(1-eta),
         (1+xi)*(1+eta),
         (1-xi)*(1+eta),
    ], dtype=np.float64)

    return N, dNdxi, dNdeta, dNdzeta

# ==============================================================
# GAUSS RULE
# ==============================================================

def gauss1D(n):
    if n == 2:
        gp = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)
        wg = np.array([1.0, 1.0], dtype=np.float64)

    elif n == 3:
        gp = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)], dtype=np.float64)
        wg = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0], dtype=np.float64)

    else:
        raise ValueError("Unsupported Gauss order.")

    return gp, wg

# ==============================================================
# DOMAIN GAUSS INTEGRATION
# ==============================================================

def domainGaussH8(coords, elem, ngauss):
    gp, wg = gauss1D(ngauss)

    Xg = []
    Wg = []

    for e in range(elem.shape[0]):
        Xe = coords[elem[e, :], :]

        for i in range(gp.size):
            for j in range(gp.size):
                for k in range(gp.size):
                    xi   = gp[i]
                    eta  = gp[j]
                    zeta = gp[k]

                    N, dNdxi, dNdeta, dNdzeta = shapeH8(xi, eta, zeta)

                    J = np.vstack((dNdxi, dNdeta, dNdzeta)) @ Xe
                    detJ = abs(np.linalg.det(J))

                    if detJ <= 1e-14:
                        raise ValueError(f"Zero Jacobian in H8 element {e+1}.")

                    xg = N @ Xe

                    Xg.append(xg)
                    Wg.append(wg[i] * wg[j] * wg[k] * detJ)

    Xg = np.array(Xg, dtype=np.float64)
    Wg = np.array(Wg, dtype=np.float64)

    return Xg, Wg

# ==============================================================
# FACE GAUSS INTEGRATION
# ==============================================================

def faceGaussH8(coords, elem, faceName, a, ngauss):
    gp, wg = gauss1D(ngauss)

    tol = 1e-8

    # all local H8 faces
    faceList = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ], dtype=np.int64)

    Xf = []
    Wf = []

    usedFaces = set()

    for e in range(elem.shape[0]):
        for lf in range(6):
            fn = elem[e, faceList[lf, :]]
            Xq = coords[fn, :]

            if faceName == "xmax":
                isBnd = np.all(np.abs(Xq[:, 0] - a) < tol)

            elif faceName == "ymax":
                isBnd = np.all(np.abs(Xq[:, 1] - a) < tol)

            elif faceName == "zmax":
                isBnd = np.all(np.abs(Xq[:, 2] - a) < tol)

            else:
                raise ValueError("Unknown faceName.")

            if not isBnd:
                continue

            # avoid duplicated faces after multiblock merging
            key = tuple(sorted(fn.tolist()))
            if key in usedFaces:
                continue
            usedFaces.add(key)

            for i in range(gp.size):
                for j in range(gp.size):
                    xi  = gp[i]
                    eta = gp[j]

                    N, dNdxi, dNdeta = shapeQ4(xi, eta)

                    xg = N @ Xq

                    dx_dxi  = dNdxi @ Xq
                    dx_deta = dNdeta @ Xq

                    dA = np.linalg.norm(np.cross(dx_dxi, dx_deta))

                    Xf.append(xg)
                    Wf.append(wg[i] * wg[j] * dA)

    if len(Xf) == 0:
        raise ValueError(f"No boundary Gauss points found for {faceName}. Check mesh boundary and tolerance.")

    Xf = np.array(Xf, dtype=np.float64)
    Wf = np.array(Wf, dtype=np.float64)

    return Xf, Wf

def shapeQ4(xi, eta):
    N = 0.25 * np.array([
        (1-xi)*(1-eta),
        (1+xi)*(1-eta),
        (1+xi)*(1+eta),
        (1-xi)*(1+eta),
    ], dtype=np.float64)

    dNdxi = 0.25 * np.array([
        -(1-eta),
         (1-eta),
         (1+eta),
        -(1+eta),
    ], dtype=np.float64)

    dNdeta = 0.25 * np.array([
        -(1-xi),
        -(1+xi),
         (1+xi),
         (1-xi),
    ], dtype=np.float64)

    return N, dNdxi, dNdeta

# ==============================================================
# NODAL DISSIPATION
# ==============================================================

def nodalDissipation(net, prob, Xnode, sigma0):
    Wraw = externalWork(net, prob)
    alpha = 1.0 / (torch.abs(Wraw) + 1e-12)

    Xnode = Xnode.clone().detach().requires_grad_(True)

    uvwRaw = net(Xnode)
    d = hardBC(Xnode, uvwRaw)
    d = alpha * d

    eps = strainRate3D(Xnode, d)
    D = dissipationDensity3D(eps, sigma0)

    Dnode = D.detach().cpu().numpy().reshape(-1)

    return Dnode

# ==============================================================
# PLOT MESH
# ==============================================================

def plotH8Mesh(coords, elem):
    faces = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ], dtype=np.int64)

    fig = plt.figure(facecolor="w")
    ax = fig.add_subplot(111, projection="3d")

    polys = []
    for e in range(elem.shape[0]):
        Xe = coords[elem[e, :], :]
        for f in faces:
            polys.append(Xe[f, :])

    pc = Poly3DCollection(
        polys,
        facecolors=(0.82, 0.85, 0.92, 0.55),
        edgecolors=(0.15, 0.15, 0.15, 0.55),
        linewidths=0.15
    )
    ax.add_collection3d(pc)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    set_axes_equal(ax)
    ax.view_init(elev=25, azim=35)
    plt.tight_layout()
    fig.savefig("cube_mesh.pdf", bbox_inches="tight")
    plt.close(fig)

# ==============================================================
# PLOT DISSIPATION
# ==============================================================

def plotDissipation3D(prob, Dnode):
    coords = prob.coords
    elem = prob.elem

    faces = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ], dtype=np.int64)

    fig = plt.figure(facecolor="w")
    ax = fig.add_subplot(111, projection="3d")

    cmap = cm.get_cmap("jet", 256)
    vmin = np.min(Dnode)
    vmax = np.max(Dnode)

    polys = []
    colors = []
    for e in range(elem.shape[0]):
        Xe = coords[elem[e, :], :]
        De = Dnode[elem[e, :]]

        for f in faces:
            polys.append(Xe[f, :])
            cval = np.mean(De[f])
            colors.append(cmap((cval - vmin) / (vmax - vmin + 1e-15)))

    pc = Poly3DCollection(
        polys,
        facecolors=colors,
        edgecolors="none",
        linewidths=0.0,
        alpha=1.0
    )
    ax.add_collection3d(pc)

    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(Dnode)
    mappable.set_clim(vmin, vmax)
    fig.colorbar(mappable, ax=ax, shrink=0.7)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Plastic dissipation density")
    set_axes_equal(ax)
    ax.view_init(elev=25, azim=35)
    plt.tight_layout()
    fig.savefig("cube_dissipation_3d.pdf", bbox_inches="tight")
    plt.close(fig)

def plotDissipation3Views(prob, Dnode):
    coords = prob.coords
    elem   = prob.elem
    Dnode  = np.asarray(Dnode, dtype=np.float64).reshape(-1)

    faces = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ], dtype=np.int64)

    views = [
        (-45, 25),
        (45, 25),
        (135, 25),
    ]

    titles = ["View 1", "View 2", "View 3"]

    fig = plt.figure(facecolor="w", figsize=(12, 3.6))
    cmap = cm.get_cmap("viridis", 256)

    vmin = np.min(Dnode)
    vmax = np.max(Dnode)

    for iv in range(3):
        ax = fig.add_subplot(1, 3, iv + 1, projection="3d")

        polys = []
        colors = []

        for e in range(elem.shape[0]):
            Xe = coords[elem[e, :], :]
            De = Dnode[elem[e, :]]

            for f in faces:
                polys.append(Xe[f, :])
                cval = np.mean(De[f])
                colors.append(cmap((cval - vmin) / (vmax - vmin + 1e-15)))

        pc = Poly3DCollection(
            polys,
            facecolors=colors,
            edgecolors="none",
            linewidths=0.0,
            alpha=1.0
        )
        ax.add_collection3d(pc)

        set_axes_equal(ax)
        ax.view_init(elev=views[iv][1], azim=views[iv][0])
        ax.set_title(titles[iv], fontname="Times New Roman", fontsize=16)
        ax.set_xlabel("x-axis", fontname="Times New Roman")
        ax.set_ylabel("y-axis", fontname="Times New Roman")
        ax.set_zlabel("z-axis", fontname="Times New Roman")
        ax.grid(False)

    plt.tight_layout()
    fig.savefig("cube_dissipation_3views.pdf", bbox_inches="tight")
    plt.close(fig)

# ==============================================================
# PLOT HISTORY
# ==============================================================

def plotHistory(iterHist, lambdaHist, nAdam):
    iterHist = np.asarray(iterHist)
    lambdaHist = np.asarray(lambdaHist)

    fig, ax = plt.subplots(facecolor="w")
    idAdam = iterHist <= nAdam

    ax.plot(iterHist[idAdam], lambdaHist[idAdam], "b-", linewidth=2)

    if np.any(~idAdam):
        ax.plot(iterHist[~idAdam], lambdaHist[~idAdam], "r-", linewidth=2)
        ax.axvline(nAdam, color="k", linestyle="--")
        ax.legend(["Adam", "LBFGS"], loc="best")
    else:
        ax.legend(["Adam"], loc="best")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\lambda^+$")
    ax.grid(True)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig("cube_history.pdf", bbox_inches="tight")
    plt.close(fig)

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)

    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)

    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":

    prob = buildProblem(prob)

    plotH8Mesh(prob.coords, prob.elem)

    # NETWORK: input [x,y,z], output [u,v,w]
    net = buildNet(3, 3, 64, 4)

    lambdaHist = []
    iterHist = []

    # ADAM TRAINING
    print("Adam training...")

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(1, nAdam + 1):
        optimizer.zero_grad()

        loss, info = modelLossInfo(net, prob, sigma0, betaInc)
        loss.backward()

        optimizer.step()

        if epoch in gridSchedule:
            newGrid = gridSchedule[epoch]
            print(f"Refining cubic B-spline grid to {newGrid} intervals")
            net.net.refine_grid(newGrid)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        lambdaHist.append(float(loss.detach().cpu().item()))
        iterHist.append(epoch)

        if epoch % 100 == 0:
            print(
                "Adam %5d | lambda = %.6f | Wint = %.4e | "
                "Linc = %.4e | Wext = %.4e | alpha = %.4e"
                % (
                    epoch,
                    float(loss.detach().cpu().item()),
                    float(info["Wint"].cpu().item()),
                    float(info["Linc"].cpu().item()),
                    float(info["Wext"].cpu().item()),
                    float(info["alpha"].cpu().item()),
                )
            )

    # LBFGS TRAINING
    print("LBFGS training...")

    optimizer_lbfgs = torch.optim.LBFGS(
        net.parameters(),
        lr=1.0,
        max_iter=1,
        max_eval=20,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    for iter in range(1, nLBFGS + 1):

        def closure():
            optimizer_lbfgs.zero_grad()
            loss = modelLossLBFGS(net, prob, sigma0, betaInc)
            loss.backward()
            return loss

        optimizer_lbfgs.step(closure)

        if iter % 25 == 0:
            loss, info = modelLossInfo(net, prob, sigma0, betaInc)

            lambdaHist.append(float(loss.detach().cpu().item()))
            iterHist.append(nAdam + iter)

            print(
                "LBFGS %5d | lambda = %.6f | Wint = %.4e | "
                "Linc = %.4e | Wext = %.4e | alpha = %.4e"
                % (
                    iter,
                    float(loss.detach().cpu().item()),
                    float(info["Wint"].cpu().item()),
                    float(info["Linc"].cpu().item()),
                    float(info["Wext"].cpu().item()),
                    float(info["alpha"].cpu().item()),
                )
            )

    # POSTPROCESS
    Xnode = torch.tensor(prob.coords, device=device)
    Dnode = nodalDissipation(net, prob, Xnode, sigma0)

    plotDissipation3Views(prob, Dnode)

    plotHistory(iterHist, lambdaHist, nAdam)
