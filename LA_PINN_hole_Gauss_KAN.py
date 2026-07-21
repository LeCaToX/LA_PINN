import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap
from kan_layers import build_kan

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# PARAMETERS
# =====================================================
sigma0 = 1.0

nx = 20
ny = nx

R = 0.2
a = 1.0

p1 = 1.0
p2 = 1.0

numGauss = 2

nAdam = 4000
nLBFGS = 300
lr = 1e-3
gridSchedule = {1500: 9, 3000: 13}

# =====================================================
# NETWORK
# =====================================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = build_kan(
            2, 2, 64, 4, grid_size=5, input_scale=a,
            device=device, dtype=torch.get_default_dtype()
        )

    def forward(self, x):
        uv = self.net(x)
        u = x[:, 0:1] * uv[:, 0:1]
        v = x[:, 1:2] * uv[:, 1:2]
        return torch.cat([u, v], dim=1)

net = Net().to(device)

# =====================================================
# GAUSS POINTS
# =====================================================
def gauss1D(n):
    if n == 2:
        gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        wg = np.array([1.0, 1.0])

    elif n == 3:
        gp = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
        wg = np.array([5/9, 8/9, 5/9])

    elif n == 5:
        gp = np.array([
            -0.9061798459386640,
            -0.5384693101056831,
             0.0000000000000000,
             0.5384693101056831,
             0.9061798459386640
        ])
        wg = np.array([
            0.2369268850561891,
            0.4786286704993665,
            0.5688888888888889,
            0.4786286704993665,
            0.2369268850561891
        ])

    else:
        raise ValueError(f"Unsupported Gauss order: {n}")

    return gp, wg

# =====================================================
# MESH GENERATION
# =====================================================
def formnode_pla(nx, ny, R, a):
    if nx % 2 == 1:
        raise ValueError("nx must be even")

    dd = 0.25
    ang = np.pi / (2 * nx)
    nk = ny + ny * (ny - 1) * dd / 2

    coords = []

    for ip in range(nx + 1):
        angi = np.pi / 2 - ip * ang

        xi = R * np.cos(angi)
        yi = R * np.sin(angi)

        if ip <= nx // 2:
            ye = a
            xe = ye / np.tan(angi)
        else:
            xe = a
            ye = xe * np.tan(angi)

        dx = 0.0
        dy = 0.0

        for iq in range(ny + 1):
            if iq > 0:
                k = ny - iq
                dx += (1 + k * dd) * (xe - xi) / nk
                dy += (1 + k * dd) * (ye - yi) / nk

            coords.append([xe - dx, ye - dy])

    return np.array(coords)

def buildQ4(nx, ny):
    elem = []

    for i in range(nx):
        for j in range(ny):
            n1 = (ny + 1) * i     + (j + 1)
            n2 = (ny + 1) * (i+1) + (j + 1)
            n3 = (ny + 1) * (i+1) + j
            n4 = (ny + 1) * i     + j
            elem.append([n1, n2, n3, n4])

    return np.array(elem)

coords = formnode_pla(nx, ny, R, a)
elem = buildQ4(nx, ny)

# =====================================================
# Q4 SHAPE FUNCTIONS
# =====================================================
def shape4Q(xi, eta):
    N = 0.25 * np.array([
        (1-xi)*(1-eta),
        (1+xi)*(1-eta),
        (1+xi)*(1+eta),
        (1-xi)*(1+eta)
    ])

    dNdxi = 0.25 * np.array([
        -(1-eta),
         (1-eta),
         (1+eta),
        -(1+eta)
    ])

    dNdeta = 0.25 * np.array([
        -(1-xi),
        -(1+xi),
         (1+xi),
         (1-xi)
    ])

    return N, dNdxi, dNdeta

# =====================================================
# HIGH-ORDER DOMAIN QUADRATURE
# =====================================================
def domainQuadQ4(coords, elem, ngauss):
    gp, wg = gauss1D(ngauss)

    Xg = []
    Wg = []

    for e in range(elem.shape[0]):
        Xe = coords[elem[e, :], :]

        for i in range(len(gp)):
            for j in range(len(gp)):
                xi = gp[i]
                eta = gp[j]

                N, dNdxi, dNdeta = shape4Q(xi, eta)

                J = np.array([dNdxi, dNdeta]) @ Xe
                detJ = np.linalg.det(J)

                if detJ <= 0:
                    raise ValueError(f"Negative or zero Jacobian in element {e}")

                xg = N @ Xe

                Xg.append(xg)
                Wg.append(wg[i] * wg[j] * detJ)

    return np.array(Xg), np.array(Wg)

# =====================================================
# HIGH-ORDER EDGE QUADRATURE
# =====================================================
def edgeQuad(coords, a, edge, ngauss):
    tol = 1e-6

    x = coords[:, 0]
    y = coords[:, 1]

    if edge == "right":
        ids = np.where(np.abs(x - a) < tol)[0]
        ids = ids[np.argsort(y[ids])]

    elif edge == "top":
        ids = np.where(np.abs(y - a) < tol)[0]
        ids = ids[np.argsort(x[ids])]

    else:
        raise ValueError("Unknown edge")

    pts = coords[ids, :]

    gp, wg = gauss1D(ngauss)

    Xg = []
    Wg = []

    for k in range(pts.shape[0] - 1):
        P0 = pts[k, :]
        P1 = pts[k+1, :]

        le = np.linalg.norm(P1 - P0)
        Jedge = le / 2

        for i in range(len(gp)):
            s = gp[i]
            xg = 0.5 * (1-s) * P0 + 0.5 * (1+s) * P1

            Xg.append(xg)
            Wg.append(wg[i] * Jedge)

    return np.array(Xg), np.array(Wg)

Xg_np, Wg_np = domainQuadQ4(coords, elem, numGauss)
Xr_np, Wr_np = edgeQuad(coords, a, "right", numGauss)
Xt_np, Wt_np = edgeQuad(coords, a, "top", numGauss)

Xg = torch.tensor(Xg_np, requires_grad=True, device=device)
Wg = torch.tensor(Wg_np.reshape(-1, 1), device=device)

Xr = torch.tensor(Xr_np, requires_grad=True, device=device)
Wr = torch.tensor(Wr_np.reshape(-1, 1), device=device)

Xt = torch.tensor(Xt_np, requires_grad=True, device=device)
Wt = torch.tensor(Wt_np.reshape(-1, 1), device=device)

# =====================================================
# STRAIN RATE
# =====================================================
def strainRate(x, d):
    u = d[:, 0:1]
    v = d[:, 1:2]

    gu = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    gv = torch.autograd.grad(
        v, x,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        retain_graph=True
    )[0]

    exx = gu[:, 0:1]
    eyy = gv[:, 1:2]
    gxy = gu[:, 1:2] + gv[:, 0:1]

    return exx, eyy, gxy

# =====================================================
# PLASTIC DISSIPATION DENSITY
# =====================================================
def dissipationDensity(exx, eyy, gxy, sigma0):
    quad = (4.0/3.0) * (exx**2 + eyy**2 + exx*eyy) + \
           (1.0/3.0) * gxy**2

    D = sigma0 * torch.sqrt(torch.clamp(quad, min=1e-18))
    return D

# =====================================================
# EXTERNAL WORK
# =====================================================
def externalWork():
    dr = net(Xr)
    ur = dr[:, 0:1]
    WextR = torch.sum(p1 * ur * Wr)

    dt = net(Xt)
    vt = dt[:, 1:2]
    WextT = torch.sum(p2 * vt * Wt)

    return WextR + WextT

# =====================================================
# LOSS FUNCTION
# =====================================================
def lossFunction():
    Wraw = externalWork()
    alpha = 1.0 / (torch.abs(Wraw) + 1e-12)

    d = net(Xg)
    d = alpha * d

    exx, eyy, gxy = strainRate(Xg, d)

    D = dissipationDensity(exx, eyy, gxy, sigma0)

    Wint = torch.sum(D * Wg)

    Wnorm = alpha * torch.abs(Wraw)

    loss = Wint

    return loss, Wraw, Wnorm, alpha

# =====================================================
# TRAINING: ADAM
# =====================================================
opt = torch.optim.Adam(net.parameters(), lr=lr)

lambdaHist = []
iterHist = []

print("Adam training...")

for epoch in range(1, nAdam + 1):
    opt.zero_grad()

    loss, Wraw, Wnorm, alpha = lossFunction()

    loss.backward()
    opt.step()

    if epoch in gridSchedule:
        newGrid = gridSchedule[epoch]
        print(f"Refining cubic B-spline grid to {newGrid} intervals")
        net.net.refine_grid(newGrid)
        opt = torch.optim.Adam(net.parameters(), lr=lr)

    lambdaHist.append(loss.item())
    iterHist.append(epoch)

    if epoch % 100 == 0:
        print(
            f"Adam {epoch:5d} | "
            f"lambda = {loss.item():.6f} | "
            f"Wnorm = {Wnorm.item():.6f} | "
            f"Wraw = {Wraw.item():.6e} | "
            f"alpha = {alpha.item():.6e}"
        )

# =====================================================
# TRAINING: LBFGS
# =====================================================
print("LBFGS training...")

optLBFGS = torch.optim.LBFGS(
    net.parameters(),
    lr=1.0,
    max_iter=20,
    tolerance_grad=1e-9,
    tolerance_change=1e-12,
    line_search_fn="strong_wolfe"
)

for it in range(1, nLBFGS + 1):

    def closure():
        optLBFGS.zero_grad()
        loss, _, _, _ = lossFunction()
        loss.backward()
        return loss

    optLBFGS.step(closure)

    if it % 25 == 0:
        loss, Wraw, Wnorm, alpha = lossFunction()

        lambdaHist.append(loss.item())
        iterHist.append(nAdam + it)

        print(
            f"LBFGS {it:5d} | "
            f"lambda = {loss.item():.6f} | "
            f"Wnorm = {Wnorm.item():.6f} | "
            f"Wraw = {Wraw.item():.6e} | "
            f"alpha = {alpha.item():.6e}"
        )

# =====================================================
# POSTPROCESS: NODAL PLASTIC DISSIPATION
# =====================================================
Xnode = torch.tensor(coords, requires_grad=True, device=device)

Wraw = externalWork()
alpha = 1.0 / (torch.abs(Wraw) + 1e-12)

dnode = net(Xnode)
dnode = alpha * dnode

exx_n, eyy_n, gxy_n = strainRate(Xnode, dnode)
Dnode = dissipationDensity(exx_n, eyy_n, gxy_n, sigma0)

Dplot = Dnode.detach().cpu().numpy().flatten()

# =====================================================
# TRIANGULATION
# =====================================================
tris = []

for e in range(elem.shape[0]):
    n1, n2, n3, n4 = elem[e, :]
    tris.append([n1, n2, n3])
    tris.append([n1, n3, n4])

tris = np.array(tris)

triang = tri.Triangulation(coords[:, 0], coords[:, 1], tris)

# =====================================================
# CUSTOM COLORMAP: green -> yellow -> red
# =====================================================
n = 256

g = np.array([0, 1, 0])
y = np.array([1, 1, 0])
r = np.array([1, 0, 0])

c1 = np.column_stack([
    np.linspace(g[0], y[0], n//2),
    np.linspace(g[1], y[1], n//2),
    np.linspace(g[2], y[2], n//2)
])

c2 = np.column_stack([
    np.linspace(y[0], r[0], n//2),
    np.linspace(y[1], r[1], n//2),
    np.linspace(y[2], r[2], n//2)
])

cmap_array = np.vstack([c1, c2])

custom_cmap = LinearSegmentedColormap.from_list(
    "green_yellow_red",
    cmap_array
)

# =====================================================
# PLOT PLASTIC DISSIPATION
# =====================================================
# =====================================================
# PLOT PLASTIC DISSIPATION - ENHANCED CONTRAST
# =====================================================

fig, ax = plt.subplots(figsize=(8, 8))

# remove invalid values
Dsafe = np.nan_to_num(Dplot, nan=0.0, posinf=0.0, neginf=0.0)

# percentile-based color range
cmin = np.percentile(Dsafe, 5)
cmax = np.percentile(Dsafe, 97)   # try 95, 97, 98

# clip values to enhance red/yellow region
Dclip = np.clip(Dsafe, cmin, cmax)

# use green-yellow-red colormap
tcf = ax.tricontourf(
    triang,
    Dclip,
    levels=150,
    cmap=custom_cmap,
    vmin=cmin,
    vmax=cmax
)

ax.set_aspect("equal")
#ax.set_xlabel("x")
#ax.set_ylabel("y")

#cbar = plt.colorbar(tcf, ax=ax)
#cbar.set_label("Plastic dissipation density")

plt.savefig(
    f"Diss_GaussFEM{nx}{numGauss}_enhanced2.pdf",
    dpi=600,
    bbox_inches="tight"
)

# =====================================================
# PLOT HISTORY: ADAM + LBFGS COLORS
# =====================================================
plt.figure(figsize=(8, 5))

adam_iter = [it for it in iterHist if it <= nAdam]
adam_lam = lambdaHist[:len(adam_iter)]

plt.semilogy(
    adam_iter,
    adam_lam,
    "b-",
    linewidth=2,
    label="Adam"
)

if len(iterHist) > len(adam_iter):
    lbfgs_iter = iterHist[len(adam_iter):]
    lbfgs_lam = lambdaHist[len(adam_iter):]

    plt.semilogy(
        lbfgs_iter,
        lbfgs_lam,
        "r-",
        linewidth=2,
        label="LBFGS"
    )

    plt.axvline(
        x=nAdam,
        color="k",
        linestyle="--",
        linewidth=1.5
    )

plt.grid(True, which="both")
plt.xlabel("Iteration")
plt.ylabel(r"$\lambda^+$")
plt.title("Upper-bound convergence")
plt.legend()

plt.savefig(
    "UB_history_adam_lbfgs.pdf",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
