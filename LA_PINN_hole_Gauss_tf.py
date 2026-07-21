import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap

torch.set_default_dtype(torch.float64)

device = torch.device("cpu")

# =====================================================
# PARAMETERS
# =====================================================

sigma0 = 1.0

R = 0.2
L = 1.0

px = 1.0
py = 1.0

numElemU = 20
numElemV = numElemU
numGauss = 5

nAdam = 4000
nLBFGS = 300

lr = 1e-3

# =====================================================
# NETWORK
# =====================================================

class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(2,64),
            nn.Tanh(),

            nn.Linear(64,64),
            nn.Tanh(),

            nn.Linear(64,64),
            nn.Tanh(),

            nn.Linear(64,64),
            nn.Tanh(),

            nn.Linear(64,2)

        )

    def forward(self,x):

        uv = self.net(x)

        u = x[:,0:1] * uv[:,0:1]
        v = x[:,1:2] * uv[:,1:2]

        return torch.cat([u,v],1)

net = Net().to(device)

# =====================================================
# GAUSS
# =====================================================

def gauss1D(n):

    if n == 2:

        gp = np.array([
            -1/np.sqrt(3),
             1/np.sqrt(3)
        ])

        wg = np.array([
            1.0,
            1.0
        ])

    elif n == 3:

        gp = np.array([
            -np.sqrt(3/5),
             0.0,
             np.sqrt(3/5)
        ])

        wg = np.array([
            5/9,
            8/9,
            5/9
        ])

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

    return gp,wg

# =====================================================
# INTERIOR GAUSS CELLS
# =====================================================

def quadInteriorCell():

    gp,wg = gauss1D(numGauss)

    dx = L/numElemU
    dy = L/numElemV

    X = []
    W = []

    for i in range(numElemU):
        for j in range(numElemV):

            x0 = i*dx
            x1 = (i+1)*dx

            y0 = j*dy
            y1 = (j+1)*dy

            J = dx*dy/4

            for a in range(len(gp)):
                for b in range(len(gp)):

                    xi  = gp[a]
                    eta = gp[b]

                    x = 0.5*(1-xi)*x0 + 0.5*(1+xi)*x1
                    y = 0.5*(1-eta)*y0 + 0.5*(1+eta)*y1

                    if x**2 + y**2 > R**2:

                        X.append([x,y])
                        W.append(wg[a]*wg[b]*J)

    return np.array(X),np.array(W)

# =====================================================
# BOUNDARY GAUSS CELLS
# =====================================================

def quadBoundaryCell():

    gp,wg = gauss1D(numGauss)

    ds = L/numElemU

    Xb = []
    Wb = []

    tx = []
    ty = []

    # right boundary
    for e in range(numElemU):

        y0 = e*ds
        y1 = (e+1)*ds

        J = ds/2

        for k in range(len(gp)):

            s = gp[k]

            y = 0.5*(1-s)*y0 + 0.5*(1+s)*y1
            x = L

            if x**2 + y**2 > R**2:

                Xb.append([x,y])
                Wb.append(wg[k]*J)

                tx.append(px)
                ty.append(0.0)

    # top boundary
    for e in range(numElemU):

        x0 = e*ds
        x1 = (e+1)*ds

        J = ds/2

        for k in range(len(gp)):

            s = gp[k]

            x = 0.5*(1-s)*x0 + 0.5*(1+s)*x1
            y = L

            if x**2 + y**2 > R**2:

                Xb.append([x,y])
                Wb.append(wg[k]*J)

                tx.append(0.0)
                ty.append(py)

    return (
        np.array(Xb),
        np.array(Wb),
        np.array(tx),
        np.array(ty)
    )

# =====================================================
# BUILD DATA
# =====================================================

Xg,Wg = quadInteriorCell()
Xb,Wb,tx,ty = quadBoundaryCell()

Xg = torch.tensor(Xg,requires_grad=True).to(device)
Wg = torch.tensor(Wg.reshape(-1,1)).to(device)

Xb = torch.tensor(Xb,requires_grad=True).to(device)
Wb = torch.tensor(Wb.reshape(-1,1)).to(device)

tx = torch.tensor(tx.reshape(-1,1)).to(device)
ty = torch.tensor(ty.reshape(-1,1)).to(device)

# =====================================================
# STRAIN RATE
# =====================================================

def strainRate(x,d):

    u = d[:,0:1]
    v = d[:,1:2]

    gu = torch.autograd.grad(
        u,x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    gv = torch.autograd.grad(
        v,x,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        retain_graph=True
    )[0]

    exx = gu[:,0:1]
    eyy = gv[:,1:2]
    gxy = gu[:,1:2] + gv[:,0:1]

    return exx,eyy,gxy

# =====================================================
# DISSIPATION
# =====================================================

def dissipationDensity(exx,eyy,gxy):

    quad = (4/3)*(exx**2 + eyy**2 + exx*eyy) + \
           (1/3)*gxy**2

    D = sigma0 * torch.sqrt(
        torch.clamp(quad,min=1e-18)
    )

    return D

# =====================================================
# EXTERNAL WORK
# =====================================================

def externalWork():

    d = net(Xb)

    u = d[:,0:1]
    v = d[:,1:2]

    Wext = torch.sum(
        (tx*u + ty*v) * Wb
    )

    return Wext

# =====================================================
# LOSS
# =====================================================

def lossFunction():

    Wraw = externalWork()

    alpha = 1.0 / (torch.abs(Wraw) + 1e-12)

    d = net(Xg)

    d = alpha * d

    exx,eyy,gxy = strainRate(Xg,d)

    D = dissipationDensity(exx,eyy,gxy)

    Wint = torch.sum(D * Wg)

    return Wint,Wraw,alpha

# =====================================================
# ADAM
# =====================================================

opt = torch.optim.Adam(
    net.parameters(),
    lr=lr
)

lambdaHist = []
iterHist = []

print("Adam training...")

for epoch in range(1,nAdam+1):

    opt.zero_grad()

    loss,Wraw,alpha = lossFunction()

    loss.backward()

    opt.step()

    lambdaHist.append(loss.item())
    iterHist.append(epoch)

    if epoch % 100 == 0:

        print(
            f"Adam {epoch:5d} | "
            f"lambda = {loss.item():.6e} | "
            f"Wext = {Wraw.item():.6e} | "
            f"alpha = {alpha.item():.6e}"
        )

# =====================================================
# LBFGS
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

for it in range(1,nLBFGS+1):

    def closure():

        optLBFGS.zero_grad()

        loss,_,_ = lossFunction()

        loss.backward()

        return loss

    optLBFGS.step(closure)

    if it % 25 == 0:

        loss,Wraw,alpha = lossFunction()

        lambdaHist.append(loss.item())
        iterHist.append(nAdam + it)

        print(
            f"LBFGS {it:5d} | "
            f"lambda = {loss.item():.6e} | "
            f"Wext = {Wraw.item():.6e} | "
            f"alpha = {alpha.item():.6e}"
        )

# =====================================================
# POSTPROCESS
# =====================================================

xv = np.linspace(0,L,200)
yv = np.linspace(0,L,200)

xx,yy = np.meshgrid(xv,yv)

Xplot = np.column_stack([
    xx.flatten(),
    yy.flatten()
])

mask = Xplot[:,0]**2 + Xplot[:,1]**2 > R**2

Xplot = Xplot[mask]

XplotT = torch.tensor(
    Xplot,
    requires_grad=True
).to(device)

Wraw = externalWork()

alpha = 1.0 / (torch.abs(Wraw) + 1e-12)

d = net(XplotT)
d = alpha * d

ux = d[:,0].detach().cpu().numpy()
uy = d[:,1].detach().cpu().numpy()

exx,eyy,gxy = strainRate(XplotT,d)

Dp = dissipationDensity(exx,eyy,gxy)

Dp = Dp.detach().cpu().numpy().flatten()

# =====================================================
# CUSTOM COLORMAP
# =====================================================

n = 256

g = np.array([0,1,0])
y = np.array([1,1,0])
r = np.array([1,0,0])

c1 = np.column_stack([
    np.linspace(g[0],y[0],n//2),
    np.linspace(g[1],y[1],n//2),
    np.linspace(g[2],y[2],n//2)
])

c2 = np.column_stack([
    np.linspace(y[0],r[0],n//2),
    np.linspace(y[1],r[1],n//2),
    np.linspace(y[2],r[2],n//2)
])

cmap_array = np.vstack([c1,c2])

custom_cmap = LinearSegmentedColormap.from_list(
    "green_yellow_red",
    cmap_array
)

# =====================================================
# PLOT DISSIPATION
# =====================================================

triang = tri.Triangulation(
    Xplot[:,0],
    Xplot[:,1]
)

fig,ax = plt.subplots(figsize=(8,8))

tcf = ax.tricontourf(
    triang,
    Dp,
    100,
    cmap=custom_cmap
)

circle = plt.Circle(
    (0,0),
    R,
    color='white'
)

ax.add_patch(circle)

ax.set_aspect('equal')

plt.colorbar(tcf)

plt.xlabel("x")
plt.ylabel("y")

#plt.title("Plastic dissipation density")

plt.savefig(
    f"Diss_GaussTf{numElemU}{numGauss}.pdf",
    dpi=600,
    bbox_inches='tight'
)

# =====================================================
# PLOT HISTORY
# =====================================================

plt.figure(figsize=(8,5))

plt.semilogy(
    iterHist[:nAdam],
    lambdaHist[:nAdam],
    'b-',
    linewidth=2,
    label='Adam'
)

if len(iterHist) > nAdam:

    plt.semilogy(
        iterHist[nAdam:],
        lambdaHist[nAdam:],
        'r-',
        linewidth=2,
        label='LBFGS'
    )

    plt.axvline(
        x=nAdam,
        color='k',
        linestyle='--',
        linewidth=1.2
    )

plt.grid(True)

plt.xlabel("Iteration")
plt.ylabel(r"$\lambda^+$")

plt.title("Upper-bound limit analysis convergence")

plt.legend()

plt.savefig(
    f"hist_GaussTf{numElemU}{numGauss}.pdf",
    dpi=600,
    bbox_inches='tight'
)

plt.show()