clc; clear; close all;
rng(1234);

%% PARAMETERS
sigma0 = 1.0;

prob.nx = 40;
prob.ny = prob.nx;
prob.R  = 0.2;
prob.a  = 1.0;
prob.p1 = 1.0;
prob.p2 = 0.0;

nEpoch = 3000;
lr = 2e-3;

%% BUILD PROBLEM
prob = buildProblem(prob);

%% BUILD NETWORK
net = buildKAN(2,2,32,3,prob.a);

if canUseGPU
    net = dlupdate(@gpuArray,net);
end

avgGrad = [];
avgSqGrad = [];

%% TRAINING
for epoch = 1:nEpoch

    [loss,grad,info] = dlfeval(@modelLoss,net,prob,sigma0);

    [net,avgGrad,avgSqGrad] = adamupdate( ...
        net,grad,avgGrad,avgSqGrad,epoch,lr);

    if mod(epoch,100)==0
        fprintf('Epoch %5d | lambda = %.6f | Wnorm = %.6f | Wraw = %.6e | alpha = %.6e\n', ...
            epoch,gather(extractdata(loss)), ...
            gather(extractdata(info.Wnorm)), ...
            gather(extractdata(info.Wraw)), ...
            gather(extractdata(info.alpha)));
    end
end

%% POSTPROCESS
Xnode = dlarray(single(prob.coords'),'CB');

if canUseGPU
    Xnode = gpuArray(Xnode);
end

Dnode = dlfeval(@nodalDissipation,net,prob,Xnode,sigma0);
plotDissipation(prob,Dnode);



%% =====================================================
% BUILD PROBLEM
%% =====================================================
function prob = buildProblem(prob)

coords = formnode_pla(prob.nx,prob.ny,prob.R,prob.a);
elem   = buildQ4(prob.nx,prob.ny);

[Xg,Wg] = domainQuadQ4(coords,elem);
[Xr,Wr] = edgeQuad(coords,prob.a,'right');
[Xt,Wt] = edgeQuad(coords,prob.a,'top');

prob.coords = coords;
prob.elem   = elem;

prob.Xg = Xg;
prob.Wg = Wg;
prob.Xr = Xr;
prob.Wr = Wr;
prob.Xt = Xt;
prob.Wt = Wt;

prob.XgDL = dlarray(single(Xg'),'CB');
prob.WgDL = dlarray(single(Wg'),'CB');

prob.XrDL = dlarray(single(Xr'),'CB');
prob.WrDL = dlarray(single(Wr'),'CB');

prob.XtDL = dlarray(single(Xt'),'CB');
prob.WtDL = dlarray(single(Wt'),'CB');

if canUseGPU
    prob.XgDL = gpuArray(prob.XgDL);
    prob.WgDL = gpuArray(prob.WgDL);

    prob.XrDL = gpuArray(prob.XrDL);
    prob.WrDL = gpuArray(prob.WrDL);

    prob.XtDL = gpuArray(prob.XtDL);
    prob.WtDL = gpuArray(prob.WtDL);
end

end

%% =====================================================
% NETWORK
%% =====================================================
function net = buildNet(inDim,outDim,width,depth)

layers = [
    featureInputLayer(inDim,'Normalization','none','Name','input')
    fullyConnectedLayer(width,'Name','fc1')
    tanhLayer('Name','tanh1')
];

for k = 2:depth
    layers = [
        layers
        fullyConnectedLayer(width,'Name',['fc',num2str(k)])
        tanhLayer('Name',['tanh',num2str(k)])
    ];
end

layers = [
    layers
    fullyConnectedLayer(outDim,'Name','out')
];

net = dlnetwork(layerGraph(layers));

end

%% =====================================================
% LOSS FUNCTION
%% =====================================================
function [loss,grad,info] = modelLoss(net,prob,sigma0)

Wraw = externalWork(net,prob);

epsw = 1e-12;
alpha = 1.0 ./ (abs(Wraw) + epsw);

X = prob.XgDL;
W = prob.WgDL;

uvRaw = forward(net,X);
d = hardBC(X,uvRaw);
d = alpha .* d;

eps = strainRate(X,d);
D = dissipationDensity(eps,sigma0);

Dint = sum(D .* W,'all');
Wnorm = alpha .* abs(Wraw);

loss = Dint;

grad = dlgradient(loss,net.Learnables);

info.Wraw  = Wraw;
info.Wnorm = Wnorm;
info.alpha = alpha;

end

%% =====================================================
% HARD BC
%% =====================================================
function d = hardBC(X,uvRaw)

x = X(1,:);
y = X(2,:);

uh = uvRaw(1,:);
vh = uvRaw(2,:);

% Symmetry:
% u = 0 on x = 0
% v = 0 on y = 0
u = x .* uh;
v = y .* vh;

d = [u; v];

end

%% =====================================================
% EXTERNAL WORK
%% =====================================================
function Wext = externalWork(net,prob)

Xr = prob.XrDL;
Wr = prob.WrDL;

uvr = forward(net,Xr);
dr  = hardBC(Xr,uvr);

ur = dr(1,:);
WextR = sum(prob.p1 .* ur .* Wr,'all');

Xt = prob.XtDL;
Wt = prob.WtDL;

uvt = forward(net,Xt);
dt  = hardBC(Xt,uvt);

vt = dt(2,:);
WextT = sum(prob.p2 .* vt .* Wt,'all');

Wext = WextR + WextT;

end

%% =====================================================
% STRAIN RATE
%% =====================================================
function eps = strainRate(X,d)

u = d(1,:);
v = d(2,:);

du = dlgradient(sum(u,'all'),X, ...
    'EnableHigherDerivatives',true, ...
    'RetainData',true);

dv = dlgradient(sum(v,'all'),X, ...
    'EnableHigherDerivatives',true);

du_dx = du(1,:);
du_dy = du(2,:);
dv_dx = dv(1,:);
dv_dy = dv(2,:);

exx = du_dx;
eyy = dv_dy;
gxy = du_dy + dv_dx;

eps = [exx; eyy; gxy];

end

%% =====================================================
% DISSIPATION DENSITY
%% =====================================================
function D = dissipationDensity(eps,sigma0)

exx = eps(1,:);
eyy = eps(2,:);
gxy = eps(3,:);

quad = (4/3).*exx.^2 + ...
       (4/3).*eyy.^2 + ...
       (4/3).*exx.*eyy + ...
       gxy.^2;

D = sigma0 .* sqrt(max(quad,1e-18));

end

%% =====================================================
% MESH GENERATION
%% =====================================================
function coords = formnode_pla(nx,ny,R,a)

if mod(nx,2)==1
    error('NX must be even.');
end

dd = 0.25;
ang = pi/(2*nx);
nk = ny + ny*(ny-1)*dd/2;

coords = zeros((nx+1)*(ny+1),2);
c = 0;

for ip = 1:nx+1

    angi = pi/2 - (ip-1)*ang;

    xi = R*cos(angi);
    yi = R*sin(angi);

    if ip <= nx/2 + 1
        ye = a;
        xe = ye/tan(angi);
    else
        xe = a;
        ye = xe*tan(angi);
    end

    dx = 0;
    dy = 0;

    for iq = 1:ny+1

        if iq > 1
            k = ny - iq + 1;
            dx = dx + (1 + k*dd)*(xe-xi)/nk;
            dy = dy + (1 + k*dd)*(ye-yi)/nk;
        end

        c = c + 1;
        coords(c,:) = [xe-dx, ye-dy];

    end
end

end

%% =====================================================
% Q4 CONNECTIVITY
%% =====================================================
function elem = buildQ4(nx,ny)

elem = zeros(nx*ny,4);
e = 0;

for i = 0:nx-1
    for j = 0:ny-1

        n1 = (ny+1)*i     + (j+1) + 1;
        n2 = (ny+1)*(i+1) + (j+1) + 1;
        n3 = (ny+1)*(i+1) + j     + 1;
        n4 = (ny+1)*i     + j     + 1;

        e = e + 1;
        elem(e,:) = [n1 n2 n3 n4];

    end
end

end

%% =====================================================
% Q4 SHAPE FUNCTION
%% =====================================================
function [N,dNdxi,dNdeta] = shape4Q(xi,eta)

N = 0.25 * [
    (1-xi)*(1-eta)
    (1+xi)*(1-eta)
    (1+xi)*(1+eta)
    (1-xi)*(1+eta)
];

dNdxi = 0.25 * [
    -(1-eta)
     (1-eta)
     (1+eta)
    -(1+eta)
];

dNdeta = 0.25 * [
    -(1-xi)
    -(1+xi)
     (1+xi)
     (1-xi)
];

end

%% =====================================================
% DOMAIN QUADRATURE
%% =====================================================
function [Xg,Wg] = domainQuadQ4(coords,elem)

gp = [-1/sqrt(3), 1/sqrt(3)];

Ng = 4*size(elem,1);
Xg = zeros(Ng,2,'single');
Wg = zeros(Ng,1,'single');

c = 0;

for e = 1:size(elem,1)

    Xe = coords(elem(e,:),:);

    for i = 1:2
        for j = 1:2

            xi  = gp(i);
            eta = gp(j);

            [N,dNdxi,dNdeta] = shape4Q(xi,eta);

            J = [dNdxi'; dNdeta'] * Xe;
            detJ = det(J);

            xg = N' * Xe;

            c = c + 1;
            Xg(c,:) = single(xg);
            Wg(c)   = single(detJ);

        end
    end
end

end

%% =====================================================
% EDGE QUADRATURE
%% =====================================================
function [Xg,Wg] = edgeQuad(coords,a,edge)

tol = 1e-6;

x = coords(:,1);
y = coords(:,2);

switch edge
    case 'right'
        id = find(abs(x-a)<tol);
        [~,ord] = sort(y(id));
        id = id(ord);

    case 'top'
        id = find(abs(y-a)<tol);
        [~,ord] = sort(x(id));
        id = id(ord);

    otherwise
        error('Unknown edge.');
end

pts = coords(id,:);

gp = [-1/sqrt(3), 1/sqrt(3)];
wg = [1,1];

Ns = size(pts,1)-1;
Xg = zeros(2*Ns,2,'single');
Wg = zeros(2*Ns,1,'single');

c = 0;

for k = 1:Ns

    P0 = pts(k,:);
    P1 = pts(k+1,:);

    le = norm(P1-P0);
    Jedge = le/2;

    for i = 1:2

        s = gp(i);
        w = wg(i);

        xg = 0.5*(1-s)*P0 + 0.5*(1+s)*P1;

        c = c + 1;
        Xg(c,:) = single(xg);
        Wg(c)   = single(w*Jedge);

    end
end

end

%% =====================================================
% NODAL DISSIPATION
%% =====================================================
function Dnode = nodalDissipation(net,prob,X,sigma0)

Wraw = externalWork(net,prob);
alpha = 1.0 ./ (abs(Wraw) + 1e-12);

uvRaw = forward(net,X);
d = hardBC(X,uvRaw);
d = alpha .* d;

eps = strainRate(X,d);
D = dissipationDensity(eps,sigma0);

Dnode = gather(extractdata(D))';
Dnode = double(Dnode);

end

%% =====================================================
% PLOT DISSIPATION
%% =====================================================
function plotDissipation(prob,Dnode)

coords = prob.coords;
elem = prob.elem;

tris = zeros(2*size(elem,1),3);

for e = 1:size(elem,1)

    n1 = elem(e,1);
    n2 = elem(e,2);
    n3 = elem(e,3);
    n4 = elem(e,4);

    tris(2*e-1,:) = [n1 n2 n3];
    tris(2*e,:)   = [n1 n3 n4];

end

figure;
trisurf(tris,coords(:,1),coords(:,2),zeros(size(coords,1),1),Dnode, ...
    'EdgeColor','none','FaceColor','interp');

view(2);
axis equal tight;
n = 256; 
g = [0 1 0];      % green for zero
y = [1 1 0];      % yellow
r = [1 0 0];      % red
cmap = [ ...
    [linspace(g(1),y(1),n/2)', ...
     linspace(g(2),y(2),n/2)', ...
     linspace(g(3),y(3),n/2)']; ...
    [linspace(y(1),r(1),n/2)', ...
     linspace(y(2),r(2),n/2)', ...
     linspace(y(3),r(3),n/2)']
];
colormap(cmap);
%colormap(jet);
colorbar;
xlabel('x');
ylabel('y');
title('Fast normalized upper-bound dissipation');

hold on;
th = linspace(0,pi/2,200);
plot(prob.R*cos(th),prob.R*sin(th),'k--','LineWidth',1.2);
exportgraphics(gcf,'fast_KAN_dissipation.pdf','ContentType','vector');

end
