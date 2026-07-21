% Limit analysis using MLP: OK May 2026
clc; clear; close all; 
rng(1234); % 

%% PARAMETERS
sigma0 = 1.0;

prob.nx = 20;
prob.ny = prob.nx;
prob.R  = 0.2;
prob.a  = 1.0;
prob.p1 = 1.0;
prob.p2 = 0;

nAdam  = 4000;
nLBFGS = 300;
lr     = 1e-3;

%% BUILD PROBLEM
prob = buildProblem(prob);

%% BUILD NETWORK
net = buildNet(2,2,64,4);
avgGrad = []; avgSqGrad = [];
%% ===================== ADAM =====================
fprintf('Adam training...\n');
lambdaHist = []; iterHist   = [];
for epoch = 1:nAdam
    [loss,grad,info] = dlfeval(@modelLossInfo,net,prob,sigma0);
    lambdaHist(end+1) = extractdata(loss);
    iterHist(end+1)   = epoch;
    [net,avgGrad,avgSqGrad] = adamupdate(net,grad,avgGrad,avgSqGrad,epoch,lr);

    if mod(epoch,100)==0
        fprintf('Adam %5d | lambda = %.6f | Wnorm = %.6f | Wraw = %.6e | alpha = %.6e\n', ...
            epoch,extractdata(loss),extractdata(info.Wnorm), ...
            extractdata(info.Wraw),extractdata(info.alpha));
    end
end

%% ===================== LBFGS =====================
fprintf('LBFGS training...\n');
lossFcn = @(net) dlfeval(@modelLossLBFGS,net,prob,sigma0);
solverState = lbfgsState;

for iter = 1:nLBFGS
    [net,solverState] = lbfgsupdate(net,lossFcn,solverState);
    if mod(iter,25)==0
        [loss,~,info] = dlfeval(@modelLossInfo,net,prob,sigma0);
        lambdaHist(end+1) = extractdata(loss);
        iterHist(end+1)   = nAdam + iter;
        fprintf('LBFGS %5d | lambda = %.6f | Wnorm = %.6f | Wraw = %.6e | alpha = %.6e\n', ...
            iter,extractdata(loss),extractdata(info.Wnorm), ...
            extractdata(info.Wraw),extractdata(info.alpha));
    end
end

%% POSTPROCESS
Xnode = dlarray(single(prob.coords'),'CB');
Dnode = dlfeval(@nodalDissipation,net,prob,Xnode,sigma0);
plotDissipation(prob,Dnode);
plotLambdaHistory(iterHist,lambdaHist,nAdam);
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

prob.XgDL = dlarray(single(Xg'),'CB');
prob.WgDL = dlarray(single(Wg'),'CB');

prob.XrDL = dlarray(single(Xr'),'CB');
prob.WrDL = dlarray(single(Wr'),'CB');

prob.XtDL = dlarray(single(Xt'),'CB');
prob.WtDL = dlarray(single(Wt'),'CB');

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
% LOSS FOR ADAM
%% =====================================================
function [loss,grad,info] = modelLossInfo(net,prob,sigma0)

[loss,info] = computeLoss(net,prob,sigma0);
grad = dlgradient(loss,net.Learnables);

end

%% =====================================================
% LOSS FOR LBFGS
%% =====================================================
function [loss,grad] = modelLossLBFGS(net,prob,sigma0)

[loss,~] = computeLoss(net,prob,sigma0);
grad = dlgradient(loss,net.Learnables);

end

%% =====================================================
% COMMON LOSS
%% =====================================================
function [loss,info] = computeLoss(net,prob,sigma0)

Wraw = externalWork(net,prob);

alpha = 1.0 ./ (abs(Wraw) + 1e-12);

X = prob.XgDL;
W = prob.WgDL;

uvRaw = forward(net,X);
d = hardBC(prob,X,uvRaw);
d = alpha .* d;

eps = strainRate(X,d);
D = dissipationDensity(eps,sigma0);

Dint = sum(D .* W,'all');
Wnorm = alpha .* abs(Wraw);

loss = Dint;

info.Wraw  = Wraw;
info.Wnorm = Wnorm;
info.alpha = alpha;

end

%% =====================================================
% HARD BC
%% =====================================================
function d = hardBC(prob,X,uvRaw) %#ok<INUSD>

x = X(1,:);
y = X(2,:);

uh = uvRaw(1,:);
vh = uvRaw(2,:);

u = x .* uh;   % u = 0 on x = 0
v = y .* vh;   % v = 0 on y = 0

d = [u; v];

end

%% =====================================================
% EXTERNAL WORK
%% =====================================================
function Wext = externalWork(net,prob)

Xr = prob.XrDL;
Wr = prob.WrDL;

uvr = forward(net,Xr);
dr  = hardBC(prob,Xr,uvr);

ur = dr(1,:);
WextR = sum(prob.p1 .* ur .* Wr,'all');

Xt = prob.XtDL;
Wt = prob.WtDL;

uvt = forward(net,Xt);
dt  = hardBC(prob,Xt,uvt);

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
gxy = eps(3,:);   % engineering shear: gxy = du_dy + dv_dx

quad = (4/3).*(exx.^2 + eyy.^2 + exx.*eyy) + ...
       (1/3).*gxy.^2;

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
% Q4 SHAPE FUNCTIONS
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

Xg = [];
Wg = [];

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

            Xg = [Xg; xg]; %#ok<AGROW>
            Wg = [Wg; detJ]; %#ok<AGROW>

        end
    end
end

Xg = single(Xg);
Wg = single(Wg);

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

Xg = [];
Wg = [];

for k = 1:size(pts,1)-1

    P0 = pts(k,:);
    P1 = pts(k+1,:);

    le = norm(P1-P0);
    Jedge = le/2;

    for i = 1:2
        s = gp(i);
        w = wg(i);

        xg = 0.5*(1-s)*P0 + 0.5*(1+s)*P1;

        Xg = [Xg; xg]; %#ok<AGROW>
        Wg = [Wg; w*Jedge]; %#ok<AGROW>
    end
end

Xg = single(Xg);
Wg = single(Wg);

end

%% =====================================================
% NODAL DISSIPATION
%% =====================================================
function Dnode = nodalDissipation(net,prob,X,sigma0)

Wraw = externalWork(net,prob);
alpha = 1.0 ./ (abs(Wraw) + 1e-12);

uvRaw = forward(net,X);
d = hardBC(prob,X,uvRaw);
d = alpha .* d;

eps = strainRate(X,d);
D = dissipationDensity(eps,sigma0);

Dnode = extractdata(D)';
Dnode = double(Dnode);

end

%% =====================================================
% PLOT
%% =====================================================
function plotDissipation(prob,Dnode)
coords = prob.coords; elem = prob.elem;
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
colormap(jet);
colorbar;
xlabel('x');
ylabel('y');
title('Normalized upper-bound dissipation');

hold on;
th = linspace(0,pi/2,200);
plot(prob.R*cos(th),prob.R*sin(th),'k--','LineWidth',1.2);

end

function plotLambdaHistory(iterHist,lambdaHist,nAdam)

figure;
hold on;

plot(iterHist(1:nAdam), ...
     lambdaHist(1:nAdam), ...
     'b','LineWidth',2);

plot(iterHist(nAdam+1:end), ...
     lambdaHist(nAdam+1:end), ...
     'r','LineWidth',2);

xlabel('Iteration');
ylabel('\lambda^+');

legend('Adam','LBFGS');

title('Upper-bound collapse multiplier history');

grid on;
box on;

end