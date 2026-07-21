% Limit analysis using MLP with multiple high-order Gauss integration
clc; clear; close all;
rng(1234);

%% PARAMETERS
sigma0 = 1.0;

baseProb.nx = 80;
baseProb.ny = baseProb.nx;
baseProb.R  = 0.2;
baseProb.a  = 1.0;
baseProb.p1 = 1.0;
baseProb.p2 = 1.0;

numGaussList = [2 3 5];

nAdam  = 4000;
nLBFGS = 900;
lr     = 1e-3;

allIterHist   = cell(length(numGaussList),1);
allLambdaHist = cell(length(numGaussList),1);

%% LOOP OVER GAUSS ORDERS
for ig = 1:length(numGaussList)

    fprintf('\n========================================\n');
    fprintf('Running numGauss = %d\n',numGaussList(ig));
    fprintf('========================================\n');

    prob = baseProb;
    prob.numGauss = numGaussList(ig);

    prob = buildProblem(prob);
    %plotInitialNodes(prob);
    net = buildNet(2,2,64,4);

    avgGrad = [];
    avgSqGrad = [];

    lambdaHist = [];
    iterHist   = [];

    %% ADAM
    fprintf('Adam training...\n');

    for epoch = 1:nAdam

        [loss,grad,info] = dlfeval(@modelLossInfo,net,prob,sigma0);

        lambdaHist(end+1) = double(extractdata(loss));
        iterHist(end+1)   = epoch;

        [net,avgGrad,avgSqGrad] = adamupdate( ...
            net,grad,avgGrad,avgSqGrad,epoch,lr);

        if mod(epoch,100)==0
            fprintf('Gauss %d | Adam %5d | lambda = %.6f | Wnorm = %.6f | Wraw = %.6e | alpha = %.6e\n', ...
                prob.numGauss,epoch, ...
                double(extractdata(loss)), ...
                double(extractdata(info.Wnorm)), ...
                double(extractdata(info.Wraw)), ...
                double(extractdata(info.alpha)));
        end
    end

    %% LBFGS
    fprintf('LBFGS training...\n');

    lossFcn = @(net) dlfeval(@modelLossLBFGS,net,prob,sigma0);
    solverState = lbfgsState;

    for iter = 1:nLBFGS

        [net,solverState] = lbfgsupdate(net,lossFcn,solverState);

        if mod(iter,25)==0

            [loss,~,info] = dlfeval(@modelLossInfo,net,prob,sigma0);

            lambdaHist(end+1) = double(extractdata(loss));
            iterHist(end+1)   = nAdam + iter;

            fprintf('Gauss %d | LBFGS %5d | lambda = %.6f | Wnorm = %.6f | Wraw = %.6e | alpha = %.6e\n', ...
                prob.numGauss,iter, ...
                double(extractdata(loss)), ...
                double(extractdata(info.Wnorm)), ...
                double(extractdata(info.Wraw)), ...
                double(extractdata(info.alpha)));
        end
    end

    allIterHist{ig}   = iterHist;
    allLambdaHist{ig} = lambdaHist;

    %% POSTPROCESS ONLY LAST GAUSS CASE
    % if ig == length(numGaussList)
    %     Xnode = dlarray(single(prob.coords'),'CB');
    %     Dnode = dlfeval(@nodalDissipation,net,prob,Xnode,sigma0);
    %     plotDissipation(prob,Dnode);
    % end

end

plotLambdaHistoryAll(allIterHist,allLambdaHist,numGaussList,nAdam,baseProb.nx);

%% =====================================================
% BUILD PROBLEM
%% =====================================================
function prob = buildProblem(prob)

coords = formnode_pla(prob.nx,prob.ny,prob.R,prob.a);
elem   = buildQ4(prob.nx,prob.ny);

[Xg,Wg] = domainQuadQ4(coords,elem,prob.numGauss);
[Xr,Wr] = edgeQuad(coords,prob.a,'right',prob.numGauss);
[Xt,Wt] = edgeQuad(coords,prob.a,'top',prob.numGauss);

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
gxy = eps(3,:);

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
% HIGH-ORDER DOMAIN QUADRATURE
%% =====================================================
function [Xg,Wg] = domainQuadQ4(coords,elem,ngauss)

[gp,wg] = gauss1D(ngauss);

ng = size(elem,1) * numel(gp)^2;

Xg = zeros(ng,2);
Wg = zeros(ng,1);

c = 0;

for e = 1:size(elem,1)

    Xe = coords(elem(e,:),:);

    for i = 1:numel(gp)
        for j = 1:numel(gp)

            xi  = gp(i);
            eta = gp(j);

            [N,dNdxi,dNdeta] = shape4Q(xi,eta);

            J = [dNdxi'; dNdeta'] * Xe;
            detJ = det(J);

            if detJ <= 0
                error('Negative or zero Jacobian in element %d.',e);
            end

            xg = N' * Xe;

            c = c + 1;
            Xg(c,:) = xg;
            Wg(c)   = wg(i) * wg(j) * detJ;

        end
    end
end

Xg = single(Xg);
Wg = single(Wg);

end

%% =====================================================
% HIGH-ORDER EDGE QUADRATURE
%% =====================================================
function [Xg,Wg] = edgeQuad(coords,a,edge,ngauss)

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

[gp,wg] = gauss1D(ngauss);

ng = numel(gp) * (size(pts,1)-1);

Xg = zeros(ng,2);
Wg = zeros(ng,1);

c = 0;

for k = 1:size(pts,1)-1

    P0 = pts(k,:);
    P1 = pts(k+1,:);

    le = norm(P1-P0);
    Jedge = le/2;

    for i = 1:numel(gp)

        s = gp(i);

        xg = 0.5*(1-s)*P0 + 0.5*(1+s)*P1;

        c = c + 1;
        Xg(c,:) = xg;
        Wg(c)   = wg(i) * Jedge;

    end
end

Xg = single(Xg);
Wg = single(Wg);

end

%% =====================================================
% GAUSS POINTS
%% =====================================================
function [gp,wg] = gauss1D(n)

switch n

    case 2
        gp = [-1/sqrt(3), 1/sqrt(3)];
        wg = [1, 1];

    case 3
        gp = [-sqrt(3/5), 0, sqrt(3/5)];
        wg = [5/9, 8/9, 5/9];

    case 4
        gp = [-0.8611363115940526, ...
              -0.3399810435848563, ...
               0.3399810435848563, ...
               0.8611363115940526];

        wg = [0.3478548451374538, ...
              0.6521451548625461, ...
              0.6521451548625461, ...
              0.3478548451374538];

    case 5
        gp = [-0.9061798459386640, ...
              -0.5384693101056831, ...
               0.0000000000000000, ...
               0.5384693101056831, ...
               0.9061798459386640];

        wg = [0.2369268850561891, ...
              0.4786286704993665, ...
              0.5688888888888889, ...
              0.4786286704993665, ...
              0.2369268850561891];

    otherwise
        error('Unsupported Gauss order.');

end

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

Dnode = double(extractdata(D))';

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

figure('Color','w');

trisurf(tris,coords(:,1),coords(:,2),zeros(size(coords,1),1),Dnode, ...
    'EdgeColor','none','FaceColor','interp');

view(2);
axis equal tight;
colormap(jet);
colorbar;
xlabel('x');
ylabel('y');
title(sprintf('Normalized upper-bound dissipation, %d x %d Gauss', ...
    prob.numGauss,prob.numGauss));

hold on;
th = linspace(0,pi/2,200);
plot(prob.R*cos(th),prob.R*sin(th),'k--','LineWidth',1.2);

exportgraphics(gcf, ...
    sprintf('UB_dissipation_gauss_%d.pdf',prob.numGauss), ...
    'ContentType','vector');

end

%% =====================================================
% PLOT ALL LAMBDA HISTORIES
%% =====================================================
function plotLambdaHistoryAll(allIterHist,allLambdaHist,numGaussList,nAdam,nx)
figure('Color','w'); hold on;
clr = lines(length(numGaussList));
lgd = {};

for k = 1:length(numGaussList)

    iterHist   = allIterHist{k};
    lambdaHist = allLambdaHist{k};

    nAdamLocal = min(nAdam,numel(iterHist));

    plot(iterHist(1:nAdamLocal), lambdaHist(1:nAdamLocal), ...
         '-','Color',clr(k,:),'LineWidth',2);
    lgd{end+1} = sprintf('%d x %d Gauss - Adam', ...
        numGaussList(k),numGaussList(k));

    if numel(iterHist) > nAdamLocal
        plot(iterHist(nAdamLocal+1:end), lambdaHist(nAdamLocal+1:end), ...
             '--','Color',clr(k,:),'LineWidth',2);
        lgd{end+1} = sprintf('%d x %d Gauss - LBFGS', ...
            numGaussList(k),numGaussList(k));
    end
end
legend(lgd,'Location','best');
lgdObj = legend(lgd,'Location','best');
lgdObj.FontSize = 14;
xlabel('Iteration'); ylabel('\lambda^+');
%title('Convergence histories for different Gauss quadrature orders');
grid on; box on;
exportgraphics(gcf, ...
    sprintf('Hist_gaussp2_%d.pdf',nx), 'ContentType','vector');

end
%---------------------------------------------------
function plotInitialNodes(prob)

coords = prob.coords;

tol = 1e-8;

% domain size inferred from mesh
Lx = max(coords(:,1));
Ly = max(coords(:,2));

r = sqrt(coords(:,1).^2 + coords(:,2).^2);

% boundary node sets
idHole   = abs(r - prob.R) < 1e-6;

idLeft   = abs(coords(:,1)) < tol;
idBottom = abs(coords(:,2)) < tol;

idRight  = abs(coords(:,1)-Lx) < tol;
idTop    = abs(coords(:,2)-Ly) < tol;

idBnd = idHole | idLeft | idBottom | idRight | idTop;
idInt = ~idBnd;
figure('Color','w');hold on;
% interior nodes
plot(coords(idInt,1),coords(idInt,2),'k.','MarkerSize',5);
% hole boundary nodes
plot(coords(idHole,1),coords(idHole,2),'co','MarkerSize',5,'LineWidth',1.2);
% left boundary
plot(coords(idLeft,1),coords(idLeft,2),'bo','MarkerSize',4,'LineWidth',1.0);
% bottom boundary
plot(coords(idBottom,1),coords(idBottom,2),'gs','MarkerSize',4,'LineWidth',1.0);
% right boundary
plot(coords(idRight,1),coords(idRight,2),'ro','MarkerSize',4,'LineWidth',1.0);
% top boundary
plot(coords(idTop,1),coords(idTop,2),'md','MarkerSize',4,'LineWidth',1.0);
% exact hole
% th = linspace(0,pi/2,400);
% plot(prob.R*cos(th), prob.R*sin(th), 'k--', 'LineWidth',1.3);
axis equal tight; box on;

%xlabel('x'); ylabel('y');

% legend( ...
%     'Interior nodes', ...
%     'Hole boundary nodes', ...
%     'Left boundary', ...
%     'Bottom boundary', ...
%     'Right boundary', ...
%     'Top boundary', ...
%     'Exact hole boundary', ...
%     'Location','bestoutside');

exportgraphics(gcf, ...
    sprintf('Data_gauss_%d.pdf',prob.nx),'ContentType','vector');

end