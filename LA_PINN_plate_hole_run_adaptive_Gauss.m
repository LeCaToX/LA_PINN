% ==============================================================
% PINN limit analysis: plate with circular hole
% Adaptive cell-wise Gauss quadrature:
%   base cells: 2x2 Gauss
%   hot cells : 3x3 Gauss
% Final lambda evaluated on fixed 5x5 Gauss over all elements
% ==============================================================

clc; clear; close all;
rng(1234);

%% PARAMETERS
prob.sigma0 = 1.0;

prob.a = 1.0;
prob.R = 0.2;

prob.nx = 40;
prob.ny = 40;

prob.px = 1.0;
prob.py = 0.0;

prob.numGaussBase = 2;
prob.numGaussHot  = 3;
prob.numGaussEval = 5;

nAdam1 = 2000;      % before adaptive update
nAdam2 = 2000;      % after adaptive update
nLBFGS = 500;
lr     = 1e-3;

hotPercentile = 80;   % top 20% cells become hot

%% BUILD PROBLEM WITH BASE 2x2 GAUSS
prob = buildProblem(prob,prob.numGaussBase);

%% BUILD NETWORK
net = buildNet(2,2,64,4);

avgGrad = [];
avgSqGrad = [];

lambdaHist = [];
iterHist   = [];

%% =====================================================
% STAGE 1: ADAM WITH BASE 2x2 GAUSS
%% =====================================================
fprintf('Stage 1: Adam with base 2x2 Gauss...\n');

for epoch = 1:nAdam1

    [loss,grad,info] = dlfeval(@modelLossInfo,net,prob);

    lambdaHist(end+1) = double(extractdata(loss));
    iterHist(end+1)   = epoch;

    [net,avgGrad,avgSqGrad] = adamupdate( ...
        net,grad,avgGrad,avgSqGrad,epoch,lr);

    if mod(epoch,100)==0
        fprintf('Adam1 %5d | lambda = %.6f | Wint = %.6e | Wext = %.6e\n', ...
            epoch, ...
            double(extractdata(loss)), ...
            double(extractdata(info.Wint)), ...
            double(extractdata(info.Wext)));
    end
end

%% =====================================================
% COMPUTE CELL DISSIPATION AND UPDATE HOT ELEMENTS
%% =====================================================
fprintf('Computing cell dissipation and adaptive Gauss update...\n');

Dcell = computeCellDissipation(net,prob);

threshold = prctile(Dcell,hotPercentile);
hotElem = Dcell >= threshold;

fprintf('Hot elements = %d / %d\n',sum(hotElem),numel(hotElem));

[Xint,Wint,elemId] = domainGaussQ4_adaptive( ...
    prob.coords,prob.elem,hotElem,prob.numGaussBase,prob.numGaussHot);

prob.XintDL = dlarray(single(Xint'),'CB');
prob.WintDL = dlarray(single(Wint'),'CB');
prob.elemId = elemId;
prob.hotElem = hotElem;

%% =====================================================
% STAGE 2: CONTINUE ADAM WITH ADAPTIVE GAUSS
%% =====================================================
fprintf('Stage 2: Adam with adaptive cell-wise Gauss...\n');

for epoch = 1:nAdam2

    globalEpoch = nAdam1 + epoch;

    [loss,grad,info] = dlfeval(@modelLossInfo,net,prob);

    lambdaHist(end+1) = double(extractdata(loss));
    iterHist(end+1)   = globalEpoch;

    [net,avgGrad,avgSqGrad] = adamupdate( ...
        net,grad,avgGrad,avgSqGrad,globalEpoch,lr);

    if mod(epoch,100)==0
        fprintf('Adam2 %5d | lambda = %.6f | Wint = %.6e | Wext = %.6e\n', ...
            epoch, ...
            double(extractdata(loss)), ...
            double(extractdata(info.Wint)), ...
            double(extractdata(info.Wext)));
    end
end

%% =====================================================
% LBFGS WITH ADAPTIVE GAUSS
%% =====================================================
fprintf('LBFGS with adaptive Gauss...\n');

lossFcn = @(net) dlfeval(@modelLossLBFGS,net,prob);
solverState = lbfgsState;

for iter = 1:nLBFGS

    [net,solverState] = lbfgsupdate(net,lossFcn,solverState);

    if mod(iter,25)==0

        [loss,~,info] = dlfeval(@modelLossInfo,net,prob);

        lambdaHist(end+1) = double(extractdata(loss));
        iterHist(end+1)   = nAdam1 + nAdam2 + iter;

        fprintf('LBFGS %5d | lambda = %.6f | Wint = %.6e | Wext = %.6e\n', ...
            iter, ...
            double(extractdata(loss)), ...
            double(extractdata(info.Wint)), ...
            double(extractdata(info.Wext)));
    end
end

%% =====================================================
% FINAL LAMBDA EVALUATED WITH FIXED 5x5 GAUSS
%% =====================================================
fprintf('Evaluating final lambda on fixed 5x5 Gauss...\n');

[Xeval,Weval] = domainGaussQ4(prob.coords,prob.elem,prob.numGaussEval);

XevalDL = dlarray(single(Xeval'),'CB');
WevalDL = dlarray(single(Weval'),'CB');

lambdaFinal = dlfeval(@evaluateFinalLambdaFixedGauss,net,prob,XevalDL,WevalDL);

fprintf('\nFinal lambda evaluated by fixed 5x5 Gauss = %.8f\n', ...
    double(extractdata(lambdaFinal)));

%% POSTPROCESS
Xnode = dlarray(single(prob.coords'),'CB');
Dnode = dlfeval(@nodalDissipation,net,prob,Xnode);

plotHotElements(prob,Dcell,hotElem);
%plotDissipation(prob,Dnode);
plotHistory(iterHist,lambdaHist,nAdam1,nAdam2);
title(sprintf('Training history, final fixed 5x5 lambda = %.6f', ...
    double(extractdata(lambdaFinal))));

coords = prob.coords; elem   = prob.elem; tris = zeros(2*size(elem,1),3);
for e = 1:size(elem,1)
    n1 = elem(e,1); n2 = elem(e,2);
    n3 = elem(e,3); n4 = elem(e,4);
    tris(2*e-1,:) = [n1 n2 n3];
    tris(2*e,:)   = [n1 n3 n4];
end
Dplot = Dnode; Dplot(isnan(Dplot)) = 0; Dplot(isinf(Dplot)) = 0;

cmin = prctile(Dplot,2); cmax = prctile(Dplot,98);

Dplot(Dplot < cmin) = cmin; Dplot(Dplot > cmax) = cmax;

figure('Color','w');
trisurf(tris,coords(:,1),coords(:,2),zeros(size(coords,1),1),Dplot, ...
    'EdgeColor','none', 'FaceColor','interp');

view(2); axis equal tight; axis off;

colormap(jet);
%colorbar; caxis([cmin cmax]);
hold on;

% % draw circular hole
% th = linspace(0,pi/2,400);
% fill(prob.R*cos(th),prob.R*sin(th),'w', ...
%     'EdgeColor','k', ...
%     'LineWidth',1.4);

% draw outer boundary
plot([0 prob.a prob.a 0 0], ...
     [0 0 prob.a prob.a 0], ...
     'k-', ...
     'LineWidth',1.0);

title('Plastic dissipation density', ...
    'FontWeight','bold');
exportgraphics(gcf, ...
    'plastic_dissipation_density.pdf', ...
    'ContentType','vector');
%% =====================================================
% BUILD PROBLEM
%% =====================================================
function prob = buildProblem(prob,ngauss)

coords = formnode_pla(prob.nx,prob.ny,prob.R,prob.a);
elem   = buildQ4(prob.nx,prob.ny);

hotElem = false(size(elem,1),1);

[Xint,Wint,elemId] = domainGaussQ4_adaptive(coords,elem,hotElem,ngauss,ngauss);

[Xr,Wr] = edgeGauss(coords,prob.a,'right',prob.numGaussEval);
[Xt,Wt] = edgeGauss(coords,prob.a,'top',prob.numGaussEval);

prob.coords = coords;
prob.elem   = elem;

prob.XintDL = dlarray(single(Xint'),'CB');
prob.WintDL = dlarray(single(Wint'),'CB');
prob.elemId = elemId;
prob.hotElem = hotElem;

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
% LOSS
%% =====================================================
function [loss,grad,info] = modelLossInfo(net,prob)

[loss,info] = computeLoss(net,prob);
grad = dlgradient(loss,net.Learnables);

end

function [loss,grad] = modelLossLBFGS(net,prob)

[loss,~] = computeLoss(net,prob);
grad = dlgradient(loss,net.Learnables);

end

function [loss,info] = computeLoss(net,prob)

Wraw = externalWork(net,prob);
alpha = 1.0 ./ (abs(Wraw) + 1e-12);

X = prob.XintDL;
W = prob.WintDL;

uvRaw = forward(net,X);

d = hardBC(X,uvRaw);
d = alpha .* d;

eps = strainRate(X,d);
D = dissipationDensityPlaneStress(eps,prob.sigma0);

Wint = sum(D .* W,'all');

loss = Wint;

info.Wint  = Wint;
info.Wext  = Wraw;
info.alpha = alpha;

end

%% =====================================================
% FINAL FIXED-GAUSS EVALUATION
%% =====================================================
function lambda = evaluateFinalLambdaFixedGauss(net,prob,XevalDL,WevalDL)

Wraw = externalWork(net,prob);
alpha = 1.0 ./ (abs(Wraw) + 1e-12);

uvRaw = forward(net,XevalDL);

d = hardBC(XevalDL,uvRaw);
d = alpha .* d;

eps = strainRate(XevalDL,d);

D = dissipationDensityPlaneStress(eps,prob.sigma0);

lambda = sum(D .* WevalDL,'all');

end

%% =====================================================
% HARD BC
%% =====================================================
function d = hardBC(X,uvRaw)

x = X(1,:);
y = X(2,:);

u = x .* uvRaw(1,:);
v = y .* uvRaw(2,:);

d = [u; v];

end

%% =====================================================
% EXTERNAL WORK
%% =====================================================
function Wext = externalWork(net,prob)

% right edge x = a, traction [px,0]
Xr = prob.XrDL;
Wr = prob.WrDL;

uvr = forward(net,Xr);
dr = hardBC(Xr,uvr);

ur = dr(1,:);
WextR = sum(prob.px .* ur .* Wr,'all');

% top edge y = a, traction [0,py]
Xt = prob.XtDL;
Wt = prob.WtDL;

uvt = forward(net,Xt);
dt = hardBC(Xt,uvt);

vt = dt(2,:);
WextT = sum(prob.py .* vt .* Wt,'all');

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
% PLANE-STRESS VON MISES PLASTIC DISSIPATION
%% =====================================================
function D = dissipationDensityPlaneStress(eps,sigma0)

exx = eps(1,:);
eyy = eps(2,:);
gxy = eps(3,:);

quad = (4/3).*(exx.^2 + eyy.^2 + exx.*eyy) + ...
       (1/3).*gxy.^2;

D = sigma0 .* sqrt(max(quad,1e-18));

end

%% =====================================================
% COMPUTE CELL DISSIPATION
%% =====================================================
function Dcell = computeCellDissipation(net,prob)

X = prob.XintDL;

D = dlfeval(@computeDAtPoints,net,prob,X);
D = double(extractdata(D))';

elemId = prob.elemId;
ne = size(prob.elem,1);

Dcell = zeros(ne,1);

for e = 1:ne
    id = elemId == e;
    if any(id)
        Dcell(e) = mean(D(id));
    end
end

end

function D = computeDAtPoints(net,prob,X)

Wraw = externalWork(net,prob);
alpha = 1.0 ./ (abs(Wraw) + 1e-12);

uvRaw = forward(net,X);

d = hardBC(X,uvRaw);
d = alpha .* d;

eps = strainRate(X,d);

D = dissipationDensityPlaneStress(eps,prob.sigma0);

end

%% =====================================================
% FORMNODE_PLA
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
% GAUSS RULE
%% =====================================================
function [gp,wg] = gauss1D(n)

switch n

    case 2
        gp = [-1/sqrt(3), 1/sqrt(3)];
        wg = [1, 1];

    case 3
        gp = [-sqrt(3/5), 0, sqrt(3/5)];
        wg = [5/9, 8/9, 5/9];

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
% FIXED DOMAIN GAUSS
%% =====================================================
function [Xg,Wg] = domainGaussQ4(coords,elem,ngauss)

hotElem = false(size(elem,1),1);
[Xg,Wg] = domainGaussQ4_adaptive(coords,elem,hotElem,ngauss,ngauss);

end

%% =====================================================
% ADAPTIVE DOMAIN GAUSS
%% =====================================================
function [Xg,Wg,elemId] = domainGaussQ4_adaptive(coords,elem,hotElem,ngaussBase,ngaussHot)

Xg = [];
Wg = [];
elemId = [];

for e = 1:size(elem,1)

    if hotElem(e)
        ngauss = ngaussHot;
    else
        ngauss = ngaussBase;
    end

    [gp,wg] = gauss1D(ngauss);

    Xe = coords(elem(e,:),:);

    for i = 1:numel(gp)
        for j = 1:numel(gp)

            xi  = gp(i);
            eta = gp(j);

            [N,dNdxi,dNdeta] = shape4Q(xi,eta);

            J = [dNdxi'; dNdeta'] * Xe;
            detJ = det(J);

            if detJ <= 0
                error('Negative Jacobian in element %d.',e);
            end

            xg = N' * Xe;

            Xg = [Xg; xg]; %#ok<AGROW>
            Wg = [Wg; wg(i)*wg(j)*detJ]; %#ok<AGROW>
            elemId = [elemId; e]; %#ok<AGROW>

        end
    end
end

Xg = single(Xg);
Wg = single(Wg);

end

%% =====================================================
% EDGE GAUSS
%% =====================================================
function [Xg,Wg] = edgeGauss(coords,a,edge,ngauss)

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

Xg = [];
Wg = [];

for k = 1:size(pts,1)-1

    P0 = pts(k,:);
    P1 = pts(k+1,:);

    le = norm(P1-P0);
    Jedge = le/2;

    for i = 1:numel(gp)

        s = gp(i);

        xg = 0.5*(1-s)*P0 + 0.5*(1+s)*P1;

        Xg = [Xg; xg]; %#ok<AGROW>
        Wg = [Wg; wg(i)*Jedge]; %#ok<AGROW>

    end
end

Xg = single(Xg);
Wg = single(Wg);

end

%% =====================================================
% NODAL DISSIPATION
%% =====================================================
function Dnode = nodalDissipation(net,prob,X)

Wraw = externalWork(net,prob);
alpha = 1.0 ./ (abs(Wraw) + 1e-12);

uvRaw = forward(net,X);

d = hardBC(X,uvRaw);
d = alpha .* d;

eps = strainRate(X,d);
D = dissipationDensityPlaneStress(eps,prob.sigma0);

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
axis off;

colormap(jet);
colorbar;

hold on;
th = linspace(0,pi/2,300);
fill(prob.R*cos(th),prob.R*sin(th),'w', ...
    'EdgeColor','k','LineWidth',1.2);

title('Plastic dissipation density');

exportgraphics(gcf,'adaptive_Gauss_dissipation.pdf','ContentType','vector');

end

%% =====================================================
% PLOT HOT ELEMENTS
%% =====================================================
function plotHotElements(prob,Dcell,hotElem)

coords = prob.coords;
elem = prob.elem;

figure('Color','w');
hold on;

for e = 1:size(elem,1)

    Xe = coords(elem(e,:),:);

    if hotElem(e)
        fc = [1 0.35 0.25];
    else
        fc = [0.85 0.85 0.85];
    end

    patch('Faces',[1 2 3 4], ...
          'Vertices',Xe, ...
          'FaceColor',fc, ...
          'EdgeColor',[0.4 0.4 0.4], ...
          'LineWidth',0.2);
end

axis equal tight;
box on;
title('Adaptive hot elements based on cell-averaged dissipation');

% hold on;
% th = linspace(0,pi/2,300);
% plot(prob.R*cos(th),prob.R*sin(th),'k-','LineWidth',1.2);

exportgraphics(gcf,'adaptive_hot_elements.pdf','ContentType','vector');

end

%% =====================================================
% PLOT HISTORY
%% =====================================================
function plotHistory(iterHist,lambdaHist,nAdam1,nAdam2)

figure('Color','w');
hold on;

nAdamTotal = nAdam1 + nAdam2;

id1 = iterHist <= nAdam1;
id2 = iterHist > nAdam1 & iterHist <= nAdamTotal;
id3 = iterHist > nAdamTotal;

h1 = plot(iterHist(id1),lambdaHist(id1),'b-','LineWidth',2);
h2 = plot(iterHist(id2),lambdaHist(id2),'m-','LineWidth',2);

h3 = gobjects(1);
if any(id3)
    h3 = plot(iterHist(id3),lambdaHist(id3),'r-','LineWidth',2);
end

hx1 = xline(nAdam1,'k--','LineWidth',1.2);
hx2 = xline(nAdamTotal,'k--','LineWidth',1.2);

hx1.Annotation.LegendInformation.IconDisplayStyle = 'off';
hx2.Annotation.LegendInformation.IconDisplayStyle = 'off';

xlabel('Iteration');
ylabel('\lambda^+');

if any(id3)
    legend([h1 h2 h3], ...
        {'Adam before adaptive','Adam after adaptive','LBFGS'}, ...
        'Location','best');
else
    legend([h1 h2], ...
        {'Adam before adaptive','Adam after adaptive'}, ...
        'Location','best');
end

grid on;
box on;

exportgraphics(gcf,'adaptive_Gauss_history.pdf','ContentType','vector');

end