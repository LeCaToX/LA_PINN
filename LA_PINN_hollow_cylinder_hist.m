% ==============================================================
% PINN upper-bound limit analysis
% Thick-walled hollow cylinder under internal pressure
% Up-right quarter model, plane strain, Q4 Gauss integration
% Parametric study for a/b = 0.3, 0.5, 0.7, 0.8
% ==============================================================

clc; clear; close all;
rng(1234);

%% ==============================================================
% GLOBAL PARAMETERS
%% ==============================================================
baseProb.sigma0 = 1.0;
baseProb.b      = 2.0;
baseProb.p      = 1.0;

baseProb.nr = 24;
baseProb.nt = 48;
baseProb.numGauss = 5;

% incompressibility penalty
baseProb.betaInc = 1;

nAdam  = 20000;
nLBFGS = 200;
lr     = 1e-3;

ratioList = [0.8];

allIter = cell(numel(ratioList),1);
allHist = cell(numel(ratioList),1);
caseNames = cell(numel(ratioList),1);

%% ==============================================================
% PARAMETRIC LOOP
%% ==============================================================
for icase = 1:numel(ratioList)

    ratio = ratioList(icase);

    prob = baseProb;
    prob.a = ratio * prob.b;

    tag = sprintf('ab_%03d',round(100*ratio));
    caseNames{icase} = sprintf('a/b = %.1f',ratio);

    refVal = 2/sqrt(3)*prob.sigma0*log(prob.b/prob.a);

    fprintf('\n=========================================\n');
    fprintf('Running case: a/b = %.2f\n',ratio);
    fprintf('a = %.4f, b = %.4f\n',prob.a,prob.b);
    fprintf('Reference lambda = %.6f\n',refVal);
    fprintf('=========================================\n');

    %% BUILD PROBLEM
    prob = buildProblem(prob);

    %plotMesh(prob.node,prob.elem,prob.a,prob.b,tag);
    tol = 1e-9;
    plotCylinderNodes(prob.node,prob.a,prob.b,tol,tag);
    

    %% NETWORK
    net = buildNet(2,2,64,4);

    avgGrad = [];
    avgSqGrad = [];

    lambdaHist = [];
    iterHist   = [];

    %% ADAM
    fprintf('Adam training...\n');

    for epoch = 1:nAdam

        [loss,grad,info] = dlfeval(@modelLossInfo,net,prob);

        lambdaHist(end+1) = double(extractdata(loss));
        iterHist(end+1)   = epoch;

        [net,avgGrad,avgSqGrad] = adamupdate( ...
            net,grad,avgGrad,avgSqGrad,epoch,lr);

        if mod(epoch,100)==0
            fprintf(['Adam %5d | lambda = %.6f | ref = %.6f | ', ...
                     'Wint = %.4e | Wext = %.4e | Linc = %.4e | alpha = %.4e\n'], ...
                epoch, ...
                double(extractdata(loss)), ...
                refVal, ...
                double(extractdata(info.Wint)), ...
                double(extractdata(info.Wext)), ...
                double(extractdata(info.Linc)), ...
                double(extractdata(info.alpha)));
        end
    end

    %% LBFGS
    fprintf('LBFGS training...\n');

    lossFcn = @(net) dlfeval(@modelLossLBFGS,net,prob);
    solverState = lbfgsState;

    for iter = 1:nLBFGS

        [net,solverState] = lbfgsupdate(net,lossFcn,solverState);

        if mod(iter,25)==0

            [loss,~,info] = dlfeval(@modelLossInfo,net,prob);

            lambdaHist(end+1) = double(extractdata(loss));
            iterHist(end+1)   = nAdam + iter;

            fprintf(['LBFGS %5d | lambda = %.6f | ref = %.6f | ', ...
                     'Wint = %.4e | Wext = %.4e | Linc = %.4e | alpha = %.4e\n'], ...
                iter, ...
                double(extractdata(loss)), ...
                refVal, ...
                double(extractdata(info.Wint)), ...
                double(extractdata(info.Wext)), ...
                double(extractdata(info.Linc)), ...
                double(extractdata(info.alpha)));
        end
    end

    %% POSTPROCESS
    Xnode = dlarray(single(prob.node'),'CB');

    [Dnode,ux,uy] = dlfeval(@nodalFields,net,prob,Xnode);

    plotDissipation(prob,Dnode,tag);
    plotVelocity(prob,ux,uy,tag);
    plotHistory(iterHist,lambdaHist,nAdam,tag);

    allIter{icase} = iterHist;
    allHist{icase} = lambdaHist;

    save(['result_' tag '.mat'], ...
        'prob','net','Dnode','ux','uy','iterHist','lambdaHist','refVal');

end

plotAllHistories(allIter,allHist,caseNames,nAdam);

%% ==============================================================
% BUILD PROBLEM
%% ==============================================================
function prob = buildProblem(prob)

[node,elem] = annulusQuarterQ4(prob.a,prob.b,prob.nr,prob.nt);

[Xg,Wg] = domainGaussQ4(node,elem,prob.numGauss);

[Xi,Wi,Ni] = innerPressureGauss(prob.a,prob.nt,prob.numGauss);

prob.node = node;
prob.elem = elem;

prob.XgDL = dlarray(single(Xg'),'CB');
prob.WgDL = dlarray(single(Wg'),'CB');

prob.XiDL = dlarray(single(Xi'),'CB');
prob.WiDL = dlarray(single(Wi'),'CB');
prob.NiDL = dlarray(single(Ni'),'CB');

end

%% ==============================================================
% NETWORK
%% ==============================================================
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

%% ==============================================================
% LOSS FUNCTIONS
%% ==============================================================
function [loss,grad,info] = modelLossInfo(net,prob)

[loss,info] = computeLoss(net,prob);
grad = dlgradient(loss,net.Learnables);

end

function [loss,grad] = modelLossLBFGS(net,prob)

[loss,~] = computeLoss(net,prob);
grad = dlgradient(loss,net.Learnables);

end

%% ==============================================================
% COMPUTE LOSS
%% ==============================================================
function [loss,info] = computeLoss(net,prob)

Wraw = externalWork(net,prob);

if double(extractdata(Wraw)) < 0
    signFix = -1.0;
else
    signFix = 1.0;
end

alpha = signFix ./ (abs(Wraw) + 1e-12);

X = prob.XgDL;
W = prob.WgDL;

uvRaw = forward(net,X);

d = hardBC(X,uvRaw);
d = alpha .* d;

eps = strainRate(X,d);

D = dissipationDensityPlaneStrain(eps,prob.sigma0);

Wint = sum(D .* W,'all');

Wext = externalWorkScaled(net,prob,alpha);

divU = eps(1,:) + eps(2,:);
Linc = sum((divU.^2).*W,'all');

loss = Wint + prob.betaInc*Linc;

info.Wint  = Wint;
info.Wext  = Wext;
info.alpha = alpha;
info.Linc  = Linc;

end

%% ==============================================================
% HARD BC FOR UP-RIGHT QUARTER
%% ==============================================================
function d = hardBC(X,uvRaw)

x = X(1,:);
y = X(2,:);

u0 = uvRaw(1,:);
v0 = uvRaw(2,:);

% symmetry:
% x = 0 -> u = 0
% y = 0 -> v = 0

u = x .* u0;
v = y .* v0;

d = [u; v];

end

%% ==============================================================
% EXTERNAL WORK FROM INTERNAL PRESSURE
%% ==============================================================
function Wext = externalWork(net,prob)

X = prob.XiDL;
W = prob.WiDL;
N = prob.NiDL;

uvRaw = forward(net,X);

d = hardBC(X,uvRaw);

un = sum(d .* N,1);

Wext = sum(prob.p .* un .* W,'all');

end

function Wext = externalWorkScaled(net,prob,alpha)

X = prob.XiDL;
W = prob.WiDL;
N = prob.NiDL;

uvRaw = forward(net,X);

d = hardBC(X,uvRaw);
d = alpha .* d;

un = sum(d .* N,1);

Wext = sum(prob.p .* un .* W,'all');

end

%% ==============================================================
% STRAIN RATE
%% ==============================================================
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

%% ==============================================================
% PLANE STRAIN VON MISES PLASTIC DISSIPATION
%% ==============================================================
function D = dissipationDensityPlaneStrain(eps,sigma0)

% eps = [exx; eyy; gxy]
% engineering shear gxy = du/dy + dv/dx
% incompressible plane-strain von Mises support function:
% D = sigma0 * sqrt((exx - eyy)^2 + gxy^2)

exx = eps(1,:);
eyy = eps(2,:);
gxy = eps(3,:);

quad = (exx - eyy).^2 + gxy.^2;

D = sigma0 .* sqrt(max(quad,1e-18));

end

%% ==============================================================
% Q4 QUARTER ANNULUS MESH
%% ==============================================================
function [node,elem] = annulusQuarterQ4(a,b,nr,nt)

s = linspace(0,1,nr+1);

% refine near inner radius
rList = a + (b-a)*(s.^1.8);

tList = linspace(0,pi/2,nt+1);

node = zeros((nr+1)*(nt+1),2);

id = @(i,j) i + (nr+1)*(j-1);

for j = 1:nt+1

    th = tList(j);

    for i = 1:nr+1

        r = rList(i);

        x = r*cos(th);
        y = r*sin(th);

        node(id(i,j),:) = [x y];

    end
end

elem = zeros(nr*nt,4);

e = 0;

for j = 1:nt
    for i = 1:nr

        n1 = id(i  ,j);
        n2 = id(i+1,j);
        n3 = id(i+1,j+1);
        n4 = id(i  ,j+1);

        e = e + 1;

        elem(e,:) = [n1 n2 n3 n4];

    end
end

end

%% ==============================================================
% Q4 DOMAIN GAUSS INTEGRATION
%% ==============================================================
function [Xg,Wg] = domainGaussQ4(node,elem,ngauss)

[gp,wg] = gauss1D(ngauss);

ng = size(elem,1)*numel(gp)^2;

Xg = zeros(ng,2);
Wg = zeros(ng,1);

c = 0;

for e = 1:size(elem,1)

    Xe = node(elem(e,:),:);

    for i = 1:numel(gp)
        for j = 1:numel(gp)

            xi  = gp(i);
            eta = gp(j);

            [N,dNdxi,dNdeta] = shapeQ4(xi,eta);

            J = [dNdxi'; dNdeta'] * Xe;

            detJ = det(J);

            if detJ <= 0
                error('Negative Jacobian in element %d.',e);
            end

            xg = N' * Xe;

            c = c + 1;

            Xg(c,:) = xg;
            Wg(c) = wg(i)*wg(j)*detJ;

        end
    end
end

Xg = single(Xg);
Wg = single(Wg);

end

%% ==============================================================
% INNER PRESSURE GAUSS INTEGRATION
%% ==============================================================
function [Xg,Wg,Ng] = innerPressureGauss(a,nt,ngauss)

[gp,w1] = gauss1D(ngauss);

thetaList = linspace(0,pi/2,nt+1);

ng = nt*numel(gp);

Xg = zeros(ng,2);
Wg = zeros(ng,1);
Ng = zeros(ng,2);

c = 0;

for e = 1:nt

    th0 = thetaList(e);
    th1 = thetaList(e+1);

    J = (th1-th0)/2;

    for i = 1:numel(gp)

        s = gp(i);

        th = 0.5*(1-s)*th0 + 0.5*(1+s)*th1;

        x = a*cos(th);
        y = a*sin(th);

        % Internal pressure direction on inner boundary
        n = [cos(th), sin(th)];

        c = c + 1;

        Xg(c,:) = [x y];
        Ng(c,:) = n;
        Wg(c) = w1(i)*J*a;

    end
end

Xg = single(Xg);
Wg = single(Wg);
Ng = single(Ng);

end

%% ==============================================================
% SHAPE FUNCTION
%% ==============================================================
function [N,dNdxi,dNdeta] = shapeQ4(xi,eta)

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

%% ==============================================================
% GAUSS RULE
%% ==============================================================
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

%% ==============================================================
% NODAL FIELDS
%% ==============================================================
function [Dnode,ux,uy] = nodalFields(net,prob,Xnode)

Wraw = externalWork(net,prob);

if double(extractdata(Wraw)) < 0
    signFix = -1.0;
else
    signFix = 1.0;
end

alpha = signFix ./ (abs(Wraw) + 1e-12);

uvRaw = forward(net,Xnode);

d = hardBC(Xnode,uvRaw);
d = alpha .* d;

eps = strainRate(Xnode,d);

D = dissipationDensityPlaneStrain(eps,prob.sigma0);

Dnode = double(extractdata(D))';
ux = double(extractdata(d(1,:)))';
uy = double(extractdata(d(2,:)))';

end

%% ==============================================================
% PLOT MESH
%% ==============================================================
function plotMesh(node,elem,a,b,tag)

figure('Color','w');

patch('Faces',elem, ...
      'Vertices',node, ...
      'FaceColor','none', ...
      'EdgeColor',[0.25 0.25 0.25], ...
      'LineWidth',0.35);

axis equal tight;
xlabel('x');
ylabel('y');
box on;

title(['Q4 mesh, ' tag]);

hold on;

th = linspace(0,pi/2,300);
plot(a*cos(th),a*sin(th),'r-','LineWidth',1.2);
plot(b*cos(th),b*sin(th),'k-','LineWidth',1.2);

exportgraphics(gcf,['mesh_' tag '.pdf'],'ContentType','vector');

end

%% ==============================================================
% PLOT DISSIPATION
%% ==============================================================
function plotDissipation(prob,Dnode,tag)

node = prob.node;
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

Dplot = Dnode(:);
Dplot(isnan(Dplot)) = 0;

cmin = prctile(Dplot,5);
cmax = prctile(Dplot,97);

Dplot(Dplot < cmin) = cmin;
Dplot(Dplot > cmax) = cmax;

figure('Color','w');

trisurf(tris,node(:,1),node(:,2), ...
    zeros(size(node,1),1),Dplot, ...
    'EdgeColor','none','FaceColor','interp');

view(2);
axis equal tight;
axis off;

colormap(jet);
colorbar;
caxis([cmin cmax]);

%title(['thick_diss_',num2str(prob.a/prob.b,'%.1f')]);

exportgraphics(gcf,['thick_diss_' tag '.pdf'],'ContentType','vector');

end

%% ==============================================================
% PLOT VELOCITY
%% ==============================================================
function plotVelocity(prob,ux,uy,tag)

node = prob.node;
elem = prob.elem;

scale = 0.2;

nodeDef = node + scale*[ux uy];

figure('Color','w');
hold on;

patch('Faces',elem,'Vertices',node, ...
    'FaceColor',[0.85 0.85 0.85], ...
    'EdgeColor','none', ...
    'FaceAlpha',0.7);

patch('Faces',elem,'Vertices',nodeDef, ...
    'FaceColor','none', ...
    'EdgeColor','r', ...
    'LineWidth',0.4);

quiver(node(:,1),node(:,2),ux,uy,0.6,'k');

axis equal tight;
axis off;

title(['Velocity mechanism, a/b = ',num2str(prob.a/prob.b,'%.1f')]);

exportgraphics(gcf,['velocity_' tag '.pdf'],'ContentType','vector');

end
function plotCylinderNodes(node,a,b,tol,tag)

% =========================================================
% Plot nodes of quarter hollow cylinder with:
% interior nodes      -> blue
% symmetry boundaries -> black
% traction boundary   -> red
%
% INPUT:
% node : nodal coordinates [x y]
% a    : inner radius
% b    : outer radius
% tol  : geometric tolerance
%
% Example:
% plotCylinderNodes(node,1.0,2.0,1e-6)
% =========================================================

if nargin < 4
    tol = 1e-6;
end

x = node(:,1);
y = node(:,2);

r = sqrt(x.^2 + y.^2);

%% ---------------------------------------------------------
% Boundary classification
%% ---------------------------------------------------------

% traction boundary (internal pressure)
idTraction = abs(r - a) < tol;

% outer boundary
idOuter = abs(r - b) < tol;

% symmetry boundaries
idSymX = abs(y) < tol;
idSymY = abs(x) < tol;

idBoundary = idTraction | idOuter | idSymX | idSymY;

% interior nodes
idInterior = ~idBoundary;

%% ---------------------------------------------------------
% Plot
%% ---------------------------------------------------------

figure('Color','w');
hold on;

% interior nodes
scatter(x(idInterior),y(idInterior), ...
    18,[0 0.447 0.741],'filled');

% symmetry + outer boundary
scatter(x(idBoundary & ~idTraction), ...
        y(idBoundary & ~idTraction), ...
    28,'k','filled');

% traction nodes
scatter(x(idTraction),y(idTraction), ...
    36,'r','filled');

%% ---------------------------------------------------------
% Draw cylinder boundaries
%% ---------------------------------------------------------

th = linspace(0,pi/2,400);

plot(a*cos(th),a*sin(th), ...
    'r-','LineWidth',1.5);

plot(b*cos(th),b*sin(th), ...
    'k-','LineWidth',1.5);

%% ---------------------------------------------------------

axis equal tight;
box on;

%xlabel('x');
%ylabel('y');

% legend({'Interior nodes', ...
%         'Boundary nodes', ...
%         'Traction nodes'}, ...
%         'Location','best');

%title('Node classification for hollow cylinder');

set(gca,'FontSize',14);
exportgraphics(gcf,['thick_nodes_' tag '.pdf'],'ContentType','vector');
end

%% ==============================================================
% PLOT HISTORY
%% ==============================================================
function plotHistory(iterHist,lambdaHist,nAdam,tag)

figure('Color','w');
hold on;

idAdam = iterHist <= nAdam;

plot(iterHist(idAdam),lambdaHist(idAdam),'b-','LineWidth',2);

if any(~idAdam)
    plot(iterHist(~idAdam),lambdaHist(~idAdam),'r-','LineWidth',2);

    hx = xline(nAdam,'k--','LineWidth',1.2);
    hx.Annotation.LegendInformation.IconDisplayStyle = 'off';

    legend('Adam','L-BFGS','Location','best');
else
    legend('Adam','Location','best');
end

xlabel('Iteration');
ylabel('\lambda^+');

grid on;
box on;

exportgraphics(gcf,['thick_history_' tag '.pdf'],'ContentType','vector');

end

%% ==============================================================
% PLOT ALL HISTORIES
%% ==============================================================
function plotAllHistories(allIter,allHist,caseNames,nAdam)

figure('Color','w');
hold on;

clr = lines(numel(caseNames));
h = gobjects(numel(caseNames),1);

for k = 1:numel(caseNames)

    iterHist   = allIter{k};
    lambdaHist = allHist{k};

    idAdam = iterHist <= nAdam;

    h(k) = plot(iterHist(idAdam),lambdaHist(idAdam), ...
        '-','Color',clr(k,:),'LineWidth',1.8);

    if any(~idAdam)
        plot(iterHist(~idAdam),lambdaHist(~idAdam), ...
            '--','Color',clr(k,:),'LineWidth',1.8);
    end
end

hx = xline(nAdam,'k--','LineWidth',1.2);
hx.Annotation.LegendInformation.IconDisplayStyle = 'off';

xlabel('Iteration');
ylabel('\lambda^+');

legend(h,caseNames,'Location','best');

grid on;
box on;

exportgraphics(gcf,'thick_history_all_ab_ratios.pdf','ContentType','vector');

end