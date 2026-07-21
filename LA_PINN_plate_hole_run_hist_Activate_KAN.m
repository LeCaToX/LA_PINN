% ==============================================================
% PINN limit analysis: plate with circular hole
% Compare 5 activation cases:
% 1 tanh
% 2 GELU
% 3 Fourier + tanh
% 4 mixed sine-tanh
% 5 full sine
% ==============================================================

clc; clear; close all;
rng(1234);

%% PARAMETERS
prob.sigma0 = 1.0;
prob.a = 1.0;
prob.R = 0.2;

prob.nx = 20;
prob.ny = 20;

prob.px = 1.0;
prob.py = 0.0;

prob.numGauss = 3;

nAdam  = 4000;
nLBFGS = 500;
lr     = 1e-3;

caseList = {'tanh','kan'};
caseNames = {'MLP-tanh','cubic B-spline KAN'};

%% BUILD PROBLEM
prob = buildProblem(prob);

allHist = cell(numel(caseList),1);
allIter = cell(numel(caseList),1);
allD    = cell(numel(caseList),1);

%% LOOP CASES
for ic = 1:numel(caseList)

    actCase = caseList{ic};

    fprintf('\n====================================\n');
    fprintf('Running case: %s\n',actCase);
    fprintf('====================================\n');

    rng(1234);

    inputDim = getInputDim(actCase);
    if strcmp(actCase,'kan')
        net = buildKAN(inputDim,2,64,4,prob.a);
    else
        net = buildNet(inputDim,2,64,4,actCase);
    end

    avgGrad = [];
    avgSqGrad = [];

    lambdaHist = [];
    iterHist   = [];

    %% ADAM
    for epoch = 1:nAdam

        [loss,grad,info] = dlfeval(@modelLossInfo,net,prob,actCase);

        lambdaHist(end+1) = double(extractdata(loss));
        iterHist(end+1)   = epoch;

        [net,avgGrad,avgSqGrad] = adamupdate( ...
            net,grad,avgGrad,avgSqGrad,epoch,lr);

        if mod(epoch,200)==0
            fprintf('%s | Adam %5d | lambda = %.6f | Wint = %.6e | Wext = %.6e\n', ...
                actCase,epoch, ...
                double(extractdata(loss)), ...
                double(extractdata(info.Wint)), ...
                double(extractdata(info.Wext)));
        end
    end

    %% LBFGS
    lossFcn = @(net) dlfeval(@modelLossLBFGS,net,prob,actCase);
    solverState = lbfgsState;

    for iter = 1:nLBFGS

        [net,solverState] = lbfgsupdate(net,lossFcn,solverState);

        if mod(iter,25)==0

            [loss,~,info] = dlfeval(@modelLossInfo,net,prob,actCase);

            lambdaHist(end+1) = double(extractdata(loss));
            iterHist(end+1)   = nAdam + iter;

            fprintf('%s | LBFGS %5d | lambda = %.6f | Wint = %.6e | Wext = %.6e\n', ...
                actCase,iter, ...
                double(extractdata(loss)), ...
                double(extractdata(info.Wint)), ...
                double(extractdata(info.Wext)));
        end
    end

    %% POSTPROCESS DISSIPATION
    Xnode = dlarray(single(prob.coords'),'CB');
    Dnode = dlfeval(@nodalDissipation,net,prob,Xnode,actCase);

    allHist{ic} = lambdaHist;
    allIter{ic} = iterHist;
    allD{ic}    = Dnode;

end

%% PLOTS
plotAllHistories(allIter,allHist,caseNames,nAdam);
plotAllDissipation(prob,allD,caseNames);

%% =====================================================
% BUILD PROBLEM
%% =====================================================
function prob = buildProblem(prob)

coords = formnode_pla(prob.nx,prob.ny,prob.R,prob.a);
elem   = buildQ4(prob.nx,prob.ny);

[Xint,Wint] = domainGaussQ4(coords,elem,prob.numGauss);
[Xr,Wr] = edgeGauss(coords,prob.a,'right',prob.numGauss);
[Xt,Wt] = edgeGauss(coords,prob.a,'top',prob.numGauss);

prob.coords = coords;
prob.elem   = elem;

prob.XintDL = dlarray(single(Xint'),'CB');
prob.WintDL = dlarray(single(Wint'),'CB');

prob.XrDL = dlarray(single(Xr'),'CB');
prob.WrDL = dlarray(single(Wr'),'CB');

prob.XtDL = dlarray(single(Xt'),'CB');
prob.WtDL = dlarray(single(Wt'),'CB');

end

%% =====================================================
% NETWORK
%% =====================================================
function inputDim = getInputDim(actCase)

switch actCase
    case 'fourier_tanh'
        nFreq = 4;
        inputDim = 2 + 2*2*nFreq;
    otherwise
        inputDim = 2;
end

end

function net = buildNet(inDim,outDim,width,depth,actCase)

layers = [
    featureInputLayer(inDim,'Normalization','none','Name','input')
    fullyConnectedLayer(width,'Name','fc1')
    activationLayer(actCase,1)
];

for k = 2:depth
    layers = [
        layers
        fullyConnectedLayer(width,'Name',['fc',num2str(k)])
        activationLayer(actCase,k)
    ];
end

layers = [
    layers
    fullyConnectedLayer(outDim,'Name','out')
];

net = dlnetwork(layerGraph(layers));

end

function layer = activationLayer(actCase,k)

switch actCase

    case {'tanh','fourier_tanh'}
        layer = tanhLayer('Name',['tanh',num2str(k)]);

    case 'gelu'
        layer = functionLayer(@geluFun, ...
            'Name',['gelu',num2str(k)], ...
            'Formattable',true);

    case 'mixed_sine_tanh'
        if mod(k,2)==1
            layer = functionLayer(@sin, ...
                'Name',['sin',num2str(k)], ...
                'Formattable',true);
        else
            layer = tanhLayer('Name',['tanh',num2str(k)]);
        end

    case 'full_sine'
        layer = functionLayer(@sin, ...
            'Name',['sin',num2str(k)], ...
            'Formattable',true);

    otherwise
        error('Unknown activation case.');

end

end

function Y = geluFun(X)

Y = 0.5 .* X .* ...
    (1 + tanh(sqrt(2/pi) .* (X + 0.044715 .* X.^3)));

end

%% =====================================================
% FORWARD WITH FEATURE TRANSFORM
%% =====================================================
function Y = forwardCase(net,X,actCase)

switch actCase

    case 'fourier_tanh'
        Xf = fourierFeatures(X);
        Y = forward(net,Xf);

    otherwise
        Y = forward(net,X);

end

end

function Xf = fourierFeatures(X)

freq = single([1 2 4 8]) * pi;

Xf = X;

for k = 1:numel(freq)
    Xf = [Xf; sin(freq(k).*X); cos(freq(k).*X)];
end

end

%% =====================================================
% LOSS
%% =====================================================
function [loss,grad,info] = modelLossInfo(net,prob,actCase)

[loss,info] = computeLoss(net,prob,actCase);
grad = dlgradient(loss,net.Learnables);

end

function [loss,grad] = modelLossLBFGS(net,prob,actCase)

[loss,~] = computeLoss(net,prob,actCase);
grad = dlgradient(loss,net.Learnables);

end

function [loss,info] = computeLoss(net,prob,actCase)

Wraw = externalWork(net,prob,actCase);
alpha = 1.0 ./ (abs(Wraw) + 1e-12);

X = prob.XintDL;
W = prob.WintDL;

uvRaw = forwardCase(net,X,actCase);

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
function Wext = externalWork(net,prob,actCase)

Xr = prob.XrDL;
Wr = prob.WrDL;

uvr = forwardCase(net,Xr,actCase);
dr = hardBC(Xr,uvr);

ur = dr(1,:);
WextR = sum(prob.px .* ur .* Wr,'all');

Xt = prob.XtDL;
Wt = prob.WtDL;

uvt = forwardCase(net,Xt,actCase);
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
% PLANE-STRESS PLASTIC DISSIPATION
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
% DOMAIN GAUSS INTEGRATION
%% =====================================================
function [Xg,Wg] = domainGaussQ4(coords,elem,ngauss)

[gp,wg] = gauss1D(ngauss);

ng = size(elem,1)*numel(gp)^2;

Xg = zeros(ng,2);
Wg = zeros(ng,1);

c = 0;

for e = 1:size(elem,1)

    Xe = coords(elem(e,:),:);

    for i = 1:numel(gp)
        for j = 1:numel(gp)

            xi = gp(i);
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
            Wg(c) = wg(i)*wg(j)*detJ;

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

ng = numel(gp)*(size(pts,1)-1);

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
        Wg(c) = wg(i)*Jedge;

    end
end

Xg = single(Xg);
Wg = single(Wg);

end

%% =====================================================
% NODAL DISSIPATION
%% =====================================================
function Dnode = nodalDissipation(net,prob,X,actCase)

Wraw = externalWork(net,prob,actCase);
alpha = 1.0 ./ (abs(Wraw) + 1e-12);

uvRaw = forwardCase(net,X,actCase);

d = hardBC(X,uvRaw);
d = alpha .* d;

eps = strainRate(X,d);
D = dissipationDensityPlaneStress(eps,prob.sigma0);

Dnode = double(extractdata(D))';

end

%% =====================================================
% PLOT HISTORIES
%% =====================================================
function plotAllHistories(allIter,allHist,caseNames,nAdam)

figure('Color','w');
hold on;

clr = lines(numel(caseNames));

h = gobjects(numel(caseNames),1);

for k = 1:numel(caseNames)

    iterHist   = allIter{k};
    lambdaHist = allHist{k};

    idAdam = iterHist <= nAdam;

    % Adam line
    h(k) = plot(iterHist(idAdam), ...
                lambdaHist(idAdam), ...
                '-','Color',clr(k,:), ...
                'LineWidth',1.8);

    % LBFGS line
    if any(~idAdam)

        plot(iterHist(~idAdam), ...
             lambdaHist(~idAdam), ...
             '--','Color',clr(k,:), ...
             'LineWidth',1.8);

    end
end

xline(nAdam,'k--','LineWidth',1.2);

xlabel('Iteration');
ylabel('\lambda^+');

legend(h,caseNames,'Location','best');

grid on;
box on;

exportgraphics(gcf,'activation_comparison_history.pdf', ...
    'ContentType','vector');

end

%% =====================================================
% PLOT ALL DISSIPATION
%% =====================================================
function plotAllDissipation(prob,allD,caseNames)

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

for k = 1:numel(caseNames)

    subplot(2,3,k);

    Dplot = allD{k};
    Dplot(isnan(Dplot)) = 0;

    cmin = prctile(Dplot,5);
    cmax = prctile(Dplot,97);

    Dplot(Dplot < cmin) = cmin;
    Dplot(Dplot > cmax) = cmax;

    trisurf(tris,coords(:,1),coords(:,2), ...
        zeros(size(coords,1),1),Dplot, ...
        'EdgeColor','none','FaceColor','interp');

    view(2);
    axis equal tight;
    axis off;

    title(caseNames{k});

    colormap(gca,localGYR());
    caxis([cmin cmax]);

    hold on;
    % th = linspace(0,pi/2,300);
    % fill(prob.R*cos(th),prob.R*sin(th),'w', ...
    %     'EdgeColor','k','LineWidth',1.0);

end

exportgraphics(gcf,'activation_comparison_dissipation.pdf','ContentType','vector');

end

function cmap = localGYR()

n = 256;

g  = [0 0.75 0];
y0 = [1 1 0];
r  = [1 0 0];

cmap = [
    linspace(g(1),y0(1),n/2)', ...
    linspace(g(2),y0(2),n/2)', ...
    linspace(g(3),y0(3),n/2)';
    linspace(y0(1),r(1),n/2)', ...
    linspace(y0(2),r(2),n/2)', ...
    linspace(y0(3),r(3),n/2)'
];

end
