% ==============================================================
% 3D PINN-based upper-bound limit analysis
% One-octant cube with spherical hole
% HEX8 Gauss integration + MLP velocity field
% Loads px, py, pz on x=a, y=a, z=a
% ==============================================================

clc; clear; close all;
rng(1234);

%% PARAMETERS
sigma0 = 1.0;

prob.a = 1.0;      % one-octant size, full cube is 2a x 2a x 2a
prob.R = 0.2;      % spherical hole radius

prob.px = 1.0;
prob.py = 0.0;
prob.pz = 0.0;

prob.nx = 20;
prob.ny = prob.nx;
prob.nz = prob.nx;

prob.numGauss = 2;

betaInc = 1;

nAdam  = 5000;
nLBFGS = 550;
lr     = 1e-3;

%% BUILD PROBLEM
prob = buildProblem(prob);
%plotCubeNodesBC(prob)
%plotH8Mesh(prob.coords,prob.elem);

%% NETWORK: input [x,y,z], output [u,v,w]
net = buildNet(3,3,64,4);

avgGrad = [];
avgSqGrad = [];

lambdaHist = [];
iterHist = [];

%% ADAM TRAINING
fprintf('Adam training...\n');

for epoch = 1:nAdam

    [loss,grad,info] = dlfeval(@modelLossInfo,net,prob,sigma0,betaInc);

    lambdaHist(end+1) = double(extractdata(loss));
    iterHist(end+1)   = epoch;

    [net,avgGrad,avgSqGrad] = adamupdate( ...
        net,grad,avgGrad,avgSqGrad,epoch,lr);

    if mod(epoch,100)==0
        fprintf(['Adam %5d | lambda = %.6f | Wint = %.4e | ', ...
                 'Linc = %.4e | Wext = %.4e | alpha = %.4e\n'], ...
            epoch, ...
            double(extractdata(loss)), ...
            double(extractdata(info.Wint)), ...
            double(extractdata(info.Linc)), ...
            double(extractdata(info.Wext)), ...
            double(extractdata(info.alpha)));
    end
end

%% LBFGS TRAINING
fprintf('LBFGS training...\n');

lossFcn = @(net) dlfeval(@modelLossLBFGS,net,prob,sigma0,betaInc);
solverState = lbfgsState;

for iter = 1:nLBFGS

    [net,solverState] = lbfgsupdate(net,lossFcn,solverState);

    if mod(iter,25)==0

        [loss,~,info] = dlfeval(@modelLossInfo,net,prob,sigma0,betaInc);

        lambdaHist(end+1) = double(extractdata(loss));
        iterHist(end+1)   = nAdam + iter;

        fprintf(['LBFGS %5d | lambda = %.6f | Wint = %.4e | ', ...
                 'Linc = %.4e | Wext = %.4e | alpha = %.4e\n'], ...
            iter, ...
            double(extractdata(loss)), ...
            double(extractdata(info.Wint)), ...
            double(extractdata(info.Linc)), ...
            double(extractdata(info.Wext)), ...
            double(extractdata(info.alpha)));
    end
end

%% POSTPROCESS
Xnode = dlarray(single(prob.coords'),'CB');
Dnode = dlfeval(@nodalDissipation,net,prob,Xnode,sigma0);

%plotDissipation3D(prob,Dnode);
coords = prob.coords;
elem   = prob.elem;
Dnode  = double(Dnode(:));

faces = [
    1 2 3 4
    5 6 7 8
    1 2 6 5
    2 3 7 6
    3 4 8 7
    4 1 5 8
];

views = [
   -45 25
    45 25
   135 25
];

titles = {'View 1','View 2','View 3'};

figure('Color','w','Position',[100 100 1200 360]);

for iv = 1:3
    subplot(1,3,iv)
    hold on
    for e = 1:size(elem,1)
        Xe = coords(elem(e,:),:); De = Dnode(elem(e,:));
        patch('Vertices',Xe,'Faces',faces,'FaceVertexCData',De(:), ...
                  'FaceColor','interp','EdgeColor','none','FaceAlpha',1.0);
    end
    axis equal tight, box off, grid off
    view(views(iv,1),views(iv,2))
    colormap(parula(256))
    %clim([prctile(Dnode,5), prctile(Dnode,99)])

    shading interp, camlight headlight
    lighting gouraud, material dull
    set(gca,'FontName','Times New Roman','FontSize',13,'LineWidth',1.0)
end
%plotHistory(iterHist,lambdaHist,nAdam);

%% ==============================================================
% BUILD PROBLEM
%% ==============================================================
function prob = buildProblem(prob)

[coords,elem] = formnode_pla_3D_multiblock( ...
    prob.nx,prob.ny,prob.nz,prob.R,prob.a);

[Xg,Wg] = domainGaussH8(coords,elem,prob.numGauss);

[Xx,Wx] = faceGaussH8(coords,elem,'xmax',prob.a,prob.numGauss);
[Xy,Wy] = faceGaussH8(coords,elem,'ymax',prob.a,prob.numGauss);
[Xz,Wz] = faceGaussH8(coords,elem,'zmax',prob.a,prob.numGauss);

prob.coords = coords;
prob.elem   = elem;

prob.XgDL = dlarray(single(Xg'),'CB');
prob.WgDL = dlarray(single(Wg'),'CB');

prob.XxDL = dlarray(single(Xx'),'CB');
prob.WxDL = dlarray(single(Wx'),'CB');

prob.XyDL = dlarray(single(Xy'),'CB');
prob.WyDL = dlarray(single(Wy'),'CB');

prob.XzDL = dlarray(single(Xz'),'CB');
prob.WzDL = dlarray(single(Wz'),'CB');

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
% LOSS
%% ==============================================================
function [loss,grad,info] = modelLossInfo(net,prob,sigma0,betaInc)

[loss,info] = computeLoss(net,prob,sigma0,betaInc);
grad = dlgradient(loss,net.Learnables);

end

function [loss,grad] = modelLossLBFGS(net,prob,sigma0,betaInc)

[loss,~] = computeLoss(net,prob,sigma0,betaInc);
grad = dlgradient(loss,net.Learnables);

end

function [loss,info] = computeLoss(net,prob,sigma0,betaInc)

Wraw = externalWork(net,prob);
alpha = 1.0 ./ (abs(Wraw) + 1e-12);

X = prob.XgDL;
W = prob.WgDL;

uvwRaw = forward(net,X);

d = hardBC(X,uvwRaw);
d = alpha .* d;

eps = strainRate3D(X,d);

D = dissipationDensity3D(eps,sigma0);
Wint = sum(D .* W,'all');

% incompressibility: exx + eyy + ezz = 0
divp = eps(1,:) + eps(2,:) + eps(3,:);
Linc = sum((divp.^2).*W,'all');

loss = Wint + betaInc * Linc;

info.Wint  = Wint;
info.Linc  = Linc;
info.Wext  = Wraw;
info.alpha = alpha;

end

%% ==============================================================
% HARD BC FOR ONE-OCTANT MODEL
%% ==============================================================
function d = hardBC(X,uvwRaw)

x = X(1,:);
y = X(2,:);
z = X(3,:);

u0 = uvwRaw(1,:);
v0 = uvwRaw(2,:);
w0 = uvwRaw(3,:);

% symmetry:
% x=0 -> u=0
% y=0 -> v=0
% z=0 -> w=0
u = x .* u0;
v = y .* v0;
w = z .* w0;

d = [u; v; w];

end

%% ==============================================================
% EXTERNAL WORK
%% ==============================================================
function Wext = externalWork(net,prob)

% face x=a, traction [px,0,0]
Xx = prob.XxDL;
Wx = prob.WxDL;

uvwx = forward(net,Xx);
dx = hardBC(Xx,uvwx);
ux = dx(1,:);

WextX = sum(prob.px .* ux .* Wx,'all');

% face y=a, traction [0,py,0]
Xy = prob.XyDL;
Wy = prob.WyDL;

uvwy = forward(net,Xy);
dy = hardBC(Xy,uvwy);
vy = dy(2,:);

WextY = sum(prob.py .* vy .* Wy,'all');

% face z=a, traction [0,0,pz]
Xz = prob.XzDL;
Wz = prob.WzDL;

uvwz = forward(net,Xz);
dz = hardBC(Xz,uvwz);
wz = dz(3,:);

WextZ = sum(prob.pz .* wz .* Wz,'all');

Wext = WextX + WextY + WextZ;

end

%% ==============================================================
% 3D STRAIN RATE
%% ==============================================================
function eps = strainRate3D(X,d)

u = d(1,:);
v = d(2,:);
w = d(3,:);

du = dlgradient(sum(u,'all'),X, ...
    'EnableHigherDerivatives',true, ...
    'RetainData',true);

dv = dlgradient(sum(v,'all'),X, ...
    'EnableHigherDerivatives',true, ...
    'RetainData',true);

dw = dlgradient(sum(w,'all'),X, ...
    'EnableHigherDerivatives',true);

du_dx = du(1,:);
du_dy = du(2,:);
du_dz = du(3,:);

dv_dx = dv(1,:);
dv_dy = dv(2,:);
dv_dz = dv(3,:);

dw_dx = dw(1,:);
dw_dy = dw(2,:);
dw_dz = dw(3,:);

exx = du_dx;
eyy = dv_dy;
ezz = dw_dz;

gxy = du_dy + dv_dx;
gyz = dv_dz + dw_dy;
gxz = du_dz + dw_dx;

eps = [exx; eyy; ezz; gxy; gyz; gxz];

end

%% ==============================================================
% 3D VON MISES PLASTIC DISSIPATION
%% ==============================================================
function D = dissipationDensity3D(eps,sigma0)

% eps = [exx; eyy; ezz; gxy; gyz; gxz]
% engineering shear strains are used.

exx = eps(1,:);
eyy = eps(2,:);
ezz = eps(3,:);
gxy = eps(4,:);
gyz = eps(5,:);
gxz = eps(6,:);

trE = exx + eyy + ezz;

exxd = exx - trE/3;
eyyd = eyy - trE/3;
ezzd = ezz - trE/3;

quad = (2/3) .* ( ...
    exxd.^2 + eyyd.^2 + ezzd.^2 + ...
    0.5*gxy.^2 + 0.5*gyz.^2 + 0.5*gxz.^2 );

D = sigma0 .* sqrt(max(quad,1e-18));

end

%% ==============================================================
% 3-BLOCK GEOMETRY-FITTED H8 MESH
%% ==============================================================
function [coords,elem] = formnode_pla_3D_multiblock(nx,ny,nz,R,a)

coords = [];
elem   = [];

[nodeX,elemX] = makeBlock(nx,ny,nz,R,a,'x');
elemX = elemX + size(coords,1);
coords = [coords; nodeX];
elem   = [elem; elemX];

[nodeY,elemY] = makeBlock(nx,ny,nz,R,a,'y');
elemY = elemY + size(coords,1);
coords = [coords; nodeY];
elem   = [elem; elemY];

[nodeZ,elemZ] = makeBlock(nx,ny,nz,R,a,'z');
elemZ = elemZ + size(coords,1);
coords = [coords; nodeZ];
elem   = [elem; elemZ];

[coords,elem] = mergeDuplicateNodes(coords,elem,1e-10);

end

function [node,elem] = makeBlock(nx,ny,nz,R,a,face)

uList = linspace(0,1,nx+1);
vList = linspace(0,1,ny+1);

% k=1: outer cube face, k=nz+1: inner spherical hole
sList = linspace(0,1,nz+1).^1.4;

node = zeros((nx+1)*(ny+1)*(nz+1),3);

id = @(i,j,k) i + (nx+1)*(j-1) + (nx+1)*(ny+1)*(k-1);

for i = 1:nx+1
    u = uList(i);

    for j = 1:ny+1
        v = vList(j);

        switch face
            case 'x'
                d = [1,u,v];
                pOuter = [a,a*u,a*v];

            case 'y'
                d = [u,1,v];
                pOuter = [a*u,a,a*v];

            case 'z'
                d = [u,v,1];
                pOuter = [a*u,a*v,a];
        end

        d = d ./ norm(d);
        pInner = R*d;

        for k = 1:nz+1

            s = sList(k);

            p = (1-s)*pOuter + s*pInner;

            node(id(i,j,k),:) = p;

        end
    end
end

elem = buildBlockH8(nx,ny,nz);

end

function elem = buildBlockH8(nx,ny,nz)

elem = zeros(nx*ny*nz,8);
e = 0;

id = @(i,j,k) i + (nx+1)*(j-1) + (nx+1)*(ny+1)*(k-1);

for i = 1:nx
    for j = 1:ny
        for k = 1:nz

            n1 = id(i  ,j  ,k);
            n2 = id(i+1,j  ,k);
            n3 = id(i+1,j+1,k);
            n4 = id(i  ,j+1,k);

            n5 = id(i  ,j  ,k+1);
            n6 = id(i+1,j  ,k+1);
            n7 = id(i+1,j+1,k+1);
            n8 = id(i  ,j+1,k+1);

            e = e + 1;
            elem(e,:) = [n1 n2 n3 n4 n5 n6 n7 n8];

        end
    end
end

end

function [nodeNew,elemNew] = mergeDuplicateNodes(node,elem,tol)

key = round(node/tol)*tol;

[~,ia,ic] = unique(key,'rows','stable');

nodeNew = node(ia,:);
elemNew = ic(elem);

end

%% ==============================================================
% H8 SHAPE FUNCTIONS
%% ==============================================================
function [N,dNdxi,dNdeta,dNdzeta] = shapeH8(xi,eta,zeta)

N = 1/8 * [
    (1-xi)*(1-eta)*(1-zeta)
    (1+xi)*(1-eta)*(1-zeta)
    (1+xi)*(1+eta)*(1-zeta)
    (1-xi)*(1+eta)*(1-zeta)
    (1-xi)*(1-eta)*(1+zeta)
    (1+xi)*(1-eta)*(1+zeta)
    (1+xi)*(1+eta)*(1+zeta)
    (1-xi)*(1+eta)*(1+zeta)
];

dNdxi = 1/8 * [
    -(1-eta)*(1-zeta)
     (1-eta)*(1-zeta)
     (1+eta)*(1-zeta)
    -(1+eta)*(1-zeta)
    -(1-eta)*(1+zeta)
     (1-eta)*(1+zeta)
     (1+eta)*(1+zeta)
    -(1+eta)*(1+zeta)
];

dNdeta = 1/8 * [
    -(1-xi)*(1-zeta)
    -(1+xi)*(1-zeta)
     (1+xi)*(1-zeta)
     (1-xi)*(1-zeta)
    -(1-xi)*(1+zeta)
    -(1+xi)*(1+zeta)
     (1+xi)*(1+zeta)
     (1-xi)*(1+zeta)
];

dNdzeta = 1/8 * [
    -(1-xi)*(1-eta)
    -(1+xi)*(1-eta)
    -(1+xi)*(1+eta)
    -(1-xi)*(1+eta)
     (1-xi)*(1-eta)
     (1+xi)*(1-eta)
     (1+xi)*(1+eta)
     (1-xi)*(1+eta)
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

    otherwise
        error('Unsupported Gauss order.');
end

end

%% ==============================================================
% DOMAIN GAUSS INTEGRATION
%% ==============================================================
function [Xg,Wg] = domainGaussH8(coords,elem,ngauss)

[gp,wg] = gauss1D(ngauss);

Xg = [];
Wg = [];

for e = 1:size(elem,1)

    Xe = coords(elem(e,:),:);

    for i = 1:numel(gp)
        for j = 1:numel(gp)
            for k = 1:numel(gp)

                xi   = gp(i);
                eta  = gp(j);
                zeta = gp(k);

                [N,dNdxi,dNdeta,dNdzeta] = shapeH8(xi,eta,zeta);

                J = [dNdxi'; dNdeta'; dNdzeta'] * Xe;
                detJ = abs(det(J));

                if detJ <= 1e-14
                    error('Zero Jacobian in H8 element %d.',e);
                end

                xg = N' * Xe;

                Xg = [Xg; xg]; %#ok<AGROW>
                Wg = [Wg; wg(i)*wg(j)*wg(k)*detJ]; %#ok<AGROW>

            end
        end
    end
end

Xg = single(Xg);
Wg = single(Wg);

end

%% ==============================================================
% FACE GAUSS INTEGRATION
%% ==============================================================
function [Xf,Wf] = faceGaussH8(coords,elem,faceName,a,ngauss)

[gp,wg] = gauss1D(ngauss);

tol = 1e-8;

% all local H8 faces
faceList = [
    1 2 3 4
    5 6 7 8
    1 2 6 5
    2 3 7 6
    3 4 8 7
    4 1 5 8
];

Xf = [];
Wf = [];

usedFaces = [];

for e = 1:size(elem,1)

    for lf = 1:6

        fn = elem(e,faceList(lf,:));
        Xq = coords(fn,:);

        switch faceName
            case 'xmax'
                isBnd = all(abs(Xq(:,1)-a) < tol);

            case 'ymax'
                isBnd = all(abs(Xq(:,2)-a) < tol);

            case 'zmax'
                isBnd = all(abs(Xq(:,3)-a) < tol);

            otherwise
                error('Unknown faceName.');
        end

        if ~isBnd
            continue;
        end

        % avoid duplicated faces after multiblock merging
        key = sort(fn);
        if ~isempty(usedFaces) && ismember(key,usedFaces,'rows')
            continue;
        end
        usedFaces = [usedFaces; key]; %#ok<AGROW>

        for i = 1:numel(gp)
            for j = 1:numel(gp)

                xi  = gp(i);
                eta = gp(j);

                [N,dNdxi,dNdeta] = shapeQ4(xi,eta);

                xg = N' * Xq;

                dx_dxi  = dNdxi'  * Xq;
                dx_deta = dNdeta' * Xq;

                dA = norm(cross(dx_dxi,dx_deta));

                Xf = [Xf; xg]; %#ok<AGROW>
                Wf = [Wf; wg(i)*wg(j)*dA]; %#ok<AGROW>

            end
        end
    end
end

if isempty(Xf)
    error('No boundary Gauss points found for %s. Check mesh boundary and tolerance.',faceName);
end

Xf = single(Xf);
Wf = single(Wf);

end

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
% NODAL DISSIPATION
%% ==============================================================
function Dnode = nodalDissipation(net,prob,Xnode,sigma0)

Wraw = externalWork(net,prob);
alpha = 1.0 ./ (abs(Wraw) + 1e-12);

uvwRaw = forward(net,Xnode);
d = hardBC(Xnode,uvwRaw);
d = alpha .* d;

eps = strainRate3D(Xnode,d);
D = dissipationDensity3D(eps,sigma0);

Dnode = double(extractdata(D))';

end


%% ==============================================================
% PLOT MESH
%% ==============================================================
function plotH8Mesh(coords,elem)

faces = [
    1 2 3 4
    5 6 7 8
    1 2 6 5
    2 3 7 6
    3 4 8 7
    4 1 5 8
];

figure('Color','w');
hold on;

for e = 1:size(elem,1)

    Xe = coords(elem(e,:),:);

    patch('Vertices',Xe, ...
          'Faces',faces, ...
          'FaceColor',[0.82 0.85 0.92], ...
          'EdgeColor',[0.15 0.15 0.15], ...
          'FaceAlpha',0.55, ...
          'LineWidth',0.15);

end

axis equal tight;
xlabel('x');
ylabel('y');
zlabel('z');
view(35,25);
camlight headlight;
camlight right;
lighting gouraud;
material dull;
box on;

end

%% ==============================================================
% PLOT DISSIPATION
%% ==============================================================
function plotDissipation3D(prob,Dnode)

coords = prob.coords;
elem = prob.elem;

faces = [
    1 2 3 4
    5 6 7 8
    1 2 6 5
    2 3 7 6
    3 4 8 7
    4 1 5 8
];

figure('Color','w');
hold on;

for e = 1:size(elem,1)

    Xe = coords(elem(e,:),:);
    De = Dnode(elem(e,:));

    patch('Vertices',Xe, ...
          'Faces',faces, ...
          'FaceVertexCData',De, ...
          'FaceColor','interp', ...
          'EdgeColor','none');

end

axis equal tight;
xlabel('x');
ylabel('y');
zlabel('z');
view(35,25);
colormap(jet);
colorbar;
camlight headlight;
lighting gouraud;
title('Plastic dissipation density');

end

%% ==============================================================
% PLOT HISTORY
%% ==============================================================
function plotHistory(iterHist,lambdaHist,nAdam)

figure('Color','w');
hold on;

idAdam = iterHist <= nAdam;

plot(iterHist(idAdam),lambdaHist(idAdam),'b-','LineWidth',2);

if any(~idAdam)
    plot(iterHist(~idAdam),lambdaHist(~idAdam),'r-','LineWidth',2);
    xline(nAdam,'k--');
    legend('Adam','LBFGS','Location','best');
else
    legend('Adam','Location','best');
end

xlabel('Iteration');
ylabel('\lambda^+');

grid on;
box on;

end

function plotCubeNodesBC(prob)

coords = prob.coords;
a      = prob.a;
R      = prob.R;

tol = 1e-8;

x = coords(:,1);
y = coords(:,2);
z = coords(:,3);

%% symmetry boundaries
symNodes = abs(x) < tol | ...
           abs(y) < tol | ...
           abs(z) < tol;

%% loaded boundaries
loadNodes = abs(x-a) < tol | ...
            abs(y-a) < tol | ...
            abs(z-a) < tol;

%% spherical hole
r = sqrt(x.^2 + y.^2 + z.^2);

holeNodes = abs(r-R) < 1e-4;

%% interior nodes
insideNodes = ~(symNodes | loadNodes | holeNodes);

%% plot
figure('Color','w');
hold on

scatter3(coords(insideNodes,1), ...
         coords(insideNodes,2), ...
         coords(insideNodes,3), ...
         8,[0.7 0.7 0.7],'filled');

scatter3(coords(symNodes,1), ...
         coords(symNodes,2), ...
         coords(symNodes,3), ...
         20,'b','filled');

scatter3(coords(loadNodes,1), ...
         coords(loadNodes,2), ...
         coords(loadNodes,3), ...
         20,'r','filled');

scatter3(coords(holeNodes,1), ...
         coords(holeNodes,2), ...
         coords(holeNodes,3), ...
         15,'k','filled');

axis equal
view(35,25)

xlabel('x')
ylabel('y')
zlabel('z')

legend({'Interior',...
        'Symmetry BC',...
        'Loaded boundary',...
        'Hole boundary'}, ...
        'Location','best')

box on
grid on

camlight headlight
lighting gouraud
exportgraphics(gcf,'cube_nodes.pdf','ContentType','vector');

end