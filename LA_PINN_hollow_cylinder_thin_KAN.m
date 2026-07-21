% ==============================================================
% PINN upper-bound limit analysis
% Thin-walled cylinder under internal pressure
% Quarter model, membrane formulation
% ==============================================================

clc; clear; close all;
rng(1);

%% PARAMETERS
prob.a      = 1.0;      % mean radius
prob.t      = 0.05;     % wall thickness
prob.p      = 1.0;      % internal pressure
prob.sigma0 = 1.0;      % yield stress

prob.ntheta = 120;

nAdam  = 4000;
nLBFGS = 400;
lr     = 1e-3;

betaInc = 1000;

%% BUILD PROBLEM
prob = buildProblem(prob);

%% NETWORK
net = buildKAN(1,2,64,4,pi/2);

avgGrad = [];
avgSqGrad = [];

lambdaHist = [];
iterHist   = [];

%% ADAM
fprintf('Adam training...\n');

for epoch = 1:nAdam

    [loss,grad,info] = dlfeval(@modelLossInfo,net,prob,betaInc);

    [net,avgGrad,avgSqGrad] = adamupdate( ...
        net,grad,avgGrad,avgSqGrad,epoch,lr);

    lambdaHist(end+1) = double(extractdata(loss));
    iterHist(end+1)   = epoch;

    if mod(epoch,100)==0
        fprintf('Adam %5d | lambda = %.6f | Wint = %.6f | Wext = %.6f | Linc = %.4e\n', ...
            epoch, ...
            double(extractdata(loss)), ...
            double(extractdata(info.Wint)), ...
            double(extractdata(info.Wext)), ...
            double(extractdata(info.Linc)));
    end
end

%% LBFGS
fprintf('LBFGS training...\n');

solverState = lbfgsState;
lossFcn = @(net) dlfeval(@modelLossLBFGS,net,prob,betaInc);

for iter = 1:nLBFGS

    [net,solverState] = lbfgsupdate(net,lossFcn,solverState);

    if mod(iter,25)==0

        [loss,~,info] = dlfeval(@modelLossInfo,net,prob,betaInc);

        lambdaHist(end+1) = double(extractdata(loss));
        iterHist(end+1)   = nAdam + iter;

        fprintf('LBFGS %5d | lambda = %.6f | Wint = %.6f | Wext = %.6f | Linc = %.4e\n', ...
            iter, ...
            double(extractdata(loss)), ...
            double(extractdata(info.Wint)), ...
            double(extractdata(info.Wext)), ...
            double(extractdata(info.Linc)));
    end
end

%% POSTPROCESS
ThetaDL = prob.ThetaDL;

[D,u,v] = dlfeval(@evaluateFields,net,prob,ThetaDL);

plotDissipation(prob.theta,D);
plotVelocity(prob.theta,u,v,prob);
plotHistory(iterHist,lambdaHist,nAdam);

%% ==============================================================
% BUILD PROBLEM
%% ==============================================================
function prob = buildProblem(prob)

theta = linspace(0,pi/2,prob.ntheta);

prob.theta  = theta;
prob.ThetaDL = dlarray(single(theta),'CB');

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
function [loss,grad,info] = modelLossInfo(net,prob,betaInc)

[loss,info] = computeLoss(net,prob,betaInc);
grad = dlgradient(loss,net.Learnables);

end

function [loss,grad] = modelLossLBFGS(net,prob,betaInc)

[loss,~] = computeLoss(net,prob,betaInc);
grad = dlgradient(loss,net.Learnables);

end

%% ==============================================================
% COMPUTE LOSS
%% ==============================================================
function [loss,info] = computeLoss(net,prob,betaInc)

Theta = prob.ThetaDL;

uvRaw = forward(net,Theta);

dRaw = hardBC(Theta,uvRaw);

Wraw = externalWork(Theta,dRaw,prob);

if double(extractdata(Wraw)) < 0
    dRaw = -dRaw;
    Wraw = -Wraw;
end

alpha = 1.0 ./ (Wraw + 1e-12);

d = alpha .* dRaw;

eps = membraneStrain(Theta,d,prob);

D = dissipationThinWall(eps,prob.sigma0);

Wint = trapz1D(Theta,D .* prob.t);

Wext = externalWork(Theta,d,prob);

% optional membrane incompressibility-like penalty
Linc = trapz1D(Theta,eps.^2);

loss = Wint + betaInc*0.0*Linc;

info.Wint = Wint;
info.Wext = Wext;
info.Linc = Linc;
info.alpha = alpha;

end

%% ==============================================================
% HARD BC
%% ==============================================================
function d = hardBC(theta,uvRaw)

th = theta;

u0 = uvRaw(1,:);
v0 = uvRaw(2,:);

% quarter symmetry:
% theta = 0    -> v = 0
% theta = pi/2 -> u = 0

u = cos(th).*u0;
v = sin(th).*v0;

d = [u; v];

end

%% ==============================================================
% MEMBRANE STRAIN
%% ==============================================================
function eps = membraneStrain(theta,d,prob)

u = d(1,:);
v = d(2,:);

dv = dlgradient(sum(v,'all'),theta, ...
    'EnableHigherDerivatives',true);

a = prob.a;

% circumferential membrane strain
% eps_theta = (dv/dtheta + u)/a
eps = (dv + u)./a;

end

%% ==============================================================
% THIN-WALL DISSIPATION
%% ==============================================================
function D = dissipationThinWall(eps,sigma0)

D = sigma0 .* sqrt(eps.^2 + 1e-18);

end

%% ==============================================================
% EXTERNAL WORK FROM INTERNAL PRESSURE
%% ==============================================================
function Wext = externalWork(theta,d,prob)

u = d(1,:);
v = d(2,:);

a = prob.a;
p = prob.p;

th = theta;

nx = cos(th);
ny = sin(th);

ur = u.*nx + v.*ny;

Wdens = p .* ur .* a;

Wext = trapz1D(theta,Wdens);

end

%% ==============================================================
% TRAPEZOID RULE FOR DLARRAY
%% ==============================================================
function I = trapz1D(theta,f)

n = size(f,2);

thetaData = extractdata(theta);
dtheta = (thetaData(end) - thetaData(1))/(n-1);

I = dtheta * ( ...
    0.5*f(:,1) + ...
    sum(f(:,2:end-1),'all') + ...
    0.5*f(:,end) );

end

%% ==============================================================
% FIELD EVALUATION
%% ==============================================================
function [D,u,v] = evaluateFields(net,prob,Theta)

uvRaw = forward(net,Theta);

dRaw = hardBC(Theta,uvRaw);

Wraw = externalWork(Theta,dRaw,prob);

if double(extractdata(Wraw)) < 0
    dRaw = -dRaw;
    Wraw = -Wraw;
end

alpha = 1.0 ./ (Wraw + 1e-12);

d = alpha .* dRaw;

eps = membraneStrain(Theta,d,prob);

D = dissipationThinWall(eps,prob.sigma0);

u = double(extractdata(d(1,:)));
v = double(extractdata(d(2,:)));
D = double(extractdata(D));

end

%% ==============================================================
% PLOT DISSIPATION
%% ==============================================================
function plotDissipation(theta,D)

figure('Color','w');

plot(theta,D,'LineWidth',2);

xlabel('\theta');
ylabel('Plastic dissipation');

grid on;
box on;

title('Thin-wall plastic dissipation');

exportgraphics(gcf,'thinwall_dissipation.pdf','ContentType','vector');

end

%% ==============================================================
% PLOT VELOCITY FIELD
%% ==============================================================
function plotVelocity(theta,u,v,prob)

a = prob.a;

x = a*cos(theta);
y = a*sin(theta);

figure('Color','w');
hold on;

th = linspace(0,pi/2,400);

plot(a*cos(th),a*sin(th),'k-','LineWidth',2);

quiver(x,y,u,v,0.6,'r','LineWidth',1.2);

axis equal tight;

xlabel('x');
ylabel('y');

grid on;
box on;

title('Thin-wall collapse mechanism');

exportgraphics(gcf,'thinwall_velocity.pdf','ContentType','vector');

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

exportgraphics(gcf,'thinwall_history.pdf','ContentType','vector');

end
