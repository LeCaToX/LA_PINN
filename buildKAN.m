function net = buildKAN(inDim,outDim,width,depth,inputScale)
%BUILDKAN Build the common cubic-B-spline KAN used by the MATLAB examples.
% Each KAN edge is represented by a smooth SiLU base term plus a linear
% combination of clamped cubic B-spline basis functions.  The first layer
% maps physical coordinates from [0,inputScale] to [-1,1]. inputScale may be
% scalar or one scale per input channel; hidden layers use tanh before
% evaluating their bounded spline basis.

if nargin < 5
    inputScale = 1.0;
end

gridSize = 5;  % coarse initial grid; refine in Python variants when needed

layers = [
    featureInputLayer(inDim,'Normalization','none','Name','input')
    functionLayer(@(X) kanBasisExpansion(X,gridSize,inputScale,true), ...
        'Formattable',true,'Name','kanBasis1')
    fullyConnectedLayer(width,'Name','kanLinear1')
];

for k = 2:depth
    layers = [
        layers
        functionLayer(@(X) kanBasisExpansion(X,gridSize,1.0,false), ...
            'Formattable',true,'Name',['kanBasis',num2str(k)])
        fullyConnectedLayer(width,'Name',['kanLinear',num2str(k)])
    ];
end

layers = [
    layers
    functionLayer(@(X) kanBasisExpansion(X,gridSize,1.0,false), ...
        'Formattable',true,'Name','kanBasisOut')
    fullyConnectedLayer(outDim,'Name','out')
];

net = dlnetwork(layerGraph(layers));

end

function Z = kanBasisExpansion(X,gridSize,inputScale,isFirst)
%KANBASISEXPANSION Evaluate one cubic spline feature map.
% X is [channels x points] for the CB data format.

if isFirst
    inputScale = reshape(inputScale,[],1);
    Xn = 2 .* X ./ inputScale - 1;
else
    Xn = tanh(X);
end

numChannels = size(Xn,1);
numPoints = size(Xn,2);
degree = 3;
inner = single(linspace(-1,1,gridSize+1));
knots = [repmat(single(-1),1,degree+1), inner(2:end-1), ...
         repmat(single(1),1,degree+1)];
numBasis = gridSize + degree;

features = cell(numChannels,1);

for i = 1:numChannels
    xi = transpose(Xn(i,:));

    % Degree-zero functions followed by Cox-de Boor recursion.
    B = (xi >= knots(1:end-1)) & (xi < knots(2:end));

    for order = 1:degree
        leftDen = knots(order+1:end-1) - knots(1:end-order-1);
        rightDen = knots(order+2:end) - knots(2:end-order);

        leftNum = xi - knots(1:end-order-1);
        rightNum = knots(order+2:end) - xi;

        idLeft = leftDen > 0;
        idRight = rightDen > 0;
        leftDenSafe = leftDen;
        rightDenSafe = rightDen;
        leftDenSafe(~idLeft) = 1;
        rightDenSafe(~idRight) = 1;
        left = (leftNum ./ leftDenSafe) .* idLeft;
        right = (rightNum ./ rightDenSafe) .* idRight;

        B = left .* B(:,1:end-1) + right .* B(:,2:end);
    end

    one = 0 .* xi + 1;
    atLeft = xi <= -1;
    atRight = xi >= 1;
    Bzero = 0 .* B;
    Bleft = [one, Bzero(:,1:numBasis-1)];
    Bright = [Bzero(:,1:numBasis-1), one];
    B = (1-atLeft-atRight) .* B + atLeft .* Bleft + atRight .* Bright;

    % SiLU is the smooth base branch; B-splines provide local edge detail.
    SiLU = xi ./ (1 + exp(-xi));
    Fi = [SiLU, B];
    features{i} = transpose(Fi);
end

Z = cat(1,features{:});

end
