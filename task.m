clear
model=load('models/forest/modelBsdsBig'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .85;     % step size of sliding window search
opts.beta  = .05;     % nms threshold for object proposals
opts.minScore = .05;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect
opts.maxAspectRatio  = 2; %[3] max aspect ratio of boxes
% opts.minBoxArea  = 500; %[1000] minimum area of boxes
% opts.kappa = 1.5; % [1.5] scale sensitivity, see equation 3 in paper
opts.gamma = 2.0;  % [2] affinity sensitivity, see equation 1 in paper

I = imread('data/2.jpg');
tic, bbs=edgeBoxes(I,model,opts); toc
bbt(:,1:2) = bbs(:,1:2);
bbt(:,3:4) = bbs(:,1:2)+bbs(:,3:4);
[bbx,score] = handle_bbx(I,bbt,bbs(:,5));

bbx(:,3:4) = bbx(:,3:4) - bbx(:,1:2);%x,y,w,h

if(size(bbs,1) ~= 0)
    RGB = insertShape(I,'Rectangle',bbx);%,'LineWidth',5
    RGB = insertText(RGB,bbx(:,1:2),score);
else
    assert(size(bbs,1) ~= 0,'no bbs generated')
end
figure;
imshow(RGB);