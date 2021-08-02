clear;
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


for iter = 201:400
    frame = 1;
    while (1)
        fprintf('episode %d frame %d \n',iter,frame);
        bbt = [];
        bbs = [];
        filename = ['~/Desktop/cm/' num2str(iter) '/' num2str(frame) '.jpg'];
        if exist(filename,'file') == 0
            frame = frame + 1;
            break;
        end
    
        I = imread(filename);
        tic, bbs=edgeBoxes(I,model,opts); toc
        bbt(:,1:2) = bbs(:,1:2);
        bbt(:,3:4) = bbs(:,1:2)+bbs(:,3:4);
        bbs(:,5) = bbs(:,5)/max(bbs(:,5));
        [bbx,score] = handle_bbx(I,bbt,bbs(:,5));
        
        bbx_im = bbx;
        bbx_im(:,3:4) = bbx_im(:,3:4) - bbx_im(:,1:2);%x,y,w,h
        RGB = insertShape(I,'Rectangle',bbx_im);
        RGB = insertText(RGB,bbx(:,1:2),score);
        
        map = zeros(320,480);
        map1 = zeros(320,480);
        for xter = 1:480
            for yter = 1:320
                prob = [];
                for jter = 1:size(score,1)
                    if xter>=bbx(jter,1) && xter<= bbx(jter,3) && yter >= bbx(jter,2) && yter <= bbx(jter,4)
                        prob = [prob score(jter)];
                    end
                end
                if size(prob,2) == 0
                    map(321 - yter,xter) = 0;
                    map1(yter,xter) = 0;
                else
                    map(321 - yter,xter) = prod(prob)^(1/length(prob));
                    map1(yter,xter) = prod(prob)^(1/length(prob));
                end
            end
        end
        
        fileoutname = ['Heatmap/' num2str(iter) '/heatmap' num2str(frame) '.txt'];
        fpout = fopen(fileoutname,'a');
        mat2txt(map1,fpout);
        frame = frame + 1;
        %     figure;
        %     subplot(1,2,1);
        %     heatmap(map1);
        %     subplot(1,2,2);
        %     imshow(RGB);
    end
end
