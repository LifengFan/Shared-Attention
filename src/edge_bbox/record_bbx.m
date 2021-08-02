clear;
model=load('models/forest/modelBsdsBig'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .75;     % step size of sliding window search
opts.beta  = .05;     % nms threshold for object proposals
opts.minScore = .05;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect
opts.maxAspectRatio  = 2; %[3] max aspect ratio of boxes
% opts.minBoxArea  = 500; %[1000] minimum area of boxes
% opts.kappa = 1.5; % [1.5] scale sensitivity, see equation 3 in paper
opts.gamma = 2.0;  % [2] affinity sensitivity, see equation 1 in paper


parfor iter = 201:400
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
        fileoutname = ['bbx/' num2str(iter) '/' num2str(frame) '.txt'];
        if exist(fileoutname,'file') >= 1
            frame = frame + 1;
            continue;
        end
        I = imread(filename);
        tic, bbs=edgeBoxes(I,model,opts); toc
        if size(bbs,1) == 0
            data = [];
            fpout = fopen(fileoutname,'a');
            mat2txt(data,fpout);
            fclose(fpout);
            frame = frame + 1;
            continue;
        end
        bbt(:,1:2) = bbs(:,1:2);
        bbt(:,3:4) = bbs(:,1:2)+bbs(:,3:4);
        bbs(:,5) = bbs(:,5)/max(bbs(:,5));
        [bbx,score] = handle_bbx(I,bbt,bbs(:,5));%x1,y1,x2,y2
%         bbx_im = bbx(:,:);
%         bbx_im(:,3:4) = bbx_im(:,3:4) - bbx_im(:,1:2);%x,y,w,h
%         RGB = insertShape(I,'Rectangle',bbx_im);
%         RGB = insertText(RGB,bbx(:,1:2),score);
%         figure;
%         imshow(RGB);
        if size(bbx,1)>25
            data = [bbx(1:25,:),score(1:25,:)];
        else
            data = [bbx,score];
        end
        
        fpout = fopen(fileoutname,'a');
        mat2txt(data,fpout);
        fclose(fpout);
        frame = frame + 1;
        %     figure;
        %     subplot(1,2,1);
        %     heatmap(map1);
        %     subplot(1,2,2);
        %     imshow(RGB);
    end
end
