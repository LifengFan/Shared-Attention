function [bbx_r,score_r] = handle_bbx(img,bbx,score)
%bbx(i,:)=[xmin,ymin,xmax,ymax]
[h,w,~] = size(img);
if size(bbx,1) ~= size(score,1) || size(score,1) == 0 || size(bbx,1) == 0
    assert(1,'input error, please check');
end

bbx_r = [];
score_r = [];


for iter = 1:size(bbx,1)
    %check if too big
    if bbx(iter,3) - bbx(iter,1) >= 0.5*w && bbx(iter,4) - bbx(iter,2) >= 0.5*h
        continue;
    end
    
    if bbx(iter,3) - bbx(iter,1) >= 0.8*w
        continue;
    end
    %check if too small
    if bbx(iter,3) - bbx(iter,1) <= 20 || bbx(iter,4) - bbx(iter,2) <= 20
        continue;
    end
    bbx_r = [bbx_r; bbx(iter,:)];
    score_r = [score_r;score(iter,1)];
    
end

end
