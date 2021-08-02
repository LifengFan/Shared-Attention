function mat2txt(X,fp)

[r,c]=size(X);
for i=1:r
    for j=1:c
        fprintf(fp,'%5d\t',X(i,j));
    end
    fprintf(fp,'\r\n');
end
end