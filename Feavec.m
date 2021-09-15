%%%%�˺�������һ�����㼯�ϵĺ˿���ķ����
%M----m*n�ľ���,ÿ�ж�Ӧһ������
%mask---m*n��0,1����,��ʶM����Ч����λ�ã�1��Ӧ��Ч����
%fvector---m*8�ľ���ÿ�ж�Ӧһ���������������
function fvector=Feavec(M,mask)
[row,col]=size(M);
fvector=[];
for i=1:row
    mask1=mask.*repmat(mask(i,:),row,1);
    M1=M.*mask1;
    rr=sum(M1);
    nr=sum(mask1);
    if nr==0
        fvector=[fvector;[0,0]];
    else
        idx=find(nr);
        rr=rr(idx);
        nr=nr(idx);
        mr=rr./nr;
        br=M(i,idx)-mr;
        fvector=[fvector;[mean(br),std(br,1)]];%,skewness(br),kurtosis(br)-3
    end
end
temp=[sum(M==ones(size(M)),2),sum(M==2*ones(size(M)),2),...
    sum(M==3*ones(size(M)),2),sum(M==4*ones(size(M)),2),...
    sum(M==5*ones(size(M)),2)]./repmat(sum(mask,2)+eps,1,5);
temp=[temp,sum(mask,2)/max(sum(mask,2))];
fvector=[fvector,temp];