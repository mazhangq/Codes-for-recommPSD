%%%%�˺�������һ�����㼯�ϵĺ˿���ķ����
%M----m*n�ľ���,ÿ�ж�Ӧһ������
%type---����
%    ---type==1:K(i,j)=exp(norm(vi-vj)^2/delta^2)
%gama---����
%K----m*m�ĺ˿���ķ�������

function K=KernelGram(fvector,gama)
[row,col]=size(fvector);
ks1=repmat(fvector,1,1,row);
ks2=permute(ks1,[3,2,1]);
temp=sum((ks2-ks1).^2,2);
clear ks1;
clear ks2;
temp=squeeze(temp);
K=exp(-temp/gama);

