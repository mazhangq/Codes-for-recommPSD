%%%%�˺����������R��������֮������ܶ�
%R----m*n�ľ���
%W----m*m�ľ���W(i,j)��ʾUI�ĵ�i�����j�е����Ҿ���

function W=Simxy(R,mask,type)
[U,I]=size(R);
singluarrows=find(sum(abs(R),2)==0);
if ~isempty(singluarrows)
    R(singluarrows,:)=repmat(sum(R)./(sum(mask)+eps),...
        numel(singluarrows),1);
end
if type==1
    temp=sqrt(sum(R.*R,2))+eps;
    temp1=R./repmat(temp,1,I);
    W=(temp1*temp1'+1)/2;
else
    W=corr(R');
    W=(W+1)/2;
end
W=W.*(eye(U)<1);


