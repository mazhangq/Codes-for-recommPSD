function ff=KBreconstucter(G,s,mask,K,alpha,beta,k,wf)
if nargin<8
    wf=[];
end
ws=numel(find(G.W));
[m,n]=size(mask);
if isempty(wf)
    LL=G.L;
else
    LL=G.U*(repmat(wf,1,G.N).*G.U');
end
klk=alpha*K+beta*K*LL*K;
Uk=G.U(:,1:k);
ff=[];
for j=1:n
    labels=find(mask(:,j));
    if ~isempty(labels)
        yL=s(labels,j);
        KL=K(labels,:);
        a=inv(Uk'*(KL'*KL+klk)*Uk)*(Uk'*KL'*yL);
        f=K*Uk*a;
    end
    ff=[ff,f];
end