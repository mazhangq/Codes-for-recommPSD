function psd=my_psd_estimate(G,s,Mask)
[N,Ns]=size(s);
c=mean(s(Mask));
p=sum(Mask,2)/Ns;
S=cov(s');
S1=diag(S)./p-(1-p)*c^2;
S2=(S-diag(diag(S)))./(p*p');

psd=sum((G.U.^2).*repmat(S1,1,N))'+...
          +sum((G.U'*S2).*G.U',2);
psd=abs(psd);
psd=smooth(psd,51);
% psd=smooth(psd,41);
% psd=smooth(psd,31);


