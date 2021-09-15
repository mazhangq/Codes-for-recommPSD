clear all;
% close all;

%/Users/yangzhihua/Documents/Lab/myDatabase/ml-100k
load('data/u1base.mat');
u{1}.base=u1base;
clear u1base;
load('data/u1test.mat');
u{1}.test=u1test;
clear u1test;
load('data/u2base.mat');
u{2}.base=u2base;
clear u2base;
load('data/u2test.mat');
u{2}.test=u2test;
clear u2test;
load('data/u3base.mat');
u{3}.base=u3base;
clear u3base;
load('data/u3test.mat');
u{3}.test=u3test;
clear u3test;
load('data/u4base.mat');
u{4}.base=u4base;
clear u4base;
load('data/u4test.mat');
u{4}.test=u4test;
clear u4test;
load('data/u5base.mat');
u{5}.base=u5base;
clear u5base;
load('data/u5test.mat');
u{5}.test=u5test;
clear u5test;
load('data/uabase.mat');
u{6}.base=uabase;
clear uabase;
load('data/uatest.mat');
u{6}.test=uatest;
clear uatest;
load('data/ubbase.mat');
u{7}.base=ubbase;
clear ubbase;
load('data/ubtest.mat');
u{7}.test=ubtest;
clear ubtest;

Un=943;
In=1682;

tstart=tic;
%%r=1:5  for u1.*-u5.*
%%r=6:7  for u6.*-u7.*
%%
%%
rmse_its=[];
mae_us=[];
rmse_us=[];
mae_it=[];
rmse_it=[];
for r=6:7 %1:5%
    ubase=u{r}.base;
    utest=u{r}.test;
    s=zeros(Un,In);
    s((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
    [s,mask,rs,cs,ff]=preprocess(s);
    [urs,itms]=size(mask);
    
    %%%%%%%%%%%%%%%user-based%%%%%%%%%%%%%%%%%%%%%%%%%%%
    alpha=0.0001;%0.01;%[0.00001 0.0001 0.001 0.01 0.1];%
    beta=0.01;%0.1;%[0.00001,0.0001 0.001 0.01 0.1];%
    gama=200;%10;%10:10:50;%50:50:200;%
    rr=200;%[5 10 20 50 100 200];
    
    [G,K,as,mr,mrc]=GKas(s,mask,gama,true);
    psd=my_psd_estimate(G,as,mask);
    psd=psd/max(psd);
    
    p=@(x)exp(-rr*x);%exp(-x.^k);% 
    wf=p(psd);
    wf=G.lmax*wf/max(wf);
    f=KBreconstucter(G,as,mask,K,alpha,beta,10,wf);
    f=f+mr+mrc;
    ff(rs,cs)=f;
    idx=find(ff>5); %if a rating is larger than 5 then set it to 5
    ff(idx)=5;
    idx=find(ff<1); %if a rating is  less than 1 then set it to 1
    ff(idx)=1;
    err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
    mae_us=[mae_us;mean(err)];
    err=(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3)).^2;
    rmse_us=[rmse_us;sqrt(mean(err))];

%     %%%%%%%%%%%%%%%item-based%%%%%%%%%%%%%%%%%%%%%%%%%%%
    alpha=0.0001;%0.1;%[0.00001 0.0001 0.001 0.01 0.1];%
    beta=0.001;%0.01%[0.00001,0.0001 0.001 0.01 0.1];%
    gama=200;%10;%10:10:50;%50:50:
    rr=200;%[5 10 20 50 100 200];
    
    [G,K,as,mr,mrc]=GKas(s',mask',gama,true);
    psd=my_psd_estimate(G,as,mask');
    psd=psd/max(psd);
    p=@(x) exp(-rr*x);
    wf=p(psd);
    wf=G.lmax*wf/max(wf);
    
    f=KBreconstucter(G,as,mask',K,alpha,beta,10,wf);
    f=f+mr+mrc;
    ff(rs,cs)=f';
    idx=find(ff>5); %if a rating is larger than 5 then set it to 5
    ff(idx)=5;
    idx=find(ff<1); %if a rating is  less than 1 then set it to 1
    ff(idx)=1;
    err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
    mae_it=[mae_it;mean(err)];
    err=(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3)).^2;
    rmse_it=[rmse_it;sqrt(mean(err))];
end
[mean(mae_us) std(mae_us) mean(rmse_us) std(rmse_us) ...
    mean(mae_it) std(mae_it) mean(rmse_it) std(rmse_it)]
% tElapsed = toc(tstart)
%ua_ub运行时间274.1761s
