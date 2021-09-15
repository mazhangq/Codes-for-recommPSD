clear all;
close all;
%load('data/ratings.mat');
% for r=1:10
%     if r<=5
%         Un=1000;
%         In=1777;
%     else
%         Un=1500;
%         In=888;
%     end
%     s=ratings{r}.data;
%     s=s';
%     [utest,s]=GetTestSet1(s,20000);
%     netfset{r}.train=s;
%     netfset{r}.test=utest;
% end
% save('data/netfset.mat','netfset');

load('data/netfset.mat');
maes=[];
rmses=[];
for r=1:10    %%%r=1_5 for item-first£»r=6-10 for user-first
    s=netfset{r}.train;
    utest=netfset{r}.test;
    [Un,In]=size(s);
    [s,mask,rs,cs,ff]=preprocess(s);
    [urs,itms]=size(mask);
    mae=[];
    rmse=[];
    %%%%%%%%===user-based=================%%%%%%%%%%%%%
    %     tic;
    if r<6
        %%%%%%%item-first(r=1-5)%%%%%%%
        alpha=0.0001;
        beta=0.0025;
        gama=200;
        rr=0.2;
    else
        %%%%%%%user-first(r=6-10)%%%%%%%
        alpha=0.0001;
        beta=0.001;
        gama=200;
        rr=0.2;
    end
    
    
    [G,K,as,mr,mrc]=GKas(s,mask,gama,true);
    
    psdparam.sm=51;
    psdparam.quantization = false;
    psd=my_psd_estimate(G,as,mask);
    psd=psd/max(psd);
    
    p=@(x) exp(-x/rr);
    wf=p(psd);
    wf=G.lmax*wf/max(wf);
    
    f=KBreconstucter(G,as,mask,K,alpha,beta,10,wf);
    f=f+mr+mrc;
    ff(rs,cs)=f;
    idx=find(ff>5); %if a rating is larger than 5 then set it to 5
    ff(idx)=5;
    idx=find(ff<1); %if a rating is  less than 1 then set it to 1
    ff(idx)=1;
    err=ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3);
    mae=[mae,mean(abs(err))];
    rmse=[rmse,sqrt(mean(err.^2))];
    
    %%%%%%%%===item-based=================%%%%%%%%%%%%%
    %     tic;
    %%%%%%%item-first(r=1-5)%%%%%%%
    if r<6
        alpha=0.0001;
        beta=0.000025;
        gama=200;
        rr=0.5;
        %%%%%%%user-first(r=6-10)%%%%%%%
    else
        alpha=0.0001;
        beta=0.00025;
        gama=200;
        rr=0.2;
    end
    
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
    err=ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3);
    mae=[mae,mean(abs(err))];
    rmse=[rmse,sqrt(mean(err.^2))];
    maes=[maes;mae];   
    rmses=[rmses;rmse]; 
end
[mean(maes(1:5,:)) mean(rmses(1:5,:)) mean(maes(6:10,:)) mean(rmses(6:10,:))]
