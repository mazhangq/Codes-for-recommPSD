clear all;
close all;
addpath('functions','data');
load('R.mat');

Un=943;
In=1682;

tstart=tic;
implement=0;
% 0: load results, 
%1: run code,runing time is about 800s depending on your computer
if implement
    t0=tic;
    mae_us=[];
    rmse_us=[];
    mae_it=[];
    rmse_it=[];
    %load('uu.mat');
    for r=1:10
        utest=R{r}.test;
        ubase=R{r}.base;
        s=zeros(Un,In);
        s((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
        [s,mask,rs,cs,ff]=preprocess(s);
        [urs,itms]=size(mask);
        
        %%%%%%%%%%%%%%%user-based%%%%%%%%%%%%%%%%%%%%%%%%%%%
        alpha=0.0001;%[0.00001 0.0001 0.001 0.01 0.1];
        beta=0.01;%[0.00001,0.0001 0.001 0.01 0.1];
        gama=200;%50:50:200;
        rr=200;%[5 10 20 50 100 200];
        
        [G,K,as,mr,mrc]=GKas(s,mask,gama,true);
        psd=my_psd_estimate(G,as,mask);
        psd=psd/max(psd);
        p=@(x) exp(-rr*x);%%exp(-x.^2/rr);%
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
        
        %%%%%%%%%%%%%%%item-based%%%%%%%%%%%%%%%%%%%%%%%%%%%
        alpha=0.0001;%[0.00001 0.0001 0.001 0.01 0.1];
        beta=0.001;%[0.00001,0.0001 0.001 0.01 0.1];
        gama=200;%50:50:200;
        rr=200;%[5 10 20 50 100 200];
        
        [G,K,as,mr,mrc]=GKas(s',mask',gama,true);
        psd=my_psd_estimate(G,as,mask');
        psd=psd/max(psd);
        
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
    toc(t0)
    PSDR=[mae_us,mae_it,rmse_us,rmse_it];
    save('data\PSDR.mat','PSDR');
else
    load('PSDR.mat');
end
disp('User-based: mae+-std------------------rmse+-std');
disp(['            ',num2str(mean(PSDR(:,1))),'+-',num2str(std(PSDR(:,1))),...
    '     ',num2str(mean(PSDR(:,3))),'+-',num2str(std(PSDR(:,3)))]);
disp('Item-based: mae+-std------------------rmse+-std');
disp(['            ',num2str(mean(PSDR(:,2))),'+-',num2str(std(PSDR(:,2))),...
    '     ',num2str(mean(PSDR(:,4))),'+-',num2str(std(PSDR(:,4)))]);



% [mean(mae_us) std(mae_us) mean(rmse_us) std(rmse_us) ...
%     mean(mae_it) std(mae_it) mean(rmse_it) std(rmse_it)]
% tElapsed = toc(tstart)
%ua_ub运行时间274.1761s
