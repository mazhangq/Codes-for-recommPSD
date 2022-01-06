clear all;
close all;
addpath('functions','data');
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


Un=943;
In=1682;

%%
implement=0;
% 0: load results,
%1: run code, running time is about 620s depending on your computer
if implement
    mae_us=[];
    rmse_us=[];
    mae_it=[];
    rmse_it=[];
    for r=1:5
        ubase=u{r}.base;
        utest=u{r}.test;
        s=zeros(Un,In);
        s((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
        [s,mask,rs,cs,ff]=preprocess(s);
        [urs,itms]=size(mask);
        
        %%%%%%%%%%%%%%%user-based%%%%%%%%%%%%%%%%%%%%%%%%%%%
        alpha=0.0001;
        gama=200;
        beta=0.01;
        rr=1/200;
        
        [G,K,as,mr,mrc]=GKas(s,mask,gama,true);
        psd=my_psd_estimate(G,as,mask);
        psd=psd/max(psd);
        
        p=@(x)exp(-x/rr);
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
        rmse_us=[rmse_us;sqrt(mean(err.^2))];
        
        % %%%%%%%%%%%%%%%item-based%%%%%%%%%%%%%%%%%%%%%%%%%%%
        alpha=0.0001;
        gama=200;
        beta=0.001;
        rr=1/200;
        
        [G,K,as,mr,mrc]=GKas(s',mask',gama,true);
        psd=my_psd_estimate(G,as,mask');
        psd=psd/max(psd);
        
        p=@(x)exp(-x/rr);
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
        rmse_it=[rmse_it;sqrt(mean(err.^2))];      
    end
    
    PSD=[mae_us,mae_it,rmse_us,rmse_it];
    save('data/PSD.mat','PSD');
else
    load('PSD.mat');
end

disp('User-based: mae+-std------------------rmse+-std');
disp(['            ',num2str(mean(PSD(:,1))),'+-',num2str(std(PSD(:,1))),...
    '     ',num2str(mean(PSD(:,3))),'+-',num2str(std(PSD(:,3)))]);
disp('Item-based: mae+-std------------------rmse+-std');
disp(['            ',num2str(mean(PSD(:,2))),'+-',num2str(std(PSD(:,2))),...
    '     ',num2str(mean(PSD(:,4))),'+-',num2str(std(PSD(:,4)))]);







