clear all;
close all;
addpath('functions','data');
implement=0;  % 0: load results, 1: run code
betas=[0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1];
if implement
    load('netfset.mat');
    alpha=0.0001;
    rr=0.25;
    gama=200;
    param_beta_us=[];
    param_beta_it=[];
    for r=1:5:10
        %     tic;
        s=netfset{r}.train;
        utest=netfset{r}.test;
        [Un,In]=size(s);
        [s,mask,rs,cs,ff]=preprocess(s);
        [urs,itms]=size(mask);
        
        %%%%%%%%===user-based=================%%%%%%%%%%%%%
        
        [G,K,as,mr,mrc]=GKas(s,mask,gama,true);
        
        psdparam.sm=51;
        psdparam.quantization = false;
        psd=my_psd_estimate(G,as,mask,psdparam);
        psd=psd/max(psd);
        
        p=@(x) exp(-x/rr);
        wf=p(psd);
        wf=G.lmax*wf/max(wf);
        aes_us=[];
        for ii=1:numel(betas)
            beta=betas(ii);
            f=KBreconstucter(G,as,mask,K,alpha,beta,10,wf);
            f=f+mr+mrc;
            ff(rs,cs)=f;
            idx=find(ff>5); %if a rating is larger than 5 then set it to 5
            ff(idx)=5;
            idx=find(ff<1); %if a rating is  less than 1 then set it to 1
            ff(idx)=1;
            err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
            aes_us=[aes_us,mean(err)];
        end
        param_beta_us=[param_beta_us;aes_us];
        
        [G,K,as,mr,mrc]=GKas(s',mask',gama,true);
        
        psdparam.sm=51;
        psdparam.quantization = false;
        psd=my_psd_estimate(G,as,mask',psdparam);
        psd=psd/max(psd);
        
        p=@(x) exp(-rr*x);
        wf=p(psd);
        wf=G.lmax*wf/max(wf);
        aes_it=[];
        for ii=1:numel(betas)
            beta=betas(ii);
            f=KBreconstucter(G,as,mask',K,alpha,beta,10,wf);
            f=f+mr+mrc;
            ff(rs,cs)=f';
            idx=find(ff>5); %if a rating is larger than 5 then set it to 5
            ff(idx)=5;
            idx=find(ff<1); %if a rating is  less than 1 then set it to 1
            ff(idx)=1;
            err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
            aes_it=[aes_it,mean(err)];
        end
        param_beta_it=[param_beta_it;aes_it];
    end
    save('data/param_beta_us.mat','param_beta_us');
    save('data/param_beta_it.mat','param_beta_it');
    
    
    path='../../../myDatabase/ml-100k/mat/';
    load([path,'u1base.mat']);
    u{1}.base=u1base;
    clear u1base;
    load([path,'u1test.mat']);
    u{1}.test=u1test;
    clear u1test;
    load([path,'uabase.mat']);
    u{6}.base=uabase;
    clear uabase;
    load([path,'uatest.mat']);
    u{6}.test=uatest;
    clear uatest;
    Un=943;
    In=1682;
    
    %r=1:5  for u1.*-u5.*
    %r=6:7  for u6.*-u7.*
    %
    %
    param_rrs_us1=[];
    param_rrs_it1=[];
    for r=1:5:6
        ubase=u{r}.base;
        utest=u{r}.test;
        s=zeros(Un,In);
        s((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
        [s,mask,rs,cs,ff]=preprocess(s);
        [urs,itms]=size(mask);
        
        %%%%%%%%%%%%%%%user-based%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [G,K,as,mr,mrc]=GKas(s,mask,gama,true);
        
        psdparam.sm=51;
        psdparam.quantization = false;
        psd=my_psd_estimate(G,as,mask,psdparam);
        psd=psd/max(psd);
        
        
        p=@(x) exp(-x/rr);
        wf=p(psd);
        wf=G.lmax*wf/max(wf);
        aes_us=[];
        for ii=1:numel(betas)
            beta=betas(ii);
            f=KBreconstucter(G,as,mask,K,alpha,beta,10,wf);
            f=f+mr+mrc;
            ff(rs,cs)=f;
            idx=find(ff>5); %if a rating is larger than 5 then set it to 5
            ff(idx)=5;
            idx=find(ff<1); %if a rating is  less than 1 then set it to 1
            ff(idx)=1;
            err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
            aes_us=[aes_us,mean(err)];
        end
        param_beta_us=[param_beta_us;aes_us];
        %%%%%%%%%%%%%%%item-based%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [G,K,as,mr,mrc]=GKas(s',mask',gama,true);
        psdparam.sm=51;
        psdparam.quantization = false;
        psd=my_psd_estimate(G,as,mask',psdparam);
        psd=psd/max(psd);
        p=@(x) exp(-x/rr);
        wf=p(psd);
        wf=G.lmax*wf/max(wf);
        aes_it=[];
        for ii=1:numel(betas)
            beta=betas(ii);
            f=KBreconstucter(G,as,mask',K,alpha,beta,10,wf);
            f=f+mr+mrc;
            ff(rs,cs)=f';
            idx=find(ff>5); %if a rating is larger than 5 then set it to 5
            ff(idx)=5;
            idx=find(ff<1); %if a rating is  less than 1 then set it to 1
            ff(idx)=1;
            err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
            aes_it=[aes_it,mean(err)];
        end
        param_beta_it=[param_beta_it;aes_it];
    end
    
    save('data/param_beta_us.mat','param_beta_us');
    save('data/param_beta_it.mat','param_beta_it');
else
    load('param_beta_us.mat');
    load('param_beta_it.mat');
    
    load('param_beta_us.mat');
    load('param_beta_it.mat');
end
hh=numel(betas);
figure;
plot(1:hh,param_beta_us(1,:),'ro-','LineWidth',2,'MarkerSize',8);
hold on;
plot(1:hh,param_beta_us(2,:),'bs-','LineWidth',2,'MarkerSize',8);
plot(1:hh,param_beta_us(3,:),'k^-','LineWidth',2,'MarkerSize',8);
plot(1:hh,param_beta_us(4,:),'mv-','LineWidth',2,'MarkerSize',8);

plot(1:hh,param_beta_it(1,:),'ro--','LineWidth',2,'MarkerSize',8);
plot(1:hh,param_beta_it(2,:),'bs--','LineWidth',2,'MarkerSize',8);
plot(1:hh,param_beta_it(3,:),'k^--','LineWidth',2,'MarkerSize',8);
plot(1:hh,param_beta_it(4,:),'mv--','LineWidth',2,'MarkerSize',8);

axis([1 9 0.68 0.86]);
legend({'Netflix IF1+UB',...
    'Netflix UF1+UB',...
    'ml-100k u1+UB',...
    'ml-100k ua+UB',...
    'Netflix IF1+IB',...
    'Netflix UF1+IB',...
    'ml-100k u1+IB',...
    'ml-100k ua+IB'},...
    'Location','northwest','NumColumns',2);
xlabel('Parameter \beta');
ylabel('MAEs');
ax=gca;
xticks(1:2:9);
ax.XTickLabel = {'0.00001','0.0001',...
    '0.001','0.01','0.1'};
%% 
ax.FontName='Times New Roman';
ax.FontSize = 20;

print('-f1','param_betas','-djpeg');