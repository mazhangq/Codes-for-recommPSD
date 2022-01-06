clear all;
close all;
addpath('functions','data');
implement=0;  % 0: load results, 1: run code
betas=[0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1];
if implement
    load('ratings.mat');
    param_pct_beta_us=[];
    param_pct_beta_it=[];
    %     tic;
    os=ratings{1}.data;
    os=os';
    temp=os>0;
    lbs=sum(temp(:));
    Un=1000;
    In=1777;
    alpha=0.0001;
    rr=0.25;
    gama=200;
    
    for pct=0.05:-0.01:0.01
        ntest=lbs-fix(Un*In*pct);
        [utest,s]=GetTestSet1(os,ntest);
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
        %%%%%%%%===item-based=================%%%%%%%%%%%%%
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
        param_pct_beta_us=[param_pct_beta_us;aes_us];
        param_pct_beta_it=[param_pct_beta_it;aes_it];
    end
    save('data/param_pct_beta_us.mat','param_pct_beta_us');
    save('data/param_pct_beta_it.mat','param_pct_beta_it');
else
    load('param_pct_beta_us.mat');
    load('param_pct_beta_it.mat');
end
hh=numel(betas);
figure;
plot(1:hh,param_pct_beta_us(1,:),'ro-','LineWidth',2,'MarkerSize',8);
hold on;
plot(1:hh,param_pct_beta_us(2,:),'bs-','LineWidth',2,'MarkerSize',8);
plot(1:hh,param_pct_beta_us(3,:),'m^-','LineWidth',2,'MarkerSize',8);
plot(1:hh,param_pct_beta_us(4,:),'kd-','LineWidth',2,'MarkerSize',8);
plot(1:hh,param_pct_beta_us(5,:),'gv-','LineWidth',2,'MarkerSize',8);

plot(1:hh,param_pct_beta_it(1,:),'ro--','LineWidth',2,'MarkerSize',8);
plot(1:hh,param_pct_beta_it(2,:),'bs--','LineWidth',2,'MarkerSize',8);
plot(1:hh,param_pct_beta_it(3,:),'m^--','LineWidth',2,'MarkerSize',8);
plot(1:hh,param_pct_beta_it(4,:),'kd--','LineWidth',2,'MarkerSize',8);
plot(1:hh,param_pct_beta_it(5,:),'gv--','LineWidth',2,'MarkerSize',8);
axis([1 9 0.68 0.88]);
legend({'5%+UB','4%+UB','3%+UB','2%+UB','1%+UB',...
    '5%+IB','4%+IB','3%+IB','2%+IB','1%+IB'},...
    'Location','northwest','NumColumns',2);
xlabel('Parameter \beta');
ylabel('MAEs');
ax=gca;
xticks(1:2:9);
ax.XTickLabel = {'0.00001','0.0001',...
    '0.001','0.01','0.1'};
ax.FontName='Times New Roman';
ax.FontSize = 20;
print('-f1','param_pct_betas','-djpeg');