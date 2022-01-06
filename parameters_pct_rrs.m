clear all;
close all;
addpath('functions','data');

implement=0;  % 0: load results, 1: run code
if implement
    load('ratings.mat');
    param_pct_rrs_us=[];
    param_pct_rrs_it=[];
    %     tic;
    os=ratings{1}.data;
    os=os';
    temp=os>0;
    lbs=sum(temp(:));
    Un=1000;
    In=1777;
    alpha=0.0001;
    gama=200;
    for pct=0.05:-0.01:0.01
        ntest=lbs-fix(Un*In*pct);
        [utest,s]=GetTestSet1(os,ntest);
        [s,mask,rs,cs,ff]=preprocess(s);
        [urs,itms]=size(mask);
        
        aes_us=[];
        aes_it=[];
        rrs=[0.5 0.4 0.3 0.2 0.1 0.05 0.04 0.03 0.02 0.01];
        %%%%%%%%===user-based=================%%%%%%%%%%%%%
        [G,K,as,mr,mrc]=GKas(s,mask,gama,true);
        psdparam.sm=51;
        psdparam.quantization = false;
        psd=my_psd_estimate(G,as,mask,psdparam);
        psd=psd/max(psd);
        for ii=1:numel(rrs)
            rr=rrs(ii);
            p=@(x) exp(-x/rr);
            wf=p(psd);
            wf=G.lmax*wf/max(wf);
            beta=0.0025;
            f=KBreconstucter(G,as,mask,K,alpha,beta,10,wf);
            f=f+mr+mrc;
            ff(rs,cs)=f;
            idx=find(ff>5); %if a rating is larger than 5 then set it to 5
            ff(idx)=5;
            idx=find(ff<1); %if a rating is  less than 1 then set it to 1
            ff(idx)=1;
            err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
            aes_us=[aes_us,mean(err)]
        end
        %%%%%%%%===item-based=================%%%%%%%%%%%%%
        [G,K,as,mr,mrc]=GKas(s',mask',gama,true);
        psdparam.sm=51;
        psdparam.quantization = false;
        psd=my_psd_estimate(G,as,mask',psdparam);
        psd=psd/max(psd);
        for ii=1:numel(rrs)
            rr=rrs(ii);
            p=@(x) exp(-rr*x);
            wf=p(psd);
            wf=G.lmax*wf/max(wf);
            beta=0.000025;
            f=KBreconstucter(G,as,mask',K,alpha,beta,10,wf);
            f=f+mr+mrc;
            ff(rs,cs)=f';
            idx=find(ff>5); %if a rating is larger than 5 then set it to 5
            ff(idx)=5;
            idx=find(ff<1); %if a rating is  less than 1 then set it to 1
            ff(idx)=1;
            err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
            aes_it=[aes_it,mean(err)]
        end
        param_pct_rrs_us=[param_pct_rrs_us;aes_us];
        param_pct_rrs_it=[param_pct_rrs_it;aes_it];
    end
    save('data/param_pct_rrs_us.mat','param_pct_rrs_us');
    save('data/param_pct_rrs_it.mat','param_pct_rrs_it');
else
    load('param_pct_rrs_us.mat');
    load('param_pct_rrs_it.mat');
end
figure;
plot(1:10,param_pct_rrs_us(1,:),'ro-','LineWidth',2,'MarkerSize',8);
hold on;
plot(1:10,param_pct_rrs_us(2,:),'bs-','LineWidth',2,'MarkerSize',8);
plot(1:10,param_pct_rrs_us(3,:),'m^-','LineWidth',2,'MarkerSize',8);
plot(1:10,param_pct_rrs_us(4,:),'kd-','LineWidth',2,'MarkerSize',8);
plot(1:10,param_pct_rrs_us(5,:),'gv-','LineWidth',2,'MarkerSize',8);
plot(1:10,param_pct_rrs_it(1,:),'ro--','LineWidth',2,'MarkerSize',8);
plot(1:10,param_pct_rrs_it(2,:),'bs--','LineWidth',2,'MarkerSize',8);
plot(1:10,param_pct_rrs_it(3,:),'m^--','LineWidth',2,'MarkerSize',8);
plot(1:10,param_pct_rrs_it(4,:),'kd--','LineWidth',2,'MarkerSize',8);
plot(1:10,param_pct_rrs_it(5,:),'gv--','LineWidth',1,'MarkerSize',8);
axis([1 10 0.68 0.86]);
legend({'5%+UB','4%+UB','3%+UB','2%+UB','1%+UB',...
    '5%+IB','4%+IB','3%+IB','2%+IB','1%+IB'},...
    'Location','northwest','NumColumns',2);
xlabel('Weight function Parameter');
ylabel('MAEs');
ax=gca;
ax.XTickLabel = {'0.5','0.4','0.3','0.2','0.1',...
    '0.05','0.04','0.03','0.02','0.01'};
ax.FontName='Times New Roman';
ax.FontSize = 20;
print('-f1','param_pct_rrs','-djpeg');