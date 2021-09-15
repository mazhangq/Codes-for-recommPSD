clear all;
close all;
load('data/ratings.mat');

netfpct_us=[];
netfpct_it=[];
netfpct_us1=[];
netfpct_it1=[];
for r=1:10
    %     tic;
    if r<6  %%%%%%%item-first(r=1-5)%%%%%%%
        Un=1000;
        In=1777;
    else   %%%%%%%user-first(r=6-10)%%%%%%%
        Un=1500;
        In=888;
    end
    
    os=ratings{r}.data;
    os=os';
    temp=os>0;
    lbs=sum(temp(:));
    aes_us=[];
    aes_it=[];
    aes_us1=[];
    aes_it1=[];
    for pct=0.05:-0.005:0.01
        ntest=lbs-fix(Un*In*pct);
        [utest,s]=GetTestSet1(os,ntest);
        [s,mask,rs,cs,ff]=preprocess(s);
        [urs,itms]=size(mask);
        %%%%%%%%===user-based=================%%%%%%%%%%%%%
        if r<6  %%%%%%%item-first(r=1-5)%%%%%%%
            alpha=0.0001;
            beta=0.0025;
            gama=200;
            rr=0.2;
        else   %%%%%%%user-first(r=6-10)%%%%%%%
            alpha=0.0001;
            beta=0.001;
            gama=200;
            rr=0.2;
        end
        [G,K,as,mr,mrc]=GKas(s,mask,gama,true);
        
        psdparam.sm=51;
        psdparam.quantization = false;
        psd=my_psd_estimate(G,as,mask,psdparam);
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
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        aes_us=[aes_us,mean(err)]
        
        f=KBreconstucter(G,as,mask,K,alpha,beta,10);
        f=f+mr+mrc;
        ff(rs,cs)=f;
        idx=find(ff>5); %if a rating is larger than 5 then set it to 5
        ff(idx)=5;
        idx=find(ff<1); %if a rating is  less than 1 then set it to 1
        ff(idx)=1;
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        aes_us1=[aes_us1,mean(err)]
        
        %%%%%%%%===item-based=================%%%%%%%%%%%%%
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
        
        psdparam.sm=51;
        psdparam.quantization = false;
        psd=my_psd_estimate(G,as,mask',psdparam);
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
        aes_it=[aes_it,mean(err)]
        
        f=KBreconstucter(G,as,mask',K,alpha,beta,10);
        f=f+mr+mrc;
        ff(rs,cs)=f';
        idx=find(ff>5); %if a rating is larger than 5 then set it to 5
        ff(idx)=5;
        idx=find(ff<1); %if a rating is  less than 1 then set it to 1
        ff(idx)=1;
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        aes_it1=[aes_it1,mean(err)]
    end
    netfpct_us=[netfpct_us;aes_us];
    netfpct_it=[netfpct_it;aes_it];
    netfpct_us1=[netfpct_us1;aes_us1];
    netfpct_it1=[netfpct_it1;aes_it1];
end
save('data/netfpct_us.mat','netfpct_us');
save('data/netfpct_it.mat','netfpct_it');
save('data/netfpct_us1.mat','netfpct_us1');
save('data/netfpct_it1.mat','netfpct_it1');
load('data/netfpct_us.mat');
load('data/netfpct_it.mat');
load('data/netfpct_us1.mat');
load('data/netfpct_it1.mat');
figure;
plot(1:9,mean(netfpct_us(1:5,:)),'ro-','LineWidth',2,'MarkerSize',8);
hold on;
plot(1:9,mean(netfpct_us1(1:5,:)),'r^:','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpct_us(6:10,:)),'bs-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpct_us1(6:10,:)),'bd:','LineWidth',2,'MarkerSize',8);
axis([1 9 0.7 0.86]);
legend({'item-first by ours',...
        'item-first by RKHS+kBR',...
        'user-first by ours',...
        'user-first by RKHS+kBR'},...
    'Location','northwest','NumColumns',1);
xlabel('Percentage of known entry (%)');
ylabel('MAEs');
ax=gca;
ax.XTickLabel = {'5','4.5','4','3.5','3','2.5','2','1.5','1'};
ax.FontName='Times New Roman';
ax.FontSize = 20;
figure;
plot(1:9,mean(netfpct_it(1:5,:)),'ro-','LineWidth',2,'MarkerSize',8);
hold on;
plot(1:9,mean(netfpct_it1(1:5,:)),'r^:','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpct_it(6:10,:)),'bs-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpct_it1(6:10,:)),'bd:','LineWidth',2,'MarkerSize',8);
axis([1 9 0.7 0.86]);
legend({'item-first by ours',...
        'item-first by RKHS+kBR',...
        'user-first by ours',...
        'user-first by RKHS+kBR'},...
    'Location','northwest','NumColumns',1);
xlabel('Percentage of known entry (%)');
ylabel('MAEs');
ax=gca;
ax.XTickLabel = {'5','4.5','4','3.5','3','2.5','2','1.5','1'};
ax.FontName='Times New Roman';
ax.FontSize = 20;
print('-f1','fig1','-djpeg');
print('-f2','fig2','-djpeg');
