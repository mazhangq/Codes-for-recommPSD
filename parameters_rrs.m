clear all;
close all;
addpath('functions','data');
implement=0;  % 0: load results, 1: run code

if implement
    rrs=[0.5 0.4 0.3 0.2 0.1 0.05 0.04 0.03 0.02 0.01];
    load('netfset.mat');
    param_rrs_us=[];
    param_rrs_it=[];
    
    for r=1:5:10
        %     tic;
        s=netfset{r}.train;
        utest=netfset{r}.test;
        [Un,In]=size(s);
        [s,mask,rs,cs,ff]=preprocess(s);
        [urs,itms]=size(mask);
        
        aes_us=[];
        aes_it=[];
        
        for ii=1:numel(rrs)
            %%%%%%%%===user-based=================%%%%%%%%%%%%%
            if r<6  %%%%%%%item-first(r=1-5)%%%%%%%
                alpha=0.0001;
                beta=0.0025;
                gama=200;
            else   %%%%%%%user-first(r=6-10)%%%%%%%
                alpha=0.0001;
                beta=0.001;
                gama=200;
            end
            [G,K,as,mr,mrc]=GKas(s,mask,gama,true);
            
            psdparam.sm=51;
            psdparam.quantization = false;
            psd=my_psd_estimate(G,as,mask,psdparam);
            psd=psd/max(psd);
            
            rr=rrs(ii);
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
            
            %%%%%%%%===item-based=================%%%%%%%%%%%%%
            if r<6
                alpha=0.0001;
                beta=0.000025;
                gama=200;
                %%%%%%%user-first(r=6-10)%%%%%%%
            else
                alpha=0.0001;
                beta=0.00025;
                gama=200;
            end
            [G,K,as,mr,mrc]=GKas(s',mask',gama,true);
            
            psdparam.sm=51;
            psdparam.quantization = false;
            psd=my_psd_estimate(G,as,mask',psdparam);
            psd=psd/max(psd);
            
            rr=rrs(ii);
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
        end
        param_rrs_us=[param_rrs_us;aes_us];
        param_rrs_it=[param_rrs_it;aes_it];
    end
    
    load('u1base.mat');
    u{1}.base=u1base;
    clear u1base;
    load('u1test.mat');
    u{1}.test=u1test;
    clear u1test;
    load('uabase.mat');
    u{6}.base=uabase;
    clear uabase;
    load('uatest.mat');
    u{6}.test=uatest;
    clear uatest;
    Un=943;
    In=1682;
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
        alpha=0.0001;%[0.00001 0.0001 0.001 0.01 0.1];
        beta=0.01;%[0.00001,0.0001 0.001 0.01 0.1];
        gama=200;%50:50:200;
        
        [G,K,as,mr,mrc]=GKas(s,mask,gama,true);
        
        psdparam.sm=51;
        psdparam.quantization = false;
        psd=my_psd_estimate(G,as,mask,psdparam);
        psd=psd/max(psd);
        aes_us=[];
        for ii=1:numel(rrs)
            rr=rrs(ii);
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
        end
        param_rrs_us1=[param_rrs_us1;aes_us];
        %%%%%%%%%%%%%%%item-based%%%%%%%%%%%%%%%%%%%%%%%%%%%
        alpha=0.0001;%[0.00001 0.0001 0.001 0.01 0.1];
        beta=0.001;%[0.00001,0.0001 0.001 0.01 0.1];
        gama=200;%50:50:200;
        rr=200;%[5 10 20 50 100 200];
        
        [G,K,as,mr,mrc]=GKas(s',mask',gama,true);
        
        psdparam.sm=51;
        psdparam.quantization = false;
        psd=my_psd_estimate(G,as,mask',psdparam);
        psd=psd/max(psd);
        
        aes_it=[];
        for ii=1:numel(rrs)
            rr=rrs(ii);
            p=@(x) exp(-x/rr);
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
        end
        param_rrs_it1=[param_rrs_it1;aes_it];
    end
    param_rrs_us=[param_rrs_us;param_rrs_us1];
    param_rrs_it=[param_rrs_it;param_rrs_it1];
    save('data/param_rrs_us.mat','param_rrs_us');
    save('data/param_rrs_it.mat','param_rrs_it');
else
    load('param_rrs_us.mat');
    load('param_rrs_it.mat');
end

figure;
plot(1:10,param_rrs_us(1,:),'ro-','LineWidth',2,'MarkerSize',8);
hold on;
plot(1:10,param_rrs_us(2,:),'bs-','LineWidth',2,'MarkerSize',8);
plot(1:10,param_rrs_us(3,:),'k^-','LineWidth',2,'MarkerSize',8);
plot(1:10,param_rrs_us(4,:),'mv-','LineWidth',2,'MarkerSize',8);

plot(1:10,param_rrs_it(1,:),'ro--','LineWidth',2,'MarkerSize',8);
plot(1:10,param_rrs_it(2,:),'bs--','LineWidth',2,'MarkerSize',8);
plot(1:10,param_rrs_it(3,:),'k^--','LineWidth',2,'MarkerSize',8);
plot(1:10,param_rrs_it(4,:),'mv--','LineWidth',2,'MarkerSize',8);

axis([1 10 0.68 0.82]);
legend({'Netflix IF1+UB',...
    'Netflix UF1+UB',...
    'ml-100k u1+UB',...
    'ml-100k ua+UB',...
    'Netflix IF1+IB',...
    'Netflix UF1+IB',...
    'ml-100k u1+IB',...
    'ml-100k ua+IB'},...
    'Location','northeast','NumColumns',2);
xlabel('Weight function Parameter');
ylabel('MAEs');
ax=gca;
xticks(1:10);
ax.XTickLabel = {'0.5','0.4','0.3','0.2','0.1',...
    '0.05','0.04','0.03','0.02','0.01'};
ax.FontName='Times New Roman';
ax.FontSize = 20;

print('-f1','param_rrs','-djpeg');