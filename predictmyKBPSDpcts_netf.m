clear all;
close all;
addpath('functions','data');
load('ratings.mat');
implement=0;
% 0: load results,
%1: run code,runing time is about 7200s depending on your computer
if implement
    t0=tic;
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
        
        os=full(ratings{r});
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
            err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
            aes_us=[aes_us,mean(err)];
            
            f=KBreconstucter(G,as,mask,K,alpha,beta,10);
            f=f+mr+mrc;
            ff(rs,cs)=f;
            idx=find(ff>5); %if a rating is larger than 5 then set it to 5
            ff(idx)=5;
            idx=find(ff<1); %if a rating is  less than 1 then set it to 1
            ff(idx)=1;
            err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
            aes_us1=[aes_us1,mean(err)];
            
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
            aes_it=[aes_it,mean(err)];
            
            f=KBreconstucter(G,as,mask',K,alpha,beta,10);
            f=f+mr+mrc;
            ff(rs,cs)=f';
            idx=find(ff>5); %if a rating is larger than 5 then set it to 5
            ff(idx)=5;
            idx=find(ff<1); %if a rating is  less than 1 then set it to 1
            ff(idx)=1;
            err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
            aes_it1=[aes_it1,mean(err)];
        end
        netfpct_us=[netfpct_us;aes_us];
        netfpct_it=[netfpct_it;aes_it];
        netfpct_us1=[netfpct_us1;aes_us1];
        netfpct_it1=[netfpct_it1;aes_it1];
    end
    toc(t0)
    save('data/netfpct_us.mat','netfpct_us');
    save('data/netfpct_it.mat','netfpct_it');
    save('data/netfpct_us1.mat','netfpct_us1');
    save('data/netfpct_it1.mat','netfpct_it1');
else
    load('netfpct_us.mat');
    load('netfpct_it.mat');
    load('netfpct_us1.mat');
    load('netfpct_it1.mat');
end
disp('RKHS+User-based:');
disp('use-first: 5--------4.5-------4-------3.5-------3-------2.5-------2-------1.5-------1');
disp(['          ',num2str(mean(netfpct_us1(6:10,1))),'  ',num2str(mean(netfpct_us1(6:10,2))),'  ',...
    num2str(mean(netfpct_us1(6:10,3))),'  ',num2str(mean(netfpct_us1(6:10,4))),'  ',...
    num2str(mean(netfpct_us1(6:10,5))),'  ',num2str(mean(netfpct_us1(6:10,6))),'  ',...
    num2str(mean(netfpct_us1(6:10,7))),'  ',num2str(mean(netfpct_us1(6:10,8))),'  ',...
    num2str(mean(netfpct_us1(6:10,9)))]);
disp('item-first: 5--------4.5-------4-------3.5-------3-------2.5-------2-------1.5-------1');
disp(['          ',num2str(mean(netfpct_us1(1:5,1))),'  ',num2str(mean(netfpct_us1(1:5,2))),'  ',...
    num2str(mean(netfpct_us1(1:5,3))),'  ',num2str(mean(netfpct_us1(1:5,4))),'  ',...
    num2str(mean(netfpct_us1(1:5,5))),'  ',num2str(mean(netfpct_us1(1:5,6))),'  ',...
    num2str(mean(netfpct_us1(1:5,7))),'  ',num2str(mean(netfpct_us1(1:5,8))),'  ',...
    num2str(mean(netfpct_us1(1:5,9)))]);
disp('RKHS+Item-based:');
disp('use-first: 5--------4.5-------4-------3.5-------3-------2.5-------2-------1.5-------1');
disp(['          ',num2str(mean(netfpct_it1(6:10,1))),'  ',num2str(mean(netfpct_it1(6:10,2))),'  ',...
    num2str(mean(netfpct_it1(6:10,3))),'  ',num2str(mean(netfpct_it1(6:10,4))),'  ',...
    num2str(mean(netfpct_it1(6:10,5))),'  ',num2str(mean(netfpct_it1(6:10,6))),'  ',...
    num2str(mean(netfpct_it1(6:10,7))),'  ',num2str(mean(netfpct_it1(6:10,8))),'  ',...
    num2str(mean(netfpct_it1(6:10,9)))]);
disp('item-first: 5--------4.5-------4-------3.5-------3-------2.5-------2-------1.5-------1');
disp(['          ',num2str(mean(netfpct_it1(1:5,1))),'  ',num2str(mean(netfpct_it1(1:5,2))),'  ',...
    num2str(mean(netfpct_it1(1:5,3))),'  ',num2str(mean(netfpct_it1(1:5,4))),'  ',...
    num2str(mean(netfpct_it1(1:5,5))),'  ',num2str(mean(netfpct_it1(1:5,6))),'  ',...
    num2str(mean(netfpct_it1(1:5,7))),'  ',num2str(mean(netfpct_it1(1:5,8))),'  ',...
    num2str(mean(netfpct_it1(1:5,9)))]);

disp('OURS+User-based:');
disp('use-first: 5--------4.5-------4-------3.5-------3-------2.5-------2-------1.5-------1');
disp(['          ',num2str(mean(netfpct_us(6:10,1))),'  ',num2str(mean(netfpct_us(6:10,2))),'  ',...
    num2str(mean(netfpct_us(6:10,3))),'  ',num2str(mean(netfpct_us(6:10,4))),'  ',...
    num2str(mean(netfpct_us(6:10,5))),'  ',num2str(mean(netfpct_us(6:10,6))),'  ',...
    num2str(mean(netfpct_us(6:10,7))),'  ',num2str(mean(netfpct_us(6:10,8))),'  ',...
    num2str(mean(netfpct_us(6:10,9)))]);
disp('item-first: 5--------4.5-------4-------3.5-------3-------2.5-------2-------1.5-------1');
disp(['          ',num2str(mean(netfpct_us(1:5,1))),'  ',num2str(mean(netfpct_us(1:5,2))),'  ',...
    num2str(mean(netfpct_us(1:5,3))),'  ',num2str(mean(netfpct_us(1:5,4))),'  ',...
    num2str(mean(netfpct_us(1:5,5))),'  ',num2str(mean(netfpct_us(1:5,6))),'  ',...
    num2str(mean(netfpct_us(1:5,7))),'  ',num2str(mean(netfpct_us(1:5,8))),'  ',...
    num2str(mean(netfpct_us(1:5,9)))]);
disp('OURS+Item-based:');
disp('use-first: 5--------4.5-------4-------3.5-------3-------2.5-------2-------1.5-------1');
disp(['          ',num2str(mean(netfpct_it(6:10,1))),'  ',num2str(mean(netfpct_it(6:10,2))),'  ',...
    num2str(mean(netfpct_it(6:10,3))),'  ',num2str(mean(netfpct_it(6:10,4))),'  ',...
    num2str(mean(netfpct_it(6:10,5))),'  ',num2str(mean(netfpct_it(6:10,6))),'  ',...
    num2str(mean(netfpct_it(6:10,7))),'  ',num2str(mean(netfpct_it(6:10,8))),'  ',...
    num2str(mean(netfpct_it(6:10,9)))]);
disp('item-first: 5--------4.5-------4-------3.5-------3-------2.5-------2-------1.5-------1');
disp(['          ',num2str(mean(netfpct_it(1:5,1))),'  ',num2str(mean(netfpct_it(1:5,2))),'  ',...
    num2str(mean(netfpct_it(1:5,3))),'  ',num2str(mean(netfpct_it(1:5,4))),'  ',...
    num2str(mean(netfpct_it(1:5,5))),'  ',num2str(mean(netfpct_it(1:5,6))),'  ',...
    num2str(mean(netfpct_it(1:5,7))),'  ',num2str(mean(netfpct_it(1:5,8))),'  ',...
    num2str(mean(netfpct_it(1:5,9)))]);

