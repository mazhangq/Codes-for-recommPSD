clear all;
close all;
addpath('functions','data');
load('uabase.mat');
u{1}.base=uabase;
clear uabase;
load('uatest.mat');
u{1}.test=uatest;
clear uatest;
load('ubbase.mat');
u{2}.base=ubbase;
clear ubbase;
load('ubtest.mat');
u{2}.test=ubtest;
clear ubtest;

Un=943;
In=1682;

tstart=tic;

implement=0;
% 0: load results,
%1: run code,the CPU time will be about 9000s depending on your computer
if implement
    bestparamter=[];
    mae_KB_us=[];
    rmse_KB_us=[];
    mae_KB_it=[];
    rmse_KB_it=[];
    mae_KBPSD_us=[];
    rmse_KBPSD_us=[];
    mae_KBPSD_it=[];
    rmse_KBPSD_it=[];
    for r=1:2
        ubase=u{r}.base;
        utest=u{r}.test;
        
        %%%%%%%%%%%%%%%user-based%%%%%%%%%%%%%%%%%%%%%%%%%%%
        alpha=0.005;
        gama=200;
        
        betas=[0.00001,0.0001 0.001 0.01 0.1];
        rrs=1./[50 100 150 200 250];
        
        %%%%%%%%%RHKS%%%%%%%%%%%%
        M=zeros(Un,In);
        M((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
        mask=(M>0);
        [aM,mU,mUI]=AdjustUI(M);
        Wi=Simxy(aM,mask,1);
        fvector=Feavec(M,mask);
        K=KernelGram(fvector,gama);
        
        D=diag(sum(Wi,2));
        L=D-Wi;
        L=size(L,1)*L/trace(L);
        [U,V]=eig(L);
        estr=zeros(size(utest,1),1);
        items=unique(utest(:,2));
        k=5;
        err0=[];
        for i=1:numel(betas)
            beta=betas(i);
            klk=alpha*K+beta*K*L*K;
            Uk=U(:,1:k);
            for j=1:numel(items)
                labels=find(mask(:,items(j)));
                us=find(utest(:,2)==items(j));  %测试集中，项目为items(j)的项目的需要预测的标签
                numlbs=numel(labels);
                if numlbs>0
                    yL=M(labels,items(j));%测试集中，项目为items(j)的项目的已知标签
                    KL=K(labels,:);
                    a=pinv(Uk'*(KL'*KL+klk)*Uk)*(Uk'*KL'*yL);
                    f=K*Uk*a;
                else
                    allus=1:U;
                    idx=find(sum(mask,2)==0);
                    if ~isempty(idx)
                        f(idx)=sum(M(:).*mask(:))/sum(mask(:));
                        allus(idx)=[];
                    end
                    f(allus)=sum(M(allus,:).*mask(allus,:),2)./sum(mask(allus,:),2);
                end
                estr(us)=f(utest(us,1));
            end
            err0=[err0,abs(estr-utest(:,3))];
        end
        [minerr,best]=min(mean(err0(1:1886,:)));
        [minerr2,best2]=min(mean(err0(1:1886,:).^2));
        temp1=[betas(best),betas(best2)];
        mae_KB_us=[mae_KB_us;mean(err0(1887:end,best))];
        rmse_KB_us=[rmse_KB_us;sqrt(mean(err0(1887:end,best2).^2))];
        
        %%%%%%%%%%%%%%%%%%%%%%%%PSD%%%%%%%%%%%%%%%%%%
        s=zeros(Un,In);
        s((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
        [s,mask,rs,cs,ff]=preprocess(s);
        [urs,itms]=size(mask);
        
        [G,K,as,mr,mrc]=GKas(s,mask,gama,true);
        psd=my_psd_estimate(G,as,mask);
        psd=psd/max(psd);
        
        err=[];
        for i=1:numel(betas)
            beta=betas(i);
            for j=1:numel(rrs)
                rr=rrs(j);
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
                err=[err,abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3))];
            end
        end
        
        [minerr,best]=min(mean(err(1:1886,:)));
        [minerr2,best2]=min(mean(err(1:1886,:).^2));
        temp=rrs(ceil(best/numel(rrs)));
        if mod(best,numel(betas))
            temp=[temp,betas(mod(best,numel(betas)))];
        else
            temp=[temp,betas(numel(betas))];
        end
        temp=[temp,rrs(ceil(best2/numel(rrs)))];
        if mod(best2,numel(betas))
            temp=[temp,betas(mod(best2,numel(betas)))];
        else
            temp=[temp,betas(numel(betas))];
        end
        
        bestparamter=[bestparamter;[temp1,temp]];
        
        mae_KBPSD_us=[mae_KBPSD_us;mean(err(1887:end,best))];
        rmse_KBPSD_us=[rmse_KBPSD_us;sqrt(mean(err(1887:end,best2).^2))];
        
        
        % %%%%%%%%%%%%%%%item-based%%%%%%%%%%%%%%%%%%%%%%%%%%%
        alpha=0.0001;
        gama=200;
        betas=[0.00001,0.0001 0.001 0.01 0.1];
        rrs=1./[50 100 150 200 250];
        %%%%%%RKHS%%%%%%%%%
        M=zeros(Un,In);
        M((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
        mask=(M>0);
        [aM,mU,mUI]=AdjustUI(M);
        Wi=Simxy(aM',mask',1);
        fvector=Feavec(M',mask');
        K=KernelGram(fvector,gama);
        
        D=diag(sum(Wi,2));
        L=D-Wi;
        L=size(L,1)*L/trace(L);
        [U,V]=eig(L);
        estr=zeros(size(utest,1),1);
        us=unique(utest(:,1));
        k=10;
        err0=[];
        for i=1:numel(betas)
            beat=betas(i);
            klk=alpha*K+beta*K*L*K;
            Uk=U(:,1:k);
            for j=1:numel(us)
                labels=find(mask(us(j),:));
                items=find(utest(:,1)==us(j));
                numlbs=numel(labels);
                if numlbs>0
                    yL=M(us(j),labels)';%测试集中，项目为items(j)的项目的已知标签
                    KL=K(labels,:);
                    a=pinv(Uk'*(KL'*KL+klk)*Uk)*(Uk'*KL'*yL);
                    f=K*Uk*a;
                else
                    allitem=1:In;
                    idx=find(sum(mask)==0);
                    if ~isempty(idx)
                        f(idx)=sum(M(:).*mask(:))/sum(mask(:));
                        allitem(idx)=[];
                    end
                    f(allitem)=sum(M(:,allitem).*mask(:,allitem))./sum(mask(:,allitem));
                end
                estr(items)=f(utest(items,2));
            end
            err0=[err0,abs(estr-utest(:,3))];
        end
        [minerr,best]=min(mean(err0(1:1886,:)));
        [minerr2,best2]=min(mean(err0(1:1886,:).^2));
        temp1=[betas(best),betas(best2)];
        mae_KB_it=[mae_KB_it;mean(err0(1887:end,best))];
        rmse_KB_it=[rmse_KB_it;sqrt(mean(err0(1887:end,best2).^2))];
        
        %%%%%%%%%PSD%%%%%%%%%
        s=zeros(Un,In);
        s((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
        [s,mask,rs,cs,ff]=preprocess(s);
        [urs,itms]=size(mask);
        [G,K,as,mr,mrc]=GKas(s',mask',gama,true);
        psd=my_psd_estimate(G,as,mask');
        psd=psd/max(psd);
        
        err=[];
        for i=1:numel(betas)
            beta=betas(i);
            for j=1:numel(rrs)
                rr=rrs(j);
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
                err=[err,abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3))];
            end
        end
        
        [minerr,best]=min(mean(err(1:1886,:)));
        [minerr2,best2]=min(mean(err(1:1886,:).^2));
        temp=rrs(ceil(best/numel(rrs)));
        if mod(best,numel(betas))
            temp=[temp,betas(mod(best,numel(betas)))];
        else
            temp=[temp,betas(numel(betas))];
        end
        temp=[temp,rrs(ceil(best2/numel(rrs)))];
        if mod(best2,numel(betas))
            temp=[temp,betas(mod(best2,numel(betas)))];
        else
            temp=[temp,betas(numel(betas))];
        end
        
        bestparamter=[bestparamter;[temp1,temp]];
        mae_KBPSD_it=[mae_KBPSD_it;mean(err(1887:end,best))];
        rmse_KBPSD_it=[rmse_KBPSD_it;sqrt(mean(err(1887:end,best2).^2))];
        
    end
    KBPSD=[mae_KB_us,mae_KBPSD_us,rmse_KB_us,rmse_KBPSD_us,...
        mae_KB_it,mae_KBPSD_it,rmse_KB_it,rmse_KBPSD_it];
    
    save('data/KBPSD.mat','KBPSD');
    save('data/bestparamter.mat','bestparamter');
    %%%%%%%%CPU time%%%%%%%%%%%%%%
    warning('off','all');
    
    CPUtime=[];
    alpha=0.0001;
    beta=0.01;
    gama=200;
    rr=1/150;
    for r=1:2
        temp=[];
        for times=1:5   %repeat 5 times
            t_KB_us=0;
            t_KBPSD_us=0;
            t_KB_it=0;
            t_KBPSD_it=0;
            
            ubase=u{r}.base;
            utest=u{r}.test;
            %%%%%%%%%%%%%%%user-based%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%RHKS%%%%%%%%%%%%
            t0=tic;
            M=zeros(Un,In);
            M((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
            mask=(M>0);
            [aM,mU,mUI]=AdjustUI(M);
            Wi=Simxy(aM,mask,1);
            fvector=Feavec(M,mask);
            K=KernelGram(fvector,gama);
            D=diag(sum(Wi,2));
            L=D-Wi;
            L=size(L,1)*L/trace(L);
            [U,V]=eig(L);
            estr=zeros(size(utest,1),1);
            items=unique(utest(:,2));
            k=5;
            klk=alpha*K+beta*K*L*K;
            Uk=U(:,1:k);
            for j=1:numel(items)
                labels=find(mask(:,items(j)));
                us=find(utest(:,2)==items(j));  %测试集中，项目为items(j)的项目的需要预测的标签
                numlbs=numel(labels);
                if numlbs>0
                    yL=M(labels,items(j));%测试集中，项目为items(j)的项目的已知标签
                    KL=K(labels,:);
                    a=pinv(Uk'*(KL'*KL+klk)*Uk)*(Uk'*KL'*yL);
                    f=K*Uk*a;
                else
                    allus=1:U;
                    idx=find(sum(mask,2)==0);
                    if ~isempty(idx)
                        f(idx)=sum(M(:).*mask(:))/sum(mask(:));
                        allus(idx)=[];
                    end
                    f(allus)=sum(M(allus,:).*mask(allus,:),2)./sum(mask(allus,:),2);
                end
                estr(us)=f(utest(us,1));
            end
            err0=abs(estr-utest(:,3));
            
            t_KB_us=toc(t0);
            
            %%%%%%%%%PSD%%%%%%%%%
            t0=tic;
            s=zeros(Un,In);
            s((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
            [s,mask,rs,cs,ff]=preprocess(s);
            [urs,itms]=size(mask);
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
            t_KBPSD_us=toc(t0);
            
            %%%%%%%%%%%%%%%%item-based%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%RKHS%%%%%%%
            t0=tic;
            M=zeros(Un,In);
            M((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
            mask=(M>0);
            [aM,mU,mUI]=AdjustUI(M);
            Wi=Simxy(aM',mask',1);
            fvector=Feavec(M',mask');
            K=KernelGram(fvector,gama);
            
            D=diag(sum(Wi,2));
            L=D-Wi;
            L=size(L,1)*L/trace(L);
            [U,V]=eig(L);
            estr=zeros(size(utest,1),1);
            us=unique(utest(:,1));
            k=10;
            klk=alpha*K+beta*K*L*K;
            Uk=U(:,1:k);
            for j=1:numel(us)
                labels=find(mask(us(j),:));
                items=find(utest(:,1)==us(j));
                numlbs=numel(labels);
                if numlbs>0
                    yL=M(us(j),labels)';%测试集中，项目为items(j)的项目的已知标签
                    KL=K(labels,:);
                    a=pinv(Uk'*(KL'*KL+klk)*Uk)*(Uk'*KL'*yL);
                    f=K*Uk*a;
                else
                    allitem=1:In;
                    idx=find(sum(mask)==0);
                    if ~isempty(idx)
                        f(idx)=sum(M(:).*mask(:))/sum(mask(:));
                        allitem(idx)=[];
                    end
                    f(allitem)=sum(M(:,allitem).*mask(:,allitem))./sum(mask(:,allitem));
                end
                
                estr(items)=f(utest(items,2));
            end
            err0=abs(estr-utest(:,3));
            t_KB_it=toc(t0);
            %%%%%%%%PSD
            t0=tic;
            s=zeros(Un,In);
            s((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
            [s,mask,rs,cs,ff]=preprocess(s);
            [urs,itms]=size(mask);
            
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
            t_KBPSD_it=toc(t0);
            temp=[temp;[t_KB_us,t_KBPSD_us,t_KB_it,t_KBPSD_it]];
        end
        CPUtime=[CPUtime;mean(temp)];
    end
    save('CPUtime.mat','CPUtime');
else
    load('KBPSD.mat');
    load('bestparamter.mat');
    load('CPUtime.mat');
end

disp('ua.test:');
disp('User-based: mae_KB-----rmse_KB---mae_KBPSD----rmseKBPSD');
disp(['            ',num2str(KBPSD(1,1)),'     ',num2str(KBPSD(1,3)),...
    '     ',num2str(KBPSD(1,2)),'     ',num2str(KBPSD(1,4))]);
disp(['beta:       ',num2str(bestparamter(1,1)),'        ',num2str(bestparamter(1,2)),...
    '         ',num2str(bestparamter(1,4)),'        ',num2str(bestparamter(1,6))]);
disp(['r:                                 ',num2str(bestparamter(1,3)),'        ',num2str(bestparamter(1,5))]);
disp(['CPUtime:     ',num2str(CPUtime(1,1)),'              ',num2str(CPUtime(1,2))]);

disp('Item-based: mae_KB-----rmse_KB---mae_KBPSD----rmseKBPSD');
disp(['            ',num2str(KBPSD(1,5)),'     ',num2str(KBPSD(1,7)),...
    '     ',num2str(KBPSD(1,6)),'     ',num2str(KBPSD(1,8))]);
disp(['beta:       ',num2str(bestparamter(2,1)),'        ',num2str(bestparamter(2,2)),...
    '         ',num2str(bestparamter(2,4)),'        ',num2str(bestparamter(2,6))]);
disp(['r:                                 ',num2str(bestparamter(2,3)),'      ',num2str(bestparamter(2,5))]);
disp(['CPUtime:     ',num2str(CPUtime(1,3)),'              ',num2str(CPUtime(1,4))]);
disp('ub.test:');
disp('User-based: mae_KB-----rmse_KB---mae_KBPSD----rmseKBPSD');
disp(['            ',num2str(KBPSD(2,1)),'     ',num2str(KBPSD(2,3)),...
    '     ',num2str(KBPSD(2,2)),'     ',num2str(KBPSD(2,4))]);
disp(['beta:       ',num2str(bestparamter(3,1)),'        ',num2str(bestparamter(3,2)),...
    '         ',num2str(bestparamter(3,4)),'        ',num2str(bestparamter(3,6))]);
disp(['r:                                 ',num2str(bestparamter(3,3)),'        ',num2str(bestparamter(3,5))]);
disp(['CPUtime:     ',num2str(CPUtime(2,1)),'              ',num2str(CPUtime(2,2))]);
disp('Item-based: mae_KB-----rmse_KB---mae_KBPSD----rmseKBPSD');
disp(['            ',num2str(KBPSD(2,5)),'     ',num2str(KBPSD(2,7)),...
    '     ',num2str(KBPSD(2,6)),'     ',num2str(KBPSD(2,8))]);
disp(['beta:       ',num2str(bestparamter(4,1)),'        ',num2str(bestparamter(4,2)),...
    '         ',num2str(bestparamter(4,4)),'        ',num2str(bestparamter(4,6))]);
disp(['r:                                 ',num2str(bestparamter(4,3)),'      ',num2str(bestparamter(4,5))]);
disp(['CPUtime:     ',num2str(CPUtime(2,3)),'              ',num2str(CPUtime(2,4))]);