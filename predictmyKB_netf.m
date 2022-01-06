clear all;
close all;
addpath('functions','data');
t0=tic;
implement=0;
% 0: load results,
%1: run code, running time is about 630s depending on your computer
if implement
    load('ratings.mat');
    alph=0.005;
    beta=0.001;
    gama=200;
    maes=[];
    rmses=[];
    for r=1:10
        if r<=5
            Un=1000;
            In=1777;
        else
            Un=1500;
            In=888;
        end
        mae=[];
        rmse=[];
        
        
        M=full(ratings{r});
        M=M';
        [utest,M]=GetTestSet1(M,20000);
        mask=(M>0);
        [aM,mU,mUI]=AdjustUI(M);
        %%%%%%%%===user-based=================%%%%%%%%%%%%%
        tic
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
        klk=alph*K+beta*K*L*K;
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
        err=abs(estr-utest(:,3));
        mae=[mae,mean(err)];
        err=(estr-utest(:,3)).^2;
        rmse=[rmse,sqrt(mean(err))];
        
        %%%%%%%%===item-based=================%%%%%%%%%%%%%
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
        klk=alph*K+beta*K*L*K;
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
        err=abs(estr-utest(:,3));
        mae=[mae,mean(err)];
        err=(estr-utest(:,3)).^2;
        rmse=[rmse,sqrt(mean(err))];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        maes=[maes;mae];
        rmses=[rmses;rmse];
    end
    KBnetf=[maes,rmses];
    toc(t0)
    save('data/KBnetf.mat','KBnetf');
else
    load('KBnetf.mat');
end
toc(t0)
disp('User-based: mae+user-first----mae+item-first----rmse+user-first----rmse+item-first');
disp(['            ',num2str(mean(KBnetf(6:10,1))),'            ',num2str(mean(KBnetf(1:5,1))),'            ',...
    num2str(mean(KBnetf(6:10,3))),'            ',num2str(mean(KBnetf(1:5,3)))]);
disp('Item-based: mae+user-first----mae+item-first----rmse+user-first----rmse+item-first');
disp(['            ',num2str(mean(KBnetf(6:10,2))),'            ',num2str(mean(KBnetf(1:5,2))),'            ',...
    num2str(mean(KBnetf(6:10,4))),'            ',num2str(mean(KBnetf(1:5,4)))]);



%[mean(maes(1:5,:)) mean(rmses(1:5,:)) mean(maes(6:10,:)) mean(rmses(6:10,:))]

