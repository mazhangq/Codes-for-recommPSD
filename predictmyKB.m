clear all;
close all;
addpath('functions','data');
implement=0;
% 0: load results, 
%1: run code,running time is about 320s depending on your computer
if implement
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
    
    
    alph=0.005;
    beta=0.001;
    gama=200;
    %%%%%%%%%%%%%%%%for user-based%%%%%%%%%%%%%%%%%
    
    mae_us=[];
    rmse_us=[];
    mae_it=[];
    rmse_it=[];
    for r=1:5%
        ubase=u{r}.base;
        utest=u{r}.test;
        M=zeros(Un,In);
        M((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
        mask=(M>0);
        [aM,mU,mUI]=AdjustUI(M);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        %         klk=beta*L+alph*Ki;
        
        klk=alph*K+beta*K*L*K;
        Uk=U(:,1:k);
        for j=1:numel(items)
            labels=find(mask(:,items(j)));
            us=find(utest(:,2)==items(j));  %测试集中，项目为items(j)的项目的需要预测的标签
            numlbs=numel(labels);
            if numlbs>0
                yL=M(labels,items(j));%测试集中，项目为items(j)的项目的已知标签
                KL=K(labels,:);
                %%%%%%%==my opion
                a=pinv(Uk'*(KL'*KL+klk)*Uk)*(Uk'*KL'*yL);
                f=K*Uk*a;
                %%%%%%%==my opion
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
        mae_us=[mae_us;mean(err)];
        err=(estr-utest(:,3)).^2;
        rmse_us=[rmse_us;sqrt(mean(err))]; 
    
    %%%%%%%%%%%%%%%%for item-based%%%%%%%%%%%%%%%%%
    
%         ubase=u{r}.base;
%         utest=u{r}.test;
%         M=zeros(Un,In);
%         M((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
%         mask=(M>0);
%         [aM,mU,mUI]=AdjustUI(M);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Wi=Simxy(aM',mask',1);
        fvector=Feavec(M',mask');
        K=KernelGram(fvector,gama);
        %K=KernelGram(M',mask',para,1);
        
        D=diag(sum(Wi,2));
        L=D-Wi;
        L=size(L,1)*L/trace(L);
        [U,V]=eig(L);
        estr=zeros(size(utest,1),1);
        us=unique(utest(:,1));
        k=10;
        klk=alph*K+beta*K*L*K;
        Uk=U(:,1:k);
        for j=1:numel(us);
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
        mae_it=[mae_it;mean(err)];
        err=(estr-utest(:,3)).^2;
        rmse_it=[rmse_it;sqrt(mean(err))];
    end
    
    KB=[mae_us,mae_it,rmse_us,rmse_it];
    save('data/KB.mat','KB');
else
    load('KB.mat');
end

disp('User-based: mae+-std------------------rmse+-std');
disp(['            ',num2str(mean(KB(:,1))),'+-',num2str(std(KB(:,1))),...
    '     ',num2str(mean(KB(:,3))),'+-',num2str(std(KB(:,3)))]);
disp('Item-based: mae+-std------------------rmse+-std');
disp(['            ',num2str(mean(KB(:,2))),'+-',num2str(std(KB(:,2))),...
    '     ',num2str(mean(KB(:,4))),'+-',num2str(std(KB(:,4)))]);
