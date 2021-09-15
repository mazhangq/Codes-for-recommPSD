clear all;
close all;
%/Users/yangzhihua/Documents/Lab/myDatabase/ml-100k
load('../../../myDatabase/ml-100k/mat/u1base.mat');
u{1}.base=u1base;
clear u1base;
load('../../../myDatabase/ml-100k/mat/u1test.mat');
u{1}.test=u1test;
clear u1test;
load('../../../myDatabase/ml-100k/mat/u2base.mat');
u{2}.base=u2base;
clear u2base;
load('../../../myDatabase/ml-100k/mat/u2test.mat');
u{2}.test=u2test;
clear u2test;
load('../../../myDatabase/ml-100k/mat/u3base.mat');
u{3}.base=u3base;
clear u3base;
load('../../../myDatabase/ml-100k/mat/u3test.mat');
u{3}.test=u3test;
clear u3test;
load('../../../myDatabase/ml-100k/mat/u4base.mat');
u{4}.base=u4base;
clear u4base;
load('../../../myDatabase/ml-100k/mat/u4test.mat');
u{4}.test=u4test;
clear u4test;
load('../../../myDatabase/ml-100k/mat/u5base.mat');
u{5}.base=u5base;
clear u5base;
load('../../../myDatabase/ml-100k/mat/u5test.mat');
u{5}.test=u5test;
clear u5test;
load('../../../myDatabase/ml-100k/mat/uabase.mat');
u{6}.base=uabase;
clear uabase;
load('../../../myDatabase/ml-100k/mat/uatest.mat');
u{6}.test=uatest;
clear uatest;
load('../../../myDatabase/ml-100k/mat/ubbase.mat');
u{7}.base=ubbase;
clear ubbase;
load('../../../myDatabase/ml-100k/mat/ubtest.mat');
u{7}.test=ubtest;
clear ubtest;

Un=943;
In=1682;


alph=0.005;
beta=0.001;
gama=200;
%%%%%%%%%%%%%%%%for user-based%%%%%%%%%%%%%%%%%
%%sim=1:3;r=1:5  for u1.*-u5.*
%%sim=3:3;r=6:7  for u6.*-u7.*
tstart=tic;
mae_us=[];
rmse_us=[];

for r=1:5%6:7 %
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
end


%%%%%%%%%%%%%%%%for item-based%%%%%%%%%%%%%%%%%
%%sim=1:3;r=1:5  for u1.*-u5.*
%%sim=3:3;r=6:7  for u6.*-u7.*
%%
%%
mae_it=[];
rmse_it=[];
for r=1:5%6:7 %
    ubase=u{r}.base;
    utest=u{r}.test;
    M=zeros(Un,In);
    M((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
    mask=(M>0);
    [aM,mU,mUI]=AdjustUI(M);
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
[mean(mae_us) std(mae_us) mean(rmse_us) std(rmse_us) ...
    mean(mae_it) std(mae_it) mean(rmse_it) std(rmse_it)]
tElapsed = toc(tstart)
%ua_ub运行时间140.6431s
