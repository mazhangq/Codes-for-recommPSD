function [s,mask,rs,cs,ff]=preprocess(s)
[Un,In]=size(s);
mask=(s>0);
% mask1=mask;
rs=1:Un;
cs=1:In;
% kr1=find(sum(mask,2)==0);
% kc1=find(sum(mask)==0);
% kr2=find(sum(mask,2)>1 & sum(mask,2)<3);
% kc2=find(sum(mask)>1 &sum(mask)<3);
% mI=sum(s.*mask)./sum(mask);
% mU=sum(s.*mask,2)./sum(mask,2);
% mUI=sum(sum(s.*mask))/sum(sum(mask));
% 
% ff=zeros(Un,In);
% if ~isempty(kr1)
%     ff(kr1,cs)=ff(kr1,cs)+...
%         repmat(mI,numel(kr1),1);
% end
% if ~isempty(kc1)
%     ff(rs,kc1)=ff(rs,kc1)+...
%         repmat(mU,1,numel(kc1));
% end
% if ~isempty(kr1) & ~isempty(kc1)
%     ff(kr,kc1)=mUI;
% end
% if ~isempty(kr2)
%     mm=sum(s(kr2,:).*mask(kr2,:),2)./sum(mask(kr2,:),2);
%     ff(kr2,cs)=ff(kr2,cs)+...
%         repmat(mm,1,In);
%     for i=1:numel(kr2)
%         cc=find(mask(kr2(i),:));
%         ff(kr2(i),cc)=s(kr2(i),cc);
%     end
% end 
% if ~isempty(kc2)
%     mm=sum(s(:,kc2).*mask(:,kc2))./sum(mask(:,kc2));
%     ff(rs,kc2)=ff(rs,kc2)+...
%         repmat(mm,Un,1);
%     for i=1:numel(kc2)
%         rr=find(mask(:,kc2(i)));
%         ff(rr,kc2(i))=s(rr,kc2(i));
%     end
% end 
% 
% kr=[kr1;kr2];
% kc=[kc1,kc2];
% rs(kr)=[];
% cs(kc)=[];
% s=s(rs,:);
% s=s(:,cs);
% mask=mask(rs,:);
% mask=mask(:,cs);


%%%%%%%%%%%%%%%%%%%%%%%%
kr=find(sum(mask,2)==0);
kc=find(sum(mask)==0);
rs(kr)=[];
cs(kc)=[];
s=s(rs,:);
s=s(:,cs);
mask=mask(rs,:);
mask=mask(:,cs);

mI=sum(s.*mask)./sum(mask);
mU=sum(s.*mask,2)./sum(mask,2);
mUI=sum(sum(s.*mask))/sum(sum(mask));
ff=zeros(Un,In);
if ~isempty(kr)
    ff(kr,cs)=ff(kr,cs)+...
        repmat(mI,numel(kr),1);
end
if ~isempty(kc)
    ff(rs,kc)=ff(rs,kc)+...
        repmat(mU,1,numel(kc));
end
if ~isempty(kr) & ~isempty(kc)
    ff(kr,kc)=mUI;
end