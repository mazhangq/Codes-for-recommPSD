%%%%�������ֱ�
%UI---��ʼ���ֱ�
%aUI----����������ֱ�
%mUI-----�ڳ�ʼ���ֱ��У������Ѵ����Ŀ��ƽ����
%mU----�ڳ�ʼ���ֱ��У�ÿ���û������д�ֵ�ƽ����

function [aUI,mU,mUI]=AdjustUI(UI)
[U,I]=size(UI);
nonEl=UI~=0;
mUI=sum(UI(:))/sum(nonEl(:));
mU=sum((UI-mUI).*nonEl,2)./(sum(nonEl,2)+eps);
aUI=(UI-repmat(mU+mUI,1,I)).*nonEl;