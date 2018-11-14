% function [x, p, f]=numunique(x) ���õ�x�����Ψһֵ����x����ÿ��Ψһֵ��Ӧ��λ������p��f����û�õ�
% �������������unique Ȼ�����ÿ��x �ҵ���Ӧ��λ������ ���������Ļ����ӶȺܸ�,�������������������һ�ּ����Ż� 
% ֪���þ��� ��������Ҫ��x��Ӧ������

function [x, p, f]=numunique(x)
if isempty(x)
    x=[];
    p=[];
    f=[];
    return
end

isrowvec=size(x,1)==1; % size(A,1)���ص��Ǿ���A����Ӧ��������

if nargout<=1  %  nargin��nargout�ֱ��ʾ����������������������ĸ���
	x=sort(x(:));
	n=[true; diff(x)~=0]; % ~=��������
	x=x(n);
	if isrowvec, x=x.'; end
	return
end

[x s]=sort(x(:));
s=s.';   % ������߾����ת�ò���
N=numel(x);
n=find([true; diff(x)~=0]);
x=x(n);
K=numel(x); % numel�������ڼ�������������ָ��������Ԫ�ظ����� K=numel(x) ��������x�ĸ���
if isrowvec, x=x.'; end

switch(nargout)
	case{2}
		if K==1
			p{1}=s;
		else
			p{K}=s(n(K):N);
			for k=1:K-1
				p{k}=s(n(k):n(k+1)-1);
			end
		end
	case{3}
		if K==1
			p{1}=s;
			f(1)=p{1}(1);
		else
			p{K}=s(n(K):N);
			f(K)=p{K}(1);
			for k=1:K-1
				p{k}=s(n(k):n(k+1)-1);
				f(k)=p{k}(1);
			end
		end
	otherwise
		error('More than 3 outputs are not allowed!')
end
end
