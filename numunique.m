% function [x, p, f]=numunique(x) ：得到x里面的唯一值集合x，和每个唯一值对应的位置索引p，f几乎没用到
% 这个函数就是先unique 然后遍历每个x 找到对应的位置索引 但是这样的话复杂度很高,所以这个函数里面用了一种技巧优化 
% 知道用就行 经常碰到要求x对应索引的

function [x, p, f]=numunique(x)
if isempty(x)
    x=[];
    p=[];
    f=[];
    return
end

isrowvec=size(x,1)==1; % size(A,1)返回的是矩阵A所对应的行数。

if nargout<=1  %  nargin和nargout分别表示这个函数的输入和输出变量的个数
	x=sort(x(:));
	n=[true; diff(x)~=0]; % ~=：不等于
	x=x(n);
	if isrowvec, x=x.'; end
	return
end

[x s]=sort(x(:));
s=s.';   % 数组或者矩阵的转置操作
N=numel(x);
n=find([true; diff(x)~=0]);
x=x(n);
K=numel(x); % numel函数用于计算数组中满足指定条件的元素个数。 K=numel(x) 返回数组x的个数
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
