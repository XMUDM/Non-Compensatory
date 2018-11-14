function [U,V,Theta] = NCR_BPR(seed,session,varargin)
rng(seed)
%% Parse parameters
params = inputParser;
params.addParameter('method','graded',@(x) ischar(x));
params.addParameter('F',5,@(x) isnumeric(x));
params.addParameter('lr',5,@(x) isnumeric(x));
params.addParameter('regU',0.3,@(x) isnumeric(x));
params.addParameter('regV',0.3,@(x) isnumeric(x));
params.addParameter('regTheta',0.001,@(x) isnumeric(x));
params.addParameter('momentum',0.8,@(x) isnumeric(x));
params.addParameter('batchNum',10,@(x) isnumeric(x));
params.addParameter('maxIter',100,@(x) isnumeric(x));
params.addParameter('K',5,@(x) isnumeric(x));
params.addParameter('adaptive',true,@(x) islogical(x));
params.addParameter('earlyStop',true,@(x) islogical(x));
params.addParameter('topN',5,@(x) isnumeric(x));
params.parse(varargin{:});
par = params.Results;
par.m = session{end}.allUser;
par.n = session{end}.allItem;
session(end) = [];
%% Run BTR and use K-folds cross validation
methodSolver = str2func([par.method,'_solver']);
temp = arrayfun(@(x) session{x}.user,(1:length(session))');
cvObj = cvpartition(temp,'KFold',par.K);
out = zeros(par.K,5);
for i = 1:cvObj.NumTestSets
    trainIdx = find(cvObj.training(i));
    testIdx = find(cvObj.test(i));
    tic
    [U,V,Theta] = feval(methodSolver,session,trainIdx,testIdx,par);
    toc
    filename = sprintf('BPR_N_fold_%i.mat',i);
    save (filename,'testIdx','U','V','Theta','par','-mat');
    out(i,:) = NCR_BPRRankEval(session,testIdx,U,V,Theta,par);
    fprintf('BPR-N:fold [%d/%d] completed\n',i,cvObj.NumTestSets);
end
fprintf('BPR-N Results: AUC = %f, NDCG = %f, MRR = %f, MAP = %f, Pre = %f\n',mean(out));
end

function [U,V,Theta] = graded_solver(session,trainIdx,testIdx,par)
D = cell(length(trainIdx),1);
for i = 1:length(trainIdx)
    sample = session{trainIdx(i)};
    buyItem = sample.buy(1,:);
    noBuyItem = sample.noBuy(1,:);
    comparePair = combvec(buyItem,noBuyItem);
    D{i} = [repmat(sample.user,[size(comparePair,2),1]),comparePair'];
end
D = cell2mat(D);
batchIdx = discretize(1:size(D,1),par.batchNum);
[~,p] = numunique(batchIdx);
fprintf('BPR-N generate data completed\n');

U = rand(par.m,par.F);
V = rand(par.n,par.F);
oldU = U;
oldV = V;
Theta = 1;
incU = zeros(par.m,par.F);
incW = zeros(par.n,par.F);
incV = zeros(par.n,par.F);
incTheta = 0;
lastLoss = 0;

for i = 1:par.maxIter
    loss = 0;
    for j = 1:par.batchNum
        u = U(D(p{j},1),:);
        w = V(D(p{j},2),:);
        v = V(D(p{j},3),:);
        x = zeros(length(D(p{j})),1);
        ixU = zeros(length(D(p{j})),par.F);
        ixW = zeros(length(D(p{j})),par.F);
        ixV = zeros(length(D(p{j})),par.F);
        ixTheta = zeros(length(D(p{j})),1);
        for m = 1:par.F
            x1 = exp(u(:,m))./sum(exp(u),2);
            x2 = exp(Theta).*(w(:,m)-v(:,m))+sum(w,2)-w(:,m)-sum(v,2)+v(:,m);
            x = x + x1.*x2;
            ixU(:,m) = x2.*(x1-x1.^2);
            ixW(:,m)=(exp(u(:,m)).*exp(Theta)+sum(exp(u),2)-exp(u(:,m)))./sum(exp(u),2);
            ixV(:,m)= -ixW(:,m);
            ixTheta = ixTheta + x1.*(exp(Theta).*(w(:,m)-v(:,m)));
        end
        ix1 = -logsig(-x);
        ixU = ix1.*ixU+par.regU*U(D(p{j},1),:);
        ixW = ix1.*ixW+par.regV*V(D(p{j},2),:);
        ixV = ix1.*ixV+par.regU*V(D(p{j},3),:);
        ixTheta = ix1.*ixTheta+par.regTheta*Theta;
        loss = loss+sum(-log(logsig(x)));
        gU = zeros(par.m,par.F);
        gW = zeros(par.n,par.F);
        gV = zeros(par.n,par.F);
        gTheta = sum(ixTheta);
        for z = 1:length(p{j})
            gU(D(p{j}(z),1),:) = gU(D(p{j}(z),1),:)+ixU(z,:);
            gW(D(p{j}(z),2),:) = gW(D(p{j}(z),2),:)+ixW(z,:);
            gV(D(p{j}(z),3),:) = gV(D(p{j}(z),3),:)+ixV(z,:);
        end
        incU = par.momentum*incU+par.lr*gU/length(p{j});
        incW = par.momentum*incW+par.lr*gW/length(p{j});
        incV = par.momentum*incV+par.lr*gV/length(p{j});
        incTheta = par.momentum*incTheta+par.lr*gTheta/length(p{j});
        U = U - incU;
        V = V - incW;
        V = V - incV;
        Theta = Theta - incTheta;
        if Theta<0
            Theta=0.1;
        end
        loss = loss+par.regU*sum(sum(U(D(p{j},1),:).^2))+par.regV*sum(sum(V(D(p{j},2),:).^2))+...
            par.regV*sum(sum(V(D(p{j},3),:).^2));
        
    end
    deltaLoss = lastLoss-0.5*loss;
    if abs(deltaLoss)<1e-5
        break;
    end
    cU=(oldU-U).^2;cV=(oldV-V).^2;
    if abs(sqrt(sum(cU(:))))<1e-4 || abs(sqrt(sum(cV(:))))<1e-4
    	break;
    end
    oldU = U;
    oldV = V;
    out = NCR_BPRRankEval(session,testIdx,U,V,Theta,par);
    if par.adaptive && i > 2
        if lastLoss > 0.5*loss
            par.lr = 1.05*par.lr;
        else
            par.lr = 0.5*par.lr;
        end
        lastLoss = 0.5*loss;
    else
        lastLoss = 0.5*loss;
    end
    if mod(i,10)==0
        fprintf('BPR_NCR:iter [%d/%d] completed, loss: %f, delta_loss = %f, lr: %f ,theta:%f\n',i,par.maxIter,0.5*loss,deltaLoss,par.lr,Theta);
        fprintf('AUC = %f, NDCG = %f, MRR = %f, MAP = %f, Precision = %f\n',out);
	end
end
end