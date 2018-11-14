function [U,V] = BPR_baseline(seed,session,varargin)
rng(seed)
%% Parse parameters
params = inputParser;
params.addParameter('method','graded',@(x) ischar(x));
params.addParameter('F',5,@(x) isnumeric(x));
params.addParameter('lr',5,@(x) isnumeric(x));
params.addParameter('regU',0.3,@(x) isnumeric(x));
params.addParameter('regV',0.3,@(x) isnumeric(x));
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
%% Run BPR_baseline and use K-folds cross validation
methodSolver = str2func([par.method,'_solver']);
temp = arrayfun(@(x) session{x}.user,(1:length(session))');
cvObj = cvpartition(temp,'KFold',par.K);
out = zeros(par.K,5);
for i = 1:cvObj.NumTestSets
    trainIdx = find(cvObj.training(i));
    testIdx = find(cvObj.test(i));
    tic
    [U,V] = feval(methodSolver,session,trainIdx,testIdx,par);
    toc
    filename = sprintf('BPR_fold_%i.mat',i);
    save (filename,'testIdx','U','V','par','-mat');
    out(i,:) = BPRRankEval(session,testIdx,U,V,par);
    fprintf('BPR_baseline:fold [%d/%d] completed\n',i,cvObj.NumTestSets);
end
fprintf('BPR_baseline Results: AUC = %f, NDCG = %f, MRR = %f, MAP = %f, Pre = %f\n',mean(out));
end

function [U,V] = graded_solver(session,trainIdx,testIdx,par)
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
fprintf('BPR-baseline generate data completed\n');
U = rand(par.m,par.F);
V = rand(par.n,par.F);
oldU = U;
oldV = V;
incU = zeros(par.m,par.F);
incW = zeros(par.n,par.F);
incV = zeros(par.n,par.F);
bestAUC = 0;
loseNum = 0;
lastLoss = 0;
for i = 1:par.maxIter
    loss = 0;
    for j = 1:par.batchNum
        winPred = sum(U(D(p{j},1),:).*V(D(p{j},2),:),2);
        losePred = sum(U(D(p{j},1),:).*V(D(p{j},3),:),2);
        compareDiff = logsig(losePred-winPred);
        loss1 = log(logsig(winPred-losePred));
        loss1(loss1==-inf)=nan;
        loss1(isnan(loss1))=min(loss1);
        loss = loss+sum(-loss1);
        ixU = -compareDiff.*(V(D(p{j},2),:)-V(D(p{j},3),:))+par.regU*U(D(p{j},1),:);
        ixW = -compareDiff.*U(D(p{j},1),:)+par.regV*V(D(p{j},2),:);
        ixV = -compareDiff.*(-U(D(p{j},1),:))+par.regV*V(D(p{j},3),:);
        gU = zeros(par.m,par.F);
        gW = zeros(par.n,par.F);
        gV = zeros(par.n,par.F);
        for z = 1:length(p{j})
            gU(D(p{j}(z),1),:) = gU(D(p{j}(z),1),:)+ixU(z,:);
            gW(D(p{j}(z),2),:) = gW(D(p{j}(z),2),:)+ixW(z,:);
            gV(D(p{j}(z),3),:) = gV(D(p{j}(z),3),:)+ixV(z,:);
        end
        incU = par.momentum*incU+par.lr*gU/length(p{j});
        incW = par.momentum*incW+par.lr*gW/length(p{j});
        incV = par.momentum*incV+par.lr*gV/length(p{j});
        U = U - incU;
        V = V - incW;
        V = V - incV;
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
    out = BPRRankEval(session,testIdx,U,V,par);
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
        fprintf('BPR_baseline:iter [%d/%d] completed, loss: %f, delta_loss = %f, lr: %f\n',i,par.maxIter,0.5*loss,deltaLoss,par.lr);
        fprintf('AUC = %f, NDCG = %f, MRR = %f, MAP = %f, Precision = %f\n',out);
    end
end
end