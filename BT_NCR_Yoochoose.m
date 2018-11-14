function out = BT_NCR_Yoochoose(seed,session,varargin)
%seed:随机数种子
rng(seed)
%% Parse parameters
params = inputParser;
params.addParameter('method','graded',@(x) ischar(x));
params.addParameter('F',5,@(x) isnumeric(x));
params.addParameter('maxIter',100,@(x) isnumeric(x));
params.addParameter('topN',5,@(x) isnumeric(x));
params.addParameter('K',5,@(x) isnumeric(x));
params.addParameter('earlyStop',false,@(x) islogical(x));
params.parse(varargin{:});
par = params.Results;
par.m = session{end}.allUser;
par.n = session{end}.allItem;
session(end) = [];
%% Run NCR_baseline and use k-folds cross validation
methodSolver = str2func([par.method,'_solver']);
temp = arrayfun(@(x) session{x}.user,(1:length(session))');
cvObj = cvpartition(temp,'KFold',par.K);
out = zeros(par.K,8);
for i = 1:cvObj.NumTestSets
    trainIdx = find(cvObj.training(i));
    testIdx = find(cvObj.test(i));
    [U,V,theta] = feval(methodSolver,session,trainIdx,testIdx,par);
%     filename = sprintf('BT_NCR_FilmTrust_fold_%i_para.mat',i);
%     filename = sprintf('BT_NCR_Tmall[buy.click]_fold_%i_para.mat',i);
    filename = sprintf('BT_NCR_Yoochoose_fold_%i_para.mat',i);
    
    save (filename,'testIdx','U','V','theta','par','-mat');%保存结果
    out(i,:) = BT_NCR_RankEval(session,testIdx,U,V,theta,par);
    fprintf('BT_NCR_baseline:fold [%d/%d] completed\n',i,cvObj.NumTestSets);
end
%---------- 保存结果---start-------
resname = sprintf('BT_NCR_Yoochoose_5fold_result.mat');
meanValue=mean(out);
allOut = out;
allOut(end+1,:)=meanValue;  
save (resname,'out','meanValue','allOut','-mat');% 保存结果
%---------- 保存结果---end-------
fprintf('Final Results: AUC = %f,Pr = %f, Re = %f, MAP = %f, Ndcg = %f, MRR = %f, oPr = %f, oMRR = %f\n',mean(out));
end

function [U,V,theta] = graded_solver(session,trainIdx,testIdx,par)
%模型函数
temp = rand(par.m,par.F);
U = temp./sum(temp,2);
temp = rand(par.n,par.F);
V = temp./sum(temp);
theta = 1.1;
thetaTop = 0;
itemLd = zeros(par.n,1);
temp = zeros(length(trainIdx),1);
cellD = cell(length(trainIdx),1);
for i = 1:length(trainIdx) 
    sample = session{trainIdx(i)};
    temp(i) = sample.user;
    buyItem = sample.buy(1,:);
    noBuyItem = sample.noBuy(1,:);
    itemLd(buyItem) = itemLd(buyItem)+length(noBuyItem);
    thetaTop = thetaTop+length(buyItem)*length(noBuyItem);
    comparePair = combvec(buyItem,noBuyItem);
    cellD{i} = [repmat(temp(i),[size(comparePair,2),1]),comparePair'];
end
thetaTop = (par.F-1)*thetaTop;
dLen = arrayfun(@(x) size(cellD{x},1),1:length(cellD));
[userSet,userP] = numunique(temp); 
userLen = arrayfun(@(x) length(userP{x}),1:length(userSet));
matD = cell2mat(cellD);
[wItemSet,wP] = numunique(matD(:,2));
[lItemSet,lP] = numunique(matD(:,3));
idx = ismember(lItemSet,wItemSet);
lItemSet(~idx) = [];
lP(~idx) = [];
fprintf('init completed\n')
bestAUC = 0;
loseNum = 0;
lastLoss = 0;

tempUU=rand(par.m,par.F);
tempVV=rand(par.n,par.F);

for i = 1:par.maxIter
    tic;
    gammaOut = cellfun(@(x) gammaFunc(x,U,V,theta,par),cellD,'UniformOutput',false); %实现U的更新公式，注意采用的是矩阵形式
    gammaOut = cell2mat(gammaOut);
    temp = arrayfun(@(x) sum(gammaOut(userP{x},:))/userLen(x),...
        (1:length(userSet))','UniformOutput',false);
    U(userSet,:) = cell2mat(temp);
    U = U./sum(U,2);
    fprintf('update U completed\n')
    tempV = zeros(par.n,par.F); %实现V的更新公式，注意采用的是矩阵形式
    gammaOut = repelem(gammaOut(:,1:end),dLen,1);
    temp = arrayfun(@(x) alphaFunc(matD(wP{x},2:3),gammaOut(wP{x},:),V,theta,par,1),(1:length(wItemSet))',...
        'UniformOutput',false);
    tempV(wItemSet,:) = tempV(wItemSet,:)+cell2mat(temp);
    temp = arrayfun(@(x) alphaFunc(matD(lP{x},2:3),gammaOut(lP{x},:),V,theta,par,0),(1:length(lItemSet))',...
        'UniformOutput',false);
    tempV(lItemSet,:) = tempV(lItemSet,:)+cell2mat(temp);
    V = itemLd./tempV;

    V(isnan(V)) = min(V(:));
    V = V./sum(V);
    
    fprintf('update V completed\n')
    temp = thetaFunc(matD(:,2:3),gammaOut,V,theta,par); %实现theta的更新公式，注意采用的是矩阵形式
    theta = thetaTop/temp;
    fprintf('update theta completed\n')
  
    cU = (tempUU-U).^2;
    deltaU = sqrt(sum(cU(:)));
    tempUU = U;
    
     cV = (tempVV-V).^2;
    deltaV = sqrt(sum(cV(:)));
    tempVV=V;
    
    if abs(deltaU)<1e-4 ||  abs(deltaV)<1e-4
        break;
    end

    out = BT_NCR_RankEval(session,testIdx,U,V,theta,par); %在测试集上计算指标结果
   
    fprintf('BT_NCR : iter [%d/%d] completed, time = %f,deltaU = %f,deltaV = %f\n',i,par.maxIter,t,abs(deltaU),abs(deltaV));
    fprintf(' AUC = %f,Pr = %f, Re = %f, MAP = %f, Ndcg = %f, MRR = %f, oPr = %f, oMRR = %f\n',out);
end
end

function r = gammaFunc(sess,U,V,theta,par)
u = U; 
w = V(sess(:,2),:);
v = V(sess(:,3),:);
temp = zeros(size(w,1),par.F);
for i = 1:par.F
    w(:,[1,i]) = w(:,[i,1]);
    v(:,[1,i]) = v(:,[i,1]);
    temp_log = log(w(:,1))-log((w(:,1)+theta*v(:,1)))+sum(log(theta*w(:,2:end))-log((v(:,2:end)+theta*w(:,2:end))),2);
    temp(:,i) = exp(temp_log);
end
p = ones(1,par.F);
for i = 1:size(temp,1)
    tp = p.*temp(i,:);
    p = tp/sum(tp);
end
r = u.*p/sum(u.*p);
end

function r = alphaFunc(sess,gammaR,V,theta,par,isWin)
w = V(sess(:,1),:);
v = V(sess(:,2),:);
r = zeros(1,par.F);
for i = 1:par.F
    w(:,[1,i]) = w(:,[i,1]);
    v(:,[1,i]) = v(:,[i,1]);
    gammaR(:,[1,i]) = gammaR(:,[i,1]);
    if isWin
        r(i) = sum(gammaR(:,1)./(w(:,1)+theta*v(:,1))+sum(theta*gammaR(:,2:end)./(v(:,1)+theta*w(:,1)),2));
    else
        r(i) = sum(theta*gammaR(:,1)./(w(:,1)+theta*v(:,1))+sum(gammaR(:,2:end)./(v(:,1)+theta*w(:,1)),2));
    end
end
end

function r = thetaFunc(sess,gammaR,V,theta,par)
w = V(sess(:,1),:);
v = V(sess(:,2),:);
r = zeros(1,par.F);
for i = 1:par.F
    w(:,[1,i]) = w(:,[i,1]);
    v(:,[1,i]) = v(:,[i,1]);
    r(i) = sum(gammaR(:,i).*(v(:,1)./(w(:,1)+theta*v(:,1))+sum(w(:,2:end)./(v(:,2:end)+theta*w(:,2:end)),2)));
end
r = sum(r);
end
