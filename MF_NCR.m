function out = MF_NCR(seed,rawData,varargin)
rng(seed);
%% Parse parameters
params = inputParser;
params.addParameter('method','graded',@(x) ischar(x));
params.addParameter('F',10,@(x) isnumeric(x));
params.addParameter('lr',5,@(x) isnumeric(x));
params.addParameter('regU',0.01,@(x) isnumeric(x));
params.addParameter('regV',0.01,@(x) isnumeric(x));
params.addParameter('regTheta',0.01,@(x) isnumeric(x));
params.addParameter('regB',0.01,@(x) isnumeric(x));
params.addParameter('momentum',0.8,@(x) isnumeric(x));
params.addParameter('batchNum',10,@(x) isnumeric(x));
params.addParameter('maxIter',1000,@(x) isnumeric(x));
params.addParameter('K',5,@(x) isnumeric(x));
params.addParameter('adaptive',true,@(x) islogical(x));
params.addParameter('topN',5,@(x) isnumeric(x));
params.parse(varargin{:});
par = params.Results;
%% Run biasedMF and use K-folds cross validation
methodSolver = str2func([par.method,'_solver']);
par.m = max(rawData(:,1));
par.n = max(max(rawData(:,2)));
temp = arrayfun(@(x) rawData(x,1),(1:length(rawData))');
cvObj = cvpartition(temp,'KFold',par.K);
out = zeros(par.K,5);
for i = 1:cvObj.NumTestSets
    trainIdx = find(cvObj.training(i));
    testIdx = find(cvObj.test(i));
    [U,V,theta] = feval(methodSolver,rawData,trainIdx,testIdx,par);
    out(i,:) = MFRankEval_NCR(rawData,testIdx,U,V,theta,par);
%     filename = sprintf('MF_NCR_fold_%i.mat',i);
%     save (filename,'testIdx','U','V','theta','par','-mat');
    fprintf('MF_NCR %d/%d fold completed\n',i,cvObj.NumTestSets);
end
fprintf('AUC = %f, NDCG = %f, RMSE = %f, MAE = %f, MRR = %f \n',mean(out));
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          end

function [U,V,theta] = graded_solver(rawData,trainIdx,testIdx,par)
trainData = rawData(trainIdx,:);
trainData = trainData(randperm(size(trainData,1)),:);
fprintf('generate data completed\n');
theta = 0.1;
U = normrnd(0,0.1,par.m,par.F);
V = normrnd(0,0.1,par.n,par.F);
incU = zeros(par.m,par.F);
incV = zeros(par.n,par.F);
incTheta = 0;
lastLoss = 0;
bestRMSE = 0;%inf;
loseNum = 0;
lastLoss = 0;
for i = 1:par.maxIter
    loss = 0;
    u = U(trainData(:,1),:);
    v = V(trainData(:,2),:); 
    x1 = zeros(length(trainIdx),par.F);
    x2 = zeros(length(trainIdx),par.F);
    ixV = zeros(length(trainIdx),par.F);
    ixTheta = zeros(length(trainIdx),1);
    sumU = sum(exp(u),2);
    for m = 1:par.F
        x1(:,m) = exp(u(:,m))./sumU;
%         if sum(isnan(x1(:,m)))~=0
%             a=1;
%         end
        x2(:,m) = (exp(theta).*v(:,m))+(sum(v,2)-v(:,m));        
        ixV(:,m)=((exp(u(:,m)).*exp(theta))+(sumU-exp(u(:,m))))./sumU;
        ixTheta = ixTheta + (x1(:,m).*exp(theta).*v(:,m));
    end   
    pred = sum(x1.*x2,2);
    ixU = x2.*(x1-x1.^2);
    error = pred-trainData(:,3);
    loss = loss+sum(error.^2);
    ixU = error.*ixU+par.regU.*u;
    ixV = error.*ixV+par.regV.*v;
    ixTheta = error.*ixTheta;
    gU = zeros(par.m,par.F);
    gV = zeros(par.n,par.F);
    gTheta = sum(ixTheta);
    for z = 1:length(trainIdx)
        gU(trainData(z,1),:) = gU(trainData(z,1),:)+ixU(z,:);
        gV(trainData(z,2),:) = gV(trainData(z,2),:)+ixV(z,:);
    end
    incU = par.momentum*incU+par.lr*gU/length(trainIdx);
    incV = par.momentum*incV+par.lr*gV/length(trainIdx);
    incTheta = par.momentum*incTheta+par.lr*gTheta/length(trainIdx);
%     if sum(isnan(incU))~=0
%         a=1;
%     end
    U = U - incU;
    V = V - incV;
    theta = theta - incTheta;
    
    
    if theta < 0
        theta = 0.10;
    end
    
%     if (theta < log(1/(par.F-1))) || (isnan(theta))
%         theta = log(1/(par.F-1));
%     end


%     U_ill = sum(U<-5,2);
%     V_ill = sum(V<-5,2);
%     for ii=1:length(U_ill)
%         if U_ill(ii)~=0
%             U(ii,:)=normrnd(0,0.1,1,par.F);
%         end
%     end
%     
%     
%     for ii=1:length(V_ill)
%         if V_ill(ii)~=0
%             V(ii,:)=normrnd(0,0.1,1,par.F);
%         end
%     end
    
%     if isnan(theta)
%         a=1;
%     end


    loss = loss+par.regU*sum(sum(U.^2))+par.regV*sum(sum(V.^2));%loss = loss+par.regU*sum(sum(U(trainData(:,1),:).^2))+par.regV*sum(sum(V(trainData(:,2),:).^2));
    deltaLoss = lastLoss-0.5*loss;
%     if abs(deltaLoss)<1e-5
%         break;
%     end
    
    out = MFRankEval_NCR(rawData,testIdx,U,V,theta,par);
%     if out(1) < bestRMSE
%         loseNum = loseNum+1;
%         if loseNum >= 100
%             U = bestU;
%             V = bestV;
%             break;
%         end
%     else
%         bestRMSE = out(1);
%         bestU = U;
%         bestV = V;
%         loseNum = 0;
%     end
% 
    if par.adaptive && i > 2
        if lastLoss > 0.5*loss
            par.lr = 1.05*par.lr;
        else
            par.lr = 0.7*par.lr;
%             par.lr = 0.5*par.lr;
        end
    end
    lastLoss = 0.5*loss;
    if mod(i,100)==0
        fprintf('MF NCR iter [%d/%d] completed, loss = %f, delta_loss: %f, lr: %f theta=%f\n',i,par.maxIter,0.5*loss,deltaLoss,par.lr,theta);
        fprintf('AUC = %f, NDCG = %f, RMSE = %f, MAE = %f, MRR = %f \n',out);
    end
end
end