function out = MFRankEval_NCR(rawData,testIdx,U,V,theta,par)
out = nan*ones(1,5);%AUC/NDCG/RMSE/MAE/MRR
rawData = rawData(testIdx,:);
rawData = [rawData,zeros(size(rawData,1),1)];
u = U(rawData(:,1),:);
v = V(rawData(:,2),:); 
x=zeros(length(testIdx),1);
for m = 1:par.F
    x1 = exp(u(:,m))./sum(exp(u),2);
    x2 = exp(theta).*v(:,m)+sum(v,2)-v(:,m);
    x = x + x1.*x2;
end
rawData(:,4) = x;
out(3) = rmseEval(rawData(:,4),rawData(:,3));
out(4) = maeEval(rawData(:,4),rawData(:,3));


[userSet,p] = numunique(rawData(:,1));%auc\NDCG\MRR
temp_out = nan*ones(length(userSet),3);
for i=1:length(userSet)
    sample = rawData(p{i},:);
    item = sample(:,2);
    rating = sample(:,3);
    pred = sample(:,4);
    [~,pred_idx] = sort(pred,'descend');
    %auc
    target = zeros(length(rating),1);
    target(rating>3) = 1;
    %全是正/负样本
    if sum(target~=zeros(length(rating),1))==0 || sum(target~=ones(length(rating),1))==0
        target = zeros(length(rating),1);
        for j=1:floor(length(rating)/2)
            target(pred_idx==j)=1;
        end
    end
    temp_out(i,1) = aucEval(target,pred);
    %ndcg
    temp_out(i,2) = ndcgEval(item(pred_idx),item(rating>3),par.topN);
    %mrr
    temp_out(i,3) = mrrEval(item(pred_idx),item(rating>3),par.topN);
end
del_idx = isnan(temp_out(:,1));
temp_out(del_idx,:) = [];
out(1) = mean(temp_out(:,1));
out(2) = mean(temp_out(:,2));
out(5) = mean(temp_out(:,3));
end








function result =aucEval(test_targets,output)
[~,I]=sort(output);
M=0;N=0;
for i=1:length(output)
    if(test_targets(i)==1)
        M=M+1;
    else
        N=N+1;
    end
end
sigma=0;
for i=M+N:-1:1
    if(test_targets(I(i))==1)
        sigma=sigma+i;
    end
end
result=(sigma-(M+1)*M/2)/(M*N);
end


function v = ndcgEval(rankedList,groundTruth,numRecs)
if numRecs>length(rankedList)
    numRecs = length(rankedList);
end
dcg = 0;
idcg = 0;
for i = 1:numRecs
    idx = find(groundTruth==rankedList(i),1);
    if ~isempty(idx)
        dcg = dcg+1/log2(i+1);
    end
    idcg = idcg + 1/log2(i+1);
end
v = dcg/idcg;
end


function v = rmseEval(pred,score)
    v=sqrt((sum((pred-score).^2))./length(pred));
end

function v = maeEval(pred,score)
    v=mean(abs(pred-score));
end



function v = mrrEval(rankedList,groundTruth,numRecs)
if numRecs>length(rankedList)
    numRecs = length(rankedList);
end
v = 0;
for i = 1:numRecs
    idx = find(groundTruth==rankedList(i),1);
    if ~isempty(idx)
        v = 1/i;
        return
    end
end
end


function v = prEval(rankedList,groundTruth,numRecs)
if numRecs>length(rankedList)
    numRecs = length(rankedList);
end
hits = sum(ismember(rankedList(1:numRecs),groundTruth));
v = hits/numRecs;
end

function v = reEval(rankedList,groundTruth,numRecs)
if numRecs>length(rankedList)
    numRecs = length(rankedList);
end
hits = sum(ismember(rankedList(1:numRecs),groundTruth));
v = hits/length(groundTruth);
end

function v = mapEval(rankedList,groundTruth,numRecs)
if numRecs>length(rankedList)
    numRecs = length(rankedList);
end
hits = 0;
sumPrecs = 0;
for i = 1:numRecs
    idx = find(groundTruth==rankedList(i),1);
    if ~isempty(idx)
        hits = hits+1;
        sumPrecs = sumPrecs+hits/i;
    end
end
v = sumPrecs/length(groundTruth);
end

function v = oPrEval(rankedList,groundTruth)
hits = sum(ismember(rankedList(1:length(groundTruth)),groundTruth));
v = hits/length(groundTruth);
end

function v = oMrrEval(rankedList,groundTruth)
v = 0;
for i = 1:length(rankedList)
    idx = find(groundTruth==rankedList(i),1);
    if ~isempty(idx)
        v = 1/i;
        return
    end
end
end