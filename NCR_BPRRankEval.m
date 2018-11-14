function out = NCR_BPRRankEval(session,testIdx,U,V,theta,par)
out = nan*ones(length(testIdx),5);
for i = 1:length(testIdx)
    sample = session{testIdx(i)};
    u = sample.user;    
    correctItems = sort(sample.buy(1,:));
    candItems = [sample.noBuy(1,:),sample.buy(1,:)];
    s = pred(candItems,U(u,:),V,theta,par);
    [~,idx] = sort(s,'descend');
    rankedItems = candItems(idx);
    target = zeros(length(candItems),1)';
    target(length(sample.noBuy(1,:))+1:end) = 1;
    out(i,1) = aucEval(target,s);
    out(i,2) = ndcgEval(rankedItems,correctItems,par.topN);
    out(i,3) = mrrEval(rankedItems,correctItems,par.topN);
    out(i,4) = mapEval(rankedItems,correctItems,par.topN);
    out(i,5) = prEval(rankedItems,correctItems,par.topN);
end
for i=1:5
    idx = isnan(out(:,i));
    out(idx,:) = [];
end
out = mean(out);
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

function r = pred(items,u,V,theta,par)
r = zeros(1,length(items));
w = V(items,:);
for i = 1:par.F
    w(:,[1,i]) = w(:,[i,1]);
    r1 = exp(u(:,i))./sum(exp(u),2);
    r2 = exp(theta).*w(:,1)+sum(w(:,2:end),2);
    r = r + r1.*(r2');
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