function out = BT_NCR_RankEval(session,testIdx,U,V,theta,par)
out = nan*ones(length(testIdx),8);  
for i = 1:length(testIdx)
    sample = session{testIdx(i)};
    u = sample.user;
    [~,g] = max(U(u,:));
    correctItems = sort(sample.buy(1,:));
    candItems = [sample.noBuy(1,:),sample.buy(1,:)];

    s = pred(candItems,g,V,theta);
    [~,idx] = sort(s,'descend');
    rankedItems = candItems(idx);
    
    label_0 = zeros(1,length(sample.noBuy(1,:)));
    label_1 = ones(1,length(sample.buy(1,:)));

    label = [label_0,label_1];
    
    out(i,1) = aucEval(label,s);
    out(i,2) = prEval(rankedItems,correctItems,par.topN);
    out(i,3) = reEval(rankedItems,correctItems,par.topN);
    out(i,4) = mapEval(rankedItems,correctItems,par.topN);
    out(i,5) = ndcgEval(rankedItems,correctItems,par.topN);
    out(i,6) = mrrEval(rankedItems,correctItems,par.topN);
    out(i,7) = oPrEval(rankedItems,correctItems);
    out(i,8) = oMrrEval(rankedItems,correctItems);
end
idx = isnan(out(:,1));
out(idx,:) = [];
out = mean(out);
end

function r = pred(items,g,V,theta)
r = zeros(1,length(items));
w = V(items,:);
w(:,[1,g]) = w(:,[g,1]);
comparePair = ones(length(items)-1,2);
comparePair(:,2) = 2:length(items);

for i = 1:length(r)
    w([1,i],:) = w([i,1],:);
     r_log = sum(log(w(comparePair(:,1),1))-log((w(comparePair(:,1),1)+theta*w(comparePair(:,2),1)))+...
        sum(log(theta*w(comparePair(:,1),2:end))-log((w(comparePair(:,2),2:end)+theta*...
        w(comparePair(:,1),2:end))),2));
    r(i) = exp(r_log);

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