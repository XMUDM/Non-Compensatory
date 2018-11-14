tic;

% load('.\data\session_FilmTrust_new.mat')
% fprintf('Start session_FilmTrust_new ……\n');

% load('.\data\session_CiaoDVD_new.mat')
% fprintf('Start session_CiaoDVD_new ……\n');

% load('.\data\session_MovieLens_new_small.mat')
% fprintf('Start session_MovieLens_new_small ……\n'); 

% load('.\data\session.mat')   
% fprintf('Start session_buy_nobuy ……\n');

% load('.\data\session_buy_click.mat')
% fprintf('Start session_buy_click ……\n');

load('.\data\session_Yoochoose.mat')
fprintf('Start session_Yoochoose ……\n'); 
% % !!!! Note : 对应的Main函数要改

seed = 1;

fprintf('Starting run BT_NCR ……\n ');
% out =BT_NCR(seed,session);
out =BT_NCR_Yoochoose(seed,session);


t=toc;
fprintf('total_time: %f\n',t);