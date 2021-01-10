clear;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 사람 8명을 맞고 반사된 레이더 신호를 깊은 인공 신경망을 이용해서 분류하는 예제 코드입니다.                            %
% 하나의 레이더 신호는 길이가 30이며, 한 사람에 대해 500번씩 신호를 받았습니다.                                        % 
% 따라서 인공 신경망을 설계하기 위해서는 길이 30의 신호가 들어왔을 때,                                                 %
% 이게 8명의 사람 중 누구에게 해당하는 신호인지 확인할 수 있는 분류기를 설계해야 합니다                                 %
% 이 예제에서는 500번 받은 신호 중 350번을 네트워크를 학습시키는 데 이용, 150번을 네트워크 성능을 검증하는 데 이용합니다. %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 데이터 로드
load('Input3.mat'); % 총 30 x 4000의 데이터 -> 사람 8명을 맞고 반사된 레이더 신호 데이터, 각 레이더 신호 데이터의 길이가 30
                    % 사람 1에 대한 레이더 신호 : 30 (신호의 길이) x 500 (측정 횟수)
                    % 사람 2에 대한 레이더 신호 : 30 (신호의 길이) x 500 (측정 횟수)
                    %                         ...
                    % 사람 8에 대한 레이더 신호 : 30 (신호의 길이) x 500 (측정 횟수)

load('output.mat'); % 총 8 x 4000의 데이터 -> 분류를 위한 one-hot encoding
                    % 사람 1에 대한 레이더 신호 : 8 (1행만 값이 1) x 500 (측정 횟수)
                    % 사람 2에 대한 레이더 신호 : 8 (2행만 값이 1) x 500 (측정 횟수)
                    %                         ...
                    % 사람 8에 대한 레이더 신호 : 8 (8행만 값이 1) x 500 (측정 횟수)

%% 이 코드의 흐름
% 1. 데이터 분류(test, train set)
% 2. NN set
%  - node 개수 설정
%  - hidden layer 개수 설정, 사이즈 설정
%  - NN 설정
%    > parrernnet() function을 이용하여 구현
%    > 디폴트로 cost 구하는 함수로 crossentropy 설정
%    > 또한 minimizing cost 함수로 Gradient Descent 설정
% 
%  - Epoch설정
%  - Parameter 설정
%    > 데이터 분류시 얻은 train set 350개를 가지고
%    > 50%를 train 시키고
%    > 25%를 검증시키고
%    > 25%를 테스트한다.
% 
%  - overfitting을 막기위해 정규화 과정을 한다.
%  - layer들의 종류를 설정한다.(sigmoid -> logsig, ReLu -> poslin, 마지막은 softmax)
%% 학습을 위한 데이터(500번 측정 중에 350번, 70%)와 테스트를 위한 데이터(500번 측정 중에 150번, 30%) 분리
% 학습과 테스트를 나눠줘야 overfitting을 방지할 수 있다.
% Total_data를 500개씩 끊어서 그중 앞의 350개를 Train에 넣고 150개를 Test에 넣는다.
% 결과적으로 Train에서 350번째, 700번째, 1050번째...는
% Total_data에서 350번째, (501+350)번째, (1001+350)번째... 값과 같다.

X_Train = [Total_input(:, 1:350), Total_input(:, 1 + 500*1 : 350 + 500*1), Total_input(:, 1 + 500*2:350 + 500*2), Total_input(:, 1 + 500*3:350 + 500*3), Total_input(:, 1 + 500*4:350 + 500*4), Total_input(:, 1 + 500*5:350 + 500*5), Total_input(:, 1 + 500*6:350 + 500*6), Total_input(:, 1 + 500*7:350 + 500*7)];
Y_Train = [Total_output(:, 1:350), Total_output(:, 1 + 500*1 : 350 + 500*1), Total_output(:, 1 + 500*2:350 + 500*2), Total_output(:, 1 + 500*3:350 +500*3), Total_output(:, 1 + 500*4:350 + 500*4), Total_output(:, 1 + 500*5:350 + 500*5), Total_output(:, 1 + 500*6:350 + 500*6), Total_output(:, 1 + 500*7:350 + 500*7)];

X_Test = [Total_input(:, 351:500), Total_input(:, 351 + 500*1 : 500 + 500*1), Total_input(:, 351 + 500*2:500 + 500*2), Total_input(:, 351 + 500*3:500 +500*3), Total_input(:, 351 + 500*4:500 + 500*4), Total_input(:, 351 + 500*5:500 + 500*5), Total_input(:, 351 + 500*6:500 + 500*6), Total_input(:, 351 + 500*7:500 + 500*7)];
Y_Test = [Total_output(:, 351:500), Total_output(:, 351 + 500*1 : 500 + 500*1), Total_output(:, 351 + 500*2:500 + 500*2), Total_output(:, 351 + 500*3:500 +500*3), Total_output(:, 351 + 500*4:500 + 500*4), Total_output(:, 351 + 500*5:500 + 500*5), Total_output(:, 351 + 500*6:500 + 500*6), Total_output(:, 351 + 500*7:500 + 500*7)];

%% 뉴럴 네트워크 구성
max_trial = 1;                       % 분류를 위해 학습을 몇 번이나 시킬 것인지 결정
num_node = 20;                       % 각 Hidden layer에서 Node 개수를 몇 개로 할 것인지 결정
                                     %(node => 하나의 신경망. 즉, node가 20개인 것은 layer 하나당 20개의 Neural로 연결된 Neural Network라고 생각하면 된다. W를 20개 구한다.)
prob_accuracy = zeros(1, max_trial); % 분류 정확도를 계산

for num_hidden = 6                  % Hidden layer의 개수 설정
    for jj = 1 : max_trial
        
        hiddenLayerSize = num_node*ones(1, num_hidden); 
        % Hidden layer의 크기 결정 -> 지금 상태에서는 Hidden layer는 총 6개이고 각 layer는
        % 20개의 node로 구성(20개의 W를 구한다)
        
        % ★★ patternnet함수에  cost funtion(crossentropy)와
        % ★★ training function(Gadient Descent) 디폴트로 들어간다.
        
        net = patternnet(hiddenLayerSize);  % Neural network 구성  
        % 중단점 설정 후 net을 입력하면, 구성한 Network에 대한 정보를 볼 수 있습니다.
        % 디폴트로 training function는 'trainscg' (Gradient Desent임, 역전파 알고리즘 중 하나), 
        % cost function은 'crossentropy'로 되어 있습니다.
        % 참고 : https://kr.mathworks.com/help/deeplearning/ref/patternnet.html
       
        net.trainParam.epochs = 1000;       % 훈련할 최대 Epoch 횟수
        net.trainParam.showCommandLine = 1; % 명령 창 ON/OFF
        net.trainParam.max_fail = 2000;     % 최대 검증 실패 횟수
        net.trainParam.goal = 0;            % 성능 목표
        % 이러한 파라미터 설정에 관련해서는 정해진 닶은 없습니다.
        % 중단점 설정 후 net.trainParam을 입력하면, 네트워크 파라미터에 대한 정보를 볼 수 있습니다.
        % 참고 : https://kr.mathworks.com/help/deeplearning/ref/trainscg.html 
        
        % ★★ 왜 ned.divideFcn을 쓰는데 잘 되고 net.divideFcn을 쓰면 이후에 에러가 뜨지?
        ned.divideFcn = 'divideind';           % 트레이닝을 위한 350개의 데이터를 랜덤으로 분할하는 함수
        net.divideMode = 'sample';             % 각 데이터를 독립적인 샘플로 취급
        net.divideParam.trainRatio = 50/100;   % 학습을 위한 데이터 비율 : 350개 중에 50%
        net.divideParam.valRatio = 25/100;     % 검증을 위한 데이터 비율 : 350개 중에 25%
        net.divideParam.testRatio = 25/100;    % 테스트를 위한 데이터 비율 : 350개 중에 25%
        net.performParam.regularization = 0.2; % ★★ Overfitting을 막기 위한 정규화과정
        % 중단점 설정 후 net.divideParam을 입력하면, 데이터 구분에 대한 정보를 볼 수 있습니다.
        % 중단점 설정 후 net.performParam을 입력하면, 성능 검증 대한 정보를 볼 수 있습니다.
        % 참고 : https://kr.mathworks.com/help/deeplearning/ug/neural-network-object-properties.html
        
        net.layers{1}.transferFcn = 'logsig'; % 첫 번째 활성 함수를 log-sigmoid로 설정
        % 중단점 설정 후 net.layers.transferFcn를 입력하면, 각 층의 활성 함수를 확인할 수 있습니다. 위 코드에서 중괄호 안에 숫자를 바꾸면 각 층에서의 활성 함수 변경 가능. 
        % transfer functions : tanh = tansig, log-sigmoid = logsig, RELU positive linear transfer function : poslin
        % 마지막 layer는 softmax를 사용한다.
        
        net.performFcn = 'mse';                                   % 네트워크의 성능을 평균 제곱 오차로서 측정
        [net, tr] = train(net, X_Train, Y_Train, 'UseGPU', 'yes'); % 네트워크를 트레이닝 데이터를 가지고 학습, GPU가 있다면 사용하는 코드.
        
        y = net(X_Test);                        % 학습된 네트워크의 성능 검증을 위한 테스트 셋 입력 (맨 처음 500개 중의 트레이닝에 사용하지 않은 150개 레이더 데이터)
        performance = perform(net, Y_Test, y);  % 실제 라벨링된 값과 추정된 값의 오차 계산
        
        [c, cm] = confusion(Y_Test, y); % 테스트 셋을 기반으로 실제 라벨링된 값과 추정된 값이 얼마나 유사한 지, Confusion matrix 계산
        % 중단점 설정 후 c를 입력하면, 오분류 확률이 나옴.
        % 중단점 설정 후 cm을 입력하면, confusion matrix가 나옴. 예를 들어 1행 1열 성분은 1번 사람의 데이터가 실제 1번 사람으로 나왔는 지를 의미함.
        % 참고 : https://kr.mathworks.com/help/deeplearning/ref/confusion.html
        
        prob_accuracy(1, jj) = 1 - c; % 정확도
        
        fprintf(': %d -th, %d-th try \n', num_hidden, jj);
        % fprintf('pass 1, patternnet, performance: %f\n', 100*(1-c));
        % fprintf('num_epochs: %d, stop: %s\n', tr.num_epochs, tr.stop);
        
    end
end

mean_accuracy = mean(prob_accuracy, 2)*100 % 평균 정확도 (%)
view(net)                                  % 최종 구성된 네트워크

