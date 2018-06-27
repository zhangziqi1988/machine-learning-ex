%% Clear and Close Figures
%��������ʣ�ȡ���ݼ�ǰ40��������Ϊѵ��������17����Ϊ���Լ����ֱ�����ݶ��½������淽�����ַ�ʽ��ƽ��������

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(1:40, 1:2);
y = data(1:40, 3);
m = length(y);


[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);


T = data(41:47, 1:2);
t = data(41:47, 3);
[T mu sigma] = featureNormalize(T);
T = [ones(length(T),1), T];

t1 = T*theta;
error = abs(t1 - t)./t;
avgError = sum(error)/length(T);
%fprintf('error is %f: \n',);
fprintf('gradientDescent formal avgError is %f: \n',avgError);
theta = normalEqn(X, y);
t1 = T*theta;
error = abs(t1 - t)./t;
avgError = sum(error)/length(T);
fprintf('normalEqn formal avgError is %f: \n',avgError);




