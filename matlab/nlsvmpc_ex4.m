%% SVM with posit constraints

close all; clear; clc;

%% Load dataset
[T, y, X_test, y_test] = load_WDBC([-1 1]);


%% Non Linear SVM with soft margin - dual model + quantile-based thresholding

% define the problem
%C = 1;
l = length(y);

% Gaussian kernel
gamma = 0.01 ; % gamma = 1 / (n_features * X.var())

K = zeros(l,l);
for i = 1 : l
    for j = 1 : l
        K(i,j) = exp(-gamma*norm(T(i,:)-T(j,:))^2);
    end
end

%% Quantile-based Thresholding
K_quantile = quantile(K(:), 0.95); % 95th percentile of the kernel matrix

% Adjust C based on the quantile
C = 1 / K_quantile; % New regularization parameter

low = 0.125; 
high = 10;
lb = low * K_quantile * ones(2*l,1); % Adjusted lower bounds
ub = high * K_quantile * ones(2*l,1); % Adjusted upper bounds

X = zeros(l,l);
for i = 1 : l
    for j = 1 : l
        X(i,j) = y(i)*y(j)*K(i,j);
    end
end

Q = [X -X; -X X];
c = [-ones(l,1); ones(l,1)];

A = [-eye(l) eye(l); eye(l) -eye(l)];
b = [zeros(l,1); C*ones(l,1)]; % Updated constraint vector b using new C

Aeq = [y; -y];
beq = 0;

% solve the problem
options = optimset('Largescale','off', 'Display', 'off');
sol = quadprog(Q, c, A, b, Aeq', beq, lb, ub, [], options);

mu = sol(1:l);
eta = sol(l+1: 2*l);

writematrix([mu eta],'mueta_exp4.csv');

la = zeros(l,1);
% compute lambdas
for i = 1 : l
   la(i) = sol(i) - sol(l+i);
end

% Compute b using a support vector
ind = find((la > 1e-2) & (la < C-1e-2));
i = ind(1);
b = 1/y(i);
for j = 1 : l
    b = b - la(j)*y(j)*K(i,j);
end

%% support vectors
supp_idxs = find(la > 1e-2);
support_vectors = la(supp_idxs);

%% Evaluation
p = zeros(length(X_test),1);
for j = 1:length(X_test)
    s = 0;
    for i = 1 : l
       s = s + la(i)*y(i)*exp(-gamma*norm(T(i,:)-X_test(j,:))^2);
    end
    s = s + b;
 
    if s > 0
         p(j) = +1;
     else 
         p(j) = -1;
     end
end

testacc = sum(p == y_test)/length(X_test);
disp(['Test Accuracy: ', num2str(testacc)]);
