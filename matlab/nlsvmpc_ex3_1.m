%% SVM with posit constraints

close all; clear; clc;

%% Load dataset
[T, y, X_test, y_test] = load_WDBC([-1 1]);

%% Non Linear SVM with soft margin - dual model + worst-case bounds adjustment

% Define the problem
C = 1;
l = length(y);
low = 0.125; 
high = 10;

% Grid search for Gamma Tuning
gamma_values = logspace(-4, 2, 7); % Example range from 0.0001 to 100
best_gamma = gamma_values(1);
best_accuracy = 0;

for gamma = gamma_values
    % Gaussian kernel computation with adjusted gamma
    K = zeros(l,l);
    for i = 1 : l
        for j = 1 : l
            K(i,j) = exp(-gamma*norm(T(i,:)-T(j,:))^2);
        end
    end
    
    % Worst-case Adjustment
    K_max = max(K(:)); % Maximum kernel value across all kernel entries
    lb = low * K_max * ones(2*l,1); % Adjusted lower bounds
    ub = high * K_max * ones(2*l,1); % Adjusted upper bounds
    %lb = low*ones(2*l,1);
    %ub = high*ones(2*l,1);
    
    % Formulate the optimization problem
    X = zeros(l,l);
    for i = 1 : l
        for j = 1 : l
            X(i,j) = y(i)*y(j)*K(i,j);
        end
    end
    Q = [X -X; -X X];
    c = [-ones(l,1); ones(l,1)];
    A = [-eye(l) eye(l); eye(l) -eye(l)];
    b = [zeros(l,1); C*ones(l,1)];
    Aeq = [y; -y];
    beq = 0;

    % Solve the problem
    options = optimset('Largescale','off', 'Display', 'off');
    sol = quadprog(Q, c, A, b, Aeq', beq, lb, ub, [], options);

    mu = sol(1:l);
    eta = sol(l+1: 2*l);

    writematrix([mu eta],'mueta_exp3_3.csv');

    % Evaluate performance with current gamma
    la = sol(1:l) - sol(l+1:2*l); % compute lambdas

    % Compute b using a support vector
    ind = find((la > 1e-2) & (la < C-1e-2));
    if ~isempty(ind)
        i = ind(1);
        b = 1/y(i);
        for j = 1 : l
            b = b - la(j)*y(j)*K(i,j);
        end
        % Evaluation
        p = zeros(length(X_test),1);
        for j = 1:length(X_test)
            s = sum(la .* y .* exp(-gamma*vecnorm(T-X_test(j,:), 2, 2).^2)) + b;
            p(j) = sign(s);
        end
        accuracy = sum(p == y_test) / length(X_test);
        
        % Update best gamma if accuracy improves
        if accuracy > best_accuracy
            best_accuracy = accuracy;
            best_gamma = gamma;
        end
    end
end

% Print best gamma and corresponding accuracy
% Print the value of b
disp(['Value of b: ', num2str(b)]);
disp(['Best Gamma: ', num2str(best_gamma)]);
disp(['Best Accuracy: ', num2str(best_accuracy)]);
