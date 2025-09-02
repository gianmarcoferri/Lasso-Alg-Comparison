% California Housing Lasso Regression Comparison

% Read dataset
dataset = readtable('california_housing_processed.csv');

% Normalize features (optional but typical for regression)
dataset{:, 1:8} = normalize(dataset{:, 1:8}, "range"); % Columns 1-8 are features

% Split data: 80% train, 20% test
cv = cvpartition(size(dataset,1),'HoldOut',0.2);
idx = cv.test;
train = dataset(~idx,:);
test  = dataset(idx,:);

X = train{:, 1:8}; % features: longitude ... median_income
Y = train{:, 9};   % target: median_house_value
X_test = test{:, 1:8};
Y_test = test{:, 9};

% Hyperparameters
iterations = 50000; 
step_size = 0.01;
l1_penalty = 1;
tolerance = 1e-4;
agents = 8;

% ISTA (Gradient Descent)
disp("GD");
lasso = LassoReg(step_size, iterations, l1_penalty, tolerance);
f1 = @() lasso.fit(X, Y, "gd");
t_gd = timeit(f1);
disp(t_gd);
disp(lasso.iterations);
Y_predicted = lasso.predict(X_test);
disp(corrcoef(Y_test, Y_predicted).^2);     % R2
plot_predict("Lasso GD", Y_test, Y_predicted);
plot_loss(lasso, "Loss GD");

% ADMM
disp("ADMM");
lasso_admm = LassoReg(step_size, iterations, l1_penalty, tolerance);
f2 = @() lasso_admm.fit(X, Y, "admm");
t_admm = timeit(f2);
disp(t_admm);
disp(lasso_admm.iterations);
Y_predicted = lasso_admm.predict(X_test);
disp(corrcoef(Y_test, Y_predicted).^2);     % R2
plot_predict("Lasso ADMM", Y_test, Y_predicted);
plot_loss(lasso_admm, "Convergence ADMM");

% Distributed ADMM
disp("Distributed ADMM");
lasso_dist = LassoReg(step_size, iterations, l1_penalty, tolerance);
f3 = @() lasso_dist.fit(X, Y, "dist", agents);
t_dist = timeit(f3);
disp(t_dist/agents);
disp(lasso_dist.iterations);
Y_predicted = lasso_dist.predict(X_test);
disp(corrcoef(Y_test, Y_predicted).^2);     % R2
plot_predict("Lasso Distributed-ADMM", Y_test, Y_predicted);
plot_loss(lasso_dist,  "Convergence Distributed-ADMM");

function plot_predict(label, Y_test, Y_predicted)
    figure
    hold on
    title(label);
    scatter(Y_test,Y_predicted)
    plot(Y_test,Y_test)
    xlabel('Actual value')
    ylabel('Predicted value')
    hold off
end

function plot_loss(lasso, label)
    if label == "Loss GD"
        figure
        hold on
        title(label);
        plot(lasso.J)
        plot(lasso.tolerance+zeros(lasso.iterations,1), "--");
        xlabel('Iterations')
        ylabel('Loss')
        hold off
    else
        figure
        subplot(2,1,1)
        title(label);
        hold on
        plot(lasso.J(1,:));
        plot(lasso.J(3,:), "--");
        xlabel('Iterations')
        ylabel('Primary residual')
        hold off
        subplot(2,1,2)
        hold on
        plot(lasso.J(2,:));
        plot(lasso.J(4,:), "--");
        xlabel('Iterations')
        ylabel('Dual residual')
        hold off
    end
end
