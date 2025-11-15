% California Housing Lasso Regression Comparison

% Read dataset
dataset = readtable('dataset/california_housing_processed.csv');

% Split data: 80% train, 20% test
cv = cvpartition(size(dataset,1),'HoldOut',0.2);
idx = cv.test;
train = dataset(~idx,:);
test  = dataset(idx,:);

X = train{:, 1:8}; % features
Y = train{:, 9};   % target
X_test = test{:, 1:8};
Y_test = test{:, 9};

% Hyperparameters
iterations = 50000; 
step_size = 0.01;
l1_penalty = 1;
tolerance = 1e-4;
agents = 8;

% ISTA (Gradient Descent)
disp("=== ISTA (Gradient Descent) ===");
lasso = LassoReg(step_size, iterations, l1_penalty, tolerance);
f1 = @() lasso.fit(X, Y, "gd");
t_gd = timeit(f1);
fprintf('Tempo: %.4f s\n', t_gd);
fprintf('Iterazioni: %d\n', lasso.iterations);
Y_predicted = lasso.predict(X_test);
r2_gd = corrcoef(Y_test, Y_predicted);
r2_gd = r2_gd(2,1)^2;
fprintf('R²: %.4f\n\n', r2_gd);
plot_predict("Lasso GD", Y_test, Y_predicted);
plot_loss(lasso, "Loss GD");

% ADMM
disp("=== ADMM ===");
lasso_admm = LassoReg(step_size, iterations, l1_penalty, tolerance);
f2 = @() lasso_admm.fit(X, Y, "admm");
t_admm = timeit(f2);
fprintf('Tempo: %.4f s\n', t_admm);
fprintf('Iterazioni: %d\n', lasso_admm.iterations);
Y_predicted = lasso_admm.predict(X_test);
r2_admm = corrcoef(Y_test, Y_predicted);
r2_admm = r2_admm(2,1)^2;
fprintf('R²: %.4f\n\n', r2_admm);
plot_predict("Lasso ADMM", Y_test, Y_predicted);
plot_loss(lasso_admm, "Convergence ADMM");

% Distributed ADMM
disp("=== Distributed ADMM ===");
lasso_dist = LassoReg(step_size, iterations, l1_penalty, tolerance);
f3 = @() lasso_dist.fit(X, Y, "dist", agents);
t_dist = timeit(f3);
fprintf('Tempo: %.4f s\n', t_dist);
fprintf('Iterazioni: %d\n', lasso_dist.iterations);
Y_predicted = lasso_dist.predict(X_test);
r2_dist = corrcoef(Y_test, Y_predicted);
r2_dist = r2_dist(2,1)^2;
fprintf('R²: %.4f\n\n', r2_dist);
plot_predict("Lasso Distributed-ADMM", Y_test, Y_predicted);
plot_loss(lasso_dist, "Convergence Distributed-ADMM");

% Tabella riassuntiva
fprintf('\n=== COMPARAZIONE ALGORITMI ===\n');
fprintf('Algoritmo\t\tR²\t\tTempo(s)\tIterazioni\n');
fprintf('ISTA\t\t\t%.4f\t\t%.4f\t\t%d\n', r2_gd, t_gd, lasso.iterations);
fprintf('ADMM\t\t\t%.4f\t\t%.4f\t\t%d\n', r2_admm, t_admm, lasso_admm.iterations);
fprintf('ADMM-Dist\t\t%.4f\t\t%.4f\t\t%d\n', r2_dist, t_dist, lasso_dist.iterations);

function plot_predict(label, Y_test, Y_predicted)
    figure
    hold on
    title(label);
    scatter(Y_test, Y_predicted, 10, 'filled', 'MarkerFaceAlpha', 0.6)
    plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--', 'LineWidth', 2)
    xlabel('Actual value')
    ylabel('Predicted value')
    grid on
    hold off
end

function plot_loss(lasso, label)
    if label == "Loss GD"
        figure
        hold on
        title(label);
        plot(1:lasso.iterations, lasso.J(1:lasso.iterations), 'LineWidth', 2)
        plot([1, lasso.iterations], [lasso.tolerance, lasso.tolerance], 'r--', 'LineWidth', 1.5)
        xlabel('Iterations')
        ylabel('Weight Change Norm')
        legend('Convergence', 'Tolerance', 'Location', 'best')
        grid on
        hold off
    else
        figure
        subplot(2,1,1)
        title(label);
        hold on
        if lasso.iterations > 0
            plot(1:lasso.iterations, lasso.J(1,1:lasso.iterations), 'LineWidth', 2);
            plot(1:lasso.iterations, lasso.J(3,1:lasso.iterations), 'r--', 'LineWidth', 1.5);
        end
        xlabel('Iterations')
        ylabel('Primary residual')
        legend('r\_norm', 'tolerance', 'Location', 'best')
        grid on
        hold off
        
        subplot(2,1,2)
        hold on
        if lasso.iterations > 0
            plot(1:lasso.iterations, lasso.J(2,1:lasso.iterations), 'LineWidth', 2);
            plot(1:lasso.iterations, lasso.J(4,1:lasso.iterations), 'r--', 'LineWidth', 1.5);
        end
        xlabel('Iterations')
        ylabel('Dual residual')
        legend('s\_norm', 'tolerance', 'Location', 'best')
        grid on
        hold off
    end
end
