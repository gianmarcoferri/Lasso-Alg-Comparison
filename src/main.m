% CONFRONTO TRA ISTA, ADMM E ADMM DISTRIBUITO PER IL LASSO
% Dataset: California Housing Prices (preprocessato)

% LETTURA DATASET
dataset = readtable('dataset/california_housing_processed.csv');

% TRAIN/TEST SPLIT
cv = cvpartition(size(dataset,1),'HoldOut',0.2);  % 80% train, 20% test
idx = cv.test;
train = dataset(~idx,:);
test  = dataset(idx,:);

% Feature (8 colonne) e target (1 colonna)
X = train{:, 1:8};
Y = train{:, 9};
X_test = test{:, 1:8};
Y_test = test{:, 9};

% IPERPARAMETRI
iterations = 50000;    % numero massimo di iterazioni 
step_size = 0.01;      % passo ISTA / rho ADMM
l1_penalty = 1;        % lambda del Lasso
tolerance = 1e-4;      % criterio di arresto
agents = 8;            % numero di agenti per Distributed ADMM


% ===============================================================
% ISTA
disp("=== ISTA ===");

% Creazione modello
lasso = LassoReg(step_size, iterations, l1_penalty, tolerance);

% Misurazione tempo usando una function handle
f1 = @() lasso.fit(X, Y, "ista");
t_ista = timeit(f1);

% Stampa del tempo impiegato e del numero di iterazioni effettivamente svolte
fprintf('Tempo: %.4f s\n', t_ista);
fprintf('Iterazioni: %d\n', lasso.iterations);

% Predizioni sul test set
Y_predicted = lasso.predict(X_test);

% Calcolo R²
r2_ista = corrcoef(Y_test, Y_predicted);
r2_ista = r2_ista(2,1)^2;
fprintf('R²: %.4f\n\n', r2_ista);

% Plot di predizione e convergenza
plot_predict("Lasso ISTA", Y_test, Y_predicted);
plot_loss(lasso, "Loss ISTA");


% ===============================================================
% ADMM
disp("=== ADMM ===");

% Creazione modello
lasso_admm = LassoReg(step_size, iterations, l1_penalty, tolerance);

% Misurazione tempo usando una function handle
f2 = @() lasso_admm.fit(X, Y, "admm");
t_admm = timeit(f2); 

% Stampa del tempo impiegato e del numero di iterazioni effettivamente svolte
fprintf('Tempo: %.4f s\n', t_admm);
fprintf('Iterazioni: %d\n', lasso_admm.iterations);

% Predizioni sul test set
Y_predicted = lasso_admm.predict(X_test);

% Calcolo R²
r2_admm = corrcoef(Y_test, Y_predicted);
r2_admm = r2_admm(2,1)^2;
fprintf('R²: %.4f\n\n', r2_admm);

% Plot di predizione e convergenza
plot_predict("Lasso ADMM", Y_test, Y_predicted);
plot_loss(lasso_admm, "Convergence ADMM");


% ===============================================================
% DISTRIBUTED ADMM
disp("=== Distributed ADMM ===");

% Creazione modello
lasso_dist = LassoReg(step_size, iterations, l1_penalty, tolerance);

% Misurazione tempo usando una function handle
f3 = @() lasso_dist.fit(X, Y, "dist", agents); % Nota: richiede parametro "agents"
t_dist = timeit(f3);

% Stampa del tempo impiegato e del numero di iterazioni effettivamente svolte
fprintf('Tempo: %.4f s\n', t_dist);
fprintf('Iterazioni: %d\n', lasso_dist.iterations);

% Predizioni sul test set
Y_predicted = lasso_dist.predict(X_test);

% Calcolo R²
r2_dist = corrcoef(Y_test, Y_predicted);
r2_dist = r2_dist(2,1)^2;
fprintf('R²: %.4f\n\n', r2_dist);

% Plot di predizione e convergenza
plot_predict("Lasso Distributed-ADMM", Y_test, Y_predicted);
plot_loss(lasso_dist, "Convergence Distributed-ADMM");


% ===============================================================  
% TABELLA FINALE DI COMPARAZIONE
fprintf('\n=== COMPARAZIONE ALGORITMI ===\n');
fprintf('Algoritmo\t\tR²\t\tTempo(s)\tIterazioni\n');
fprintf('ISTA\t\t\t%.4f\t\t%.4f\t\t%d\n', r2_ista, t_ista, lasso.iterations);
fprintf('ADMM\t\t\t%.4f\t\t%.4f\t\t%d\n', r2_admm, t_admm, lasso_admm.iterations);
fprintf('ADMM-Dist\t\t%.4f\t\t%.4f\t\t%d\n', r2_dist, t_dist, lasso_dist.iterations);


% ===============================================================        
% FUNZIONE DI PLOT PREVISIONI
function plot_predict(label, Y_test, Y_predicted)
    figure
    hold on
    title(label);

    % Scatter delle predizioni
    scatter(Y_test, Y_predicted, 10, 'filled', 'MarkerFaceAlpha', 0.6)

    % Linea ideale y = x
    plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--', 'LineWidth', 2)

    xlabel('Actual value')
    ylabel('Predicted value')
    grid on
    hold off
end


% ===============================================================                   
% FUNZIONE DI PLOT PER LA CONVERGENZA
function plot_loss(lasso, label)

    % ISTA ha una sola curva (norma dei pesi)
    if label == "Loss ISTA"

        figure
        hold on
        title(label);

        % Norma del cambiamento dei pesi
        plot(1:lasso.iterations, lasso.J(1:lasso.iterations), 'LineWidth', 2)

        % Linea della tolleranza
        plot([1, lasso.iterations], [lasso.tolerance, lasso.tolerance], 'r--', 'LineWidth', 1.5)

        xlabel('Iterations')
        ylabel('Weight Change Norm')
        legend('Convergence', 'Tolerance', 'Location', 'best')
        grid on
        hold off

    % ADMM e Distributed ADMM hanno r_norm e s_norm
    else
        figure

        % Residuo primale
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
        
        % Residuo duale
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
