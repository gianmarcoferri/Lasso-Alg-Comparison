classdef LassoReg < handle
    properties
        step_size        % passo di aggiornamento (ISTA) / rho (ADMM)
        max_iterations   % numero massimo di iterazioni
        iterations       % numero di iterazioni effettive fatte
        l1_penalty       % parametro di regolarizzazione L1 (lambda)
        tolerance        % tolleranza per la convergenza
        m                % numero campioni
        n                % numero feature
        W                % vettore dei pesi (1 x n)
        X                % dataset input
        Y                % target
        J                % memorizza residui / norme per i grafici
    end
    
    methods
        % COSTRUTTORE
        function obj = LassoReg(step_size, max_iterations, l1_penalty, tolerance)
            obj.step_size = step_size;
            obj.max_iterations = max_iterations;
            obj.l1_penalty = l1_penalty;
            obj.tolerance = tolerance;
        end
        
        % FIT: selezione dell’algoritmo (ISTA, ADMM, ADMM distribuito)
        function fit(obj, X, Y, algo, agents)
            obj.m = size(X, 1); % numero campioni
            obj.n = size(X, 2); % numero feature
            
            obj.W = zeros(1, obj.n); % inizializzazione w (vettore riga)
            obj.X = X;
            obj.Y = Y;
            
            % Selezione algoritmo
            if algo == "ista"
                obj.ista();
            elseif algo == "admm"
                obj.admm();
            else % "dist"
                obj.distributed_admm(agents);
            end
        end

        % ISTA
        function ista(obj)
            for i = 1:obj.max_iterations

                % Predizione
                Y_predict = obj.predict(obj.X);
                
                % Gradiente del loss MSE
                gradient = -obj.X' * (obj.Y - Y_predict) / obj.m;
                
                % Passo di gradiente
                new_W_unthresholded = obj.W - obj.step_size * gradient';

                % Soft-thresholding
                new_W = obj.soft_threshold(new_W_unthresholded, obj.step_size * obj.l1_penalty);
                
                % Controllo convergenza
                if norm(new_W - obj.W) < obj.tolerance
                    break
                end   
                
                obj.J(i) = norm(new_W - obj.W);  % Salvataggio progresso
                obj.W = new_W;
                obj.iterations = i;
            end
        end
        
        % ADMM
        function admm(obj)
            rho = obj.step_size; % parametro rho ADMM (controlla la forza del vincolo di consenso)
            z = zeros(1, obj.n); % variabile ausiliaria
            u = zeros(1, obj.n); % moltiplicatori di Lagrange
            I = eye(obj.n); % matrice identità
            
            abs_tol = obj.tolerance; % tolleranza assoluta
            rel_tol = abs_tol * 100; % tolleranza relativa
            
            for i = 1:obj.max_iterations
                last_z = z;
                
                % Aggiornamento w (risoluzione sistema lineare)
                obj.W = (obj.X' * obj.X + rho * I) \ (obj.X' * obj.Y + rho * (z - u)');
                obj.W = obj.W'; % conversione in vettore riga
                
                % Aggiornamento z con soft-thresholding
                z = obj.soft_threshold(obj.W + u, obj.l1_penalty / rho);

                % Aggiornamento dei moltiplicatori duali
                u = u + obj.W - z;
                
                % Norme dei residui per la convergenza
                r_norm  = norm(obj.W - z);
                s_norm  = norm(-rho * (z - last_z));

                % Soglie di convergenza
                tol_prim = sqrt(obj.n) * abs_tol + rel_tol * max(norm(obj.W), norm(z));
                tol_dual = sqrt(obj.n) * abs_tol + rel_tol * norm(rho * u);
                
                % Salvataggio andamento
                obj.iterations = i;
                obj.J(1,i) = r_norm;
                obj.J(2,i) = s_norm;
                obj.J(3,i) = tol_prim;
                obj.J(4,i) = tol_dual;
                
                % Condizione di arresto
                if r_norm < tol_prim && s_norm < tol_dual
                    break
                end
            end
        end
        
        % ADMM DISTRIBUITO
        function distributed_admm(obj, agents)
            rho = obj.step_size; % parametro rho ADMM (controlla la forza del vincolo di consenso)
            z = zeros(1, obj.n); % variabile consenso
            I = eye(obj.n);
            
            abs_tol = obj.tolerance; % tolleranza assoluta
            rel_tol = 1; % tolleranza relativa
            
            % Divisione dei dati sugli agenti
            [r, ~] = size(obj.X); % numero totale di campioni
            samples_per_agent = floor(r/agents); % numero di campioni assegnati a ciascun agente (gli scarti vanno all'ultimo)
            
            splitted_X = cell(agents, 1); % celle per gli X_j locali
            splitted_Y = cell(agents, 1); % celle per gli Y_j locali
            
            indices = randperm(r); % shuffle casuale
            
            for j = 1:agents
                start_idx = (j-1)*samples_per_agent + 1;
                if j == agents
                    end_idx = r; % l’ultimo agente prende anche gli scarti
                else
                    end_idx = j * samples_per_agent;
                end
                
                agent_indices = indices(start_idx:end_idx); % assegno al j-esimo agente i campioni selezionati
                splitted_X{j} = obj.X(agent_indices, :);  % dataset locale X_j
                splitted_Y{j} = obj.Y(agent_indices); % dataset locale Y_j
            end
            
            % Ogni agente ha un proprio w_j e un duale u_j
            W_agents = zeros(agents, obj.n);
            u_agents = zeros(agents, obj.n);
            
            % Loop ADMM
            for i = 1:obj.max_iterations
                last_z = z; % salvataggio di z per calcolare il residuo duale
                
                % Aggiornamento locale degli agenti
                for j = 1:agents
                    X_j = splitted_X{j};
                    Y_j = splitted_Y{j};

                    w_temp = (X_j' * X_j + rho * I) \ (X_j' * Y_j + rho * (z - u_agents(j,:))');
                    W_agents(j,:) = w_temp';
                end
                
                % Aggiornamento del consenso z
                z = obj.soft_threshold(mean(W_agents + u_agents, 1), obj.l1_penalty / (rho * agents));
                
                % Aggiornamento dei moltiplicatori duali
                for j = 1:agents
                    u_agents(j,:) = u_agents(j,:) + (W_agents(j,:) - z);
                end
                
                % Norme residui
                r_norm = norm(mean(W_agents, 1) - z);
                s_norm = norm(-rho * (z - last_z));
                tol_prim = sqrt(obj.n) * abs_tol + rel_tol * max(norm(mean(W_agents, 1)), norm(z));
                tol_dual = sqrt(obj.n) * abs_tol + rel_tol * norm(rho * mean(u_agents, 1));
                
                % Salvataggio andamento
                obj.iterations = i;
                obj.J(1,i) = r_norm;
                obj.J(2,i) = s_norm;
                obj.J(3,i) = tol_prim;
                obj.J(4,i) = tol_dual;
                
                % Condizione di arresto
                if r_norm < tol_prim && s_norm < tol_dual
                    break
                end
            end
            
            % Pesi finali = media dei pesi locali (consenso)
            obj.W = mean(W_agents, 1);
        end
                
        % Predizione
        function Y_predict = predict(obj, X)
            Y_predict = X * obj.W'; % w è vettore riga -> trasposto
        end
        
        % Funzione di loss (non usata nell'algoritmo, solo di supporto)
        function loss = loss_function(obj, Y, Y_predict, W)
            loss = (1/2 * sum((Y - Y_predict).^2) + obj.l1_penalty * norm(W, 1));
        end
      
        % Operatore di soft-thresholding
        function soft_term = soft_threshold(~, w, th)
            soft_term = sign(w) .* max(0, abs(w) - th);
        end
    end
end
