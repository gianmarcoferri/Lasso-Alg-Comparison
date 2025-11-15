classdef LassoReg < handle
    properties
        step_size
        max_iterations
        iterations
        l1_penalty
        tolerance
        m
        n
        W
        X
        Y
        J
    end
    
    methods
        function obj = LassoReg(step_size, max_iterations, l1_penalty, tolerance)
            obj.step_size = step_size;
            obj.max_iterations = max_iterations;
            obj.l1_penalty = l1_penalty;
            obj.tolerance = tolerance;
        end
        
        function fit(obj, X, Y, algo, agents)
            obj.m = size(X, 1); % number of samples
            obj.n = size(X, 2); % number of features
            
            obj.W = zeros(1, obj.n); % vettore riga per tutti gli algoritmi
            obj.X = X;
            obj.Y = Y;
           
            if algo == "gd"
                obj.gradient_descent();
            elseif algo == "admm"
                obj.admm();
            else % "dist"
                obj.distributed_admm(agents);
            end
        end
        
        function admm(obj)
            rho = obj.step_size;
            z = zeros(1, obj.n); % vettore riga
            u = zeros(1, obj.n); % vettore riga
            I = eye(obj.n, obj.n);
            
            abs_tol = obj.tolerance;
            rel_tol = abs_tol * 100; 
            
            for i = 1:obj.max_iterations
                last_z = z;
                
                % Soluzione sistema lineare
                obj.W = (obj.X' * obj.X + rho * I) \ (obj.X' * obj.Y + rho * (z - u)');
                obj.W = obj.W'; % converti a vettore riga
                
                z = obj.soft_threshold(obj.W + u, obj.l1_penalty / rho);
                u = u + obj.W - z;
                
                r_norm  = norm(obj.W - z);
                s_norm  = norm(-rho * (z - last_z));
                tol_prim = sqrt(obj.n) * abs_tol + rel_tol * max(norm(obj.W), norm(z));
                tol_dual = sqrt(obj.n) * abs_tol + rel_tol * norm(rho * u);
                
                obj.iterations = i;
                obj.J(1,i) = r_norm;
                obj.J(2,i) = s_norm;
                obj.J(3,i) = tol_prim;
                obj.J(4,i) = tol_dual;
                
                if r_norm < tol_prim && s_norm < tol_dual
                    break
                end
            end
        end
        
        function distributed_admm(obj, agents)
            rho = obj.step_size;
            z = zeros(1, obj.n); % vettore riga
            I = eye(obj.n, obj.n);
            
            abs_tol = obj.tolerance;
            rel_tol = 1; 
            
            % Split dati
            [r, ~] = size(obj.X);
            samples_per_agent = floor(r/agents);
            
            splitted_X = cell(agents, 1);
            splitted_Y = cell(agents, 1);
            
            indices = randperm(r);
            for j = 1:agents
                start_idx = (j-1)*samples_per_agent + 1;
                if j == agents
                    end_idx = r;
                else
                    end_idx = j*samples_per_agent;
                end
                agent_indices = indices(start_idx:end_idx);
                splitted_X{j} = obj.X(agent_indices, :);
                splitted_Y{j} = obj.Y(agent_indices);
            end
            
            W_agents = zeros(agents, obj.n); % matrice agents x n
            u_agents = zeros(agents, obj.n); % matrice agents x n
            
            for i = 1:obj.max_iterations
                last_z = z;
                
                % Aggiornamento agenti
                for j = 1:agents
                    X_j = splitted_X{j};
                    Y_j = splitted_Y{j};
                    w_temp = (X_j' * X_j + rho * I) \ (X_j' * Y_j + rho * (z - u_agents(j,:))');
                    W_agents(j,:) = w_temp';
                end
                
                % Aggiornamento consenso
                z = obj.soft_threshold(mean(W_agents + u_agents, 1), obj.l1_penalty / (rho * agents));
                
                % Aggiornamento duale
                for j = 1:agents
                    u_agents(j,:) = u_agents(j,:) + (W_agents(j,:) - z);
                end
                
                % Convergenza
                r_norm = norm(mean(W_agents, 1) - z);
                s_norm = norm(-rho * (z - last_z));
                tol_prim = sqrt(obj.n) * abs_tol + rel_tol * max(norm(mean(W_agents, 1)), norm(z));
                tol_dual = sqrt(obj.n) * abs_tol + rel_tol * norm(rho * mean(u_agents, 1));
                
                obj.iterations = i;
                obj.J(1,i) = r_norm;
                obj.J(2,i) = s_norm;
                obj.J(3,i) = tol_prim;
                obj.J(4,i) = tol_dual;
                
                if r_norm < tol_prim && s_norm < tol_dual
                    break
                end
            end
            obj.W = mean(W_agents, 1); % vettore riga
        end
        
        function gradient_descent(obj)
            % ISTA implementation
            for i = 1:obj.max_iterations
                Y_predict = obj.predict(obj.X);
                
                % Calcolo gradiente
                gradient = -obj.X' * (obj.Y - Y_predict) / obj.m;
                
                % ISTA update con soft-thresholding
                new_W_unthresholded = obj.W - obj.step_size * gradient';
                new_W = obj.soft_threshold(new_W_unthresholded, obj.step_size * obj.l1_penalty);
                
                % Check convergenza
                if norm(new_W - obj.W) < obj.tolerance
                    break
                end   
                
                obj.J(i) = norm(new_W - obj.W);
                obj.W = new_W;
                obj.iterations = i;
            end
        end
        
        function Y_predict = predict(obj, X)  
            Y_predict = X * obj.W'; % W è sempre vettore riga
        end
        
        function loss = loss_function(obj, Y, Y_predict, W)
            loss = (1/2 * sum((Y - Y_predict).^2) + obj.l1_penalty * norm(W, 1));
        end
      
        function soft_term = soft_threshold(~, w, th)
            soft_term = sign(w) .* max(0, abs(w) - th);
        end
    end
end
