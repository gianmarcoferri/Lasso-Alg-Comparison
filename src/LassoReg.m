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
            
            obj.W = zeros(1, obj.n);
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
            z = 0;
            u = 0;
            I = eye(obj.n,obj.n);
            
            abs_tol = obj.tolerance;
            rel_tol = abs_tol * 100; 
            
            for i = 1:obj.max_iterations
                last_z = z;
                
                obj.W = (obj.X'*obj.X + rho*I)^(-1) * (obj.X'*obj.Y + rho*(z-u));
                z = obj.soft_threshold(obj.W + u, obj.l1_penalty/rho);
                u = u + obj.W - z;
                
                r_norm  = norm(obj.W - z);
                s_norm  = norm(-rho*(z - last_z));
                tol_prim = sqrt(obj.n)*abs_tol + rel_tol*max(norm(obj.W), norm(-z));
                tol_dual= sqrt(obj.n)*abs_tol + rel_tol*norm(rho*u);
                
                obj.iterations = i;
                obj.J(1,i) = r_norm;
                obj.J(2,i) = s_norm;
                obj.J(3,i) = tol_prim;
                obj.J(4,i) = tol_dual;
                
                if r_norm < tol_prim && s_norm < tol_dual
                    break
                end
            end
            obj.W = obj.W';
        end
        
        function distributed_admm(obj, agents)
            rho = obj.step_size;
            z = 0;
            I = eye(obj.n,obj.n);
            
            abs_tol = obj.tolerance;
            rel_tol = abs_tol * 100; 
            converged = 0;
            
            [r,c] = size(obj.X);
            splitted_X   = permute(reshape(obj.X',[c,r/agents,agents]),[2,1,3]);
            splitted_Y = reshape(obj.Y,[r/agents,agents]);
            obj.W = zeros([agents c]);
            u = zeros([agents c]);
                        
            for i = 1:obj.max_iterations
                last_z = z;
                for j = 1:agents                    
                    obj.W(j,:) = (permute(splitted_X(:,:,j), [2,1,3])*splitted_X(:,:,j) + rho*I)^(-1) * ...
                                 (permute(splitted_X(:,:,j), [2,1,3])*splitted_Y(:,j) + rho*(z-u(j,:))');
                    z = obj.soft_threshold(mean(obj.W) + mean(u), obj.l1_penalty/rho);
                    u(j,:) = u(j,:) + (obj.W(j,:) - z);
                    
                    r_norm  = norm(mean(obj.W) - z);
                    s_norm  = norm(-rho*(z - last_z));
                    tol_prim = sqrt(obj.n)*abs_tol + rel_tol*max(norm(mean(obj.W)), norm(-z));
                    tol_dual= sqrt(obj.n)*abs_tol + rel_tol*norm(rho*mean(u));
                    
                    obj.J(1,i,j) = r_norm;
                    obj.J(2,i,j) = s_norm;
                    obj.J(3,i,j) = tol_prim;
                    obj.J(4,i,j) = tol_dual;
                    
                    if r_norm < tol_prim && s_norm < tol_dual
                        converged = 1;
                        obj.J = obj.J(:,:,j);
                        break
                    end
                end
                obj.iterations = i;
                if converged
                    break
                end
            end
            obj.W = mean(obj.W);
        end
        
        function gradient_descent(obj)
            for i = 1:obj.max_iterations
                Y_predict = obj.predict(obj.X);
                
                soft_term = obj.soft_threshold(obj.W, obj.l1_penalty);
                dW = (-2 * obj.X' * (obj.Y - Y_predict) + soft_term') / obj.m;
                new_W = obj.W - obj.step_size * dW';
                
                if mean(abs(new_W - obj.W)) < obj.tolerance
                    break
                end   
                
                obj.J(i) = mean(abs(new_W - obj.W));
                obj.W = new_W;
                obj.iterations = i;
            end
        end
        
        function Y_predict = predict(obj, X)  
            Y_predict = X * obj.W';
        end
        
        function loss = loss_function(obj, Y, Y_predict, W)
            loss = ( 1/2*sum((Y - Y_predict).^2) + obj.l1_penalty*norm(W,1) );
        end
      
        function soft_term = soft_threshold(~, w, th)
            soft_term = max(0, w-th) - max(0, -w-th);
        end
    end
end
