function [rho,uncertainty] = estimate_rho_Gauss_2(f,centroid,R,tol,epoch_bounds,m)

%% create edge to endpoint mapping
edge_to_endpoints = NaN(m*(m - 1)/2,2);
k = 0;
for i = 1:m-1
    for j = i+1:m
        k = k+1;
        edge_to_endpoints(k,:) = [i,j];
    end
end

%% preallocate
rhos = NaN(epoch_bounds(1),1);

%% loop over epochs
epoch = 0;
stop = 0;
while stop == 0
    %% update epoch count
    epoch = epoch + 1;
    
    %% sample competitors
    X = repmat(centroid,m,1) + randn(m,length(centroid))*R';
    
    %% Calculate performance using the performance function
    xs = X(edge_to_endpoints(:,1),:);
    ys = X(edge_to_endpoints(:,2),:);
    fs = f(xs,ys);
    competition = sparse(edge_to_endpoints(:,1), edge_to_endpoints(:,2),fs,m,m) - ...
        sparse(edge_to_endpoints(:,2), edge_to_endpoints(:,1),fs,m,m);
    competition = full(competition);
    
    %% performing the HHD for a complete graph
    ratings = (1/m)*sum(competition,2);
        
        
    %% another way to do this without G
    F_t = ratings - ratings';
    F_c = competition - F_t;
    
    %% sizes
    Trans = norm(F_t,'fro')/sqrt(2);
    Intrans = norm(F_c,'fro')/sqrt(2);
    
    %% compute rho
    E = m*(m-1)/2;
    L = E - (m-1);
    
    rhos(epoch) = (1 - (E/L)*(Intrans^2/(Trans^2 + Intrans^2)))/2;
    
    
    %% check stopping
    if epoch > epoch_bounds(1) % ran enough, check if convergence
        if std(rhos(1:epoch))/sqrt(epoch - 1) <= tol % stop if uncertainty in average rho is small enough
            stop = 1;
        end
    end
    
    if epoch == epoch_bounds(2) % exceed max epochs
        stop = 1;
    end
end

%% output
rho = mean(rhos);
uncertainty = std(rhos);

end