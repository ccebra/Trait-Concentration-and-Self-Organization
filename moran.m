function f = moran(payout,strategy_x,strategy_y,num_individuals)

%% populate vector with half instances of each strategy and create fitnesses array
population_strategies = NaN(num_individuals,1);
for i = 1:num_individuals/2
    population_strategies(i) = strategy_x;
    population_strategies(num_individuals/2+i) = strategy_y;
end
fitnesses = zeros(num_individuals,2);
fitnesses(:,1) = 1:num_individuals;

while range(population_strategies) ~= 0
    for i = 1:num_individuals-1
        for j = i+1:num_individuals
            %% Round-robin play
            rand_1 = rand;
            rand_2 = rand;
            if rand_1 <= population_strategies(i) && rand_2 <= population_strategies(j)
                fitnesses(i,2) = fitnesses(i,2)+payout{1,1}(1);
                fitnesses(j,2) = fitnesses(j,2)+payout{1,1}(2);
            elseif rand_1 <= population_strategies(i) && rand_2 > population_strategies(j)
                fitnesses(i,2) = fitnesses(i,2)+payout{1,2}(1);
                fitnesses(j,2) = fitnesses(j,2)+payout{1,2}(2);
            elseif rand_1 > population_strategies(i) && rand_2 <= population_strategies(j)
                fitnesses(i,2) = fitnesses(i,2)+payout{2,1}(1);
                fitnesses(j,2) = fitnesses(j,2)+payout{2,1}(2);
            elseif rand_1 > population_strategies(i) && rand_2 > population_strategies(j)
                fitnesses(i,2) = fitnesses(i,2)+payout{2,2}(1);
                fitnesses(j,2) = fitnesses(j,2)+payout{2,2}(2);
            end
        end
    end
    
    %% reproduction
    zeta = rand;
    r_index = find(cumsum(fitnesses(:,2))/sum(fitnesses(:,2)) > zeta, 1,'first');%Identify reproducing individual
    
    %% death
    d_index = randperm(num_individuals,1);
    
    %% next generation
    population_strategies(d_index) = population_strategies(r_index);
    
end
f = population_strategies(1);
end
