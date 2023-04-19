%% Plot PD competitor distributions
load('evolution_test_moran_results_pd_comptraits.mat')

%% Clean up array
comptraits = results.competitorarray;
%for i = 1:results.parameters.num_experiments
%    for j = 1:results.parameters.num_epochs
%        if max(comptraits(:,j,i) == 0)
%            comptraits(:,j,i) = comptraits(:,j-1,i);
%        end
%    end
%end
%% Draw plots
firstcomps = squeeze(comptraits(:,1,:));
firstrow = firstcomps(:);

%% analyze for clusters      
% AIC = zeros(1,round(length(firstrow)/10));
% GMModels = cell(1,round(length(firstrow)/10));
% options = statset('MaxIter',100);
% for num_clusters = 1:(round(length(firstrow)/10))%Getting an ill-conditioned covariance error
%     GMModels{num_clusters} = fitgmdist(firstrow,num_clusters,'Options',options,'CovarianceType','full', 'RegularizationValue',results.parameters.genetic_drift/100);
%     AIC(num_clusters)= GMModels{num_clusters}.AIC;
% end
% 
% [minAIC,numComponents] = min(AIC);
% n_classes = numComponents;
% 
% BestModel = GMModels{numComponents};
% mu = BestModel.mu;
% mu(end+1:round(num_competitors/10)) = 0;
% sigma = BestModel.Sigma;
% sigma(end+1:round(num_competitors/10)) = 0;
% 
% mu_array(epoch,:) = mu;
% sigma_array(epoch,:) = sigma;
        
%% Plots

midcomps = squeeze(comptraits(:,3,:));
midrow = midcomps(:);

latecomps = squeeze(comptraits(:,5,:));
laterow = latecomps(:);

finalcomps = squeeze(comptraits(:,50,:));
finalrow = finalcomps(:);

figure(1)
clf
histogram(firstrow,100,'normalization','pdf')
grid on
axis square
xlim([-0.1,1.1])
xlabel('Percentage of times swerving','FontSize',28)
ylabel('Normalized concentration','FontSize',28)
title('Stag hunt competitor locations at step 1','FontSize',28)
drawnow

figure(2)
clf
histogram(midrow,100,'normalization','pdf')
grid on
axis square
xlim([-0.1,1.1])
xlabel('Percentage of times hunting the stag','FontSize',28)
ylabel('Normalized concentration','FontSize',28)
title('Stag hunt competitor locations at step 5','FontSize',28)
drawnow

figure(3)
clf
histogram(laterow,100,'normalization','pdf')
grid on
axis square
xlim([-0.1,1.1])
xlabel('Percentage of times hunting the stag','FontSize',28)
ylabel('Normalized concentration','FontSize',28)
title('Stag hunt competitor locations at step 10','FontSize',28)
drawnow

figure(4)
clf
histogram(finalrow,100,'normalization','pdf')
grid on
axis square
xlim([-0.1,1.1])
drawnow

figure(5)
clf
hold on
set(gca,'fontsize',20)
xline(6.6107*10^-5,'-k','LineWidth',2)
histogram(firstrow,200,'normalization','pdf','FaceColor','r','EdgeAlpha',0,'FaceAlpha',0.75)
histogram(midrow,200,'normalization','pdf','FaceColor','b','EdgeAlpha',0.05,'FaceAlpha',0.75)
%plot([median(midrow),median(midrow)],[0,12],'b')
histogram(laterow,200,'normalization','pdf','FaceColor','m','EdgeAlpha',0.05,'FaceAlpha',0.75)
%plot([median(laterow),median(laterow)],[0,12],'m')
[N,edges]= histcounts(laterow)
histogram(finalrow,200,'BinEdges',edges,'normalization','pdf','FaceColor','#77AC30','EdgeAlpha',0.05,'FaceAlpha',0.75)
%plot([median(finalrow),median(finalrow)],[0,12],'g')
%plot([0.25,0.25],[0,12],'k')
grid on
axis square
xlim([-0.01,1.01])
xlabel('Percentage of times defecting','FontSize',28)
%ylabel('Normalized concentration','FontSize',28)
title('PD','FontSize',28)
drawnow

figure(6)
clf
hold on
load('log_odds_moran_stag_50000.mat')
%colorscale = [-1:1]
%imagesc(log_odds_moran)
bin_centers = (edges(1:end-1) + edges(2:end))/2;
joint_dist = N'*N;
surf(linspace(0,1,21),linspace(0,1,21),log_odds_moran,'EdgeColor','none','Facecolor','interp')
contour3(bin_centers,bin_centers,joint_dist,10,'k','Linewidth',0.5)
contour3(bin_centers,bin_centers,joint_dist,[1/5,2/5,3/5,4/5]*max(max(joint_dist)),'k','Linewidth',2)
axis square

grid on
colorbar
caxis([-0.85 0.85])
set(gca,'FontSize',24)
title('Stag Hunt','FontSize',28)
xlabel('Percentage of times hunting the stag','FontSize',28)
ylabel('Percentage of times hunting the stag','FontSize',28)


drawnow 
% bin_levels = 10.^(linspace(1/log(10),log(max(max(joint_dist)))/log(10),20));
% contour(bin_centers,bin_centers,joint_dist,bin_levels,'k','Linewidth',2)