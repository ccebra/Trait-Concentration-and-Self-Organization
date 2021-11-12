%% Test rho prediction
clear;
clf;

%% parameters
num_traits = 5; % we want to generalize to more dimensions
num_frequencies = 3;% In performance function, number of trigonometric frequencies
trig_amplitude = 1;% Amplitude of trigonometric components in performance function
linear_amplitude = 0;% Amplitude of linear components in performance function

n_trait_pairs = 10; % should be less than num_traits^2, number of coupled trait pairs
n_trait_pairs = min(n_trait_pairs,num_traits^2);
trait_pairs = nan(n_trait_pairs,num_frequencies);
for freq = 1:num_frequencies
    trait_pairs(:,freq) = randperm(num_traits^2,n_trait_pairs); % indexes which trait pairs to include in the sum defining performance
end
alpha = trig_amplitude*randn([n_trait_pairs,num_frequencies])/n_trait_pairs; % second dimension is the upper frequency of sines used to construct the random performance function
phase = 2*pi*rand([n_trait_pairs,num_frequencies]);

%% define performance function
f = @(x,y) example_performance_6(x,y,trait_pairs,alpha,linear_amplitude,phase);

%% pick centroid and Sigma
centroid = randn([1,num_traits]);
S = randn([num_traits,num_traits]);
Sigma = S*S';
Sigma = num_traits*Sigma/trace(Sigma); % normalize so unit std in each trait
[U,Lambda,~] = svd(Sigma);
R = U*Lambda^(1/2); % then Sigma = R*R'

%% Compute gradient and Hessian at centroid
[g,H] = Hessian_for_example_performance_6(centroid,trait_pairs,alpha,linear_amplitude,phase);


%% pick set of stds to test and parameters for empirical estimation
std_bounds = [-4,0]; % bounds on range of stds to test for convergence (log base 10)
num_stds = 100;% Number of standard deviations that we run convergence rate analysis over
V = 100; % size of network to sample per trial in convergence test
E = V*(V - 1)/2; % number of edges in complete graph
L = E - (V - 1); % number of loops in complete graph
epoch_bounds = [10,10^3]; % max and min number of trials to run in convergence test
tol = 10^(-4); % desired tolerance on estimated rho in convergence test

%% compute/estimate rho over the range of stds
stds = 10.^(linspace(std_bounds(2),std_bounds(1),num_stds));

%% empirical
for k = 1:length(stds)
    %% get std
    s = stds(k);
    
    %% compute theoretical rho, epsilon, intransitivity, radius of locally transitive region
    delta = 0.05;
    [intrans.rel.analytic(k),rhos.analytic(k),epsilons(k),r_trans] = Estimate_rho_Gaussian_traits(g,H,s^2*Sigma,V,E,delta);
    
    %% estimate rho empirically
    [rhos.mean(k),rhos.std(k)] = estimate_rho_Gauss_2(f,centroid,s*R,tol,epoch_bounds,V);
    
    %% convert to relative intransitivity averaged over the networks
    intrans.rel.empirical(k) = sqrt((1 - 2*rhos.mean(k))*(L/E));
    
    %% display
    figure(1)
    clf
    hold on
    plot(stds(1:k),0.5 - rhos.mean(1:k),'b-','Linewidth',2)
    fill([stds(1:k),fliplr(stds(1:k))],[0.5 - rhos.mean(1:k) + rhos.std(1:k),fliplr(0.5 - rhos.mean(1:k) - rhos.std(1:k))],...
        'b','FaceAlpha',0.4,'Linestyle','none')
    fill([stds(1:k),fliplr(stds(1:k))],[0.5 - rhos.mean(1:k) + 2*rhos.std(1:k),fliplr(0.5 - rhos.mean(1:k) - 2*rhos.std(1:k))],...
        'm','FaceAlpha',0.2,'Linestyle','none')
    fill([stds(1:k),fliplr(stds(1:k))],[0.5 - rhos.mean(1:k) + 3*rhos.std(1:k),fliplr(0.5 - rhos.mean(1:k) - 3*rhos.std(1:k))],...
        'r','FaceAlpha',0.1,'Linestyle','none')
    plot(stds(1:k),0.5 - rhos.analytic(1:k),'k-','Linewidth',2)
    plot([r_trans,r_trans],0.5 - [min(min(rhos.mean(1:k)),min(rhos.analytic(1:k))),...
        max(max(rhos.mean(1:k)),max(rhos.analytic(1:k)))],'k--','Linewidth',1)
    plot([stds(1),stds(k)],0.5*(1 - 1./(1 + delta))*[1,1],'k--','Linewidth',1)
    grid on
    set(gca,'yscale','log','xscale','log')
    axis tight
    xlim([min(stds),max(stds)])
    ylim(0.5 - [max(max(rhos.mean(1:k)),max(rhos.analytic(1:k))),...
        min(min(rhos.mean(1:k)),min(rhos.analytic(1:k)))]);
    axis square
    xlabel('Standard Deviation: $\sqrt{E[||X - \bar{x}||^2]/T}$','FontSize',16,'interpreter','latex')
    ylabel('$0.5 - \rho$','FontSize',16,'interpreter','latex')
    title('Correlation Coefficient and Trait Concentration ','FontSize',16,'interpreter','latex')
    l = legend('Empirical Estimate','$1 \sigma$','$2 \sigma$','$3 \sigma$', 'Analytic Prediction','Transitive Radius','$0.5 - \rho(\delta)$');
    set(l,'FontSize',14,'interpreter','latex','location','best')
    
    
    figure(2)
    clf
    hold on
    plot(stds(1:k),intrans.rel.empirical(1:k),'b-','Linewidth',2)
    plot(stds(1:k),intrans.rel.analytic(1:k),'k-','Linewidth',2)
    grid on
    set(gca,'yscale','log','xscale','log')
    axis tight
    axis square
    xlabel('Standard Deviation','FontSize',16,'interpreter','latex')
    ylabel('$||f_c||/||f||$','FontSize',16,'interpreter','latex')
    title('Relative Intransitivity and Trait Concentration ','FontSize',16,'interpreter','latex')
    l = legend('Empirical Estimate','Analytic Prediction');
    set(l,'FontSize',14,'interpreter','latex','location','best')
    
    drawnow
    
end