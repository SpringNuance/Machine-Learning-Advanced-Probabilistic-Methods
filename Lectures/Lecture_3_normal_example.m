% Author: P. Marttinen (2015)
clear
clf;
rng(1); % Set random number generator.


% SIMULATE THE TRUE DATA SET
num_samples = 10; % Try e.g. values between 5 and 10000
lambda_true = 4; % precision
mu_true = 2; % mean
sigma_true = 1/sqrt(lambda_true); % standard deviation
data_set = normrnd(mu_true, sigma_true, num_samples, 1);

histogram_edges = 0:0.25:4;
hist(data_set, histogram_edges);
title(['Histogram of ' num2str(num_samples) ' samples from N(' num2str(mu_true) ',' num2str(sigma_true) '^2).']);
pause


% SPECIFY PRIORS

% lambda is the precision parameter of the unknown Gaussian
% and it is given a prior distribution Gamma(a0,b0),
% (a0 is the 'shape' and b0 the 'rate')
a0 = 0.01;
b0 = 0.01; % These correspond to a noninformative prior

% mu is the mean parameter of the unknown Gaussian
% and it is given a prior distribution that depends on
% lambda: N(mu0, (beta0*lambda)^-1)
mu0 = 0;
beta0 = 0.001; % Low precision corresponds to high variance

% (This is the so-called Normal-Gamma(mu0, beta0, a0, b0)
% prior distribution for mu and lambda)


% LEARN THE POSTERIOR DISTRIBUTION

% Due to conjugacy, the posterior distribution is also
% Normal-Gamma(mu_n, beta_n, a_n, b_n)

sample_mean = sum(data_set) / num_samples;
sample_var = sum((data_set - sample_mean).^2) / num_samples;

a_n = a0 + num_samples / 2;

b_n = b0 + (num_samples * sample_var ...
    + (beta0 * num_samples * (sample_mean-mu0)^2) ...
    / (beta0 + num_samples)) / 2;

mu_n = (mu0 * beta0 + num_samples * sample_mean) ...
    / (beta0 + num_samples);

beta_n = beta0 + num_samples;



% PLOT THE PRIOR AND THE POSTERIOR DISTRIBUTIONS

% Plot distribution of lambda, the precision
lambda_range = 0:0.01:10;
prior_lambda_pdf = gampdf(lambda_range, a0, 1/b0);
posterior_lambda_pdf = gampdf(lambda_range, a_n, 1/b_n);

clf
set(gcf,'DefaultAxesFontSize',14);
set(gcf,'DefaultTextFontSize',16);
set(gcf,'DefaultLineLineWidth',2);
set(gca,'ytick',[],'box','on');
plot(lambda_range, [prior_lambda_pdf;posterior_lambda_pdf]');
hold on
max_y = max(get(gca, 'YLim'));
line([lambda_true,lambda_true],[0,max_y], 'Color','black'); % Show the true value
title('lambda')
legend('Prior','Posterior','Location','northeast')
pause

hold off

% Plot distribution of mu, the mean
mu_range = 1:0.01:3;
% Because mu depends on lambda, we need to integrate over 
% lambda. We do this by Monte Carlo integration (i.e. 
% average over multiple simulated lambdas)
gamma_prior_samples = gamrnd(a0, 1/b0, 1, 100);
sum_prior_mu_pdf = zeros(1,length(mu_range));
for gamma_sample = gamma_prior_samples
    prior_mu_pdf = normpdf(mu_range, mu0, 1/sqrt((beta0*gamma_sample)));
    sum_prior_mu_pdf = sum_prior_mu_pdf + prior_mu_pdf;
end
prior_mu_pdf = sum_prior_mu_pdf / length(gamma_prior_samples);

gamma_posterior_samples = gamrnd(a_n, 1/b_n, 1, 100);
sum_posterior_mu_pdf = zeros(1, length(mu_range));
for gamma_sample = gamma_posterior_samples
    posterior_mu_pdf = normpdf(mu_range, mu_n, 1/sqrt(beta_n*gamma_sample));
    sum_posterior_mu_pdf = sum_posterior_mu_pdf + posterior_mu_pdf;
end
posterior_mu_pdf = sum_posterior_mu_pdf / length(gamma_prior_samples);

clf
plot(mu_range, [prior_mu_pdf;posterior_mu_pdf]');
hold on
max_y = max(get(gca, 'YLim'));
line([mu_true,mu_true],[0,max_y], 'Color','black'); % Show the true value.
title('mu')
legend('Prior','Posterior','Location','northeast')
pause
hold off


% PLOT THE TRUE AND ESTIMATED DISTRIBUTIONS OF THE SAMPLES

% We estimate the parameters with the mean of the posterior distribution
mu_hat = sum(posterior_mu_pdf .* mu_range) / sum(posterior_mu_pdf);
lambda_hat = sum(posterior_lambda_pdf .* lambda_range) / sum(posterior_lambda_pdf);

full_dist_range = -2:0.1:6;
true_pdf = normpdf(full_dist_range, mu_true, sigma_true);
estimated_pdf = normpdf(full_dist_range, mu_hat, 1/sqrt(lambda_hat));
clf
plot(full_dist_range, [true_pdf;estimated_pdf]');
hold on
title('Distribution of the samples')
legend('True','Estimated','Location','northeast')

