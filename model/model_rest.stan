data{
    int N; // number of observations
    vector[N] y; // target margin
    int<lower=0,upper=1> z[N]; // target win or loss
    int team_i[N]; // team i vector
    int team_j[N]; // team j vector
    int h_i[N]; // home team i
    int h_j[N]; // home team j
    int N_g; // number of players (groups)
    real rest_i[N]; // covariate rest i
    real rest_j[N]; // covariate rest j
}
parameters{
    vector[2] beta_p[N_g]; // vector for slope and intercept
    vector<lower=0>[2] sigma_p; // sd for intercept and slope
    vector[2] beta; // intercept and slope hyper priors
    corr_matrix[2] Omega; // correlation matrix
    vector<lower=0>[N_g] omega;
    real<lower=0> sigma;
    real<lower=0> tau_omega;
}

model{
    // vector for conditional mean storage
    vector[N] mu;

    // priors
    Omega ~ lkj_corr(2);
    sigma_p ~ exponential(1);
    beta ~ normal(0,1);
    tau_omega ~ cauchy(0,.25)T[0,];
    sigma ~ cauchy(0,1)T[0,];
    beta_p ~ multi_normal(beta, quad_form_diag(Omega,sigma_p));
    omega ~ lognormal(0,tau_omega);

    // define mu for the Gaussian
    for( t in 1:N ) {
      mu[t] = (beta_p[team_i[t],1] + beta_p[team_i[t],2]*h_i[t] -omega[team_i[t]]*rest_i[t]) -
              (beta_p[team_j[t],1] + beta_p[team_j[t],2]*h_j[t] -omega[team_j[t]]*rest_j[t]);
    }

// the likelihood
    y ~ normal(mu,sigma);
    z ~ bernoulli_logit(mu + sigma); // regularize huge blowouts

}