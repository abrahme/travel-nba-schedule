data{
    int N; // number of observations
    vector[N] y; // target margin
    int team_i[N]; // team i vector
    int team_j[N]; // team j vector
    int h_i[N]; // home team i
    int h_j[N]; // home team j
    int N_g; // number of teams (groups)
    int games_i[N]; // number of games played i (time step)
    int games_j[N]; // number of games played j (time step)
    int N_t; // max number of games played 
    real initial_prior; // initial theta values
    real fatigue_i[N]; // rest for team i
    real fatigue_j[N]; // rest for team j
}
parameters{
  real a; // correlation parameter
  real<lower=0> tau; // random variation
  matrix[N_g, N_t] theta;
  real<lower=0> sigma; // standard deviation of total model
  real<lower=0> w; // random noise from state space
  real home; // intercept for home indicator
  real fatigue;
}
model{
  ### state space model
    theta[,1] ~ normal(initial_prior, tau);
    for(t in 2:N_t) {
      for (g in 1:N_g){
        theta[g,t] ~ normal(a*theta[g,t-1],w);
  }
}
   for (i in 1:N){
     y[i] ~ normal(theta[team_i[i],games_i[i]] + h_i[i] * home + fatigue_i[i]*fatigue  - theta[team_j[i],games_j[i]] - h_j[i] * home - fatigue_j[i]*fatigue, sigma);
   }
}
