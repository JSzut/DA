data { 
    int<lower=1> N; // Number of observations 
    int<lower=1> M; // Number of covariates 
    matrix[N, M] X; // Covariate design matrix 
    int<lower=0, upper=1> y[N]; // Binary variates 
} 

transformed data { 
    vector[N] ones_N = rep_vector(1, N); 
}

parameters { 
    vector[M] beta; 
    real alpha; 
}

model { 
    beta ~ normal(0, 2); // Prior model 
    alpha ~ normal(0, 2); // Prior model 
    y ~ bernoulli_logit(X * beta + alpha); // Observational model 
}
 
// Simulate empirical probabilities from the current value of the parameters 
generated quantities { 
    int y_ppc[N] = bernoulli_logit_rng(X * beta + ones_N*alpha); 
    vector[N] prob_ppc = inv_logit(X * beta + ones_N*alpha); 
} 
