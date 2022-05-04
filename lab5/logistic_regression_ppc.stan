data { 
    int<lower=1> N; // Number of observations 
    int<lower=1> M; // Number of covariates 
    matrix[N, M] X; // Covariate design matrix
    real sigma; //prior standard deviation 
} 

transformed data { 
    vector[N] ones_N = rep_vector(1, N);
    vector[M] ones_M = rep_vector(1, M); 
} 

// Simulate empirical probabilities from the current value of the parameters 
generated quantities { 
    vector[N] prob_ppc; 
    real beta[M] = normal_rng(0, ones_M*sigma); // Prior model 
    real alpha = normal_rng(0, sigma); // Prior model 
    prob_ppc = inv_logit(X * to_vector(beta) + ones_N*alpha);
} 