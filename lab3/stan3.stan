data {
   int<lower=1> N;
   vector[N] y;
 }
 
 parameters {
   real mu;
   real<lower=0> sigma;
 }

 model {
   mu ~ normal(0,1);
   sigma ~ normal(0,1);

   y ~ normal(mu, sigma);
 }

 generated quantities {
    vector[N] y_rep;
    for(i in 1:N) {
        y_rep[i] = normal_rng(mu, sigma);
    }
 }