data {
    int<lower=1> N;
}

generated quantities {
   real mu = normal_rng(0, 1);
   real<lower=0> sigma = abs(normal_rng(0,1));

   array[N] real y_prior;
   for (n in 1:N) {
       y_prior[n] = normal_rng(mu, sigma);
   }
}