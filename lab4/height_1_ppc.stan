generated quantities {
   real mu = normal_rng(178, 20);
   real sigma = exponential_rng(0.067);
   real height = normal_rng(mu, sigma);
}