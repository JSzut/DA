parameters{
    real<lower=0> theta;
}

model {
    theta ~ gamma(1.25, 1.25);
}