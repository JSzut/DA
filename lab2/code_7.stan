data {
    int<lower=1> N;
}

parameters {
    real y[N];
    real theta;
}

model {
    y ~ normal(theta, 1);
    theta ~ normal(0, 1);
}