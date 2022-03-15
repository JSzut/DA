data {
    int<lower=1> N;
}

parameters {
    real y[N];
    real theta;
}

model {
    for (n in 1:N)
        target += normal_lpdf(y[n] | theta, 1);
        target += normal_lpdf(theta | 0, 1);
}