parameters{
    real theta;
}

model {
    theta ~ gamma(1.25, 1.25);
}