data{
    int N;
    real weight[N];
}

generated quantities {
   real alpha = normal_rng(178, 20);
   real beta = normal_rng(0, 10);
   real sigma = exponential_rng(0.067);
   real height[N];
   for (i in 1:N) {
      height[N] = normal_rng(weight[N]*beta*alpha, sigma);
   }
}