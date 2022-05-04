data {
	int N; //number of samples
	array[N] real y; //measurements
}
parameters {
	real<lower=0> sigma;
	real mu;
}
model {
	target+=2*log(sigma);
	target+=normal_lpdf(y|mu,sigma);
}
generated quantities {
	vector[N] log_lik;
	array[N] real y_hat;
	for (j in 1:N) {
		log_lik[j] = normal_lpdf(y[j]|mu,sigma);
		y_hat[j] = normal_rng(mu,sigma);
	}
}