data {
	int<lower = 1> N;
	int<lower=1> M;
	matrix[M,N] X;
	array[N] real y;
}

parameters {
	vector[M] beta;
	real alpha;
	real<lower = 0> sigma;
}

model{
	alpha ~normal(0,2);
	sigma ~normal(0,2);
	
	y~normal(X' * beta + alpha, sigma);
}