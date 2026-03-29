#ifndef GAMMA_H
#define GAMMA_H

// Generating a G(a,1) random variable using the Marsaglia and Tsang method
__device__ double generate_gamma(double a, curandState *state); 


#endif // GAMMA_H