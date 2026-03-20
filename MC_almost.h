#ifndef MC_ALMOST_H
#define MC_ALMOST_H


#include <curand_kernel.h>


__global__ void MC_almost(float rho, float v_0, float S_0, float r, float sigma, float k, float theta, float dt, float K, int N,
            curandState* state, float* sum, float* sum2, int n);

#endif // MC_ALMOST_H