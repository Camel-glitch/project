#ifndef MC_EULER_H
#define MC_EULER_H

__global__ void MC_euler(float rho, float v_0, float S_0, float r, float sigma, float k, float theta, float dt, float K, int N,
              curandState* state, float* sum, float* sum2, int n);

#endif // MC_EULER_H