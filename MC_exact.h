#ifndef MC_EXACT_H
#define MC_EXACT_H

void MC_exact(float rho, float v_0, float S_0, float r, float sigma, float k, float theta, float dt, float K, int N,
              curandState* state, float* sum, int n);

#endif // MC_EXACT_H    