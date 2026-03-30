
#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>
#include "gamma.h"
#include "utils.h"









// Monte Carlo with Reduction in shared memory




__global__ void MC_almost(float rho, float v_0, float S_0, float r, float sigma, float k, float theta, float dt, float K, int N,
            curandState* state, float* sum, float* sum2, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    extern __shared__ float A[]; 
    
    float* R1s = A;             
    float* R2s = &A[blockDim.x]; 
    


    float payoff = 0.0f;

    if (idx < n) {
      curandState localState = state[idx];
      float log_S = log(S_0);
      float2 G; 
      float rho_sigma = rho / sigma;
      float k0 = (-rho_sigma * k * theta) * dt;
      float k1 = (rho_sigma * k - 0.5f) * dt - rho_sigma;
      float k2 = rho_sigma;
      float v0 = v_0 ;
      float v1 = step_variance(v0, k, theta, sigma, dt,&localState); 
      for (int i = 0; i<N;i++){
        G = curand_normal2(&localState);
        log_S = log_S + k0 +k1*v0 + k2*v1  + sqrt((1-rho*rho)*v0*dt)*(rho*G.x + sqrt(1-rho*rho)*G.y);
        v0 = v1; 
        v1 = step_variance(v0, k, theta, sigma, dt,&localState);
      }
      
      payoff = fmaxf(0.0f, exp(log_S) - K);
    

    // All threads write to shared memory (inactive threads write 0.0f)
    R1s[threadIdx.x] = payoff;
    R2s[threadIdx.x] = payoff * payoff;
    __syncthreads();

    // Parallel reduction in shared memory (Requires blockDim.x to be a power of 2)
    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            R1s[threadIdx.x] += R1s[threadIdx.x + i];
            R2s[threadIdx.x] += R2s[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Thread 0 of each block adds its block's total to the global sum safely
    if (threadIdx.x == 0) {
        atomicAdd(sum, R1s[0]);
        atomicAdd(sum2, R2s[0]);
        
    }
    /*Copy new state to global memory */
    //state[idx]=localState;



}
            }



