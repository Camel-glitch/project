#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>
#include <curand_kernel.h>
#include "utils.h"
#include "gamma.h"


// Monte Carlo simulation kernel
__global__ void MC_exact(float rho, float v_0, float S_0, float r, float sigma, float k, float theta, float dt, float K, int N,
            curandState* state, float* sum, float* sum2, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float A[];
    float* R1s = A;
    float* R2s = &A[blockDim.x];
    float payoff = 0.0f;
    if (idx < n) {
    

    curandState localState = state[idx];
    float2 G1;
    float2 G2; 
    float S = S_0 ;
    float v0 = v_0 ;
    float v1 = step_variance(v0, k, theta, sigma, dt,state); 
    float vI = 0.0 ; 
    for (int i = 0; i<N;i++){
      //update v1 and v0 
      vI += 0.5*(v0+v1)*dt;
      v0 = v1;
      v1 = step_variance(v0, k, theta, sigma, dt,state); 
    

    }
    float V = (v1 - v_0 - k*theta + k*vI)/sigma;
    float m = rho*V - 0.5*vI;
    float sigma = sqrt((1-rho*rho)*vI);
    float2 G = curand_normal2(&localState);
    payoff = fmaxf(0.0f, exp(m + sigma * G.x) - K);

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
}
    /*Copy new state to global memory */
    //state[idx]=localState;

}
