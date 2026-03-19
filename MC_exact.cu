#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>
#include <curand_kernel.h>
#include "gamma.h"


// Monte Carlo simulation kernel
__global__ void MC_exact(float rho, float v_0, float S_0, float r, float sigma, float k, float theta, float dt, float K, int N,
            curandState* state, float* PayGPU){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = state[idx];
    float2 G1;
    float2 G2; 
    float S = S_0 ;
    float v0 = v_0 ;
    float v1 = step_variance(v0, kappa, theta, sigma, dt,state); 
    float vI = 0.0 ; 
    for (int i = 0; i<N;i++){
      //update v1 and v0 
      vI += 0.5*(v0+v1)*dt;
      v0 = v1;
      v1 = step_variance(v0, kappa, theta, sigma, dt,state); 
    

    }
    float V = (v1 - v_0 - k*theta + k*vI)/sigma;
    float m = rho*V - 0.5*vI;
    float sigma2 = (1-rho*rho)*vI;
    double G = curand_normal_double(state);

    //PayGPU[idx]= expf(-r *dt*dt*N)*fmaxf(0.0f,S-K);
    PayGPU[idx] = exp(m + Sigma * G);
    /*Copy new state to global memory */
    //state[idx]=localState;

}
