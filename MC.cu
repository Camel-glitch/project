/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>
#include <curand_kernel.h>


// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))




__device__ double step_variance(double vt, double kappa, double theta, double sigma, double dt, curandState *state) {
    // 1. Précalcul des constantes pour ce pas de temps
    double exp_kdt = exp(-kappa * dt);
    double sigma2 = sigma * sigma;
    double scale = (sigma2 * (1.0 - exp_kdt)) / (4.0 * kappa); // Facteur multiplicatif final

    // 2. Calcul des paramètres d et lambda
    // Note : d dans ton énoncé est le "degré de liberté" divisé par 2
    double d = (2.0 * kappa * theta) / sigma2;
    double lambda = (4.0 * kappa * exp_kdt * vt) / (sigma2 * (1.0 - exp_kdt));

    // 3. Simulation de la composante Poisson (N)
    // N ~ Poisson(lambda / 2)
    unsigned int N = curand_poisson(state, lambda / 2.0);

    // 4. Simulation de la Gamma standard G(alpha)
    // alpha = d + N
    double alpha = d + N;
    double G = generate_gamma(alpha, state); // Utilise ta device function Marsaglia-Tsang

    // 5. Calcul de vt+dt
    // La formule vt+dt = scale * 2 * G(d + N) correspond à la loi du Khi-deux non-centrale
    return scale * 2.0 * G;
}
}

// Set the state for each thread
__global__ void init_curand_state_k(curandStateXORWOW *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    /* curand_init parameters:
       1. seed: Use the same one for all threads (e.g., 1234).
       2. subsequence: USE THE THREAD ID. This ensures each thread 
          gets a unique, non-overlapping sequence of numbers.
       3. offset: Usually 0 (starts at the beginning of the sequence).
       4. state: Pointer to this thread's specific memory slot.
    */
    curand_init(seed, idx, 0, &state[idx]);
}


// Monte Carlo with Reduction in shared memory
__global__ void MC_k2(float S_0, float r, float sigma, float dt, float K,
	int N, curandState* state, float* sum, int n) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = state[idx];
	float2 G;
	float S = S_0;
	extern __shared__ float A[];

	float* R1s, * R2s;
	R1s = A;
	R2s = R1s + blockDim.x;

	for (int i = 0; i < N; i++) {
		G = curand_normal2(&localState);
		S *= (1 + r * dt * dt + sigma * dt * G.x);
	}
	R1s[threadIdx.x] = expf(-r * dt * dt * N) * fmaxf(0.0f, S - K) / n;
	R2s[threadIdx.x] = R1s[threadIdx.x] * R1s[threadIdx.x] * n;

	__syncthreads();
	int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			R1s[threadIdx.x] += R1s[threadIdx.x + i];
			R2s[threadIdx.x] += R2s[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		atomicAdd(sum, R1s[0]);
		atomicAdd(sum + 1, R2s[0]);
	}

	/* Copy state back to global memory */
	//state[idx] = localState;
}

// Monte Carlo simulation kernel
__global__ void MC_euler(float rho, float v_0, float S_0, float r, float sigma, float k, float theta, float dt, float K, int N,
            curandState* state, float* sum, int n) {
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Dynamic shared memory for reduction
    extern __shared__ float R1s[]; 
    // Add variance computation for confidence interval if needed (not implemented here)

    float payoff = 0.0f;

    // Only active threads compute the simulation
    if (idx < n) {
        curandState localState = state[idx];
        float2 G; 
        float S = S_0;
        float v = v_0;
        
        // Simulation of price and variance up to time N 
        for (int i = 0; i < N; i++) {
            G = curand_normal2(&localState);
            // Euler discretization
            S = S * (1.0f + r * dt + sqrtf(v * dt) * (rho * G.x + sqrtf(1.0f - rho * rho) * G.y)); 
            // Full truncation / reflective boundary for variance
            v = fabsf(v + k * (theta - v) * dt + sigma * sqrtf(v * dt) * G.x);
        }

        payoff = fmaxf(0.0f, S - K) / (float)n;
        
        // Save state back to global memory for future kernel calls
        state[idx] = localState;
    }

    // All threads write to shared memory (inactive threads write 0.0f)
    R1s[threadIdx.x] = payoff;
    __syncthreads();

    // Parallel reduction in shared memory (Requires blockDim.x to be a power of 2)
    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            R1s[threadIdx.x] += R1s[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Thread 0 of each block adds its block's total to the global sum safely
    if (threadIdx.x == 0) {
        atomicAdd(sum, R1s[0]);
    }
}

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


__global__ void MC_almost(float rho, float v_0, float S_0, float r, float sigma, float k, float theta, float dt, float K, int N,
            curandState* state, float* PayGPU){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = state[idx];
    float2 log_S = log(S_0);
    float2 G; 
    float rho_sigma = rho / sigma;
    float k0 = (-rho_sigma * kappa * theta) * dt;
    float k1 = (rho_sigma * kappa - 0.5f) * dt - rho_sigma;
    float k2 = rho_sigma;
    float v0 = v_0 ;
    float v1 = step_variance(v0, kappa, theta, sigma, dt,state) 

    for (int i = 0; i<N;i++){
      G = curand_normal2(&localState);
      log_S = log_S + k0 +k1*v0 + k2*v1 +  + sqrt((1-rho*rho)*v0*dt)*(rho*G.x + sqrt(1-rho*rho)*G.y));
      v0 = v1; 
      v1 = step_variance(v0, kappa, theta, sigma, dt,state);
    }
    //PayGPU[idx]= expf(-r *dt*dt*N)*fmaxf(0.0f,S-K);
    PayGPU[idx] = exp(log_S);
    /*Copy new state to global memory */
    //state[idx]=localState;



            }

// DO THE SUM USING REDUCTION !!! 



int main(void) {
    int scale = 1 ; 
    int NTPB = 512;
    int NB = 512;
    long long n = scale * NB * NTPB; // Number of trajectories
    
    // Model Parameters
    float T = 1.0f;
    float rho = 0.5f; 
    float k = 0.5;
    float theta = 0.1;
    float S_0 = 1.0f;
    float K = S_0;
    float sigma = 0.3f;
    float r = 0.0f;
    int N = 100; // Time steps
    float dt = T / (float)N; // Standard time step
    float v_0 = 0.1f;

    // CPU Memory Allocation
    //float *PayCPU = (float*)malloc(n * sizeof(float));
    float *PayCPU = (float*)malloc(1 * sizeof(float)); // Only one float for the sum

    // GPU Memory Allocation
    float *PayGPU;
    //cudaMalloc((void**)&PayGPU, n * sizeof(float));
    cudaMalloc((void**)&PayGPU, 1 * sizeof(float));

    // RNG State Allocation
    curandStateXORWOW *state; 
    cudaMalloc((void**)&state, n * sizeof(curandStatePhilox4_32_10_t));

    // Random Seed (using system time or fixed value)
    unsigned long seed = 1234ULL; 

    // 1. Initialize RNG States
    // Arguments: State array, Global Seed
    init_curand_state_k<<<NB, NTPB>>>(state, seed);

    // Timing Setup
    float Tim;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 2. Launch Monte Carlo Kernel
    // Arguments: state, output array, and model parameters
    MC_euler<<<NB, NTPB>>>(rho,v_0,S_0, r, sigma, k, theta, dt, K, N,
            state, PayGPU, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&Tim, start, stop);
    
    // 3. Copy Results back to Host
    //cudaMemcpy(PayCPU, PayGPU, n* sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(PayCPU, PayGPU, 1* sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = PayCPU[0]; // pricing estimation
    //float sum2 = 0.0f; // We would need to compute this separately if we want the confidence interval

    /* Reduction performed on the host
    float sum = 0.0f;
    float sum2 = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += PayCPU[i];
        sum2 += PayCPU[i] * PayCPU[i];
    }
    
    sum = sum / n;
    sum2 = sum2 / n; 
    */

    printf("The estimated price is equal to %f\n", sum);
    //printf("error (95%% CI) = %f\n", 1.96 * sqrt((double)(1.0f / (n - 1)) * (n * sum2 - (n * sum * sum))) / sqrt((double)n));
    
    // Note: The Black-Scholes formula in your print statement is a simplified version for comparison
    printf("Execution time %f ms\n", Tim);

    // 4. Cleanup
    cudaFree(PayGPU);
    cudaFree(state);
    free(PayCPU);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}