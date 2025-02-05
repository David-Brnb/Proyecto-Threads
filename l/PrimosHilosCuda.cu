// // =================================================================
// //
// // File: SumaParesCuda.cpp
// // Author: David Bernabe
// // Description: This file implements the counting of pair numbers in an array
// //              To compile:
// //		        !nvcc -arch=sm_75 -o app SumaParesCuda.cpp
// //
// // Copyright (c) 2024 by Tecnologico de Monterrey.
// // All Rights Reserved. May be reproduced for any non-commercial
// // purpose.
// //
// // =================================================================

// #include <iostream>
// #include <iomanip>
// #include <chrono>
// #include <algorithm>
// #include <climits>
// #include <cuda_runtime.h>

// using namespace std;
// using namespace std::chrono;

// #define SIZE 5000000 //5e6
// #define THREADS 512
// #define BLOCKS	32
// #define N 10

// __global__ void pares(long long *results) {
//     __shared__ long long cache[THREADS];

//     int tid = threadIdx.x + (blockIdx.x * blockDim.x);
//     int cacheIndex = threadIdx.x;

//     long long aux = 0;
//     while (tid < SIZE) {
//         if(tid>=2){
//           bool res=1;

//           for(int j=2; j*j <= tid; j++){
//               if(tid%j==0) res=0;
//           }

//           if(res) aux+=tid;
//         }
        
//         tid += blockDim.x * gridDim.x;
//     }

//     cache[cacheIndex] = aux;

//     __syncthreads();

//     int i = blockDim.x / 2;
//     while (i > 0) {
//         if (cacheIndex < i) {
//             cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + i];
//         }
//         __syncthreads();
//         i /= 2;
//     }

//     if (cacheIndex == 0) {
//         results[blockIdx.x] = cache[cacheIndex];
//     }
// }

// int main(int argc, char* argv[]) {
//     long long *results, *d_r;
//     int i;
    
//     // These variables are used to keep track of the execution time.
//     high_resolution_clock::time_point start, end;
//     double timeElapsed;

//     results = new long long[BLOCKS];

//     cudaMalloc( (void**) &d_r, BLOCKS * sizeof(long long) );

//     cout << "Starting...\n";
//     timeElapsed = 0;
//     for (int j = 0; j < N; j++) {
//         start = high_resolution_clock::now();

//         pares<<<BLOCKS, THREADS>>> (d_r);

//         end = high_resolution_clock::now();
//         timeElapsed += 
//             duration<double, std::milli>(end - start).count();
//     }

//     cudaMemcpy(results, d_r, BLOCKS * sizeof(long long), cudaMemcpyDeviceToHost);

//     long long aux = 0;
//     for (i = 0; i < BLOCKS; i++) {
//         aux += results[i];
//     }

//     cout << "result = " << aux << "\n";
//     cout << "avg time = " << fixed << setprecision(3) 
//         << (timeElapsed / N) <<  " ms\n";

//     cudaFree(d_r);

//     delete [] results;

//     return 0;
// }