// =================================================================
//
// File: SumaPares.cpp
// Author: David Bernabe
// Description: This file implements the counting of pair numbers in an array
//		using C/C++ threads. To compile:
//		g++ -o app SumaPares.cpp
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

using namespace std;
using namespace std::chrono;

#define SIZE   5000000 // 1e9
#define THREADS std::thread::hardware_concurrency()
#define N 10

void getSum(int start, int end, long long &sum) {
    sum=0;

    for (int i = start; i < end; i++) {
        if(i<2){
            continue;
        }
        bool res=1;

        for(int j=2; j*j <= i; j++){
            if(i%j==0) res=0;
        }

        if(res) sum+=i;
    }
}

int main(int argc, char* argv[]) {

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point startTime, endTime;
    double timeElapsed;

    int end, blockSize;
    thread threads[THREADS];
    long long result[THREADS];
    blockSize = SIZE / THREADS;

    cout << "Starting...\n";
    timeElapsed = 0;
    long long totalSum;
    for (int j = 0; j < N; j++) {
        startTime = high_resolution_clock::now();

        totalSum=0;
        for (int i = 0; i < THREADS; i++) {
            end = (i != (THREADS - 1))? (i + 1) * blockSize : SIZE;
            threads[i] = thread(getSum, (i * blockSize), end, std::ref(result[i]));
        }

        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
            totalSum+=result[i];
        }
        cout << totalSum << endl;

        endTime = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(endTime - startTime).count();
    }
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    return 0;
}