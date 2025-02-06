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

#define SIZE   5000000 // 5e9
#define THREADS std::thread::hardware_concurrency()
#define N 10

/*
    Definimos getSum
    Este método nos permitirá sumar a todos
    de números primos desde start hasta end. 
    La complejidad temporal es O(n*√n)
*/
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

    // Estas variables son usadas para el manejo de hilos
    int end, blockSize;
    thread threads[THREADS];
    long long result[THREADS];
    blockSize = SIZE / THREADS;

    // Se comienza el proceso pero ahora delegando la tarea a 
    // varios hilos a la vez, de forma segmentada.
    cout << "Starting...\n";
    timeElapsed = 0;
    long long totalSum;
    for (int j = 0; j < N; j++) {
        // Inico del conteo del tiempo
        startTime = high_resolution_clock::now();

        // Se realiza el llamado de los hilos dandoles
        // un determinado inicio, así como un determinado fin,
        // de acuerdo a la lógica  correspondiente.
        for (int i = 0; i < THREADS; i++) {
            end = (i != (THREADS - 1))? (i + 1) * blockSize : SIZE;
            threads[i] = thread(getSum, (i * blockSize), end, std::ref(result[i]));
        }

        // Se espera a que termine cada hilo, y tras haber 
        // finalizado, se recopila toda la cantidad de
        // primos calculada por cada hilo en la variable
        // totalSum.
        totalSum=0;
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
            totalSum+=result[i];
        }
        cout << totalSum << endl;

        // Fin del conteo del tiempo y recopilación de los tiempos 
        endTime = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(endTime - startTime).count();
    }

    // Se imprime el promedio del tiempo que se dedico al métdo
    // mediante el uso del hilos.
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    return 0;
}