#include <stdio.h>
#include <algorithm>
#include <pthread.h>
#include <math.h>
#include <immintrin.h>

#include "CycleTimer.h"
#include "sqrt_ispc.h"

using namespace ispc;

extern void sqrtSerial(int N, float startGuess, float* values, float* output);

static void verifyResult(int N, float* result, float* gold) {
    for (int i=0; i<N; i++) {
        if (fabs(result[i] - gold[i]) > 1e-4) {
            printf("Error: [%d] Got %f expected %f\n", i, result[i], gold[i]);
        }
    }
}

void sqrtAVX(int N,
                float initialGuess,
                float values[],
                float output[])
{

    static const float kThreshold = 0.00001f;
    __mmask8 mask;
    __m256 tmp,left,right; 
    __m256 ONE = _mm256_set1_ps(1.f);//几个常量
    __m256 PointFive = _mm256_set1_ps(0.5f);
    __m256 THREE = _mm256_set1_ps(3.f);
    __m256 ABS = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));//用来求绝对值，and ABS使符号位变为0
    __m256 vector_Threshold = _mm256_set1_ps(kThreshold);

    for (int i=0; i<N; i+=8) {
        mask=0xff;
        __m256 origx = _mm256_load_ps(&values[i]);//origx = values[i]
        __m256 vector_guess = _mm256_set1_ps(initialGuess);
        
        tmp = _mm256_mask_mul_ps(vector_guess,mask,vector_guess,vector_guess);//tmp = guess*guess
        tmp = _mm256_mask_mul_ps(tmp,mask,tmp,origx);//tmp=tmp*origx
        tmp = _mm256_mask_sub_ps(tmp,mask,tmp,ONE);//tmp=tmp-1
        tmp = _mm256_mask_and_ps(tmp,mask,tmp,ABS);//tmp=|tmp|
        mask = mask & (_mm256_cmp_ps_mask(tmp,vector_Threshold,(unsigned char)30));//获取error大于threshold的mask

        while(mask){
            left = _mm256_mask_mul_ps(vector_guess,mask,vector_guess,THREE);//left = guess*3
            right = _mm256_mask_mul_ps(vector_guess,mask,vector_guess,vector_guess);//right = guess*guess
            right = _mm256_mask_mul_ps(right,mask,right,vector_guess);//right = right*guess
            right = _mm256_mask_mul_ps(right,mask,origx,right);//right = origx*right
            left = _mm256_mask_sub_ps(left,mask,left,right);//left = left - right
            vector_guess = _mm256_mask_mul_ps(left,mask,left,PointFive);//guess = left * 0.5

            tmp = _mm256_mask_mul_ps(vector_guess,mask,vector_guess,vector_guess);
            tmp = _mm256_mask_mul_ps(tmp,mask,tmp,origx);
            tmp = _mm256_mask_sub_ps(tmp,mask,tmp,ONE);
            tmp = _mm256_mask_and_ps(tmp,mask,tmp,ABS);
            mask = mask & (_mm256_cmp_ps_mask(tmp,vector_Threshold,(unsigned char)30));
        }

        _mm256_store_ps(&output[i],vector_guess);//output[i] = guess
    }
}


int main() {

    const unsigned int N = 20 * 1000 * 1000;
    const float initialGuess = 1.0f;

    float* values = new float[N];
    float* output = new float[N];
    float* gold = new float[N];

    for (unsigned int i=0; i<N; i++)
    {
        // TODO: CS149 students.  Attempt to change the values in the
        // array here to meet the instructions in the handout: we want
        // to you generate best and worse-case speedups
        
        // starter code populates array with random input values
        values[i] = 0.001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;
        // for(unsigned int j= i+1;j<i+8;j++)
        //     values[j]=1.f;
    }

    // generate a gold version to check results
    for (unsigned int i=0; i<N; i++)
        gold[i] = sqrt(values[i]);

    //
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrtSerial(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[sqrt serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    verifyResult(N, output, gold);

    //
    // Compute the image using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    printf("[sqrt ispc]:\t\t[%.3f] ms\n", minISPC * 1000);

    verifyResult(N, output, gold);

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;

    //
    // Tasking version of the ISPC code
    //
    double minTaskISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc_withtasks(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minTaskISPC, endTime - startTime);
    }

    printf("[sqrt task ispc]:\t[%.3f] ms\n", minTaskISPC * 1000);

    verifyResult(N, output, gold);

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;
    
    double minAVX = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrtAVX(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minAVX, endTime - startTime);
    }

    printf("[sqrt AVX]:\t[%.3f] ms\n", minAVX * 1000);

    verifyResult(N, output, gold);

    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);
    printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial/minTaskISPC);
    printf("\t\t\t\t(%.2fx speedup from AVX)\n", minSerial/minAVX);

    delete [] values;
    delete [] output;
    delete [] gold;

    return 0;
}
