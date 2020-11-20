#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>


#define SHFL_MASK 0xffffffff

namespace {

template <typename U, typename V>	
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {	
    return (a + b - 1) / b;	
}


template<int FS, int SB, int padding_l, typename scalar_t>
__inline__ __device__
void zeroSharedMem(scalar_t* data) {
    /*
    Given an array of length FS + SB, zero out the first padding_l and last
    (FS - padding_l) values in the array
    */

    int tid = threadIdx.x;

    if (FS < SB) {

    // zero all if we have enough threads in a block to do all of them
    if (tid < padding_l || tid > SB - FS + padding_l - 1) {
        data[tid] = scalar_t(0.0);
    }
    } else {

    // otherwise zero out one block at a time
    const int numIterations = divUp<int, int>(FS, SB);
    for (int i = 0; i < numIterations; i++) {
        int offset = i * SB;
        if (tid + offset < padding_l) {
        data[tid + offset] = scalar_t(0.0);
        } else if (tid + offset < FS) {
        data[SB + tid + offset] = scalar_t(0.0);
        }
    }
    }
}

template<typename scalar_t>
__inline__ __device__
scalar_t warpReduce(scalar_t data) {
    /*
    Reduce an array within each warp. After processing all values in warp will
    caontain the sum of all original values in that warp.

    data - pointer to data to reduce
    */
    data += __shfl_xor_sync(SHFL_MASK, data, 16);
    data += __shfl_xor_sync(SHFL_MASK, data, 8);
    data += __shfl_xor_sync(SHFL_MASK, data, 4);
    data += __shfl_xor_sync(SHFL_MASK, data, 2);
    data += __shfl_xor_sync(SHFL_MASK, data, 1);
    return data;
}

template<typename scalar_t>
__inline__ __device__
scalar_t blockReduce(scalar_t data) {
    /*
        Reduce an entire array on the block level. After processing, the
        first value in the array will contain the reduced sum.

        data - pointer to data to reduce
    */

    static __shared__ scalar_t warpSum[32];
    const int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    __syncthreads();

    // reduce each warp then write to shared memory
    scalar_t sum = warpReduce(data);
    if (lane == 0) {
    warpSum[wid] = sum;
    }
    
    __syncthreads();

    scalar_t v;
    // perform final sum of partial warp sums
    if (tid < blockDim.x / 32) {
    v = warpSum[lane];
    } else {
    v = scalar_t(0.0);
    }

    if (wid == 0) {
    v = warpReduce(v);
    }
    __syncthreads();

    return v;
}

void checkCudaStatus(cudaError_t status, int lineNumber = -1) {

    if (status != cudaSuccess) {
    std::cout << cudaGetErrorString(status)
                << " at line " << lineNumber << std::endl;
    std::cout << "Exiting" << std::endl;
    exit(1);
    }
}

template<int FS, int SB, int padding_l, typename scalar_t>
__device__
void load_input_to_shared(const scalar_t* input, // global memory
                            int inputOffset, int sequenceLength,
                            int iteration, int numIterations,
                            bool no_prev, scalar_t* output /* shared memory */) {
    /*
    Load a block size of input into shared memory with
    right and left overhang of total size FS. If previously
    loaded memory, overlap will be shifted over to reduce
    global memory access

    input - pointer to start of channel sequence
    inputOffset - how far in the sequence to start loading
    sequenceLength - total length of sequence
    iteration - which block of sequence we are loading
    numIterations - total number of blocks to load
    no_prev - whether to load the whole block if the previous block
                wasn't loaded
    output - shared memory to write input to
    */

    const int tid = threadIdx.x;

    // Load the left "overhang" of input
    if (iteration > 0) {
    if (padding_l < SB) {

        // load all at once
        if (tid < padding_l) {
        output[tid] = (no_prev) ? input[inputOffset - padding_l + tid] : output[tid + SB];
        }
    } else {

        // load in chunks of size SB
        int numIterations = divUp<int, int>(padding_l, SB);
        for (int i = 0; i < numIterations; i++) {
        int offset = i * SB;
        if ((tid + offset) < padding_l) {
            output[tid + offset] = (no_prev) ? input[inputOffset - padding_l + tid + offset] : output[tid + offset + SB];
        }
        }
    }
    }

    // Load the right "overhang" of input
    if (iteration < (numIterations - 1)) {
    const int elementsLeft = sequenceLength - (iteration+1) * SB;

    if ((FS - padding_l) < SB) {

        // load all at once
        if (tid < (FS - padding_l)) {
            output[padding_l + SB + tid] = (tid < elementsLeft) ? input[inputOffset + SB + tid] : scalar_t(0.0);
        }
    } else {

        // load in chunks of size SB
        int numIterations = divUp<int, int>(FS - padding_l, SB);
        for (int i = 0; i < numIterations; i++) {
        int offset = i * SB;
        if ((tid + offset) < (FS - padding_l)) {
            output[padding_l + SB + tid + offset] = ((tid + offset) < elementsLeft) ? input[inputOffset + SB + tid + offset] : scalar_t(0.0);
        }
        }
    }
    }

    // We should also clear out the right "overhang"
    if (iteration == (numIterations - 1)) {
    if ((FS - padding_l) < SB) {

        // clear out all at once
        if (tid < (FS - padding_l)) {
            output[padding_l + SB + tid] = scalar_t(0.0);
        }
    } else {

        // clear in chunks of size SB
        int numIterations = divUp<int, int>(FS - padding_l, SB);
        for (int i = 0; i < numIterations; i++) {
        int offset = i * SB;
        if ((tid + offset) < (FS - padding_l)) {
            output[padding_l + SB + tid + offset] = scalar_t(0.0);
        }
        }
    }
    }
    output[tid + padding_l] = ((inputOffset + tid) < sequenceLength) ? input[inputOffset + tid] : scalar_t(0.0);
}

} // namespace

// FS is filter size and kernels are specialized for filter sizes
template<int FS, int SB, int padding_l, typename scalar_t>
__global__
void dynamicconv_forward_kernel(const scalar_t* input,
                                const scalar_t* weight,
                                int minibatch,
                                int sequenceLength,
                                int numFeatures,
                                int numFiltersInBlock,
                                int numHeads,
                                scalar_t* output) {
    assert(blockDim.x == SB);

    const int tid = threadIdx.x;
    const int batchIdx = blockIdx.x;
    const int featureIdx = blockIdx.y;
    const int head = featureIdx / numFiltersInBlock;

    const int IOOffset = batchIdx * numFeatures * sequenceLength
                        + featureIdx * sequenceLength;
    const scalar_t* inputFeature = &input[IOOffset];
    scalar_t* outputFeature = &output[IOOffset];

    scalar_t filter[FS];

    __shared__ scalar_t tempInput[SB + FS];
    zeroSharedMem<FS, SB, padding_l>(tempInput);

    const int numIterations = divUp<int, int>(sequenceLength, SB);

    for (int i = 0; i < numIterations; ++i) {
    __syncthreads();
    const int inputOffset = i * SB;
    load_input_to_shared<FS, SB, padding_l>(inputFeature, inputOffset,
                                            sequenceLength, i,
                                            numIterations, false, tempInput);
    __syncthreads();
    if (inputOffset + tid < sequenceLength) {

        #pragma unroll
        for (int k = 0; k < FS; ++k) {
        const int filterOffset = batchIdx * numHeads * FS * sequenceLength
                                    + head * FS * sequenceLength
                                    + k * sequenceLength
                                    + i * SB + tid;
        filter[k] = weight[filterOffset];
        }

        scalar_t out = scalar_t(0.0);
        #pragma unroll
        for (int k = 0; k < FS; ++k) {
        out += filter[k] * tempInput[tid + k];
        }

        outputFeature[inputOffset + tid] = out;

    }
    }
}

template<int FS, int SB, int padding_l, typename scalar_t>
__global__
void dynamicconv_backward_kernel(
    const scalar_t* gradOutput, // B * C * T
    const scalar_t* input, // B * C * T
    const scalar_t* weight,
    int minibatch,
    int sequenceLength,
    int numFeatures,
    int numFiltersInBlock,
    int numHeads,
    scalar_t* gradWeight,
    scalar_t* gradInput) { // B * H * k * T

    assert(blockDim.x == SB);

    // each block operates on a single batch and filter head
    const int tid = threadIdx.x;
    const int batchIdx = blockIdx.x;
    const int headIdx = blockIdx.y;
    const int chunkIdx = blockIdx.z;

    const int numChunks = divUp<int, int>(sequenceLength, SB);
    const int inputOffset = chunkIdx * SB;

    // initialize shared memory for output gradient and input
    __shared__ scalar_t tempGradOutput[SB + FS];
    __shared__ scalar_t tempInput[SB + FS];
    const int padding = FS - padding_l - 1;

    zeroSharedMem<FS, SB, padding>(tempGradOutput);
    zeroSharedMem<FS, SB, padding_l>(tempInput);

    // initialize local filter and weight gradient sum arrays
    scalar_t tempGradSum[FS];
    scalar_t bfilter[FS];
    for (int k = 0; k < FS; ++k) {
    tempGradSum[k] = scalar_t(0.0);

    int idxOffset = inputOffset + tid + k - padding;
    if (idxOffset >= 0 && idxOffset < sequenceLength) {
        int bfilterOffset = batchIdx * numHeads * FS * sequenceLength
                            + headIdx * FS * sequenceLength
                            + (FS - k  - 1) * sequenceLength
                            + idxOffset;
        bfilter[k] = weight[bfilterOffset];
    } else {
        bfilter[k] = scalar_t(0.0);
    }
    }


    // iterate over filter block
    for (int featureIdx = 0; featureIdx < numFiltersInBlock; ++featureIdx) {
    __syncthreads();

    // load input and output gradient for this channel and chunk
    const int IOOffset = batchIdx * numFeatures * sequenceLength
                            + (headIdx * numFiltersInBlock + featureIdx) * sequenceLength;
    const scalar_t* inputFeature = &input[IOOffset];
    const scalar_t* gradOutputFeature = &gradOutput[IOOffset];
    scalar_t* gradInputFeature = &gradInput[IOOffset];

    load_input_to_shared<FS, SB, padding>(gradOutputFeature, inputOffset,
                                            sequenceLength, chunkIdx,
                                            numChunks, true, tempGradOutput);
    load_input_to_shared<FS, SB, padding_l>(inputFeature, inputOffset,
                                            sequenceLength, chunkIdx,
                                            numChunks, true, tempInput);
    __syncthreads();
    
    // sum input and weight gradients
    scalar_t out = scalar_t(0.0);
    #pragma unroll
    for (int k = 0; k < FS; ++k) {
        tempGradSum[k] += tempInput[tid + k] * tempGradOutput[tid + padding];
        out += bfilter[k] * tempGradOutput[tid + k];
    }
    
    if (inputOffset + tid < sequenceLength) {
        gradInputFeature[inputOffset + tid] = out;
    }
    }

    const int gradOffset = batchIdx * numHeads * FS * sequenceLength
                + headIdx * FS * sequenceLength;
    scalar_t *gradWeightFeature = &gradWeight[gradOffset];

    // write weight gradient
    if (inputOffset + tid < sequenceLength) {
    for (int k = 0; k < FS; ++k) {
        const int outputOffset = k * sequenceLength + inputOffset + tid;
        gradWeightFeature[outputOffset] = tempGradSum[k];
    }
    }
}
