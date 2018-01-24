/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <device_functions.h>

/*
Parallel sum reduction using shared memory
- takes log(n) steps for n input elements
- uses n/2 threads
- only works for power-of-2 arrays

This version adds multiple elements per thread sequentially.  This reduces the overall
cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
(Brent's Theorem optimization)

See the CUDA SDK "reduction" sample for more information.
*/

template <unsigned int blockSize>
__device__ void
reduceBlock(volatile float *sdata, float mySum, int *sid, int myId, const unsigned int tid, int *dev_loc, int *dev_cid)
{
	sdata[tid] = mySum;
	sid[tid] = myId;
	__syncthreads();

	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{			
			if(sdata[tid] > sdata[tid + 256]){
				sdata[tid] = sdata[tid + 256];
				sid[tid] = sid[tid + 256];
			}
			if(sdata[tid] == sdata[tid + 256] && sid[tid + 256] < sid[tid]){
				sid[tid] = sid[tid + 256];
			}

			//sdata[tid] = mySum = mySum + sdata[tid + 256];
			//printf("tid %3d mySum %5f\n",tid,mySum);
		}

		__syncthreads();

	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			if(sdata[tid] > sdata[tid + 128]){
				sdata[tid] = sdata[tid + 128];
				sid[tid] = sid[tid + 128];
			}
			if(sdata[tid] == sdata[tid + 128] && sid[tid + 128] < sid[tid]){
				sid[tid] = sid[tid + 128];
			}
			//sdata[tid] = mySum = mySum + sdata[tid + 128];
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			if(sdata[tid] > sdata[tid + 64]){
				sdata[tid] = sdata[tid + 64];
				sid[tid] = sid[tid + 64];
			}
			if(sdata[tid] == sdata[tid + 64] && sid[tid + 64] < sid[tid]){
				sid[tid] = sid[tid + 64];
			}
			//sdata[tid] = mySum = mySum + sdata[tid + 64];
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		if (blockSize >= 64)
		{
			if(sdata[tid] > sdata[tid + 32]){
				sdata[tid] = sdata[tid + 32];
				sid[tid] = sid[tid + 32];
				//printf("tid = %2d -- mySum = %f myId = %2d\n",tid,sdata[tid],sid[tid]);
			}
			if(sdata[tid] == sdata[tid + 32] && sid[tid + 32] < sid[tid]){
				sid[tid] = sid[tid + 32];
			}
			//printf("tid = %2d -- mySum = %f myId = %2d\n",tid,sdata[tid],sid[tid]);
			//sdata[tid] = mySum = mySum + sdata[tid + 32];
		}
		__syncthreads();
		if (blockSize >= 32)
		{
			if(sdata[tid] > sdata[tid + 16]){
				sdata[tid] = sdata[tid + 16];
				sid[tid] = sid[tid + 16];
			}
			if(sdata[tid] == sdata[tid + 16] && sid[tid + 16] < sid[tid]){
				sid[tid] = sid[tid + 16];
			}
			//printf("tid = %2d -- mySum = %f myId = %2d\n",tid,sdata[tid],sid[tid]);
			//sdata[tid] = mySum = mySum + sdata[tid + 16];
		}
		__syncthreads();
		if (blockSize >= 16)
		{
			if(sdata[tid] > sdata[tid + 8]){
				sdata[tid] = sdata[tid + 8];
				sid[tid] = sid[tid + 8];
			}
			if(sdata[tid] == sdata[tid + 8] && sid[tid + 8] < sid[tid]){
				sid[tid] = sid[tid + 8];
			}
			//sdata[tid] = mySum = mySum + sdata[tid + 8];
		}
		__syncthreads();
		if (blockSize >= 8)
		{

			if(sdata[tid] > sdata[tid + 4]){
				sdata[tid] = sdata[tid + 4];
				sid[tid] = sid[tid + 4];
			}
			if(sdata[tid] == sdata[tid + 4] && sid[tid + 4] < sid[tid]){
				sid[tid] = sid[tid + 4];
			}
			//sdata[tid] = mySum = mySum + sdata[tid + 4];
		}
		__syncthreads();
		if (blockSize >= 4)
		{
			if(sdata[tid] > sdata[tid + 2]){
				sdata[tid] = sdata[tid + 2];
				sid[tid] = sid[tid + 2];
			}
			if(sdata[tid] == sdata[tid + 2] && sid[tid + 2] < sid[tid]){
				sid[tid] = sid[tid + 2];
			}
			//sdata[tid] = mySum = mySum + sdata[tid + 2];
		}
		__syncthreads();
		if (blockSize >= 2)
		{
			if(sdata[tid] > sdata[tid + 1]){
				sdata[tid] = sdata[tid + 1];
				sid[tid] = sid[tid + 1];
			}
			if(sdata[tid] == sdata[tid + 1] && sid[tid + 1] < sid[tid]){
				sid[tid] = sid[tid + 1];
			}
			//sdata[tid] = mySum = mySum + sdata[tid + 1];
		}
		__syncthreads();
	}
}

template <unsigned int blockSize, bool nIsPow2>
__device__ void
reduceBlocks(const float *g_idata, float *g_odata, unsigned int n, int *dev_loc, int *dev_cid,int rank)
{
	extern __shared__ float sdata[];
	__shared__ int sid[512];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	float mySum = 999999;
	int myId = i;
	int shift = n * rank;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread

	while (i < n)
	{
		if(g_idata[i + shift] < mySum && g_idata[i + shift] > 0){
			mySum = g_idata[i + shift];
			myId = dev_cid[i + shift];
		}

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays

		if (nIsPow2 || i + blockSize < n){
			//printf("2tid = %2d -- g_idata = %f g_iid = %2d mySum = %f myId = %2d\n",i,g_idata[i + blockSize],dev_cid[i + blockSize],mySum,myId);
			if(g_idata[i + blockSize + shift] < mySum && g_idata[i + blockSize + shift] > 0){
				mySum = g_idata[i + blockSize + shift];
				myId = dev_cid[i + blockSize + shift];
			}
		}

		i += gridSize;
	}

	__syncthreads();

	// do reduction in shared mem
	reduceBlock<blockSize>(sdata, mySum, sid, myId, tid, dev_loc, dev_cid);
	__syncthreads();
	// write result for this block to global mem
	if (tid == 0){
		g_odata[blockIdx.x] = sdata[0];
		dev_loc[blockIdx.x] = sid[0];
	} 
	__syncthreads();
}


// Global variable used by reduceSinglePass to count how many blocks have finished
__device__ unsigned int retirementCount = 0;

cudaError_t setRetirementCount(int retCnt)
{
	return cudaMemcpyToSymbol(retirementCount, &retCnt, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
}

// This reduction kernel reduces an arbitrary size array in a single kernel invocation
// It does so by keeping track of how many blocks have finished.  After each thread
// block completes the reduction of its own block of data, it "takes a ticket" by
// atomically incrementing a global counter.  If the ticket value is equal to the number
// of thread blocks, then the block holding the ticket knows that it is the last block
// to finish.  This last block is responsible for summing the results of all the other
// blocks.
//
// In order for this to work, we must be sure that before a block takes a ticket, all
// of its memory transactions have completed.  This is what __threadfence() does -- it
// blocks until the results of all outstanding memory transactions within the
// calling thread are visible to all other threads.
//
// For more details on the reduction algorithm (notably the multi-pass approach), see
// the "reduction" sample in the CUDA SDK.
template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceSinglePass(const float *g_idata, float *g_odata, unsigned int n, int *dev_loc, int *dev_cid, int rank)
{
		const unsigned int tid = threadIdx.x;
	//
	// PHASE 1: Process all inputs assigned to this block
	//
	reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n, dev_loc, dev_cid, rank);
	__syncthreads();
	//
	// PHASE 2: Last block finished will process all partial sums
	//
	
	if (gridDim.x > 1)
	{
		
		__shared__ bool amLast;
		extern float __shared__ smem[];
		//__shared__ int sid[512];

		// wait until all outstanding memory instructions in this thread are finished
		__threadfence();

		// Thread 0 takes a ticket
		if (tid == 0)
		{
			unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
			// If the ticket ID is equal to the number of blocks, we are the last block!
			amLast = (ticket == gridDim.x - 1);
		}

		__syncthreads();

		// The last block sums the results of all other blocks
		if (amLast)
		{
			int i = tid;
			float mySum = 99999;
			int myId = 0;

			while (i < gridDim.x)
			{
				
				//printf("mySum = %f tid =%d\n",mySum,i);

				if(g_odata[i] < mySum && g_odata[i] > 0){

					mySum = g_odata[i];
					myId = dev_loc[i];
				}
				if(g_odata[i] == mySum && dev_loc[i] < myId){

					myId = dev_loc[i];
				}
				//printf("ID= %d  data=%f loc=%d\n",i,g_odata[i],dev_loc[i]);
				//mySum += g_odata[i];
				i += blockSize;
			}
			__syncthreads();
			if(tid == 0){
				float min = 99999;
				for(int i = 0; i < gridDim.x; i++){
					if(g_odata[i] < min && g_odata[i] > 0){

						mySum = g_odata[i];
						myId = dev_loc[i];
					}
					if(g_odata[i] == mySum && dev_loc[i] < myId){
						myId = dev_loc[i];
					}
				}
			}
			
			//reduceBlock<blockSize>(smem, mySum, sid, myId, tid, dev_loc, dev_cid);
			__syncthreads();
			if (tid == 0)
			{
				g_odata[0] = mySum;
				dev_loc[0] = myId;
				retirementCount = 0;
			}
		}
	}
}

bool isPow2(unsigned int x)
{
	return ((x&(x - 1)) == 0);
}


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
extern "C"
void reduceSinglePass(int size, int threads, int blocks, float *d_idata, float *d_odata, int *dev_loc, int *dev_cid, int rank)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = threads * sizeof(float);

	// choose which of the optimized versions of reduction to launch
	if (isPow2(size))
	{
		switch (threads)
		{
		case 512:
			//printf("512\n");
			reduceSinglePass<512, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case 256:
		//printf("256\n");
			reduceSinglePass<256, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case 128:
		//printf("128\n");
			reduceSinglePass<128, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case 64:
		//printf("64\n");
			reduceSinglePass< 64, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case 32:
		//printf("32\n");
			reduceSinglePass< 32, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case 16:
		//printf("16\n");
			reduceSinglePass< 16, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case  8:
		//printf("8\n");
			reduceSinglePass<  8, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case  4:
		//printf("4\n");
			reduceSinglePass<  4, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case  2:
		//printf("2\n");
			reduceSinglePass<  2, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case  1:
		//printf("1\n");
			reduceSinglePass<  1, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;
		}
	}
	else
	{
		switch (threads)
		{
		case 512:
		//printf("5122\n");
			reduceSinglePass<512, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case 256:
		//printf("2562\n");
			reduceSinglePass<256, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case 128:
		//printf("1282\n");
			reduceSinglePass<128, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case 64:
		//printf("642\n");
			reduceSinglePass< 64, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case 32:
		//printf("322\n");
			reduceSinglePass< 32, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case 16:
		//printf("162\n");
			reduceSinglePass< 16, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case  8:
		//printf("82\n");
			reduceSinglePass<  8, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case  4:
		//printf("42\n");
			reduceSinglePass<  4, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case  2:
		//printf("22\n");
			reduceSinglePass<  2, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;

		case  1:
		//printf("12\n");
			reduceSinglePass<  1, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size, dev_loc, dev_cid, rank);
			break;
		}
	}
}

#endif // #ifndef _REDUCE_KERNEL_H_
