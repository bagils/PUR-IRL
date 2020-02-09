
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <cmath>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

using namespace std;

#define swap(a, b) {float *hold = a; a = b; b = hold;}
#define swapD(a, b) {double *hold = a; a = b; b = hold;}
#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif
#define	kBlockSize	32

/**
 * Matrix multiplication and addition on the device: {@code result = (mulMatrix * fxnMatrix) + addMatrix}
 * {@code widthMul} is {@code mulMatrix}'s width and {@code widthFxn} is {@code fxnMatrix}'s width
 * {@code result} and {@code addMatrix} have height {@code heightMul and width {@code widthFxn}
 */
template <int BLOCK_SIZE, class T> __global__ void
matrixMulCUDA (T *result, T *mulMatrix, T *fxnMatrix, T *addMatrix, int widthMul, int heightMul, int widthFxn, int heightFxn)
{
	// Block index
	int		blockX = blockIdx.x * BLOCK_SIZE;
	int		blockY = blockIdx.y * BLOCK_SIZE;
	
	// Thread index
	int		threadX = threadIdx.x;
	int		threadY = threadIdx.y;
	int		maxFxn = widthFxn * heightFxn;
	int		maxMul = widthMul * heightMul;
	bool	deadFxnX, deadMulY;
	bool	deadResX = deadFxnX = (blockX + threadX) >= widthFxn;	// Are these within the valid range of the results matrix?
	bool	deadResY = deadMulY = (blockY + threadY) >= heightMul;
	
	// Offsets from upper left corner to the cells of interest for this thread
	int	mulOffset = (widthMul * threadY) + threadX;
	int	fxnOffset = (widthFxn * threadY) + threadX;
	
	// Index of the first sub-matrix of mulMatrix processed by the block
	int	mulBegin = widthMul * blockY;
	
	// Index of the first sub-matrix of mulMatrix NOT processed by the block
	int	mulEnd = mulBegin + widthMul;
	
	// Step size used to iterate through the sub-matrices of mulMatrix
	int	mulStep = BLOCK_SIZE;
	
	// Index of the first sub-matrix of fxnMatrix processed by the block
	int	fxnBegin = blockX;
	
	// Step size used to iterate through the sub-matrices of fxnMatrix
	int	fxnStep = BLOCK_SIZE * widthFxn;
	
	// resultSub is used to store the element of the block sub-matrix
	// that is computed by the thread
	T	resultSub = 0;
	
	// Loop over all the sub-matrices of mulMatrix and fxnMatrix
	// required to compute the block sub-matrix
	for (int mulUpperLeft = mulBegin, fxnUpperLeft = fxnBegin; mulUpperLeft < mulEnd; mulUpperLeft += mulStep, fxnUpperLeft += fxnStep)
	{
		__shared__ T	mulSub[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ T	fxnSub[BLOCK_SIZE][BLOCK_SIZE];
		
//		printf ("Loading data: Block[%d, %d]: Thread[%d, %d]\n", blockX, blockY, threadX, threadY);
		
		// Load the matrices from device memory to shared memory; each thread loads
		// one element of each matrix.	If block size doesn't evenly divide into array 
		// size, will be attempting to read past the edge.	Make those 0, so nothing 
		// extra gets added
		int mulPos = mulUpperLeft + mulOffset;
		if (!deadMulY && (mulPos < maxMul))
			mulSub[threadY][threadX] = mulMatrix[mulPos];
		else
			mulSub[threadY][threadX] = 0.0f;
		
		int fxnPos = fxnUpperLeft + fxnOffset;
		if (!deadFxnX && (fxnPos < maxFxn))
			fxnSub[threadY][threadX] = fxnMatrix[fxnPos];
		else
			fxnSub[threadY][threadX] = 0.0f;
		
		// Synchronize to make sure the matrices are loaded
		__syncthreads ();
		
//		printf ("Loaded data: Block[%d, %d]: Thread[%d, %d]\n", blockX, blockY, threadX, threadY);
		
		// Multiply the two matrices together; each thread computes one element
		// of the block sub-matrix
#pragma unroll
		
		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			resultSub += mulSub[threadY][k] * fxnSub[k][threadX];
		}
		
		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of mulMatrix and fxnMatrix in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory; each thread writes one element
	if (!deadResX && !deadResY)
	{
		int	pos = ((blockY + threadY) * widthFxn) + blockX + threadX;	// (numRows * width) + numCols
		result[pos] = resultSub + addMatrix[pos];
	}
}


/**
 * Compare the contents of two T[].  If any value of {@code first} differs from {@code second} 
 * by {@code epsilon} or more, return false.  Else return true
 * 
 * @param first		double[] holding values to test, same length as {@code second}
 * @param second	double[] holding values to test, same length as {@code first}
 * @param size		Length of both {@code first} and {@code second}
 * @param epsilon	Test value.  All matching elements of {@code first} and {@code second} must 
 * differ by less than this
 * @param diffCount	Count of values in {@code first} that differ from {@code second} by {@code epsilon} or more, 
 */
template <int BLOCK_SIZE, class T> __global__ void
compare (T *first, T *second, size_t size, T epsilon, size_t *diffCount)
{
	size_t	blockX = blockIdx.x * BLOCK_SIZE;
	size_t	threadX = threadIdx.x;
	size_t	pos = blockX + threadX;
	
	if (pos < size)
	{
		T	value = first[pos] - second[pos];
		
		if (value < 0)
			value = -value;
		
		if (value > epsilon)
			++(*diffCount);
	}
}


/**
 * Copy the contents of {@code source} into {@code target}
 * 
 * @param source	float[] / double[] to read from.  Must not be null, and length {@code theSize}
 * @param target	float[] / double[] to write to.  Must not be null, and length {@code theSize}
 * @param theSize	Size of both arrays
 */
template <int BLOCK_SIZE, class T> __global__ void
copy (const T *source, T *target, int theSize)
{
	// Block index
	int	blockX = blockIdx.x * BLOCK_SIZE;
	
	// Thread index
	int	threadX = threadIdx.x;
	
	//  Source location
	int	pos = blockX + threadX;
	
	if (pos < theSize)
		target[pos] = source[pos];
}


/**
 * Write the transpose of {@code source}, which is in column:row order (i.e. laid out with the first 
 * column being in positions 0 - {@code numRows - 1} into {@code target} in row:column order
 * 
 * @param source	float[] / double[] to read from.  Must not be null, and length {@code numRows * numCols}
 * @param target	float[] / double[] to write to.  Must not be null, and length {@code numRows * numCols}
 * @param numRows	Number of rows in {@code source}, will be number of cols in {@code target}
 * @param numCols	Number of cols in {@code source}, will be number of rows in {@code target}
 */
template <int BLOCK_SIZE, class T, class U> __global__ void
transposeCR (const T *source, U *target, int numRows, int numCols)
{
	// Block index
	int	blockX = blockIdx.x * BLOCK_SIZE;
	int	blockY = blockIdx.y * BLOCK_SIZE;
	
	// Thread index
	int	threadX = threadIdx.x;
	int	threadY = threadIdx.y;
	
	//  Source location
	int	row = blockY + threadY;
	int	col = blockX + threadX;
	
	if ((row < numRows) && (col < numCols))
	{
		int	readPos = (col * numRows) + row;
		int	writePos = (row * numCols) + col;
		
		target[writePos] = (U) source[readPos];
	}
}


/**
 * Write the transpose of {@code source}, which is in row:column order (i.e. laid out with the first 
 * row being in positions 0 - {@code numCols - 1} into {@code target} in column:row order<br>
 * Runs on the Device, on arrays on the device
 * 
 * @param source	float[] / double[] to read from.  Must not be null, and length {@code numRows * numCols}
 * @param target	float[] / double[] to write to.  Must not be null, and length {@code numRows * numCols}
 * @param numRows	Number of rows in {@code source}, will be number of cols in {@code target}
 * @param numCols	Number of cols in {@code source}, will be number of rows in {@code target}
 */
template <int BLOCK_SIZE, class T, class U> __global__ void
transposeRC (const T *source, U *target, int numRows, int numCols)
{
	// Block index
	int	blockX = blockIdx.x * BLOCK_SIZE;
	int	blockY = blockIdx.y * BLOCK_SIZE;
	
	// Thread index
	int	threadX = threadIdx.x;
	int	threadY = threadIdx.y;
	
	//  Source location
	int	row = blockY + threadY;
	int	col = blockX + threadX;
	
	if ((row < numRows) && (col < numCols))
	{
		int	readPos = (row * numCols) + col;
		int	writePos = (col * numRows) + row;
		
		target[writePos] = (U) (source[readPos]);
	}
}


/**
 * Write the transpose of {@code source} into {@code target}, on the Host
 * 
 * @param source	float / double[] to read from.  Must not be null, and length {@code numRows * numCols}
 * @param target	float / double[] to write to.  Must not be null, and length {@code numRows * numCols}
 * @param numRows	Number of rows in {@code source}, will be number of cols in {@code target}
 * @param numCols	Number of cols in {@code source}, will be number of rows in {@code target}
 */
template <class T, class U> void transposeRCHost (const T source[], U target[], int numRows, int numCols)
{
	int	i, readPos = 0;
	
	for (i = 0; i < numRows; ++i)
	{
		int	j, writePos = i;
		
		for (j = 0; j < numCols; ++j)
		{
			target[writePos] = (U) (source[readPos]);
			++readPos;
			writePos += numRows;
		}
	}
}


/**
 *	Chose which GPU to use, exiting on any error
 */
void cSetDevice (int whichGPU, int line)
{
	cudaError_t	error = cudaSetDevice (whichGPU);

	if (error != cudaSuccess)
	{
		printf ("cudaSetDevice (%d) returned error %s (code %d), line (%d)\n", whichGPU, cudaGetErrorString (error), error, line);
		exit (EXIT_FAILURE);
	}
//	printf ("cudaSetDevice (%d) succeeded\n", whichGPU);
}


/**
 *	Chose which GPU to use, exiting on any error
 */
void cuBlasSetup (cublasHandle_t *handle, int whichGPU, int line)
{
	cSetDevice (whichGPU, line);
	
	cublasStatus_t	stat = cublasCreate (handle);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf ("CUBLAS initialization failed, cublasCreate () returned error %s (code %d), line (%d)\n", _cudaGetErrorEnum (stat), stat, line);
		exit (EXIT_FAILURE);
	}
}


/**
 * Matrix multiplication and addition on the device: {@code result = (mulMatrix * fxnMatrix) + addMatrix}
 * {@code widthMul} is {@code mulMatrix}'s width and {@code widthFxn} is {@code fxnMatrix}'s width
 * {@code result} and {@code addMatrix} have height {@code heightMul and width {@code widthFxn}
 */
void cuBlasMul (cublasHandle_t handle, float *result, float *mulMatrix, float *fxnMatrix, 
				float *addMatrix, int colsMul, int rowsMul, int colsFxn, int rowsFxn, int line)
{
	cublasStatus_t	stat;
	const float		alpha = 1.0f;
	const float		beta = 1.0f;
	int				arraySize = colsFxn * rowsFxn;
	
	// Replace contents of result with addMatrix, so can use the beta add, rather than a separate operation
	copy<1024, float><<< (arraySize + 1023) / 1024, 1024 >>>(addMatrix, result, arraySize);
	cudaDeviceSynchronize ();
	
	stat = cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, rowsMul, colsFxn, rowsFxn, &alpha, 
						mulMatrix, rowsMul, fxnMatrix, rowsFxn, &beta, result, rowsMul);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf ("cublasSgemm () returned error %s (code %d), line (%d)\n", _cudaGetErrorEnum (stat), stat, line);
		exit (EXIT_FAILURE);
	}
}


/**
 * Matrix multiplication and addition on the device: {@code result = (mulMatrix * fxnMatrix) + addMatrix}
 * {@code widthMul} is {@code mulMatrix}'s width and {@code widthFxn} is {@code fxnMatrix}'s width
 * {@code result} and {@code addMatrix} have height {@code heightMul and width {@code widthFxn}
 */
void cuBlasMul (cublasHandle_t handle, double *result, double *mulMatrix, double *fxnMatrix, 
				double *addMatrix, int colsMul, int rowsMul, int colsFxn, int rowsFxn, int line)
{
	cublasStatus_t	stat;
	const double	alpha = 1.0f;
	const double	beta = 1.0f;
	int				arraySize = colsFxn * rowsFxn;
	
	// Replace contents of result with addMatrix, so can use the beta add, rather than a separate operation
	copy<1024, double><<< (arraySize + 1023) / 1024, 1024 >>>(addMatrix, result, arraySize);
	stat = cublasDgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, rowsMul, colsFxn, rowsFxn, &alpha, 
						mulMatrix, rowsMul, fxnMatrix, rowsFxn, &beta, result, rowsMul);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf ("cublasDgemm () returned error %s (code %d), line (%d)\n", _cudaGetErrorEnum (stat), stat, line);
		exit (EXIT_FAILURE);
	}
}


/**
 *	Allocate cuda memory, exiting on any error
 */
void cMalloc (void **target, size_t theSize, const char *name, int line)
{
	cudaError_t	error = cudaMalloc (target, theSize);

	if (error != cudaSuccess)
	{
		printf ("cudaMalloc %s returned error %s (code %d), line (%d)\n", name, cudaGetErrorString (error), error, line);
		exit (EXIT_FAILURE);
	}
}


/**
 *	Copy to or from cuda memory, exiting on any error
 */
void cCopy (void *target, const void *source, size_t theSize, enum cudaMemcpyKind kind, const char *name, int line)
{
	cudaError_t	error = cudaMemcpy (target, source, theSize, kind);

	if (error != cudaSuccess)
	{
		printf ("cudaMemcpy %s returned error %s (code %d), line (%d)\n", name, cudaGetErrorString (error), error, line);
		exit (EXIT_FAILURE);
	}
}


cublasHandle_t *handles;

/**
 *	Routine to let us know the library really is loaded
 */
void initCuda (int numGPUs)
{
	handles = (cublasHandle_t *) malloc (numGPUs * sizeof(cublasHandle_t));
	
	for (int whichGPU = 0; whichGPU < numGPUs; ++whichGPU)
	{
		cuBlasSetup (handles + whichGPU, whichGPU, __LINE__);	// Create a cublasHandle_t for each GPU
	}
}


bool convergeMatrixCudaD (double *fxnMatrixH, double *mulMatrixH, double *addMatrixH, double *results, 
						  int numRows, int numCols, int maxIter, int whichGPU, double epsilon)
{
	cSetDevice (whichGPU, __LINE__);	// Assign code to the appropriate GPU
	
	size_t	arraySize = numRows * numCols;
	size_t	memSizeFxn = arraySize * sizeof(double);
	size_t	memSizeMul = numRows * numRows * sizeof(double);
	double	*mulMatrix, *fxnMatrix, *addMatrix, *resultHold;
	size_t	changeCountH;
	size_t	*changeCount;
	bool	converged = false;
	
	struct timeval startTV;
	gettimeofday (&startTV, NULL);
	
	// Allocate device memory
	cMalloc ((void **) &mulMatrix, memSizeMul, "mulMatrix", __LINE__);
	cMalloc ((void **) &fxnMatrix, memSizeFxn, "fxnMatrix", __LINE__);
	cMalloc ((void **) &addMatrix, memSizeFxn, "addMatrix", __LINE__);
	cMalloc ((void **) &resultHold, memSizeFxn, "resultHold", __LINE__);
	cMalloc ((void **) &changeCount, sizeof(size_t), "changeCount", __LINE__);
	
	// copy host memory to device
	cCopy (mulMatrix, mulMatrixH, memSizeMul, cudaMemcpyHostToDevice, "(mulMatrix, mulMatrixH)", __LINE__);
	cCopy (fxnMatrix, fxnMatrixH, memSizeFxn, cudaMemcpyHostToDevice, "(fxnMatrix, fxnMatrixH)", __LINE__);
	cCopy (addMatrix, addMatrixH, memSizeFxn, cudaMemcpyHostToDevice, "(addMatrix, addMatrixH)", __LINE__);
	
	// Setup execution parameters
	dim3	threads (kBlockSize, kBlockSize);
	dim3	grid ((numCols + kBlockSize - 1) / kBlockSize, (numRows + kBlockSize - 1) / kBlockSize);
	int		i, numBlocks = (arraySize + 1023) / 1024;
	
	for (i = 0; i < maxIter; ++i)
	{
		matrixMulCUDA<kBlockSize, double><<< grid, threads >>>(resultHold, mulMatrix, fxnMatrix, addMatrix, numRows, numRows, numCols, numRows);
		cudaDeviceSynchronize ();
		swapD(resultHold, fxnMatrix);	// Want fxnMatrix to always hold the final results when we exit
		changeCountH = 0;
		cCopy (changeCount, &changeCountH, sizeof(size_t), cudaMemcpyHostToDevice, "(changeCount, changeCountH)", __LINE__);
		compare<1024, double><<< numBlocks, 1024 >>>(fxnMatrix, resultHold, arraySize, epsilon, changeCount);
		cudaDeviceSynchronize ();
		cCopy (&changeCountH, changeCount, sizeof(size_t), cudaMemcpyDeviceToHost, "(changeCountH, changeCount)", __LINE__);
		if (changeCountH == 0)
		{
			converged = true;
			break;
		}
	}
	
	struct timeval endTV;
	gettimeofday (&endTV, NULL);
	unsigned long	theStart = (1000000 * startTV.tv_sec) + startTV.tv_usec;
	unsigned long	done = (1000000 * endTV.tv_sec) + endTV.tv_usec;
	printf ("matrixMulCUDA (double) used %lu microseconds for %d iterations\n", done - theStart, i);
	
	transposeRC<kBlockSize, double, double><<< grid, threads >>>(fxnMatrix, resultHold, numRows, numCols);
	cCopy (results, resultHold, memSizeFxn, cudaMemcpyDeviceToHost, "(results, resultHold)", __LINE__);
	
	cudaFree (mulMatrix);
	cudaFree (fxnMatrix);
	cudaFree (addMatrix);
	cudaFree (resultHold);
	cudaFree (changeCount);
	
	return converged;
}


bool convergeMatrixCuBLASD (double *fxnMatrixH, double *mulMatrixH, double *addMatrixH, double *results, 
						    int numRows, int numCols, int maxIter, int whichGPU, double epsilon)
{
	cSetDevice (whichGPU, __LINE__);	// Assign code to the appropriate GPU
	cublasHandle_t	handle = handles[whichGPU];
	
	size_t	arraySize = numRows * numCols;
	size_t	memSizeFxn = arraySize * sizeof(double);
	size_t	memSizeMul = numRows * numRows * sizeof(double);
	double	*mulMatrix, *fxnMatrix, *addMatrix, *resultHold;
	size_t	changeCountH;
	size_t	*changeCount;
	bool	converged = false;
	
	struct timeval startTV;
	gettimeofday (&startTV, NULL);
	
	// Allocate device memory
	cMalloc ((void **) &mulMatrix, memSizeMul, "mulMatrix", __LINE__);
	cMalloc ((void **) &fxnMatrix, memSizeFxn, "fxnMatrix", __LINE__);
	cMalloc ((void **) &addMatrix, memSizeFxn, "addMatrix", __LINE__);
	cMalloc ((void **) &resultHold, memSizeFxn, "resultHold", __LINE__);
	cMalloc ((void **) &changeCount, sizeof(size_t), "changeCount", __LINE__);
	
	// copy host memory to device
	cCopy (mulMatrix, mulMatrixH, memSizeMul, cudaMemcpyHostToDevice, "(mulMatrix, mulMatrixH)", __LINE__);
	cCopy (fxnMatrix, fxnMatrixH, memSizeFxn, cudaMemcpyHostToDevice, "(fxnMatrix, fxnMatrixH)", __LINE__);
	cCopy (addMatrix, addMatrixH, memSizeFxn, cudaMemcpyHostToDevice, "(addMatrix, addMatrixH)", __LINE__);
	
	
	int		i, numBlocks = (arraySize + 1023) / 1024;
	
	for (i = 0; i < maxIter; ++i)
	{
		cuBlasMul (handle, resultHold, mulMatrix, fxnMatrix, addMatrix, numRows, numRows, numCols, numRows, __LINE__);
		cudaDeviceSynchronize ();
		swapD(resultHold, fxnMatrix);	// Want fxnMatrix to always hold the final results when we exit
		changeCountH = 0;
		cCopy (changeCount, &changeCountH, sizeof(size_t), cudaMemcpyHostToDevice, "(changeCount, changeCountH)", __LINE__);
		compare<1024, double><<< numBlocks, 1024 >>>(fxnMatrix, resultHold, arraySize, epsilon, changeCount);
		cudaDeviceSynchronize ();
		cCopy (&changeCountH, changeCount, sizeof(size_t), cudaMemcpyDeviceToHost, "(changeCountH, changeCount)", __LINE__);
		if (changeCountH == 0)
		{
			converged = true;
			break;
		}
	}
	
	struct timeval endTV;
	gettimeofday (&endTV, NULL);
	unsigned long	theStart = (1000000 * startTV.tv_sec) + startTV.tv_usec;
	unsigned long	done = (1000000 * endTV.tv_sec) + endTV.tv_usec;
	printf ("cuBlasMul (double) used %lu microseconds for %d iterations\n", done - theStart, i);
	
	cCopy (results, fxnMatrix, memSizeFxn, cudaMemcpyDeviceToHost, "(results, resultHold)", __LINE__);
	
	cudaFree (mulMatrix);
	cudaFree (fxnMatrix);
	cudaFree (addMatrix);
	cudaFree (resultHold);
	cudaFree (changeCount);
	
	return converged;
}


bool convergeMatrixCuda (float *fxnMatrixH, float *mulMatrixH, float *addMatrixH, float *results, 
						 int numRows, int numCols, int maxIter, int whichGPU, float epsilon)
{
	cSetDevice (whichGPU, __LINE__);	// Assign code to the appropriate GPU
	
	size_t	arraySize = numRows * numCols;
	size_t	memSizeFxn = arraySize * sizeof(float);
	size_t	memSizeMul = numRows * numRows * sizeof(float);
	float	*mulMatrix, *fxnMatrix, *addMatrix, *resultHold;
	size_t	changeCountH;
	size_t	*changeCount;
	bool	converged = false;
	
	struct timeval startTV;
	gettimeofday (&startTV, NULL);
	
	// Allocate device memory
	cMalloc ((void **) &mulMatrix, memSizeMul, "mulMatrix", __LINE__);
	cMalloc ((void **) &fxnMatrix, memSizeFxn, "fxnMatrix", __LINE__);
	cMalloc ((void **) &addMatrix, memSizeFxn, "addMatrix", __LINE__);
	cMalloc ((void **) &resultHold, memSizeFxn, "resultHold", __LINE__);
	cMalloc ((void **) &changeCount, sizeof(size_t), "changeCount", __LINE__);
	
	// copy host memory to device
	cCopy (mulMatrix, mulMatrixH, memSizeMul, cudaMemcpyHostToDevice, "(mulMatrix, mulMatrixH)", __LINE__);
	cCopy (fxnMatrix, fxnMatrixH, memSizeFxn, cudaMemcpyHostToDevice, "(fxnMatrix, fxnMatrixH)", __LINE__);
	cCopy (addMatrix, addMatrixH, memSizeFxn, cudaMemcpyHostToDevice, "(addMatrix, addMatrixH)", __LINE__);
	
	// Setup execution parameters
	dim3	threads (kBlockSize, kBlockSize);
	dim3	grid ((numCols + kBlockSize - 1) / kBlockSize, (numRows + kBlockSize - 1) / kBlockSize);
	int		i, numBlocks = (arraySize + 1023) / 1024;
	
	for (i = 0; i < maxIter; ++i)
	{
		matrixMulCUDA<kBlockSize, float><<< grid, threads >>>(resultHold, mulMatrix, fxnMatrix, addMatrix, numRows, numRows, numCols, numRows);
		cudaDeviceSynchronize ();
		swap(resultHold, fxnMatrix);	// Want fxnMatrix to always hold the final results when we exit
		changeCountH = 0;
		cCopy (changeCount, &changeCountH, sizeof(size_t), cudaMemcpyHostToDevice, "(changeCount, changeCountH)", __LINE__);
		compare<1024, float><<< numBlocks, 1024 >>>(fxnMatrix, resultHold, arraySize, epsilon, changeCount);
		cudaDeviceSynchronize ();
		cCopy (&changeCountH, changeCount, sizeof(size_t), cudaMemcpyDeviceToHost, "(changeCountH, changeCount)", __LINE__);
		if (changeCountH == 0)
		{
			converged = true;
			break;
		}
	}
	
	struct timeval endTV;
	gettimeofday (&endTV, NULL);
	unsigned long	theStart = (1000000 * startTV.tv_sec) + startTV.tv_usec;
	unsigned long	done = (1000000 * endTV.tv_sec) + endTV.tv_usec;
	printf ("matrixMulCUDA (float) used %lu microseconds for %d iterations\n", done - theStart, i);
	
	transposeRC<kBlockSize, float, float><<< grid, threads >>>(fxnMatrix, resultHold, numRows, numCols);
	cCopy (results, resultHold, memSizeFxn, cudaMemcpyDeviceToHost, "(results, resultHold)", __LINE__);
	
	cudaFree (mulMatrix);
	cudaFree (fxnMatrix);
	cudaFree (addMatrix);
	cudaFree (resultHold);
	cudaFree (changeCount);
	
	return converged;
}


bool convergeMatrixCuBLAS (float *fxnMatrixH, float *mulMatrixH, float *addMatrixH, float *results, 
						   int numRows, int numCols, int maxIter, int whichGPU, float epsilon)
{
	cublasHandle_t	handle = handles[whichGPU];
	
	size_t	arraySize = numRows * numCols;
	size_t	memSizeFxn = arraySize * sizeof(float);
	size_t	memSizeMul = numRows * numRows * sizeof(float);
	float	*mulMatrix, *fxnMatrix, *addMatrix, *resultHold;
	size_t	changeCountH;
	size_t	*changeCount;
	bool	converged = false;
	
	struct timeval startTV;
	gettimeofday (&startTV, NULL);
	
	// Allocate device memory
	cMalloc ((void **) &mulMatrix, memSizeMul, "mulMatrix", __LINE__);
	cMalloc ((void **) &fxnMatrix, memSizeFxn, "fxnMatrix", __LINE__);
	cMalloc ((void **) &addMatrix, memSizeFxn, "addMatrix", __LINE__);
	cMalloc ((void **) &resultHold, memSizeFxn, "resultHold", __LINE__);
	cMalloc ((void **) &changeCount, sizeof(size_t), "changeCount", __LINE__);
	
	// copy host memory to device
	cCopy (mulMatrix, mulMatrixH, memSizeMul, cudaMemcpyHostToDevice, "(mulMatrix, mulMatrixH)", __LINE__);
	cCopy (fxnMatrix, fxnMatrixH, memSizeFxn, cudaMemcpyHostToDevice, "(fxnMatrix, fxnMatrixH)", __LINE__);
	cCopy (addMatrix, addMatrixH, memSizeFxn, cudaMemcpyHostToDevice, "(addMatrix, addMatrixH)", __LINE__);
	
	
	int		i, numBlocks = (arraySize + 1023) / 1024;
	
	for (i = 0; i < maxIter; ++i)
	{
		cuBlasMul (handle, resultHold, mulMatrix, fxnMatrix, addMatrix, numRows, numRows, numCols, numRows, __LINE__);
		cudaDeviceSynchronize ();
		swap(resultHold, fxnMatrix);	// Want fxnMatrix to always hold the final results when we exit
		changeCountH = 0;
		cCopy (changeCount, &changeCountH, sizeof(size_t), cudaMemcpyHostToDevice, "(changeCount, changeCountH)", __LINE__);
		compare<1024, float><<< numBlocks, 1024 >>>(fxnMatrix, resultHold, arraySize, epsilon, changeCount);
		cudaDeviceSynchronize ();
		cCopy (&changeCountH, changeCount, sizeof(size_t), cudaMemcpyDeviceToHost, "(changeCountH, changeCount)", __LINE__);
		if (changeCountH == 0)
		{
			converged = true;
			break;
		}
	}
	
	struct timeval endTV;
	gettimeofday (&endTV, NULL);
	unsigned long	theStart = (1000000 * startTV.tv_sec) + startTV.tv_usec;
	unsigned long	done = (1000000 * endTV.tv_sec) + endTV.tv_usec;
	printf ("cuBlasMul (float) used %lu microseconds for %d iterations\n", done - theStart, i);
	
	cCopy (results, fxnMatrix, memSizeFxn, cudaMemcpyDeviceToHost, "(results, resultHold)", __LINE__);
	
	cudaFree (mulMatrix);
	cudaFree (fxnMatrix);
	cudaFree (addMatrix);
	cudaFree (resultHold);
	cudaFree (changeCount);
	
	return converged;
}


template <class T> void dumpMatrix (T *matrix, const char *title, int numRows, int numCols)
{
	printf ("Row Major %s: ", title);
	
	for (int i = 0; (i < 10) && (i < numCols); ++i)
		printf ("%lf\t", (double) matrix[i]);
	printf ("\n");
	
	for (int i = 0; (i < 10) && (i < numCols); ++i)
		printf ("%lf\t", (double) matrix[i + numRows]);
	printf ("\nCol Major %s: ", title);
	
	for (int i = 0; (i < 10) && (i < numRows); ++i)
		printf ("%lf\t", (double) matrix[i]);
	printf ("\n");
	
	for (int i = 0; (i < 10) && (i < numRows); ++i)
		printf ("%lf\t", (double) matrix[i + numCols]);
	printf ("\n");
}


	/**
	 * Compare results from two different matrix multiplications 
	 * 
	 * @param first		Array that's assumed to have the correct answers
	 * @param second	Array whose values we are testing
	 * @param numRows	Number of rows in {@code first} and {@code second}
	 * @param numCols	GPU which produced the results, which will tell us what method was used
	 * @param whichGPU	GPU which produced the results, which will tell us what method was used
	 * @param maxIter	Maximum number of iterations allowed
	 * @param whichRun	Which run we're executing
	 * @param epsilon	Allowed difference between values
	 */
template <class T> void compareMatrix (T *first, T *second, int numRows, int numCols, int maxIter, int whichGPU, int whichRun, T epsilon)
{
	int	numErrors = 0;
	int	numZero = 0;
	int	numNaN = 0;
	int	numInf = 0;
	int	pos = 0;
	
	for (int i = 0; i < numRows; ++i)
	{
		for (int j = 0; j < numCols; ++j, ++pos)
		{
			T	fValue = first[pos];
			T	sValue = second[pos];
			T	value = fValue - sValue;
			
			if (value < 0.0)
				value = -value;
			
			if (value > epsilon)
				++numErrors;
			else if (fValue == 0.0)
				++numZero;
			else if (isnan (fValue) || isnan (sValue))
				++numNaN;
			else if (isinf (fValue) || isinf (sValue))
				++numInf;
		}
	}
	
	printf ("For run %d, GPU %d, %d max iterations, there were %d zero values, %d NaN, %d Inf, and %d errors out of %d values\nFirst results: ", 
			whichRun, whichGPU, maxIter, numZero, numNaN, numInf, numErrors, pos);
	
	printf ("");
	for (int i = 0; (i < 10) && (i < numCols); ++i)
		printf ("%10.4f\t", first[i + numCols]);
	printf ("\nSecond results: ");
	
	for (int i = 0; (i < 10) && (i < numCols); ++i)
		printf ("%10.4f\t", second[i + numCols]);
	printf ("\n");
}


template <class T> void referenceSolve (T *fxnMatrix, T *mulMatrix, T *addMatrix, T *results, int numRows, int numCols)
{
	int	writePos = 0;
	
	for (int i = 0; i < numRows; ++i)
	{
		T	*mulRow = mulMatrix + (i * numRows);
		
		for (int j = 0; j < numCols; ++j)
		{
			T	value = addMatrix[writePos];
			T	*fxnCol = fxnMatrix + j;
			
			for (int k = 0; k < numRows; ++k)
			{
				value += mulRow[k] * fxnCol[k * numCols];
			}
			
			results[writePos] = value;
			++writePos;
		}
	}
}


/**
 * Done with data in column major order, rather than row major order
 */
template <class T> void referenceSolveCol (T *fxnMatrix, T *mulMatrix, T *addMatrix, T *results, int numRows, int numCols)
{
	int	writePos = 0;
	
	for (int i = 0; i < numCols; ++i)
	{
		T	*fxnCol = fxnMatrix + (i * numRows);
		
		for (int j = 0; j < numRows; ++j)
		{
			T	value = addMatrix[writePos];
			T	*mulRow = mulMatrix + j;
			
			for (int k = 0; k < numRows; ++k)
			{
				value += mulRow[k * numRows] * fxnCol[k];
			}
			
			results[writePos] = value;
			++writePos;
		}
	}
}


void testMulD (int numRows, int numCols, int maxIter, int whichGPU, int whichRun)
{
	double	epsilon = (double) 1e-12;
	size_t	arraySize = numRows * numCols;
	size_t	memSizeFxn = arraySize * sizeof(double);
	size_t	memSizeMul = numRows * numRows * sizeof(double);
	double	*mulMatrix, *fxnMatrix, *addMatrix, *results, *results1;
	
	mulMatrix = (double *) malloc (memSizeMul);
	addMatrix = (double *) malloc (memSizeFxn);
	fxnMatrix = (double *) malloc (memSizeFxn);
	results = (double *) malloc (memSizeFxn);
	results1 = (double *) malloc (memSizeFxn);
	
	for (int row = 0; row < numRows; ++row)
	{
		double	rowF = row;
		int		pos = row * numCols;	// Row offset
		
		for (int col = 0; col < numCols; ++col)
		{
			double	value = rowF + col;
			
			addMatrix[pos + col] = fxnMatrix[pos + col] = value;
		}
		
		pos = row * numRows;	// Row offset
		
		for (int col = 0; col < numRows; ++col)
		{
			double	value = rowF + col;
			
			mulMatrix[pos + col] = value;
		}
	}
	
	double	*reference = (double *) malloc (memSizeFxn);
	referenceSolve<double> (fxnMatrix, mulMatrix, addMatrix, results, numRows, numCols);
	transposeRCHost<double, double> (results, reference, numRows, numCols);
	convergeMatrixCudaD (fxnMatrix, mulMatrix, addMatrix, results1, numRows, numCols, 1, whichGPU, epsilon);	// Only iterate once
	convergeMatrixCudaD (fxnMatrix, mulMatrix, addMatrix, results, numRows, numCols, maxIter, whichGPU, epsilon);
	
	double	*mulMatrixT, *fxnMatrixT, *addMatrixT, *resultsBLAS, *resultsBLAS1;
	
	mulMatrixT = (double *) malloc (memSizeMul);
	addMatrixT = (double *) malloc (memSizeFxn);
	fxnMatrixT = (double *) malloc (memSizeFxn);
	resultsBLAS = (double *) malloc (memSizeFxn);
	resultsBLAS1 = (double *) malloc (memSizeFxn);
	
	transposeRCHost<double, double> (fxnMatrix, fxnMatrixT, numRows, numCols);
	transposeRCHost<double, double> (addMatrix, addMatrixT, numRows, numCols);
	transposeRCHost<double, double> (mulMatrix, mulMatrixT, numRows, numRows);
	convergeMatrixCuBLASD (fxnMatrixT, mulMatrixT, addMatrixT, resultsBLAS1, numRows, numCols, 1, whichGPU, epsilon);	// Only iterate once
	convergeMatrixCuBLASD (fxnMatrixT, mulMatrixT, addMatrixT, resultsBLAS, numRows, numCols, maxIter, whichGPU, epsilon);
	
//	dumpMatrix<double> (fxnMatrix, "fxnMatrix", numRows, numCols);
//	dumpMatrix<double> (addMatrix, "addMatrix", numRows, numCols);
//	dumpMatrix<double> (mulMatrix, "mulMatrix", numRows, numRows);
	
	compareMatrix<double> (reference, results1, numRows, numCols, 1, whichGPU, whichRun, epsilon);
	compareMatrix<double> (reference, resultsBLAS1, numRows, numCols, 1, whichGPU, whichRun, epsilon);
	compareMatrix<double> (results, resultsBLAS, numRows, numCols, maxIter, whichGPU, whichRun, epsilon);
	
	free ((void *) addMatrix);
	free ((void *) fxnMatrix);
	free ((void *) mulMatrix);
	free ((void *) results);
	free ((void *) results1);
	free ((void *) addMatrixT);
	free ((void *) fxnMatrixT);
	free ((void *) mulMatrixT);
	free ((void *) resultsBLAS);
	free ((void *) resultsBLAS1);
}


void testMul (int numRows, int numCols, int maxIter, int whichGPU, int whichRun)
{
	float	epsilon = (float) 1e-12;
	size_t	arraySize = numRows * numCols;
	size_t	memSizeFxn = arraySize * sizeof(float);
	size_t	memSizeMul = numRows * numRows * sizeof(float);
	float	*mulMatrix, *fxnMatrix, *addMatrix, *results, *results1;
	
	mulMatrix = (float *) malloc (memSizeMul);
	addMatrix = (float *) malloc (memSizeFxn);
	fxnMatrix = (float *) malloc (memSizeFxn);
	results = (float *) malloc (memSizeFxn);
	results1 = (float *) malloc (memSizeFxn);
	
	for (int row = 0; row < numRows; ++row)
	{
		float	rowF = row;
		int		pos = row * numCols;	// Row offset
		
		for (int col = 0; col < numCols; ++col)
		{
			float	value = rowF + col;
			
			addMatrix[pos + col] = fxnMatrix[pos + col] = value;
		}
		
		pos = row * numRows;	// Row offset
		
		for (int col = 0; col < numRows; ++col)
		{
			float	value = rowF + col;
			
			mulMatrix[pos + col] = value;
		}
	}
	
	float	*reference = (float *) malloc (memSizeFxn);
	referenceSolve<float> (fxnMatrix, mulMatrix, addMatrix, results, numRows, numCols);
	transposeRCHost<float, float> (results, reference, numRows, numCols);
	convergeMatrixCuda (fxnMatrix, mulMatrix, addMatrix, results1, numRows, numCols, 1, whichGPU, epsilon);
	convergeMatrixCuda (fxnMatrix, mulMatrix, addMatrix, results, numRows, numCols, maxIter, whichGPU, epsilon);
	
	float	*mulMatrixT, *fxnMatrixT, *addMatrixT, *resultsBLAS, *resultsBLAS1;
	
	mulMatrixT = (float *) malloc (memSizeMul);
	addMatrixT = (float *) malloc (memSizeFxn);
	fxnMatrixT = (float *) malloc (memSizeFxn);
	resultsBLAS = (float *) malloc (memSizeFxn);
	resultsBLAS1 = (float *) malloc (memSizeFxn);
	
	transposeRCHost<float, float> (fxnMatrix, fxnMatrixT, numRows, numCols);
	transposeRCHost<float, float> (addMatrix, addMatrixT, numRows, numCols);
	transposeRCHost<float, float> (mulMatrix, mulMatrixT, numRows, numRows);
	convergeMatrixCuBLAS (fxnMatrixT, mulMatrixT, addMatrixT, resultsBLAS1, numRows, numCols, 1, whichGPU, epsilon);	// Only iterate once
	convergeMatrixCuBLAS (fxnMatrixT, mulMatrixT, addMatrixT, resultsBLAS, numRows, numCols, maxIter, whichGPU, epsilon);
	
//	dumpMatrix<float> (fxnMatrix, "fxnMatrix", numRows, numCols);
//	dumpMatrix<float> (addMatrix, "addMatrix", numRows, numCols);
//	dumpMatrix<float> (mulMatrix, "mulMatrix", numRows, numRows);
	
	compareMatrix<float> (reference, results1, numRows, numCols, 1, whichGPU, whichRun, epsilon);
	compareMatrix<float> (reference, resultsBLAS1, numRows, numCols, 1, whichGPU, whichRun, epsilon);
	compareMatrix<float> (results, resultsBLAS, numRows, numCols, maxIter, whichGPU, whichRun, epsilon);
	
	free ((void *) addMatrix);
	free ((void *) fxnMatrix);
	free ((void *) mulMatrix);
	free ((void *) results);
	free ((void *) results1);
	free ((void *) addMatrixT);
	free ((void *) fxnMatrixT);
	free ((void *) mulMatrixT);
	free ((void *) resultsBLAS);
	free ((void *) resultsBLAS1);
}


void testMulBLAS (int numRows, int numCols, int maxIter, int whichGPU, int whichRun)
{
	float	epsilon = (float) 1e-12;
	size_t	arraySize = numRows * numCols;
	size_t	memSizeFxn = arraySize * sizeof(float);
	size_t	memSizeMul = numRows * numRows * sizeof(float);
	float	*mulMatrix, *fxnMatrix, *addMatrix, *resultsBLAS;
	
	mulMatrix = (float *) malloc (memSizeMul);
	addMatrix = (float *) malloc (memSizeFxn);
	fxnMatrix = (float *) malloc (memSizeFxn);
	resultsBLAS = (float *) malloc (memSizeFxn);
	
	for (int col = 0; col < numCols; ++col)
	{
		float	colF = col;
		int		pos = col * numRows;	// Col offset
		
		for (int row = 0; row < numRows; ++row)
		{
			float	value = colF + row;
			
			addMatrix[pos + row] = fxnMatrix[pos + row] = value;
		}
	}
	
	for (int col = 0; col < numRows; ++col)
	{
		float	colF = col;
		int		pos = col * numRows;	// Col offset
		
		for (int row = 0; row < numRows; ++row)
		{
			float	value = colF + row;
			
			mulMatrix[pos + row] = value;
		}
	}
	
	float	*reference = (float *) malloc (memSizeFxn);
	referenceSolveCol<float> (fxnMatrix, mulMatrix, addMatrix, reference, numRows, numCols);
	convergeMatrixCuBLAS (fxnMatrix, mulMatrix, addMatrix, resultsBLAS, numRows, numCols, maxIter, whichGPU, epsilon);
	
//	dumpMatrix<float> (fxnMatrix, "fxnMatrix", numRows, numCols);
//	dumpMatrix<float> (addMatrix, "addMatrix", numRows, numCols);
//	dumpMatrix<float> (mulMatrix, "mulMatrix", numRows, numRows);
	
	compareMatrix<float> (reference, resultsBLAS, numRows, numCols, maxIter, whichGPU, whichRun, epsilon);
	
	free ((void *) addMatrix);
	free ((void *) fxnMatrix);
	free ((void *) mulMatrix);
	free ((void *) resultsBLAS);
}


int main (int argc, char **argv)
{
	int	numRows = 1758;
	int	numCols = 296;
	int	maxIter = 1000;
	int	numGPUs = 4;
	int	numRuns = 1;
	
	initCuda (numGPUs);
	for (int run = 0; run < numRuns; ++run)
	{
		for (int whichGPU = 0; whichGPU < numGPUs; ++whichGPU)
		{
			testMulD (numRows, numCols, maxIter, whichGPU, run);
//			testMul (numRows, numCols, maxIter, whichGPU, run);
//			testMulBLAS (numRows, numCols, maxIter, whichGPU, run);
		}
	}
}
