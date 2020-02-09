
#include <assert.h>
#include <stdio.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include "CRC_Prediction_InferenceAlgoCancer.h"

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
#define	kBlockRounding	31
#define	kBigBlockSize	1024
#define	kBigBlockRounding	1023

cublasHandle_t *handles = NULL;


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
		
		target[writePos] = (U) (source[readPos]);
	}
}


/**
 * Write the transpose of {@code source}, which is in row:column order (i.e. laid out with the first 
 * row being in positions 0 - {@code numCols - 1} into {@code target} in column:row order
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
 * Write the transpose of {@code source} into {@code target}
 * 
 * @param source	float / double[] to read from.  Must not be null, and length {@code numRows * numCols}
 * @param target	float / double[] to write to.  Must not be null, and length {@code numRows * numCols}
 * @param numRows	Number of rows in {@code source}, will be number of cols in {@code target}
 * @param numCols	Number of cols in {@code source}, will be number of rows in {@code target}
 */
template <class T> void transpose (T source[], T target[], int numRows, int numCols)
{
	int	i, readPos = 0;
	
	for (i = 0; i < numRows; ++i)
	{
		int	j, writePos = i;
		
		for (j = 0; j < numCols; ++j)
		{
			target[writePos] = source[readPos];
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
 * {@code result} and {@code addMatrix} have height {@code rowsMul} and width {@code colsFxn}
 *
 * @param handle	{@link cublasHandle_t} that says what CuBLAS settings to use
 * @param result	{@code float *} to fill in with results of processing.  Current contents ignored
 * @param mulMatrix	{@code float *} A array for AB + C matrix multiplication and addition
 * @param fxnMatrix	{@code float *} B array for AB + C matrix multiplication and addition
 * @param addMatrix	{@code float *} C array for AB + C matrix multiplication and addition
 * @param colsMul	{@code mulMatrix}'s width, must equal {@code rowsFxn}
 * @param rowsMul	{@code mulMatrix}'s height, also height of {@code result} and {@code addMatrix}
 * @param colsFxn	{@code fxnMatrix}'s width, also width of {@code result} and {@code addMatrix}
 * @param rowsFxn	{@code fxnMatrix}'s height, must equal {@code colsMul}
 * @param line		Line routine was called from, used when reporting errors
 * @param iteration	Iteration calling routine was on when called this, used when reporting errors
 */
void cuBlasMul (cublasHandle_t handle, float *result, float *mulMatrix, float *fxnMatrix, float *addMatrix, 
				int colsMul, int rowsMul, int colsFxn, int rowsFxn, int line, int iteration)
{
	cublasStatus_t	stat;
	const float		alpha = 1.0f;
	const float		beta = 1.0f;
	int				arraySize = colsFxn * rowsFxn;
	
	// Replace contents of result with addMatrix, so can use the beta add, rather than a separate operation
//	printf ("Calling copy\n");
	copy<kBigBlockSize, float><<< (arraySize + kBigBlockRounding) / kBigBlockSize, kBigBlockSize >>>(addMatrix, result, arraySize);
//	printf ("Calling cudaDeviceSynchronize\n");
	cudaDeviceSynchronize ();
	
//	printf ("Calling cublasSgemm\n");
	stat = cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, rowsMul, colsFxn, rowsFxn, &alpha, 
						mulMatrix, rowsMul, fxnMatrix, rowsFxn, &beta, result, rowsMul);
//	printf ("testing return code\n");
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf ("cublasSgemm () iteration: %d, returned error %s (code %d), call line (%d)\n", iteration, _cudaGetErrorEnum (stat), stat, line);
		exit (EXIT_FAILURE);
	}
}


/**
 * Matrix multiplication and addition on the device: {@code result = (mulMatrix * fxnMatrix) + addMatrix}
 * {@code result} and {@code addMatrix} have height {@code rowsMul} and width {@code colsFxn}
 *
 * @param handle	{@link cublasHandle_t} that says what CuBLAS settings to use
 * @param result	{@code double *} to fill in with results of processing.  Current contents ignored
 * @param mulMatrix	{@code double *} A array for AB + C matrix multiplication and addition
 * @param fxnMatrix	{@code double *} B array for AB + C matrix multiplication and addition
 * @param addMatrix	{@code double *} C array for AB + C matrix multiplication and addition
 * @param colsMul	{@code mulMatrix}'s width, must equal {@code rowsFxn}
 * @param rowsMul	{@code mulMatrix}'s height, also height of {@code result} and {@code addMatrix}
 * @param colsFxn	{@code fxnMatrix}'s width, also width of {@code result} and {@code addMatrix}
 * @param rowsFxn	{@code fxnMatrix}'s height, must equal {@code colsMul}
 * @param line		Line routine was called from, used when reporting errors
 * @param iteration	Iteration calling routine was on when called this, used when reporting errors
 */
void cuBlasMul (cublasHandle_t handle, double *result, double *mulMatrix, double *fxnMatrix, double *addMatrix, 
				int colsMul, int rowsMul, int colsFxn, int rowsFxn, int line, int iteration)
{
	cublasStatus_t	stat;
	const double	alpha = 1.0f;
	const double	beta = 1.0f;
	int				arraySize = colsFxn * rowsFxn;
	
	// Replace contents of result with addMatrix, so can use the beta add, rather than a separate operation
	copy<kBigBlockSize, double><<< (arraySize + kBigBlockRounding) / kBigBlockSize, kBigBlockSize >>>(addMatrix, result, arraySize);
	cudaDeviceSynchronize ();
	
	stat = cublasDgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, rowsMul, colsFxn, rowsFxn, &alpha, 
						mulMatrix, rowsMul, fxnMatrix, rowsFxn, &beta, result, rowsMul);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf ("cublasDgemm () iteration: %d, returned error %s (code %d), call line (%d)\n", iteration, _cudaGetErrorEnum (stat), stat, line);
		exit (EXIT_FAILURE);
	}
}


/**
 * Matrix multiplication on the device: {@code result = firstMatrix * secondMatrix}
 * {@code result} has height {@code rowsFirst} and width {@code colsSecond}
 *
 * @param handle		{@link cublasHandle_t} that says what CuBLAS settings to use
 * @param result		{@code double *} to fill in with results of processing.  Current contents ignored
 * @param firstMatrix	{@code double *} A array for AB matrix multiplication
 * @param secondMatrix	{@code double *} B array for AB matrix multiplication
 * @param colsFirst		{@code firstMatrix}'s width, must equal {@code rowsSecond}
 * @param rowsFirst		{@code firstMatrix}'s height, also height of {@code result}
 * @param colsSecond	{@code secondMatrix}'s width, also width of {@code result}
 * @param rowsSecond	{@code secondMatrix}'s height, must equal {@code colsFirst}
 * @param line			Line routine was called from, used when reporting errors
 */
void cuBlasMul (cublasHandle_t handle, double *result, double *firstMatrix, double *secondMatrix, 
				int colsFirst, int rowsFirst, int colsSecond, int rowsSecond, int line)
{
	cublasStatus_t	stat;
	const double	alpha = 1.0f;
	const double	beta = 0.0f;
	
	stat = cublasDgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, colsFirst, rowsSecond, colsSecond, &alpha, 
						firstMatrix, colsFirst, secondMatrix, colsSecond, &beta, result, colsFirst);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf ("cublasDgemm () returned error %s (code %d), call line (%d)\n", _cudaGetErrorEnum (stat), stat, line);
		exit (EXIT_FAILURE);
	}
}


/**
 * Matrix multiplication on the device: {@code result = firstMatrix * secondMatrix}<br>
 * <b>Arrays are in C stype row major order, unlike all other calls</b>
 * {@code result} has height {@code rowsFirst} and width {@code colsSecond}
 *
 * @param handle		{@link cublasHandle_t} that says what CuBLAS settings to use
 * @param result		{@code double *} to fill in with results of processing.  Current contents ignored
 * @param firstMatrix	{@code double *} A array for AB matrix multiplication
 * @param secondMatrix	{@code double *} B array for AB matrix multiplication
 * @param colsFirst		{@code firstMatrix}'s width, must equal {@code rowsSecond}
 * @param rowsFirst		{@code firstMatrix}'s height, also height of {@code result}
 * @param colsSecond	{@code secondMatrix}'s width, also width of {@code result}
 * @param rowsSecond	{@code secondMatrix}'s height, must equal {@code colsFirst}
 * @param line			Line routine was called from, used when reporting errors
 */
void cuBlasMulC (cublasHandle_t handle, double *result, double *firstMatrix, double *secondMatrix, 
				 int colsFirst, int rowsFirst, int colsSecond, int rowsSecond, int line)
{
	cublasStatus_t	stat;
	const double	alpha = 1.0f;
	const double	beta = 0.0f;
	
	stat = cublasDgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, colsSecond, rowsFirst, colsFirst, &alpha, 
						secondMatrix, colsSecond, firstMatrix, colsFirst, &beta, result, colsSecond);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf ("cublasDgemm () returned error %s (code %d), call line (%d)\n", _cudaGetErrorEnum (stat), stat, line);
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
	
	printf ("Finished initCuda (%d)\n", numGPUs);
}


/**
 *	Routine to let us know the library really is loaded
 */
JNIEXPORT void JNICALL Java_CRC_1Prediction_InferenceAlgoCancer_initCuda (JNIEnv *env, jclass clazz, jint numGPUsJ)
{
	int	numGPUs = (int) numGPUsJ;
	
	initCuda (numGPUs);
}


/**
 * Cuda processing, using single precision for speed<br>
 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
 * as {@code epsilon}<br>
 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
 * All matrices are laid out with the first col being in positions 0 - {@code numRows - 1}, 
 * <b>this is the layout of the {@link DoubleMatrix} data, and different from all the other 
 * JNI versions of this code
 * 
 * @param fxnMatrixJ	Key matrix
 * @param mulMatrixJ	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
 * @param addMatrixJ	Matrix with same dimensions as {@code fxnMatrix}
 * @param numRowsJ	Number of rows in {@code fxnMatrix} and {@code addMatrix}.<br>
 * Number of rows and columns in {@code mulMatrix}
 * @param numColsJ	Number of cols in {@code fxnMatrix} and {@code addMatrix}
 * @param maxIterations	Maximum number of iterations to run if don't converge
 * @param epsilonJ	Result is converged if every element of {@code fxnMatrix} changes by less than 
 * {@code epsilon} during one round of processing
 * @param whichGPUJ	Which GPU to target
 * @return	The transpose of the final matrix, whether or not it converged, as a 1-d matrix
 */
JNIEXPORT void JNICALL Java_CRC_1Prediction_InferenceAlgoCancer_multiplyMatrixCuBLASD (
	JNIEnv *env, jclass clazz, jdoubleArray firstMatrixJ, jdoubleArray secondMatrixJ, 
	jdoubleArray resultsJ, jint resultRowsJ, jint resultColsJ, jint sharedDimJ, jint whichGPUJ)
{
	int	whichGPU = (int) whichGPUJ;
	
	if (handles == NULL)
		initCuda (max(4, whichGPU + 1));
	
	cublasHandle_t	handle = handles[whichGPU];
	cSetDevice (whichGPU, __LINE__);
	
	int		resultRows = (int) resultRowsJ;
	int		numCols = (int) resultColsJ;
	int		sharedDim = (int) sharedDimJ;
	double	*firstMatrixH = env->GetDoubleArrayElements (firstMatrixJ, NULL);
	double	*secondMatrixH = env->GetDoubleArrayElements (secondMatrixJ, NULL);
	size_t	resultSize = resultRows * numCols;
	size_t	firstSize = resultRows * sharedDim;
	size_t	secondSize = sharedDim * numCols;
	size_t	memSizeFirst = firstSize * sizeof(double);
	size_t	memSizeSecond = secondSize * sizeof(double);
	size_t	memSizeResult = resultSize * sizeof(double);
	double	*firstMatrix, *secondMatrix, *resultHold;
	
//	struct timeval startTV;
//	gettimeofday (&startTV, NULL);
	
	// Allocate device memory
	cMalloc ((void **) &firstMatrix, memSizeFirst, "firstMatrix", __LINE__);
	cMalloc ((void **) &secondMatrix, memSizeSecond, "secondMatrix", __LINE__);
	cMalloc ((void **) &resultHold, memSizeResult, "resultHold", __LINE__);
	
	// copy host memory to device
	cCopy (firstMatrix, firstMatrixH, memSizeFirst, cudaMemcpyHostToDevice, "(firstMatrix, firstMatrixH)", __LINE__);
	cCopy (secondMatrix, secondMatrixH, memSizeSecond, cudaMemcpyHostToDevice, "(secondMatrix, secondMatrixH)", __LINE__);
	
	cuBlasMulC (handle, resultHold, firstMatrix, secondMatrix, sharedDim, resultRows, numCols, sharedDim, __LINE__);
	cudaDeviceSynchronize ();
	
//	struct timeval endTV;
//	gettimeofday (&endTV, NULL);
//	unsigned long	theStart = (1000000 * startTV.tv_sec) + startTV.tv_usec;
//	unsigned long	done = (1000000 * endTV.tv_sec) + endTV.tv_usec;
//	printf ("cuBlasMulC (double) used %lu microseconds for %d rows, %d cols, and %d sharedDim\n", done - theStart, resultRows, numCols, sharedDim);
	
	double	*results = env->GetDoubleArrayElements (resultsJ, NULL);
	cCopy (results, resultHold, memSizeResult, cudaMemcpyDeviceToHost, "(results, resultHold)", __LINE__);
	env->ReleaseDoubleArrayElements (resultsJ, results, 0);	// Copy these results back
	
	cudaFree (firstMatrix);
	cudaFree (secondMatrix);
	cudaFree (resultHold);
	env->ReleaseDoubleArrayElements (secondMatrixJ, secondMatrixH, JNI_ABORT);
	env->ReleaseDoubleArrayElements (firstMatrixJ, firstMatrixH, JNI_ABORT);
}


/**
 * Cuda processing, using single precision for speed<br>
 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
 * as {@code epsilon}<br>
 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
 * All matrices are laid out with the first col being in positions 0 - {@code numRows - 1}, 
 * <b>this is the layout of the {@link DoubleMatrix} data, and different from all the other 
 * JNI versions of this code
 * 
 * @param fxnMatrixJ	Key matrix
 * @param mulMatrixJ	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
 * @param addMatrixJ	Matrix with same dimensions as {@code fxnMatrix}
 * @param numRowsJ	Number of rows in {@code fxnMatrix} and {@code addMatrix}.<br>
 * Number of rows and columns in {@code mulMatrix}
 * @param numColsJ	Number of cols in {@code fxnMatrix} and {@code addMatrix}
 * @param maxIterations	Maximum number of iterations to run if don't converge
 * @param epsilonJ	Result is converged if every element of {@code fxnMatrix} changes by less than 
 * {@code epsilon} during one round of processing
 * @param whichGPUJ	Which GPU to target
 * @return	The transpose of the final matrix, whether or not it converged, as a 1-d matrix
 */
JNIEXPORT jboolean JNICALL Java_CRC_1Prediction_InferenceAlgoCancer_convergeMatrixCuBLAS (
	JNIEnv *env, jclass clazz, jfloatArray fxnMatrixJ, jfloatArray mulMatrixJ, jfloatArray addMatrixJ, 
	jfloatArray resultsJ, jint numRowsJ, jint numColsJ, jint maxIterations, jint whichGPUJ, jfloat epsilonJ)
{
	int	whichGPU = (int) whichGPUJ;
	
	if (handles == NULL)
		initCuda (max(4, whichGPU + 1));
	
	cublasHandle_t	handle = handles[whichGPU];
	cSetDevice (whichGPU, __LINE__);
	
	int		numRows = (int) numRowsJ;
	int		numCols = (int) numColsJ;
	int		maxIter = (int) maxIterations;
	float	epsilon = (float) epsilonJ;
	float	*fxnMatrixH = env->GetFloatArrayElements (fxnMatrixJ, NULL);
	float	*mulMatrixH = env->GetFloatArrayElements (mulMatrixJ, NULL);
	float	*addMatrixH = env->GetFloatArrayElements (addMatrixJ, NULL);
	size_t	arraySize = numRows * numCols;
	size_t	memSizeFxn = arraySize * sizeof(float);
	size_t	memSizeMul = numRows * numRows * sizeof(float);
	float	*mulMatrix, *fxnMatrix, *addMatrix, *resultHold;
	size_t	changeCountH;
	size_t	*changeCount;
	jboolean	converged = false;
	
#ifdef DoPrint
	printf ("In convergeMatrixCudaCuBLAS, numRows: %d, numCols: %d, maxIter: %d\n", numRows, numCols, maxIter);
	printArray ("mulMatrix", mulMatrixH, numRows * numRows);
	printArray ("fxnMatrix", fxnMatrixH, arraySize);
	printArray ("addMatrix", addMatrixH, arraySize);
#endif
	
//	struct timeval startTV;
//	gettimeofday (&startTV, NULL);
	
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
	
	
	int		i, numBlocks = (arraySize + kBigBlockRounding) / kBigBlockSize;
	for (i = 0; i < maxIter; ++i)
	{
#ifdef DoPrint
		printArray ("first", first, arraySize);
#endif
		cuBlasMul (handle, resultHold, mulMatrix, fxnMatrix, addMatrix, numRows, numRows, numCols, numRows, __LINE__, i);
		cudaDeviceSynchronize ();
		swap(resultHold, fxnMatrix);	// Want fxnMatrix to always hold the final results when we exit
		changeCountH = 0;
		cCopy (changeCount, &changeCountH, sizeof(size_t), cudaMemcpyHostToDevice, "(changeCount, changeCountH)", __LINE__);
		compare<kBigBlockSize, float><<< numBlocks, kBigBlockSize >>>(fxnMatrix, resultHold, arraySize, epsilon, changeCount);
		cudaDeviceSynchronize ();
		cCopy (&changeCountH, changeCount, sizeof(size_t), cudaMemcpyDeviceToHost, "(changeCountH, changeCount)", __LINE__);
		if (changeCountH == 0)
		{
			converged = true;
			break;
		}
	}
	
//	struct timeval endTV;
//	gettimeofday (&endTV, NULL);
//	unsigned long	theStart = (1000000 * startTV.tv_sec) + startTV.tv_usec;
//	unsigned long	done = (1000000 * endTV.tv_sec) + endTV.tv_usec;
//	printf ("cuBlasMul (float) used %lu microseconds for %d iterations\n", done - theStart, i);
	
	float	*results = env->GetFloatArrayElements (resultsJ, NULL);
	cCopy (results, fxnMatrix, memSizeFxn, cudaMemcpyDeviceToHost, "(results, result)", __LINE__);
//	cCopy (fxnMatrixH, results, memSizeFxn, cudaMemcpyDeviceToHost, "(results, result)", __LINE__);
	env->ReleaseFloatArrayElements (resultsJ, results, 0);	// Copy these results back
	
	cudaFree (mulMatrix);
	cudaFree (fxnMatrix);
	cudaFree (addMatrix);
	cudaFree (resultHold);
	cudaFree (changeCount);
	env->ReleaseFloatArrayElements (fxnMatrixJ, fxnMatrixH, JNI_ABORT);
	env->ReleaseFloatArrayElements (mulMatrixJ, mulMatrixH, JNI_ABORT);
	env->ReleaseFloatArrayElements (addMatrixJ, addMatrixH, JNI_ABORT);
	
	return converged;
}


/**
 * Cuda processing, using single precision for speed<br>
 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
 * as {@code epsilon}<br>
 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
 * All matrices are laid out with the first col being in positions 0 - {@code numRows - 1}, 
 * <b>this is the layout of the {@link DoubleMatrix} data, and different from all the other 
 * JNI versions of this code
 * 
 * @param fxnMatrixJ	Key matrix
 * @param mulMatrixJ	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
 * @param addMatrixJ	Matrix with same dimensions as {@code fxnMatrix}
 * @param numRowsJ	Number of rows in {@code fxnMatrix} and {@code addMatrix}.<br>
 * Number of rows and columns in {@code mulMatrix}
 * @param numColsJ	Number of cols in {@code fxnMatrix} and {@code addMatrix}
 * @param maxIterations	Maximum number of iterations to run if don't converge
 * @param epsilonJ	Result is converged if every element of {@code fxnMatrix} changes by less than 
 * {@code epsilon} during one round of processing
 * @param whichGPUJ	Which GPU to target
 * @return	The transpose of the final matrix, whether or not it converged, as a 1-d matrix
 */
JNIEXPORT jboolean JNICALL Java_CRC_1Prediction_InferenceAlgoCancer_convergeMatrixCuBLASD (
	JNIEnv *env, jclass clazz, jdoubleArray fxnMatrixJ, jdoubleArray mulMatrixJ, jdoubleArray addMatrixJ, 
	jdoubleArray resultsJ, jint numRowsJ, jint numColsJ, jint maxIterations, jint whichGPUJ, jdouble epsilonJ)
{
	int	whichGPU = (int) whichGPUJ;
	
	if (handles == NULL)
		initCuda (max(4, whichGPU + 1));
	
	cublasHandle_t	handle = handles[whichGPU];
	cSetDevice (whichGPU, __LINE__);
	
	int		numRows = (int) numRowsJ;
	int		numCols = (int) numColsJ;
	int		maxIter = (int) maxIterations;
	double	epsilon = (double) epsilonJ;
	double	*fxnMatrixH = env->GetDoubleArrayElements (fxnMatrixJ, NULL);
	double	*mulMatrixH = env->GetDoubleArrayElements (mulMatrixJ, NULL);
	double	*addMatrixH = env->GetDoubleArrayElements (addMatrixJ, NULL);
	size_t	arraySize = numRows * numCols;
	size_t	memSizeFxn = arraySize * sizeof(double);
	size_t	memSizeMul = numRows * numRows * sizeof(double);
	double	*mulMatrix, *fxnMatrix, *addMatrix, *resultHold;
	size_t	changeCountH;
	size_t	*changeCount;
	jboolean	converged = false;
	
#ifdef DoPrint
	printf ("In convergeMatrixCudaCuBLASD, numRows: %d, numCols: %d, maxIter: %d\n", numRows, numCols, maxIter);
	printArray ("mulMatrix", mulMatrixH, numRows * numRows);
	printArray ("fxnMatrix", fxnMatrixH, arraySize);
	printArray ("addMatrix", addMatrixH, arraySize);
#endif
	
//	struct timeval startTV;
//	gettimeofday (&startTV, NULL);
	
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
	
	int	i, numBlocks = (arraySize + kBigBlockRounding) / kBigBlockSize;
	
	for (i = 0; i < maxIter; ++i)
	{
		cuBlasMul (handle, resultHold, mulMatrix, fxnMatrix, addMatrix, numRows, numRows, numCols, numRows, __LINE__, i);
		cudaDeviceSynchronize ();
		swapD(resultHold, fxnMatrix);	// Want fxnMatrix to always hold the final results when we exit
		changeCountH = 0;
		cCopy (changeCount, &changeCountH, sizeof(size_t), cudaMemcpyHostToDevice, "(changeCount, changeCountH)", __LINE__);
		compare<kBigBlockSize, double><<< numBlocks, kBigBlockSize >>>(fxnMatrix, resultHold, arraySize, epsilon, changeCount);
		cudaDeviceSynchronize ();
		cCopy (&changeCountH, changeCount, sizeof(size_t), cudaMemcpyDeviceToHost, "(changeCountH, changeCount)", __LINE__);
		if (changeCountH == 0)
		{
			converged = true;
			break;
		}
	}
	
//	struct timeval endTV;
//	gettimeofday (&endTV, NULL);
//	unsigned long	theStart = (1000000 * startTV.tv_sec) + startTV.tv_usec;
//	unsigned long	done = (1000000 * endTV.tv_sec) + endTV.tv_usec;
//	printf ("cuBlasMul (double) used %lu microseconds for %d iterations\n", done - theStart, i);
	
	double	*results = env->GetDoubleArrayElements (resultsJ, NULL);
	cCopy (results, fxnMatrix, memSizeFxn, cudaMemcpyDeviceToHost, "(results, resultHold)", __LINE__);
	env->ReleaseDoubleArrayElements (resultsJ, results, 0);	// Copy these results back
	
	cudaFree (mulMatrix);
	cudaFree (fxnMatrix);
	cudaFree (addMatrix);
	cudaFree (resultHold);
	cudaFree (changeCount);
	env->ReleaseDoubleArrayElements (fxnMatrixJ, fxnMatrixH, JNI_ABORT);
	env->ReleaseDoubleArrayElements (mulMatrixJ, mulMatrixH, JNI_ABORT);
	env->ReleaseDoubleArrayElements (addMatrixJ, addMatrixH, JNI_ABORT);
	
	return converged;
}


/**
 * Cuda processing, using single precision for speed<br>
 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
 * as {@code epsilon}<br>
 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
 * Input matrices are in row:column order (i.e. laid out with the first being in positions 0 - {@code numCols - 1}
 * 
 * @param fxnMatrixJ	Key matrix
 * @param mulMatrixJ	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
 * @param addMatrixJ	Matrix with same dimensions as {@code fxnMatrix}
 * @param numRowsJ	Number of rows in {@code fxnMatrix} and {@code addMatrix}.<br>
 * Number of rows and columns in {@code mulMatrix}
 * @param numColsJ	Number of cols in {@code fxnMatrix} and {@code addMatrix}
 * @param maxIterations	Maximum number of iterations to run if don't converge
 * @param epsilonJ	Result is converged if every element of {@code fxnMatrix} changes by less than 
 * {@code epsilon} during one round of processing
 * @param whichGPUJ	Which GPU to target
 * @return	The transpose of the final matrix, whether or not it converged, as a 1-d matrix
 */
JNIEXPORT jboolean JNICALL Java_CRC_1Prediction_InferenceAlgoCancer_convergeMatrixCuda (
	JNIEnv *env, jclass clazz, jfloatArray fxnMatrixJ, jfloatArray mulMatrixJ, jfloatArray addMatrixJ, 
	jfloatArray resultsJ, jint numRowsJ, jint numColsJ, jint maxIterations, jint whichGPUJ, jfloat epsilonJ)
{
	int	whichGPU = (int) whichGPUJ;
	cSetDevice (whichGPU, __LINE__);	// Assign code to the appropriate GPU
	
	int		numRows = (int) numRowsJ;
	int		numCols = (int) numColsJ;
	int		maxIter = (int) maxIterations;
	float	epsilon = (float) epsilonJ;
	float	*fxnMatrixH = env->GetFloatArrayElements (fxnMatrixJ, NULL);
	float	*mulMatrixH = env->GetFloatArrayElements (mulMatrixJ, NULL);
	float	*addMatrixH = env->GetFloatArrayElements (addMatrixJ, NULL);
	size_t	arraySize = numRows * numCols;
	size_t	memSizeFxn = arraySize * sizeof(float);
	size_t	memSizeMul = numRows * numRows * sizeof(float);
	float	*mulMatrix, *fxnMatrix, *addMatrix, *resultHold;
	size_t	changeCountH;
	size_t	*changeCount;
	jboolean	converged = false;
	
//	struct timeval startTV;
//	gettimeofday (&startTV, NULL);
	
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
	dim3	grid ((numCols + kBlockRounding) / kBlockSize, (numRows + kBlockRounding) / kBlockSize);
	int		i, numBlocks = (arraySize + kBigBlockRounding) / kBigBlockSize;
	
	for (i = 0; i < maxIter; ++i)
	{
		matrixMulCUDA<kBlockSize, float><<< grid, threads >>>(resultHold, mulMatrix, fxnMatrix, addMatrix, numRows, numRows, numCols, numRows);
		cudaDeviceSynchronize ();
		swap(resultHold, fxnMatrix);	// Want fxnMatrix to always hold the final results when we exit
		changeCountH = 0;
		cCopy (changeCount, &changeCountH, sizeof(size_t), cudaMemcpyHostToDevice, "(changeCount, changeCountH)", __LINE__);
		compare<kBigBlockSize, float><<< numBlocks, kBigBlockSize >>>(fxnMatrix, resultHold, arraySize, epsilon, changeCount);
		cudaDeviceSynchronize ();
		cCopy (&changeCountH, changeCount, sizeof(size_t), cudaMemcpyDeviceToHost, "(changeCountH, changeCount)", __LINE__);
		if (changeCountH == 0)
		{
			converged = true;
			break;
		}
	}
	
//	struct timeval endTV;
//	gettimeofday (&endTV, NULL);
//	unsigned long	theStart = (1000000 * startTV.tv_sec) + startTV.tv_usec;
//	unsigned long	done = (1000000 * endTV.tv_sec) + endTV.tv_usec;
//	printf ("matrixMulCUDA (float) used %lu microseconds for %d iterations\n", done - theStart, i);
	
	float	*results = env->GetFloatArrayElements (resultsJ, NULL);
	transposeRC<kBlockSize, float, float><<< grid, threads >>>(fxnMatrix, resultHold, numRows, numCols);
	cCopy (results, resultHold, memSizeFxn, cudaMemcpyDeviceToHost, "(results, resultHold)", __LINE__);
	env->ReleaseFloatArrayElements (resultsJ, results, 0);	// Copy these results back
	
	cudaFree (mulMatrix);
	cudaFree (fxnMatrix);
	cudaFree (addMatrix);
	cudaFree (resultHold);
	cudaFree (changeCount);
	env->ReleaseFloatArrayElements (fxnMatrixJ, fxnMatrixH, JNI_ABORT);
	env->ReleaseFloatArrayElements (mulMatrixJ, mulMatrixH, JNI_ABORT);
	env->ReleaseFloatArrayElements (addMatrixJ, addMatrixH, JNI_ABORT);
	
	return converged;
}


/**
 * Cuda processing, using double precision rather than single precision<br>
 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
 * as {@code epsilon}<br>
 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
 * Input matrices are in row:column order (i.e. laid out with the first being in positions 0 - {@code numCols - 1}
 * 
 * @param fxnMatrixJ	Key matrix
 * @param mulMatrixJ	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
 * @param addMatrixJ	Matrix with same dimensions as {@code fxnMatrix}
 * @param numRowsJ	Number of rows in {@code fxnMatrix} and {@code addMatrix}.<br>
 * Number of rows and columns in {@code mulMatrix}
 * @param numColsJ	Number of cols in {@code fxnMatrix} and {@code addMatrix}
 * @param maxIterations	Maximum number of iterations to run if don't converge
 * @param epsilonJ	Result is converged if every element of {@code fxnMatrix} changes by less than 
 * {@code epsilon} during one round of processing
 * @param whichGPUJ	Which GPU to target
 * @return	The transpose of the final matrix, whether or not it converged, as a 1-d matrix
 */
JNIEXPORT jboolean JNICALL Java_CRC_1Prediction_InferenceAlgoCancer_convergeMatrixCudaD (
	JNIEnv *env, jclass clazz, jdoubleArray fxnMatrixJ, jdoubleArray mulMatrixJ, jdoubleArray addMatrixJ, 
	jdoubleArray resultsJ, jint numRowsJ, jint numColsJ, jint maxIterations, jint whichGPUJ, jdouble epsilonJ)
{
	int	whichGPU = (int) whichGPUJ;
	cSetDevice (whichGPU, __LINE__);	// Assign code to the appropriate GPU
	
	int		numRows = (int) numRowsJ;
	int		numCols = (int) numColsJ;
	int		maxIter = (int) maxIterations;
	double	epsilon = (double) epsilonJ;
	double	*fxnMatrixH = env->GetDoubleArrayElements (fxnMatrixJ, NULL);
	double	*mulMatrixH = env->GetDoubleArrayElements (mulMatrixJ, NULL);
	double	*addMatrixH = env->GetDoubleArrayElements (addMatrixJ, NULL);
	size_t	arraySize = numRows * numCols;
	size_t	memSizeFxn = arraySize * sizeof(double);
	size_t	memSizeMul = numRows * numRows * sizeof(double);
	double	*mulMatrix, *fxnMatrix, *addMatrix, *resultHold;
//	size_t	changeCountTotal = 0;
	size_t	changeCountH;
	size_t	*changeCount;
	jboolean	converged = false;
	
#ifdef DoPrint
	printf ("In convergeMatrixCudaD, numRows: %d, numCols: %d, maxIter: %d\n", numRows, numCols, maxIter);
	printArray ("mulMatrix", mulMatrixH, numRows * numRows);
	printArray ("fxnMatrix", fxnMatrixH, arraySize);
	printArray ("addMatrix", addMatrixH, arraySize);
#endif
	
//	struct timeval startTV;
//	gettimeofday (&startTV, NULL);
	
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
	dim3	grid ((numCols + kBlockRounding) / kBlockSize, (numRows + kBlockRounding) / kBlockSize);
	int		i, numBlocks = (arraySize + kBigBlockRounding) / kBigBlockSize;
	
	for (i = 0; i < maxIter; ++i)
	{
#ifdef DoPrint
		printArray ("first", first, arraySize);
#endif
		matrixMulCUDA<kBlockSize, double><<< grid, threads >>>(resultHold, mulMatrix, fxnMatrix, addMatrix, numRows, numRows, numCols, numRows);
		cudaDeviceSynchronize ();
		swapD(resultHold, fxnMatrix);	// Want fxnMatrix to always hold the final results when we exit
		changeCountH = 0;
		cCopy (changeCount, &changeCountH, sizeof(size_t), cudaMemcpyHostToDevice, "(changeCount, changeCountH)", __LINE__);
		compare<kBigBlockSize, double><<< numBlocks, kBigBlockSize >>>(fxnMatrix, resultHold, arraySize, epsilon, changeCount);
		cudaDeviceSynchronize ();
		cCopy (&changeCountH, changeCount, sizeof(size_t), cudaMemcpyDeviceToHost, "(changeCountH, changeCount)", __LINE__);
		if (changeCountH == 0)
		{
			converged = true;
			break;
		}
//		else
//			changeCountTotal += changeCountH;
	}
	
//	struct timeval endTV;
//	gettimeofday (&endTV, NULL);
//	unsigned long	theStart = (1000000 * startTV.tv_sec) + startTV.tv_usec;
//	unsigned long	done = (1000000 * endTV.tv_sec) + endTV.tv_usec;
//	printf ("matrixMulCUDA (double) used %lu microseconds for %d iterations\n", done - theStart, i);
	
	double	*results = env->GetDoubleArrayElements (resultsJ, NULL);
	transposeRC<kBlockSize, double, double><<< grid, threads >>>(fxnMatrix, resultHold, numRows, numCols);
	cCopy (results, resultHold, memSizeFxn, cudaMemcpyDeviceToHost, "(results, resultHold)", __LINE__);
	env->ReleaseDoubleArrayElements (resultsJ, results, 0);	// Copy these results back
	
	cudaFree (mulMatrix);
	cudaFree (fxnMatrix);
	cudaFree (addMatrix);
	cudaFree (resultHold);
	cudaFree (changeCount);
	env->ReleaseDoubleArrayElements (fxnMatrixJ, fxnMatrixH, JNI_ABORT);
	env->ReleaseDoubleArrayElements (mulMatrixJ, mulMatrixH, JNI_ABORT);
	env->ReleaseDoubleArrayElements (addMatrixJ, addMatrixH, JNI_ABORT);
	
	return converged;
}


/**
 * Cuda processing, using single precision for speed<br>
 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
 * as {@code epsilon}<br>
 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
 * Input matrices are in row:column order (i.e. laid out with the first being in positions 0 - {@code numCols - 1}
 * 
 * @param fxnMatrixJ	Key matrix
 * @param mulMatrixJ	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
 * @param addMatrixJ	Matrix with same dimensions as {@code fxnMatrix}
 * @param numRowsJ	Number of rows in {@code fxnMatrix} and {@code addMatrix}.<br>
 * Number of rows and columns in {@code mulMatrix}
 * @param numColsJ	Number of cols in {@code fxnMatrix} and {@code addMatrix}
 * @param maxIterations	Maximum number of iterations to run if don't converge
 * @param epsilonJ	Result is converged if every element of {@code fxnMatrix} changes by less than 
 * {@code epsilon} during one round of processing
 * @param whichGPUJ	Which GPU to target
 * @return	The transpose of the final matrix, (i.e. in column:row order) whether or not it converged, as a 1-d matrix
 */
JNIEXPORT jboolean JNICALL Java_CRC_1Prediction_InferenceAlgoCancer_convergeMatrixCudaOld (
	JNIEnv *env, jclass clazz, jfloatArray fxnMatrixJ, jfloatArray mulMatrixJ, jfloatArray addMatrixJ, 
	jfloatArray resultsJ, jint numRowsJ, jint numColsJ, jint maxIterations, jint whichGPUJ, jfloat epsilonJ)
{
	int	whichGPU = (int) whichGPUJ;
	cSetDevice (whichGPU, __LINE__);	// Assign code to the appropriate GPU
	
	int		numRows = (int) numRowsJ;
	int		numCols = (int) numColsJ;
	int		maxIter = (int) maxIterations;
	float	epsilon = (float) epsilonJ;
	float	*fxnMatrixH = env->GetFloatArrayElements (fxnMatrixJ, NULL);
	float	*mulMatrixH = env->GetFloatArrayElements (mulMatrixJ, NULL);
	float	*addMatrixH = env->GetFloatArrayElements (addMatrixJ, NULL);
	size_t	arraySize = numRows * numCols;
	size_t	memSizeFxn = arraySize * sizeof(float);
	size_t	memSizeMul = numRows * numRows * sizeof(float);
	float	*mulMatrix, *fxnMatrix, *addMatrix, *resultHold;
//	size_t	changeCountTotal = 0;
	size_t	changeCountH;
	size_t	*changeCount;
	jboolean	converged = false;
	
#ifdef DoPrint
	printf ("In convergeMatrixCuda, numRows: %d, numCols: %d, maxIter: %d\n", numRows, numCols, maxIter);
	printArray ("mulMatrix", mulMatrixH, numRows * numRows);
	printArray ("fxnMatrix", fxnMatrixH, arraySize);
	printArray ("addMatrix", addMatrixH, arraySize);
#endif
	
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
	dim3	grid ((numCols + kBlockRounding) / kBlockSize, (numRows + kBlockRounding) / kBlockSize);
	int		i, numBlocks = (arraySize + kBigBlockRounding) / kBigBlockSize;
	
//	struct timeval	midTV;
//	gettimeofday (&midTV, NULL);
	for (i = 0; i < maxIter; ++i)
	{
#ifdef DoPrint
		printArray ("first", first, arraySize);
#endif
		matrixMulCUDA<kBlockSize, float><<< grid, threads >>>(resultHold, mulMatrix, fxnMatrix, addMatrix, numRows, numRows, numCols, numRows);
		cudaDeviceSynchronize ();
		swap(resultHold, fxnMatrix);	// Want fxnMatrix to always hold the final results when we exit
		changeCountH = 0;
		cCopy (changeCount, &changeCountH, sizeof(size_t), cudaMemcpyHostToDevice, "(changeCount, changeCountH)", __LINE__);
		compare<kBigBlockSize, float><<< numBlocks, kBigBlockSize >>>(fxnMatrix, resultHold, arraySize, epsilon, changeCount);
		cudaDeviceSynchronize ();
		cCopy (&changeCountH, changeCount, sizeof(size_t), cudaMemcpyDeviceToHost, "(changeCountH, changeCount)", __LINE__);
		if (changeCountH == 0)
		{
			converged = true;
			break;
		}
//		else
//			changeCountTotal += changeCountH;
	}
	
	struct timeval endTV;
	gettimeofday (&endTV, NULL);
	unsigned long	theStart = (1000000 * startTV.tv_sec) + startTV.tv_usec;
//	unsigned long	theMid = (1000000 * midTV.tv_sec) + midTV.tv_usec;
	unsigned long	done = (1000000 * endTV.tv_sec) + endTV.tv_usec;
	printf ("matrixMulCUDA (float) used %lu microseconds for %d iterations\n", done - theStart, i);
//	printf ("Just matrixMulCUDA used %lu microseconds\n", done - theMid);
//	printf ("changeCountTotal = %lu\n", changeCountTotal);
	
	float	*results = env->GetFloatArrayElements (resultsJ, NULL);
//	cCopy (fxnMatrixH, fxnMatrix, memSizeFxn, cudaMemcpyDeviceToHost, "(results, result)", __LINE__);
//	transpose<float> (fxnMatrixH, results, numRows, numCols);
	transposeRC<kBlockSize, float, float><<< grid, threads >>>(fxnMatrix, resultHold, numRows, numCols);
	cCopy (results, resultHold, memSizeFxn, cudaMemcpyDeviceToHost, "(results, resultHold)", __LINE__);
	env->ReleaseFloatArrayElements (resultsJ, results, 0);	// Copy these results back
	
	cudaFree (mulMatrix);
	cudaFree (fxnMatrix);
	cudaFree (addMatrix);
	cudaFree (resultHold);
	cudaFree (changeCount);
	env->ReleaseFloatArrayElements (fxnMatrixJ, fxnMatrixH, JNI_ABORT);
	env->ReleaseFloatArrayElements (mulMatrixJ, mulMatrixH, JNI_ABORT);
	env->ReleaseFloatArrayElements (addMatrixJ, addMatrixH, JNI_ABORT);
	
	return converged;
}


/**
 * Cuda processing, using double precision rather than single precision<br>
 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
 * as {@code epsilon}<br>
 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
 * Input matrices are in row:column order (i.e. laid out with the first being in positions 0 - {@code numCols - 1}
 * 
 * @param fxnMatrixJ	Key matrix
 * @param mulMatrixJ	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
 * @param addMatrixJ	Matrix with same dimensions as {@code fxnMatrix}
 * @param numRowsJ	Number of rows in {@code fxnMatrix} and {@code addMatrix}.<br>
 * Number of rows and columns in {@code mulMatrix}
 * @param numColsJ	Number of cols in {@code fxnMatrix} and {@code addMatrix}
 * @param maxIterations	Maximum number of iterations to run if don't converge
 * @param epsilonJ	Result is converged if every element of {@code fxnMatrix} changes by less than 
 * {@code epsilon} during one round of processing
 * @param whichGPUJ	Which GPU to target
 * @return	The transpose of the final matrix, (i.e. in column:row order) whether or not it converged, as a 1-d matrix
 */
JNIEXPORT jboolean JNICALL Java_CRC_1Prediction_InferenceAlgoCancer_convergeMatrixCudaDOld (
	JNIEnv *env, jclass clazz, jdoubleArray fxnMatrixJ, jdoubleArray mulMatrixJ, jdoubleArray addMatrixJ, 
	jdoubleArray resultsJ, jint numRowsJ, jint numColsJ, jint maxIterations, jint whichGPUJ, jdouble epsilonJ)
{
	int	whichGPU = (int) whichGPUJ;
	cSetDevice (whichGPU, __LINE__);	// Assign code to the appropriate GPU
	
	int		numRows = (int) numRowsJ;
	int		numCols = (int) numColsJ;
	int		maxIter = (int) maxIterations;
	double	epsilon = (double) epsilonJ;
	double	*fxnMatrixH = env->GetDoubleArrayElements (fxnMatrixJ, NULL);
	double	*mulMatrixH = env->GetDoubleArrayElements (mulMatrixJ, NULL);
	double	*addMatrixH = env->GetDoubleArrayElements (addMatrixJ, NULL);
	size_t	arraySize = numRows * numCols;
	size_t	memSizeFxn = arraySize * sizeof(double);
	size_t	memSizeMul = numRows * numRows * sizeof(double);
	double	*mulMatrix, *fxnMatrix, *addMatrix, *resultHold;
//	size_t	changeCountTotal = 0;
	size_t	changeCountH;
	size_t	*changeCount;
	jboolean	converged = false;
	
#ifdef DoPrint
	printf ("In convergeMatrixCuda, numRows: %d, numCols: %d, maxIter: %d\n", numRows, numCols, maxIter);
	printArray ("mulMatrix", mulMatrixH, numRows * numRows);
	printArray ("fxnMatrix", fxnMatrixH, arraySize);
	printArray ("addMatrix", addMatrixH, arraySize);
#endif
	
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
	dim3	grid ((numCols + kBlockRounding) / kBlockSize, (numRows + kBlockRounding) / kBlockSize);
	int		i, numBlocks = (arraySize + kBigBlockRounding) / kBigBlockSize;
	
//	struct timeval	midTV;
//	gettimeofday (&midTV, NULL);
	for (i = 0; i < maxIter; ++i)
	{
#ifdef DoPrint
		printArray ("first", first, arraySize);
#endif
		matrixMulCUDA<kBlockSize, double><<< grid, threads >>>(resultHold, mulMatrix, fxnMatrix, addMatrix, numRows, numRows, numCols, numRows);
		cudaDeviceSynchronize ();
		swapD(resultHold, fxnMatrix);	// Want fxnMatrix to always hold the final results when we exit
		changeCountH = 0;
		cCopy (changeCount, &changeCountH, sizeof(size_t), cudaMemcpyHostToDevice, "(changeCount, changeCountH)", __LINE__);
		compare<kBigBlockSize, double><<< numBlocks, kBigBlockSize >>>(fxnMatrix, resultHold, arraySize, epsilon, changeCount);
		cudaDeviceSynchronize ();
		cCopy (&changeCountH, changeCount, sizeof(size_t), cudaMemcpyDeviceToHost, "(changeCountH, changeCount)", __LINE__);
		if (changeCountH == 0)
		{
			converged = true;
			break;
		}
//		else
//			changeCountTotal += changeCountH;
	}
	
	struct timeval endTV;
	gettimeofday (&endTV, NULL);
	unsigned long	theStart = (1000000 * startTV.tv_sec) + startTV.tv_usec;
//	unsigned long	theMid = (1000000 * midTV.tv_sec) + midTV.tv_usec;
	unsigned long	done = (1000000 * endTV.tv_sec) + endTV.tv_usec;
	printf ("matrixMulCUDA (double) used %lu microseconds for %d iterations\n", done - theStart, i);
//	printf ("Just matrixMulCUDA used %lu microseconds\n", done - theMid);
//	printf ("changeCountTotal = %lu\n", changeCountTotal);
	
	double	*results = env->GetDoubleArrayElements (resultsJ, NULL);
//	cCopy (fxnMatrixH, fxnMatrix, memSizeFxn, cudaMemcpyDeviceToHost, "(results, result)", __LINE__);
//	transpose<double> (fxnMatrixH, results, numRows, numCols);
	transposeRC<kBlockSize, double, double><<< grid, threads >>>(fxnMatrix, resultHold, numRows, numCols);
	cCopy (results, resultHold, memSizeFxn, cudaMemcpyDeviceToHost, "(results, resultHold)", __LINE__);
	env->ReleaseDoubleArrayElements (resultsJ, results, 0);	// Copy these results back
	
	cudaFree (mulMatrix);
	cudaFree (fxnMatrix);
	cudaFree (addMatrix);
	cudaFree (resultHold);
	cudaFree (changeCount);
	env->ReleaseDoubleArrayElements (fxnMatrixJ, fxnMatrixH, JNI_ABORT);
	env->ReleaseDoubleArrayElements (mulMatrixJ, mulMatrixH, JNI_ABORT);
	env->ReleaseDoubleArrayElements (addMatrixJ, addMatrixH, JNI_ABORT);
	
	return converged;
}


