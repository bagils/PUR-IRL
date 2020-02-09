#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define UseOMP
#undef UseOMP

#include "mkl.h"
#ifdef UseOMP
#include "omp.h"
#endif

#include "CRC_Prediction_InferenceAlgoCancer.h"

//#define DoPrint
#undef DoPrint
#define swap(a, b) {double *hold = a; a = b; b = hold;}

/**
 * Write the transpose of {@code source} into {@code target}
 * 
 * @param source	double[] to read from.  Must not be null, and length {@code numRows * numCols}
 * @param target	double[] to write to.  Must not be null, and length {@code numRows * numCols}
 * @param numRows	Number of rows in {@code source}, will be number of cols in {@code target}
 * @param numCols	Number of cols in {@code source}, will be number of rows in {@code target}
 */
void transpose (double source[], double target[], int numRows, int numCols)
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
 * Multiply two matrices together, storing the result in {@code target}<br>
 * Requires that the {@code second} matrix be transposed
 * 
 * @param first		double[] representing a double[][] with {@code fRows} rows and {@code fCols} cols
 * @param second	double[] representing a transposed double[][] that started out with 
 * {@code fCols} rows and {@code sCols} cols
 * @param target	double[] to write to, representing a double[][] with {@code fRows} rows and {@code sCols} cols
 * @param fRows		Number of rows in {@code first} and rows in {@code target}
 * @param fCols		Number of cols in {@code first} and rows in transposed {@code second}
 * @param sCols		Number of cols in transposed {@code second} and cols in {@code target}
 */
void mmul (double first[], double second[], double target[], int fRows, int fCols, int sCols)
{
#ifdef UseOMP
#pragma omp parallel for schedule(static)
#endif
	int	i, j, k;
	
	for (i = 0; i < fRows; ++i)
	{
		double	*firstPos = first + (i * fCols);
		double	*secondPos = second;
		double	*targetPos = target + (i * sCols);
		
		for (j = 0; j < sCols; ++j)
		{
			double	*firstVal = firstPos;
			double	result = 0.0;
			
			for (k = 0; k < fCols; ++k)
			{
				result += (*firstVal) * (*secondPos);
#ifdef DoPrint
				printf ("result: %lf firstVal: %lf secondPos: %lf\n", result, *firstVal, *secondPos);
#endif
				++firstVal;
				++secondPos;
			}
			
			(*targetPos) = result;
			++targetPos;
		}
	}
}


/**
 * Add the values in {@code first} to the values of {@code second}, and store the results in {@code second}
 * 
 * @param first		double[] holding values to add, same length as {@code second}
 * @param second	double[] holding values to add to, same length as {@code first}
 * @param size		Length of both {@code first} and {@code second}
 */
void madd (double *first, double *second, int size)
{
	int	i;
	
	for (i = 0; i < size; ++i)
	{
		*second += *first;
		++first;
		++second;
	}
}


/**
 * Compare the contents of two double[].  If any value of {@code first} differs from {@code second} 
 * by {@code epsilon} or more, return false.  Else return true
 * 
 * @param first		double[] holding values to test, same length as {@code second}
 * @param second	double[] holding values to test, same length as {@code first}
 * @param size		Length of both {@code first} and {@code second}
 * @param epsilon	Test value.  All matching elements of {@code first} and {@code second} must 
 * differ by less than this
 * @return False if any value in {@code first} differs from {@code second} by {@code epsilon} or more, 
 * else return true
 */
bool compare (const double *first, const double *second, int size, double epsilon)
{
	int	i;
	
	for (i = 0; i < size; ++i)
	{
		double	value = *first - *second;
		
		if (value < 0)
		{
			if ((value + epsilon) <= 0)
				return false;
		}
		else if (value >= epsilon)
			return false;
		
		++first;
		++second;
	}
	
	return true;
}


#ifdef DoPrint
void printArray (char *title, double *theArray, int size)
{
	printf ("%s\n", title);
	
	int	i;
	
	for (i = 0; i < size; ++i)
	{
		if (i != 0)
			printf (", ");
		
		printf ("%lf", theArray[i]);
	}
	
	printf ("\n");
}
#endif


JNIEXPORT jboolean JNICALL Java_CRC_1Prediction_InferenceAlgoCancer_compareMatrices (
	JNIEnv *env, jclass clazz, jdoubleArray firstJ, jdoubleArray secondJ, jint sizeJ, jdouble epsilonJ)
{
	int		size = (int) sizeJ;
	double	epsilon = (double) epsilonJ;
	double	*first = (*env)->GetDoubleArrayElements (env, firstJ, NULL);
	double	*second = (*env)->GetDoubleArrayElements (env, secondJ, NULL);
	bool	same = compare (first, second, size, epsilon);
	
	(*env)->ReleaseDoubleArrayElements (env, firstJ, first, JNI_ABORT);
	(*env)->ReleaseDoubleArrayElements (env, secondJ, second, JNI_ABORT);
	
	return (jboolean) same;
}


/**
 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
 * as {@code epsilon}<br>
 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}
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
 * @return	The transpose of the final matrix, whether or not it converged, as a 1-d matrix
 */
JNIEXPORT jboolean JNICALL Java_CRC_1Prediction_InferenceAlgoCancer_convergeMatrixJNI (
	JNIEnv *env, jclass clazz, jdoubleArray fxnMatrixJ, jdoubleArray mulMatrixJ, jdoubleArray addMatrixJ, 
	jdoubleArray resultsJ, jint numRowsJ, jint numColsJ, jint maxIterations, jdouble epsilonJ)
{
	int		numRows = (int) numRowsJ;
	int		numCols = (int) numColsJ;
	int		maxIter = (int) maxIterations;
	double	epsilon = (double) epsilonJ;
	double	*fxnMatrix = (*env)->GetDoubleArrayElements (env, fxnMatrixJ, NULL);
	double	*mulMatrix = (*env)->GetDoubleArrayElements (env, mulMatrixJ, NULL);
	double	*addMatrix = (*env)->GetDoubleArrayElements (env, addMatrixJ, NULL);
	int		i, arraySize = (*env)->GetArrayLength (env, fxnMatrixJ);
	double	*trans = (double *) malloc (arraySize * sizeof(double));
	double	*second = (double *) malloc (arraySize * sizeof(double));
	double	*first = fxnMatrix;
	jboolean	converged = false;
	
#ifdef DoPrint
	printArray ("mulMatrix", mulMatrix, numRows * numRows);
	printArray ("fxnMatrix", fxnMatrix, arraySize);
	printArray ("addMatrix", addMatrix, arraySize);
#endif
	
	for (i = 0; i < maxIter; ++i)
	{
		transpose (first, trans, numRows, numCols);
#ifdef DoPrint
		printArray ("first", first, arraySize);
		printArray ("trans", trans, arraySize);
#endif
		mmul (mulMatrix, trans, second, numRows, numRows, numCols);
		madd (addMatrix, second, arraySize);
		
		swap(first, second);	// Want first to always hold the final results when we exit
		if (compare (first, second, arraySize, epsilon))
		{
			converged = true;
			break;
		}
	}
	
	double	*results = (*env)->GetDoubleArrayElements (env, resultsJ, NULL);
	transpose (first, results, numRows, numCols);
	(*env)->ReleaseDoubleArrayElements (env, resultsJ, results, 0);	// Copy these results back
	
	if (first == fxnMatrix)
		free (second);
	else
		free (first);
	free (trans);
	(*env)->ReleaseDoubleArrayElements (env, fxnMatrixJ, fxnMatrix, JNI_ABORT);
	(*env)->ReleaseDoubleArrayElements (env, mulMatrixJ, mulMatrix, JNI_ABORT);
	(*env)->ReleaseDoubleArrayElements (env, addMatrixJ, addMatrix, JNI_ABORT);
	
	return converged;
}


/**
 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
 * as {@code epsilon}<br>
 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}
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
 * @return	The transpose of the final matrix, whether or not it converged, as a 1-d matrix
 */
JNIEXPORT void JNICALL Java_CRC_1Prediction_InferenceAlgoCancer_multiplyMatrixMKL (
	JNIEnv *env, jclass clazz, jdoubleArray firstMatrixJ, jdoubleArray secondMatrixJ, 
	jdoubleArray resultsJ, jint resultRowsJ, jint resultColsJ, jint sharedDimJ)
{
	int		resultRows = (int) resultRowsJ;
	int		numCols = (int) resultColsJ;
	int		sharedDim = (int) sharedDimJ;
	double	*firstMatrix = (*env)->GetDoubleArrayElements (env, firstMatrixJ, NULL);
	double	*secondMatrix = (*env)->GetDoubleArrayElements (env, secondMatrixJ, NULL);
	double	*results = (*env)->GetDoubleArrayElements (env, resultsJ, NULL);
	size_t	resultSize = resultRows * numCols;
	size_t	memSizeResult = resultSize * sizeof(double);
#ifdef DoPrint
	size_t	firstSize = resultRows * sharedDim;
	size_t	secondSize = sharedDim * numCols;
	
	printArray ("firstMatrix", firstMatrix, firstSize);
	printArray ("secondMatrix", secondMatrix, secondSize);
#endif
	
	struct timeval startTV;
	gettimeofday (&startTV, NULL);
	
	cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, resultRows, numCols, sharedDim, 1.0, firstMatrix, 
				 sharedDim, secondMatrix, numCols, 0.0, results, numCols);
	
	struct timeval endTV;
	gettimeofday (&endTV, NULL);
	unsigned long	theStart = (1000000 * startTV.tv_sec) + startTV.tv_usec;
	unsigned long	done = (1000000 * endTV.tv_sec) + endTV.tv_usec;
//	printf ("cblas_dgemm used %lu microseconds for %d rows, %d cols, and %d sharedDim\n", done - theStart, resultRows, numCols, sharedDim);
	
	(*env)->ReleaseDoubleArrayElements (env, resultsJ, results, 0);	// Copy these results back
	
	(*env)->ReleaseDoubleArrayElements (env, firstMatrixJ, firstMatrix, JNI_ABORT);
	(*env)->ReleaseDoubleArrayElements (env, secondMatrixJ, secondMatrix, JNI_ABORT);
}


/**
 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
 * as {@code epsilon}<br>
 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}
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
 * @return	The transpose of the final matrix, whether or not it converged, as a 1-d matrix
 */
JNIEXPORT jboolean JNICALL Java_CRC_1Prediction_InferenceAlgoCancer_convergeMatrixMKL (
	JNIEnv *env, jclass clazz, jdoubleArray fxnMatrixJ, jdoubleArray mulMatrixJ, jdoubleArray addMatrixJ, 
	jdoubleArray resultsJ, jint numRowsJ, jint numColsJ, jint maxIterations, jdouble epsilonJ)
{
	int		numRows = (int) numRowsJ;
	int		numCols = (int) numColsJ;
	int		maxIter = (int) maxIterations;
	double	epsilon = (double) epsilonJ;
	double	*fxnMatrix = (*env)->GetDoubleArrayElements (env, fxnMatrixJ, NULL);
	double	*mulMatrix = (*env)->GetDoubleArrayElements (env, mulMatrixJ, NULL);
	double	*addMatrix = (*env)->GetDoubleArrayElements (env, addMatrixJ, NULL);
	int		i, arraySize = (*env)->GetArrayLength (env, fxnMatrixJ);
	double	*second = (double *) malloc (arraySize * sizeof(double));
	double	*first = fxnMatrix;
	jboolean	converged = false;
	
#ifdef DoPrint
	printArray ("mulMatrix", mulMatrix, numRows * numRows);
	printArray ("fxnMatrix", fxnMatrix, arraySize);
	printArray ("addMatrix", addMatrix, arraySize);
#endif
	
	struct timeval startTV;
	gettimeofday (&startTV, NULL);
	
	for (i = 0; i < maxIter; ++i)
	{
#ifdef DoPrint
		printArray ("first", first, arraySize);
#endif
		cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, numRows, numCols, numRows, 1.0, mulMatrix, numRows, first, numCols, 0.0, second, numCols);
		madd (addMatrix, second, arraySize);
		
		swap(first, second);	// Want first to always hold the final results when we exit
		if (compare (first, second, arraySize, epsilon))
		{
			converged = true;
			break;
		}
	}
	
	struct timeval endTV;
	gettimeofday (&endTV, NULL);
	unsigned long	theStart = (1000000 * startTV.tv_sec) + startTV.tv_usec;
	unsigned long	done = (1000000 * endTV.tv_sec) + endTV.tv_usec;
	printf ("cblas_dgemm used %lu microseconds for %d iterations\n", done - theStart, i);
	
	double	*results = (*env)->GetDoubleArrayElements (env, resultsJ, NULL);
	transpose (first, results, numRows, numCols);
	(*env)->ReleaseDoubleArrayElements (env, resultsJ, results, 0);	// Copy these results back
	
	if (first == fxnMatrix)
		free (second);
	else
		free (first);
	(*env)->ReleaseDoubleArrayElements (env, fxnMatrixJ, fxnMatrix, JNI_ABORT);
	(*env)->ReleaseDoubleArrayElements (env, mulMatrixJ, mulMatrix, JNI_ABORT);
	(*env)->ReleaseDoubleArrayElements (env, addMatrixJ, addMatrix, JNI_ABORT);
	
	return converged;
}





