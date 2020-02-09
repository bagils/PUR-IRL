
package CRC_Prediction;


import org.apache.commons.math3.util.CombinatoricsUtils;
import org.apache.commons.math3.util.Pair;
import org.jblas.DoubleMatrix;
import CRC_Prediction.Utils.*;
import com.datastax.driver.core.Row;
import com.datastax.driver.core.Session;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import com.jprofiler.api.controller.Controller;
import org.apache.commons.lang3.time.StopWatch;
import gnu.trove.set.hash.THashSet;
import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.text.*;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;


/**
 * @author John Kalantari
 * PUR-IRL
 * Copyright 2019, Mayo Foundation for Medical Education and Research
 * 
 * PUR-IRL is licensed under the terms of GPLv3 for open source use, or
 * alternatively under the terms of the Mayo Clinic Commercial License for commercial use.
 * You may use PUR-IRL according to either of these licenses as is most appropriate
 * for your project on a case-by-case basis.
 * 
 * You should have received a copy of the GNU General Public License
 * along with PUR-IRL.  If not, see <https://www.gnu.org/licenses/>.
 *
 */
public class InferenceAlgoCancer
{
	private static final int	kOneMicroSecond = 1000;
	protected static final long kOneHundredthSecond = 10;
	protected static final long kOneQuarterSecond = 250;
	protected static final long kOneSecond = 1000;
	protected static final long kFiveSeconds = 5000;
	private static final String	kTimeStampFormat = "yyy-MM-dd-HH-mm-ss";
	private static final String	kDateTimeFormat = "yyy-MM-dd HH:mm:ss.SSS";
	private static final String	kTimeNumberFormat = "###,###.###";
	private static final String	kNumberFormat = "###,###";
	private static final int	kCompare = 0;
	private static final int	kCompareJNI = kCompare + 1;
	private static final int	kCompareBlas = kCompareJNI + 1;
	private static final boolean	kPrintDebug = false;
	
	private static final DateFormat		timeStampFormat = new SimpleDateFormat (kTimeStampFormat);
	private static final DateFormat		dateTimeFormat = new SimpleDateFormat (kDateTimeFormat);
	private static final DecimalFormat	timeFormat = new DecimalFormat (kTimeNumberFormat);
	private static final DecimalFormat	numberFormat = new DecimalFormat (kNumberFormat);
	/** String to add to thread name, followed by the number of the GPU the thread should be targeted at */
	public static final String	kGPU = "GPU";
	private static final int	kGPULen = kGPU.length ();
	private static final int	kInvalidGPU = -1;
	
	
	/**
	 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
	 * as {@code epsilon}<br>
	 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
	 * All matrices are laid out with the first row being in positions 0 - {@code numCols - 1}
	 * 
	 * @param fxnMatrix	Key matrix
	 * @param mulMatrix	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
	 * @param addMatrix	Matrix with same dimensions as {@code fxnMatrix}
	 * @param results	Matrix same size as {@code fxnMatrix}, will be filled in with transpose of 
	 * final matrix, whether or not it converged, as a 1-d matrix
	 * @param numRows	Number of rows in {@code fxnMatrix} and {@code addMatrix}.<br>
	 * Number of rows and columns in {@code mulMatrix}
	 * @param numCols	Number of cols in {@code fxnMatrix} and {@code addMatrix}
	 * @param maximumIterations	Maximum number of iterations to run if don't converge
	 * @param epsilon	Result is converged if every element of {@code fxnMatrix} changes by less than 
	 * {@code epsilon} during one round of processing
	 * @return	True if converged, false if didn't
	 */
	private static final native boolean convergeMatrixMKL (double[] fxnMatrix, double[] mulMatrix, double[] addMatrix, double[] results, 
															int numRows, int numCols, int maximumIterations, double epsilon);
	
	
	/**
	 * MKL BLAS library processing<br>
	 * Processing: {@code firstMatrix x secondMatrix}<br>
	 * All matrices are laid out with the first row being in positions 0 - {@code numCols - 1}
	 * 
	 * @param firstMatrix	Matrix whose rows define the number of output rows
	 * @param secondMatrix	Matrix whose columns define the number of output columns
	 * @param results	Matrix of size {@code firstMatrix.rows x secondMatrix.cols}, will be filled in with transpose of 
	 * final matrix, whether or not it converged, as a 1-d matrix
	 * @param numRows	Number of rows in {@code firstMatrix} and {@code results}.<br>
	 * @param numCols	Number of cols in {@code secondMatrix} and {@code secondMatrix}
	 * @param sharedDim	Number of rows in {@code secondMatrix} and cols in {@code firstMatrix}
	 * @param whichGPU	GPU to target
	 */
	private static final native void multiplyMatrixMKL (double[] firstMatrix, double[] secondMatrix, double[] results, 
													 	int resultRows, int resultCols, int sharedDim);
	
	
	/**
	 * Make a call into the library, to insure it was loaded
	 * 
	 * @param numGPUs	Number of GPUs to use
	 */
	public static final native void initCuda (int numGPUs);
	
	
	/**
	 * Cuda processing, using single precision for speed<br>
	 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
	 * as {@code epsilon}<br>
	 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
	 * All matrices are laid out with the first row being in positions 0 - {@code numCols - 1}
	 * 
	 * @param fxnMatrix	Key matrix
	 * @param mulMatrix	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
	 * @param addMatrix	Matrix with same dimensions as {@code fxnMatrix}
	 * @param results	Matrix same size as {@code fxnMatrix}, will be filled in with transpose of 
	 * final matrix, whether or not it converged, as a 1-d matrix
	 * @param numRows	Number of rows in {@code fxnMatrix} and {@code addMatrix}.<br>
	 * Number of rows and columns in {@code mulMatrix}
	 * @param numCols	Number of cols in {@code fxnMatrix} and {@code addMatrix}
	 * @param maximumIterations	Maximum number of iterations to run if don't converge
	 * @param epsilon	Result is converged if every element of {@code fxnMatrix} changes by less than 
	 * {@code epsilon} during one round of processing
	 * @param whichGPU	GPU to target
	 * @return	True if converged, false if didn't
	 */
	private static final native boolean convergeMatrixCuda (float[] fxnMatrix, float[] mulMatrix, float[] addMatrix, float[] results, 
															int numRows, int numCols, int maximumIterations, int whichGPU, float epsilon);
	
	
	/**
	 * Cuda processing, using double precision rather than single precision<br>
	 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
	 * as {@code epsilon}<br>
	 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
	 * All matrices are laid out with the first row being in positions 0 - {@code numCols - 1}
	 * 
	 * @param fxnMatrix	Key matrix
	 * @param mulMatrix	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
	 * @param addMatrix	Matrix with same dimensions as {@code fxnMatrix}
	 * @param results	Matrix same size as {@code fxnMatrix}, will be filled in with transpose of 
	 * final matrix, whether or not it converged, as a 1-d matrix
	 * @param numRows	Number of rows in {@code fxnMatrix} and {@code addMatrix}.<br>
	 * Number of rows and columns in {@code mulMatrix}
	 * @param numCols	Number of cols in {@code fxnMatrix} and {@code addMatrix}
	 * @param maximumIterations	Maximum number of iterations to run if don't converge
	 * @param epsilon	Result is converged if every element of {@code fxnMatrix} changes by less than 
	 * {@code epsilon} during one round of processing
	 * @param whichGPU	GPU to target
	 * @return	True if converged, false if didn't
	 */
	private static final native boolean convergeMatrixCudaD (double[] fxnMatrix, double[] mulMatrix, double[] addMatrix, double[] results, 
															 int numRows, int numCols, int maximumIterations, int whichGPU, double epsilon);
	
	
	/**
	 * CuBLAS library processing, using single precision for speed<br>
	 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
	 * as {@code epsilon}<br>
	 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
	 * All matrices are laid out with the first col being in positions 0 - {@code numRows - 1}, 
	 * <b>this is the layout of the {@link DoubleMatrix} data, and different from all the other 
	 * JNI versions of this code</b>
	 * 
	 * @param fxnMatrix	Key matrix
	 * @param mulMatrix	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
	 * @param addMatrix	Matrix with same dimensions as {@code fxnMatrix}
	 * @param results	Matrix same size as {@code fxnMatrix}, will be filled in with transpose of 
	 * final matrix, whether or not it converged, as a 1-d matrix
	 * @param numRows	Number of rows in {@code fxnMatrix} and {@code addMatrix}.<br>
	 * Number of rows and columns in {@code mulMatrix}
	 * @param numCols	Number of cols in {@code fxnMatrix} and {@code addMatrix}
	 * @param maximumIterations	Maximum number of iterations to run if don't converge
	 * @param epsilon	Result is converged if every element of {@code fxnMatrix} changes by less than 
	 * {@code epsilon} during one round of processing
	 * @param whichGPU	GPU to target
	 * @return	True if converged, false if didn't
	 */
	private static final native boolean convergeMatrixCuBLAS (float[] fxnMatrix, float[] mulMatrix, float[] addMatrix, float[] results, 
															  int numRows, int numCols, int maximumIterations, int whichGPU, float epsilon);
	
	
	/**
	 * CuBLAS library processing, using double precision rather than single precision<br>
	 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
	 * as {@code epsilon}<br>
	 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
	 * All matrices are laid out with the first col being in positions 0 - {@code numRows - 1}, 
	 * <b>this is the layout of the {@link DoubleMatrix} data, and different from all the other 
	 * JNI versions of this code</b>
	 * 
	 * @param fxnMatrix	Key matrix
	 * @param mulMatrix	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
	 * @param addMatrix	Matrix with same dimensions as {@code fxnMatrix}
	 * @param results	Matrix same size as {@code fxnMatrix}, will be filled in with transpose of 
	 * final matrix, whether or not it converged, as a 1-d matrix
	 * @param numRows	Number of rows in {@code fxnMatrix} and {@code addMatrix}.<br>
	 * Number of rows and columns in {@code mulMatrix}
	 * @param numCols	Number of cols in {@code fxnMatrix} and {@code addMatrix}
	 * @param maximumIterations	Maximum number of iterations to run if don't converge
	 * @param epsilon	Result is converged if every element of {@code fxnMatrix} changes by less than 
	 * {@code epsilon} during one round of processing
	 * @param whichGPU	GPU to target
	 * @return	True if converged, false if didn't
	 */
	private static final native boolean convergeMatrixCuBLASD (double[] fxnMatrix, double[] mulMatrix, double[] addMatrix, double[] results, 
															   int numRows, int numCols, int maximumIterations, int whichGPU, double epsilon);
	
	
	/**
	 * CuBLAS library processing, using double precision rather than single precision<br>
	 * Processing: {@code firstMatrix x secondMatrix}<br>
	 * All matrices are laid out with the first row being in positions 0 - {@code numRows - 1}, 
	 * <b>this is the layout of Java arrays, and different from all the other CuBLAS versions of this code</b>
	 * 
	 * @param firstMatrix	Matrix whose rows define the number of output rows
	 * @param secondMatrix	Matrix whose columns define the number of output columns
	 * @param results	Matrix of size {@code firstMatrix.rows x secondMatrix.cols}, will be filled in with transpose of 
	 * final matrix, whether or not it converged, as a 1-d matrix
	 * @param numRows	Number of rows in {@code firstMatrix} and {@code results}.<br>
	 * @param numCols	Number of cols in {@code secondMatrix} and {@code secondMatrix}
	 * @param sharedDim	Number of rows in {@code secondMatrix} and cols in {@code firstMatrix}
	 * @param whichGPU	GPU to target
	 */
	private static final native void multiplyMatrixCuBLASD (double[] firstMatrix, double[] secondMatrix, double[] results, 
														 	int resultRows, int resultCols, int sharedDim, int whichGPU);
	
	
	/**
	 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
	 * as {@code epsilon}<br>
	 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
	 * All matrices are laid out with the first row being in positions 0 - {@code numCols - 1}
	 * 
	 * @param fxnMatrix	Key matrix
	 * @param mulMatrix	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
	 * @param addMatrix	Matrix with same dimensions as {@code fxnMatrix}
	 * @param results	Matrix same size as {@code fxnMatrix}, will be filled in with transpose of 
	 * final matrix, whether or not it converged, as a 1-d matrix
	 * @param numRows	Number of rows in {@code fxnMatrix} and {@code addMatrix}.<br>
	 * Number of rows and columns in {@code mulMatrix}
	 * @param numCols	Number of cols in {@code fxnMatrix} and {@code addMatrix}
	 * @param maximumIterations	Maximum number of iterations to run if don't converge
	 * @param epsilon	Result is converged if every element of {@code fxnMatrix} changes by less than 
	 * {@code epsilon} during one round of processing
	 * @return	True if converged, false if didn't
	 */
	private static final native boolean convergeMatrixJNI (double[] fxnMatrix, double[] mulMatrix, double[] addMatrix, double[] results, 
															int numRows, int numCols, int maximumIterations, double epsilon);
	
	
	/**
	 * Compare contents of two matrices, returning true if every element in {@code first} is less 
	 * than {@code epsilon} from its matching element in {@code second}, else false
	 * 
	 * @param first		First matrix
	 * @param second	Second matrix
	 * @param size		Length of both matrices
	 * @param epsilon	Matrices are equal if every element of {@code first} differs by less than 
	 * {@code epsilon} from matching element in {@code second}
	 * @return	True if "equal", false if not
	 */
	private static final native boolean compareMatrices (double[] first, double[] second, int size, double epsilon);
	
	
	private static BlockingQueue<Integer>	gpuQueue = null;
	
	/**
	 * Create the GPU access {@link BlockingQueue} and fill it with the GPUs that can be used, 
	 * then initialize the Cuda and CuBLAS side
	 * 
	 * @param numGPUs	Number of GPUs to use
	 */
	public static final void initGPUs (int numGPUs)
	{
		gpuQueue = new ArrayBlockingQueue<> (numGPUs);
		
		for (int i = 0; i < numGPUs; ++i)
			gpuQueue.add (Integer.valueOf (i));
		
		initCuda (numGPUs);
	}
	
	
	/**
	 * Get the next available GPU, waiting if all GPUs are in use
	 * 
	 * @return	Number of the GPU to use, from 0 to {@code numGPUs - 1}, or {@value #kInvalidGPU} if 
	 * not to use a GPU
	 */
	private static final int getGPU ()
	{
		if (gpuQueue == null)
			return kInvalidGPU;
		
		try
		{
			return gpuQueue.take ().intValue ();
		}
		catch (InterruptedException oops)
		{
			oops.printStackTrace ();
			return kInvalidGPU;
		}
	}
	
	
	/**
	 * Release a GPU to be used by a different thread
	 * 
	 * @param theGPU	GPU to be available.  Must be > {@value #kInvalidGPU}
	 */
	private static final void releaseGPU (int theGPU)
	{
		if (theGPU > kInvalidGPU)
			gpuQueue.add (Integer.valueOf (theGPU));
	}
	
	
	/**
	 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
	 * as {@code epsilon}<br>
	 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
	 * All matrices are laid out with the first row being in positions 0 - {@code numCols - 1}
	 * 
	 * @param fxnMatrix	Key matrix
	 * @param mulMatrix	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
	 * @param addMatrix	Matrix with same dimensions as {@code fxnMatrix}
	 * @param maximumIterations	Maximum number of iterations to run if don't converge
	 * @param epsilon	Result is converged if every element of {@code fxnMatrix} changes by less than 
	 * {@code epsilon} during one round of processing
	 * @return	The transpose of the final matrix, whether or not it converged
	 */
	public static final double[][] convergeMatrix (DoubleMatrix fxnMatrix, DoubleMatrix mulMatrix, DoubleMatrix addMatrix, 
													int maximumIterations, double epsilon)
	{
		return convergeMatrixCuBLAS (fxnMatrix, mulMatrix, addMatrix, maximumIterations, epsilon);
//		int	gpu = getGPU ();
//		if (gpu != kInvalidGPU)
//			return convergeMatrixCuda (fxnMatrix, mulMatrix, addMatrix, maximumIterations, gpu, epsilon);
//		
//		int			numRows = fxnMatrix.rows;
//		int			numCols = fxnMatrix.columns;
//		int			matrixLen = numRows * numCols;
//		double[]	intResults = new double[matrixLen];
////		StopWatch	watch = StopWatch.createStarted ();
//		
//		convergeMatrixMKL (toArray (fxnMatrix), toArray (mulMatrix), toArray (addMatrix), intResults, numRows, numCols, maximumIterations, epsilon);
////		watch.stop ();
////		reportTime ("Time required for convergeMatrixMKL: ", watch.getTime ());
////		printMatrix ("intResults", intResults);
//		double[][]	results = new double[numCols][];
//		int			pos = 0;
//		
//		for (int i = 0; i < numCols; ++i)
//		{
//			double[]	row = results[i] = new double[numRows];
//			
//			System.arraycopy (intResults, pos, row, 0, numRows);
//			pos += numRows;
//		}
//		
//		return results;
	}
	
	
	/**
	 * Processing: {@code firstMatrix x secondMatrix}<br>
	 * 
	 * @param firstMatrix	Matrix whose rows define the number of output rows
	 * @param secondMatrix	Matrix whose columns define the number of output columns
	 * @param results		double[] in which to put the results
	 * @return	double[] of size {@code firstMatrix.rows x secondMatrix.cols}, will be filled in with 
	 * final matrix, in row-major order if there's more than one column
	 */
	public static final double[] multiplyMatrix1 (double[][] firstMatrix, double[][] secondMatrix, double[] results)
	{
//		int	gpu = getGPUT ();
//		if (gpu != kInvalidGPU)
//			return multiplyMatrixCuda1 (firstMatrix, secondMatrix, results, gpu);
		
		int			numRows = firstMatrix.length;
		int			sharedDim = secondMatrix.length;
		int			numCols = secondMatrix[0].length;
//		int			matrixLen = numRows * numCols;
//		double[]	results = new double[matrixLen];
//		StopWatch	watch = StopWatch.createStarted ();
		
		multiplyMatrixMKL (toArray (firstMatrix), toArray (secondMatrix), results, numRows, numCols, sharedDim);
//		watch.stop ();
//		reportTime ("Time required for multiplyMatrixMKL1: ", watch.getTime ());
//		printMatrix ("intResults", intResults);
		
		return results;
	}
	
	
	/**
	 * Processing: {@code firstMatrix x secondMatrix}<br>
	 * 
	 * @param firstMatrix	Matrix whose rows define the number of output rows
	 * @param secondMatrix	Matrix whose columns define the number of output columns
	 * @param whichGPU	GPU to target
	 * @return	double[] of size {@code firstMatrix.rows x secondMatrix.cols}, will be filled in with 
	 * final matrix, in row-major order if there's more than one column
	 */
	private static final double[] multiplyMatrixCuda1 (double[][] firstMatrix, double[][] secondMatrix, double[] results, int whichGPU)
	{
		int			numRows = firstMatrix.length;
		int			sharedDim = secondMatrix.length;
		int			numCols = secondMatrix[0].length;
//		int			matrixLen = numRows * numCols;
//		double[]	results = new double[matrixLen];
//		StopWatch	watch = StopWatch.createStarted ();
		
		multiplyMatrixCuBLASD (toArray (firstMatrix), toArray (secondMatrix), results, numRows, numCols, sharedDim, whichGPU);
//		watch.stop ();
//		reportTime ("Time required for multiplyMatrixCuBLASD: ", watch.getTime ());
//		printMatrix ("intResults", intResults);
		
		return results;
	}
	
	
	/**
	 * Processing: {@code firstMatrix x secondMatrix}<br>
	 * 
	 * @param firstMatrix	Matrix whose rows define the number of output rows
	 * @param secondMatrix	Matrix whose columns define the number of output columns
	 * @return	Matrix of size {@code firstMatrix.rows x secondMatrix.cols}, will be filled in with 
	 * final matrix, as a normal Java matrix
	 */
	public static final double[][] multiplyMatrix (double[][] firstMatrix, double[][] secondMatrix)
	{
		int	gpu = getGPUT ();
		if (gpu != kInvalidGPU)
			return multiplyMatrixCuda (firstMatrix, secondMatrix, gpu);
		
		int			numRows = firstMatrix.length;
		int			sharedDim = secondMatrix.length;
		int			numCols = secondMatrix[0].length;
		int			matrixLen = numRows * numCols;
		double[]	intResults = new double[matrixLen];
		StopWatch	watch = StopWatch.createStarted ();
		
		multiplyMatrixMKL (toArray (firstMatrix), toArray (secondMatrix), intResults, numRows, numCols, sharedDim);
		watch.stop ();
		reportTime ("Time required for multiplyMatrixMKL: ", watch.getTime ());
//		printMatrix ("intResults", intResults);
		double[][]	results = new double[numRows][];
		int			pos = 0;
		
		for (int i = 0; i < numRows; ++i)
		{
			double[]	row = results[i] = new double[numCols];
			
			System.arraycopy (intResults, pos, row, 0, numCols);
			pos += numCols;
		}
		
		return results;
	}
	
	
	/**
	 * Processing: {@code firstMatrix x secondMatrix}<br>
	 * 
	 * @param firstMatrix	Matrix whose rows define the number of output rows
	 * @param secondMatrix	Matrix whose columns define the number of output columns
	 * @param whichGPU	GPU to target
	 * @return	Matrix of size {@code firstMatrix.rows x secondMatrix.cols}, will be filled in with 
	 * final matrix, as a normal Java matrix
	 */
	private static final double[][] multiplyMatrixCuda (double[][] firstMatrix, double[][] secondMatrix, int whichGPU)
	{
		int			numRows = firstMatrix.length;
		int			sharedDim = secondMatrix.length;
		int			numCols = secondMatrix[0].length;
		int			matrixLen = numRows * numCols;
		double[]	intResults = new double[matrixLen];
		StopWatch	watch = StopWatch.createStarted ();
		
		multiplyMatrixCuBLASD (toArray (firstMatrix), toArray (secondMatrix), intResults, numRows, numCols, sharedDim, whichGPU);
		watch.stop ();
		reportTime ("Time required for multiplyMatrixCuBLASD: ", watch.getTime ());
//		printMatrix ("intResults", intResults);
		double[][]	results = new double[numRows][];
		int			pos = 0;
		
		for (int i = 0; i < numRows; ++i)
		{
			double[]	row = results[i] = new double[numCols];
			
			System.arraycopy (intResults, pos, row, 0, numCols);
			pos += numCols;
		}
		
		return results;
	}
	
	
	/**
	 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
	 * as {@code epsilon}<br>
	 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
	 * All matrices are laid out with the first row being in positions 0 - {@code numCols - 1}
	 * 
	 * @param fxnMatrix	Key matrix
	 * @param mulMatrix	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
	 * @param addMatrix	Matrix with same dimensions as {@code fxnMatrix}
	 * @param maximumIterations	Maximum number of iterations to run if don't converge
	 * @param epsilon	Result is converged if every element of {@code fxnMatrix} changes by less than 
	 * {@code epsilon} during one round of processing
	 * @return	The transpose of the final matrix, whether or not it converged
	 */
	private static final double[][] convergeMatrixMKL (DoubleMatrix fxnMatrix, DoubleMatrix mulMatrix, DoubleMatrix addMatrix, 
														int maximumIterations, double epsilon)
	{
		int			numRows = fxnMatrix.rows;
		int			numCols = fxnMatrix.columns;
		int			matrixLen = numRows * numCols;
		double[]	intResults = new double[matrixLen];
//		StopWatch	watch = StopWatch.createStarted ();
		
		convergeMatrixMKL (toArray (fxnMatrix), toArray (mulMatrix), toArray (addMatrix), intResults, numRows, numCols, maximumIterations, epsilon);
//		watch.stop ();
//		reportTime ("Time required for convergeMatrixMKL: ", watch.getTime ());
//		printMatrix ("intResults", intResults);
		double[][]	results = new double[numCols][];
		int			pos = 0;
		
		for (int i = 0; i < numCols; ++i)
		{
			double[]	row = results[i] = new double[numRows];
			
			System.arraycopy (intResults, pos, row, 0, numRows);
			pos += numRows;
		}
		
		return results;
	}
	
	
	/**
	 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
	 * as {@code epsilon}<br>
	 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
	 * All matrices are laid out with the first row being in positions 0 - {@code numCols - 1}
	 * 
	 * @param fxnMatrix	Key matrix
	 * @param mulMatrix	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
	 * @param addMatrix	Matrix with same dimensions as {@code fxnMatrix}
	 * @param maximumIterations	Maximum number of iterations to run if don't converge
	 * @param epsilon	Result is converged if every element of {@code fxnMatrix} changes by less than 
	 * {@code epsilon} during one round of processing
	 * @param whichGPU	GPU to target
	 * @return	The transpose of the final matrix, whether or not it converged
	 */
	private static final double[][] convergeMatrixCuda (DoubleMatrix fxnMatrix, DoubleMatrix mulMatrix, DoubleMatrix addMatrix, 
														int maximumIterations, int whichGPU, double epsilon)
	{
//		boolean	useBLAS = (whichGPU & 0B10) != 0;	// GPUs < 2 use my code, >=2 use BLAS
		boolean	useBLAS = true;
		if (useBLAS)
			return convergeMatrixCuBLASO (fxnMatrix, mulMatrix, addMatrix, maximumIterations, whichGPU, epsilon);
		
		int			numRows = fxnMatrix.rows;
		int			numCols = fxnMatrix.columns;
		int			matrixLen = numRows * numCols;
		boolean		useFloat = (whichGPU & 1) == 0;	// Odd GPUs get doubles, even get floats
//		boolean		useFloat = false;
		double[][]	results = new double[numCols][];
//		StopWatch	watch = StopWatch.createStarted ();
		int			pos = 0;
		
		if (useFloat)
		{
			float[]	intResults = new float[matrixLen];
			
			convergeMatrixCuda (toFArray (fxnMatrix), toFArray (mulMatrix), toFArray (addMatrix), intResults, 
								numRows, numCols, maximumIterations, whichGPU, (float) epsilon);
//			printMatrix ("intResults", intResults);
			
			for (int i = 0; i < numCols; ++i)
			{
				double[]	row = results[i] = new double[numRows];
				
				for (int j = 0; j < numRows; ++j, ++pos)
					row[j] = intResults[pos];
			}
		}
		else
		{
			double[]	intResults = new double[matrixLen];
			
			convergeMatrixCudaD (toArray (fxnMatrix), toArray (mulMatrix), toArray (addMatrix), 
								 intResults, numRows, numCols, maximumIterations, whichGPU, epsilon);
//			printMatrix ("intResults", intResults);
			
			for (int i = 0; i < numCols; ++i)
			{
				double[]	row = results[i] = new double[numRows];
				
				System.arraycopy (intResults, pos, row, 0, numRows);
				pos += numRows;
			}
		}
		
//		watch.stop ();
//		if (useFloat)
//			reportTime ("Time required for convergeMatrixCuda (float): ", watch.getTime ());
//		else
//			reportTime ("Time required for convergeMatrixCuda (double): ", watch.getTime ());
		return results;
	}
	
	
	/**
	 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
	 * as {@code epsilon}<br>
	 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
	 * All matrices are laid out with the first row being in positions 0 - {@code numCols - 1}
	 * 
	 * @param fxnMatrix	Key matrix
	 * @param mulMatrix	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
	 * @param addMatrix	Matrix with same dimensions as {@code fxnMatrix}
	 * @param maximumIterations	Maximum number of iterations to run if don't converge
	 * @param epsilon	Result is converged if every element of {@code fxnMatrix} changes by less than 
	 * {@code epsilon} during one round of processing
	 * @return	The transpose of the final matrix, whether or not it converged
	 */
	private static final double[][] convergeMatrixCuBLAS (DoubleMatrix fxnMatrix, DoubleMatrix mulMatrix, DoubleMatrix addMatrix, 
														  int maximumIterations, double epsilon)
	{
		int			numRows = fxnMatrix.rows;
		int			numCols = fxnMatrix.columns;
		int			matrixLen = numRows * numCols;
		double[][]	results = new double[numCols][];
//		StopWatch	watch = StopWatch.createStarted ();
		int			pos = 0;
		double[]	intResults = new double[matrixLen];
		int			whichGPU = getGPU ();
		
		if (whichGPU == kInvalidGPU)
			return convergeMatrixMKL (fxnMatrix, mulMatrix, addMatrix, maximumIterations, epsilon);
		
		convergeMatrixCuBLASD (fxnMatrix.toArray (), mulMatrix.toArray (), addMatrix.toArray (), 
							   intResults, numRows, numCols, maximumIterations, whichGPU, epsilon);
		releaseGPU (whichGPU);
//		printMatrix ("intResults", intResults);
		
		for (int i = 0; i < numCols; ++i)
		{
			double[]	row = results[i] = new double[numRows];
			
			System.arraycopy (intResults, pos, row, 0, numRows);
			pos += numRows;
		}
		
//		watch.stop ();
//		if (useFloat)
//			reportTime ("Time required for convergeMatrixCuBLAS (float): ", watch.getTime ());
//		else
//			reportTime ("Time required for convergeMatrixCuBLAS (double): ", watch.getTime ());
		return results;
	}
	
	
	/**
	 * Process {@code fxnMatrix} {@code maximumIterations} times, or until no element changes as much 
	 * as {@code epsilon}<br>
	 * Processing: {@code (mulMatrix x fxnMatrix) + addMatrix}<br>
	 * All matrices are laid out with the first row being in positions 0 - {@code numCols - 1}
	 * 
	 * @param fxnMatrix	Key matrix
	 * @param mulMatrix	Square matrix with same number of rows and columns as {@code fxnMatrix} has rows
	 * @param addMatrix	Matrix with same dimensions as {@code fxnMatrix}
	 * @param maximumIterations	Maximum number of iterations to run if don't converge
	 * @param epsilon	Result is converged if every element of {@code fxnMatrix} changes by less than 
	 * {@code epsilon} during one round of processing
	 * @param whichGPU	GPU to target
	 * @return	The transpose of the final matrix, whether or not it converged
	 */
	private static final double[][] convergeMatrixCuBLASO (DoubleMatrix fxnMatrix, DoubleMatrix mulMatrix, DoubleMatrix addMatrix, 
															int maximumIterations, int whichGPU, double epsilon)
	{
		int			numRows = fxnMatrix.rows;
		int			numCols = fxnMatrix.columns;
		int			matrixLen = numRows * numCols;
//		boolean		useFloat = (whichGPU & 1) == 0;	// Odd GPUs get doubles, even get floats
		boolean		useFloat = false;
		double[][]	results = new double[numCols][];
//		StopWatch	watch = StopWatch.createStarted ();
		int			pos = 0;
		
		if (useFloat)
		{
			float[]	intResults = new float[matrixLen];
			
			convergeMatrixCuBLAS (toArrayF (fxnMatrix), toArrayF (mulMatrix), toArrayF (addMatrix), intResults, 
								  numRows, numCols, maximumIterations, whichGPU, (float) epsilon);
//			printMatrix ("intResults", intResults);
			
			for (int i = 0; i < numCols; ++i)
			{
				double[]	row = results[i] = new double[numRows];
				
				for (int j = 0; j < numRows; ++j, ++pos)
					row[j] = intResults[pos];
			}
		}
		else
		{
			double[]	intResults = new double[matrixLen];
			
			convergeMatrixCuBLASD (fxnMatrix.toArray (), mulMatrix.toArray (), addMatrix.toArray (), 
								   intResults, numRows, numCols, maximumIterations, whichGPU, epsilon);
//			printMatrix ("intResults", intResults);
			
			for (int i = 0; i < numCols; ++i)
			{
				double[]	row = results[i] = new double[numRows];
				
				System.arraycopy (intResults, pos, row, 0, numRows);
				pos += numRows;
			}
		}
		
//		watch.stop ();
//		if (useFloat)
//			reportTime ("Time required for convergeMatrixCuBLAS (float): ", watch.getTime ());
//		else
//			reportTime ("Time required for convergeMatrixCuBLAS (double): ", watch.getTime ());
		return results;
	}
	
	
	/**
	 * Compare contents of two matrices, returning true if every element in {@code first} is less 
	 * than {@code epsilon} from its matching element in {@code second}, else false
	 * 
	 * @param first		First matrix
	 * @param second	Second matrix
	 * @param epsilon	Matrices are equal if every element of {@code first} differs by less than 
	 * {@code epsilon} from matching element in {@code second}
	 * @return	True if "equal", false if not
	 */
	private static final boolean compareMatricesJNI (DoubleMatrix first, DoubleMatrix second, double epsilon)
	{
		return compareMatrices (first.toArray (), second.toArray (), first.length, epsilon);
	}
	
	
	/**
	 * Compare contents of two matrices, returning true if every element in {@code first} is less 
	 * than {@code epsilon} from its matching element in {@code second}, else false
	 * 
	 * @param first		First matrix
	 * @param second	Second matrix
	 * @param epsilon	Matrices are equal if every element of {@code first} differs by less than 
	 * {@code epsilon} from matching element in {@code second}
	 * @return	True if "equal", false if not
	 */
	public static final boolean compareMatrices (DoubleMatrix first, DoubleMatrix second, double epsilon)
	{
		int			pos = 0;
		double[]	secondVals = second.toArray ();
		
		for (double fValue : first.toArray ())
		{
			double	sValue = secondVals[pos];
			double	value = fValue - sValue;
			
			if (value < 0.0)
			{
				if ((value + epsilon) <= 0)
					return false;
			}
			else if (value >= epsilon)
				return false;
			++pos;
		}
		
		return true;
	}
	
	
	/**
	 * 
	 * 
	 * @param nRows		7032
	 * @param nCols		299
	 * @param epsilon	1e-12
	 */
	public static final void matrixTest (int nRows, int nCols, double epsilon)
	{
		int		numIterations = 20;
		double	testLevel = 0.1;
		
		for (int i = 1; i <= 5; ++i)
			matrixTest (nRows, nCols, epsilon, testLevel * i, numIterations);
	}
	
	
	private static final void matrixTest (int nRows, int nCols, double epsilon, double testLevel, int numIterations)
	{
		long	seed = new Random ().nextLong ();
		
//		for (int whichTest = kCompare; whichTest <= kCompareBlas; ++whichTest)
		for (int whichTest = kCompareBlas; whichTest >= kCompare; --whichTest)
			matrixTestCompare (seed, nRows, nCols, epsilon, testLevel, numIterations, whichTest);
	}
	
	
	/**
	 * 
	 * 
	 * @param seed
	 * @param nRows
	 * @param nCols
	 * @param epsilon
	 * @param testLevel
	 * @param numIterations
	 * @param whichTest
	 */
	private static void matrixTestCompare (long seed, int nRows, int nCols, double epsilon, double testLevel, int numIterations, int whichTest)
	{
		double[][]		testMatrix = new double[nRows][nCols];
		double[][]		minusMatrix = new double[nRows][nCols];
		DoubleMatrix	first = new DoubleMatrix (testMatrix);
		DoubleMatrix	second = new DoubleMatrix (minusMatrix);
		
		Random	rng = new Random (seed);
		long	start = new Date ().getTime ();
		long	matTotal = 0;
		int		numEqual = 0;
		
		for (int iter = 0; iter < numIterations; ++iter)
		{
			long	matStart = new Date ().getTime ();
			makeNearMatrices (testMatrix, minusMatrix, nRows, nCols, epsilon * testLevel, rng);
			matTotal += new Date ().getTime () - matStart;
			
			badCopy (testMatrix, first);
			badCopy (minusMatrix, second);
			
			switch (whichTest)
			{
				case kCompare: 
					if (compareMatrices (first, second, epsilon))
						++numEqual;
					break;
					
				case kCompareJNI: 
					if (compareMatricesJNI (first, second, epsilon))
						++numEqual;
					break;
					
				case kCompareBlas: 
					if (MatrixUtilityJBLAS.compareREALMatrices (first, second, epsilon))
						++numEqual;
					break;
			}
		}
		
		switch (whichTest)
		{
			case kCompare: 
				reportTime ("compareMatrices", start, matTotal, numIterations, numEqual, testLevel);
				break;
				
			case kCompareJNI: 
				reportTime ("compareMatricesJNI", start, matTotal, numIterations, numEqual, testLevel);
				break;
				
			case kCompareBlas: 
				reportTime ("MatrixUtilityJBLAS.compareREALMatrices", start, matTotal, numIterations, numEqual, testLevel);
				break;
		}
	}
	
	
	/**
	 * 
	 * 
	 * @param title
	 * @param start
	 * @param matTotal
	 * @param numIterations
	 * @param numEqual
	 * @param testLevel
	 */
	private static void reportTime (String title, long start, long matTotal, int numIterations, int numEqual, double testLevel)
	{
		long	end = new Date ().getTime ();
		long	elapsed = end - start;
		
		System.out.print (title);
		System.out.print (" took ");
		System.out.print (Long.toString (elapsed));
		System.out.print (" to compare ");
		System.out.print (Integer.toString (numIterations));
		System.out.print (" matrices, of which ");
		System.out.print (Long.toString (matTotal));
		System.out.print (" was spent updating the matrices, for a total of ");
		System.out.print (Long.toString (elapsed - matTotal));
		System.out.print (" spent comparing, found ");
		System.out.print (Integer.toString (numEqual));
		System.out.print (" that were converged, testLevel was ");
		System.out.println (Double.toString (testLevel));
	}


	/**
	 * Do an invalid copy from {@code testMatrix} to {@code first}
	 * 
	 * @param testMatrix
	 * @param first
	 */
	private static final void badCopy (double[][] testMatrix, DoubleMatrix first)
	{
		int	pos = 0;
		for (double[] row : testMatrix)
		{
			int	len = row.length;
			
			System.arraycopy (row, 0, first.data, pos, len);
			pos += len;
		}
		
	}
	
	
	/**
	 * Fill in two matrices, the first with random number between 0 and 10,000, the second with 
	 * the values from the first, plus a Gausian determined offset with mean of 0 and 
	 * std deviation of epsilon
	 * 
	 * @param baseMatrix
	 * @param nearMatrix
	 * @param nRows
	 * @param nCols
	 * @param epsilon
	 * @param rng
	 */
	private static final void makeNearMatrices (double[][] baseMatrix, double[][] nearMatrix, int nRows, int nCols, double epsilon, Random rng)
	{
		for (int i = 0; i < nRows; ++i)
		{
			double[] row = baseMatrix[i];
			double[] fillRow = nearMatrix[i];
			
			for (int j = 0; j < nCols; ++j)
			{
				double	value = row[j] = rng.nextDouble ();
				double	offset = rng.nextGaussian () * epsilon;	
				fillRow[j] = value + offset;
			}
		}
	}


	/**
	 * Print out the contents of a 1d matrix
	 * 
	 * @param title	Title line to print out, if not null and not empty
	 * @param row	The double[] to print out
	 */
	protected static final void printMatrix (String title, double[] row)
	{
		if ((title != null) && !title.isEmpty ())
			System.out.println (title);
		
		int	numCols = Math.min (row.length, 100);
		
		for (int j = 0; j < numCols; ++j)
		{
			if (j != 0)
				System.out.print (", ");
			System.out.print (row[j]);
		}
		System.out.println ();
	}
	
	
	/**
	 * Create an array of doubles in row / column order, meaning all the columns of the first row, 
	 * followed by all the columns of the second row, etc
	 * 
	 * @param theMatrix	{@link DoubleMatrix} to map
	 * @return	double[]
	 */
	private static final double[] toArray (double[][] theMatrix)
	{
		int			numRows = theMatrix.length;
		int			numCols = theMatrix[0].length;
		double[]	results = new double[numRows * numCols];
		int			pos = 0;
		
		for (int i = 0; i < numRows; ++i)
		{
			double[]	row = theMatrix[i];
			
			System.arraycopy (row, 0, results, pos, numCols);
			pos += numCols;
		}
		
		return results;
	}
	
	
	/**
	 * Create an array of doubles in row / column order, meaning all the columns of the first row, 
	 * followed by all the columns of the second row, etc
	 * 
	 * @param theMatrix	{@link DoubleMatrix} to map
	 * @return	double[]
	 */
	private static final double[] toArray (DoubleMatrix theMatrix)
	{
		int			numRows = theMatrix.rows;
		int			numCols = theMatrix.columns;
		double[][]	array2D = theMatrix.toArray2 ();
		double[]	results = new double[numRows * numCols];
		int			pos = 0;
		
		for (int i = 0; i < numRows; ++i)
		{
			double[]	row = array2D[i];
			
			System.arraycopy (row, 0, results, pos, numCols);
			pos += numCols;
		}
		
		return results;
	}
	
	
	/**
	 * Create an array of floats in row / column order, meaning all the columns of the first row, 
	 * followed by all the columns of the second row, etc
	 * 
	 * @param theMatrix	{@link DoubleMatrix} to map
	 * @return	float[]
	 */
	private static final float[] toFArray (DoubleMatrix theMatrix)
	{
//		int			numRows = theMatrix.rows;
//		int			numCols = theMatrix.columns;
//		double[][]	array2D = theMatrix.toArray2 ();
//		float[]		results = new float[numRows * numCols];
//		int			pos = 0;
//		
//		for (int i = 0; i < numRows; ++i)
//		{
//			double[]	row = array2D[i];
//			
//			for (int j = 0; j < numCols; ++j, ++pos)
//				results[pos] = (float) row[j];
//		}
//		
//		return results;
		int		numRows = theMatrix.rows;
		int		numCols = theMatrix.columns;
		float[]	results = new float[numRows * numCols];
		int		pos = 0;
		
		for (int i = 0; i < numRows; ++i)
		{
			for (int j = 0; j < numCols; ++j, ++pos)
				results[pos] = (float) theMatrix.get (i, j);
		}
		
		return results;
	}
	
	
	/**
	 * Create an array of floats in column / row order, meaning all the rows of the first column, 
	 * followed by all the rows of the second column, etc
	 * 
	 * @param theMatrix	{@link DoubleMatrix} to map
	 * @return	float[]
	 */
	private static final float[] toArrayF (DoubleMatrix theMatrix)
	{
		int			numRows = theMatrix.rows;
		int			numCols = theMatrix.columns;
		float[]		results = new float[numRows * numCols];
		double[]	data = theMatrix.data;
		int			pos = 0;
		
		for (int i = 0; i < numCols; ++i)
		{
			for (int j = 0; j < numRows; ++j, ++pos)
				results[pos] = (float) data[pos];
		}
		
		return results;
	}
	
	
	/** RNG to use */
	public static MersenneTwisterFastIRL	RNG;
	/** {@link Session} used */
	public static Session					_session;
	
	// trajectorySet is matrix of dimension : nTrajs x nSteps x 2; each trajectory is of length
	// (nSteps), where each step consists of the PAIR state and action
	
	/**
	 * 
	 * 
	 * @param environment
	 * @param trajectorySet
	 * @param irlAlgo
	 * @param mhSampledRestaurants
	 * @param bestSampledRestaurant
	 * @param seed
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param maxTables
	 * @param boolStartFromScratch
	 * @param restaurantFactory
	 * @param mdpFactory
	 * @throws Exception
	 * @throws FileNotFoundException
	 * @throws IOException
	 * @throws ClassNotFoundException
	 * @throws InstantiationException
	 * @throws IllegalAccessException
	 * @throws NoSuchMethodException
	 * @throws IllegalArgumentException
	 * @throws InvocationTargetException
	 */
	public static void ChineseRestaurantProcessInference (MDPCancer environment, List<double[][]> trajectorySet, IRLAlgorithmCancer irlAlgo, 
														  THashSet<IRLRestaurant> mhSampledRestaurants, IRLRestaurant bestSampledRestaurant, 
														  long seed, int numThreads, int numGPUs, int maxTables, boolean boolStartFromScratch, 
														  IRLRestaurantFactory restaurantFactory, MDPCancerFactory mdpFactory, boolean profile) 
		throws Exception, FileNotFoundException, IOException, ClassNotFoundException, InstantiationException, IllegalAccessException, 
				NoSuchMethodException, IllegalArgumentException, InvocationTargetException
	{
		if (seed != 0)
			RNG = new MersenneTwisterFastIRL (seed);
		else
			RNG = new MersenneTwisterFastIRL ();
		
		// XXX:Trajectory data: NOTE: each trajectory is defined by 2 sequences: 1. a sequence of
		// states, and 2. a sequence of actions.
		int numberTrajectories = trajectorySet.size (); // should be same as MDP._numberTrajectories
		System.out.println ("Total numTrajectories: " + numberTrajectories);
		
//		boolean addInferredTablesDuringRestaurantInitialization = false; //JK 6.20.2019: added
		if (maxTables == 0)
			maxTables = numberTrajectories;
		int	numberRewardFeatures = environment.getNumRewardFeatures ();
		
		// map of state-action counts for trajectories of interest
		// includes the 'count' of each 'observed' state-action pair in the subset of trajectory set.
		// Thus, if we consider all trajectories to extract 'count' information, the maximum size of
		// this trajectoryInfo dataset is: nTrajs x nSteps x 3
		Multimap<Integer, double[]> stateActionPairCountsInfoForAllTrajectories = ArrayListMultimap.create ();
		reportEvent ("Going to compute occupancy, one customer at a time before table initialization ");
		for (int i = 0; i < numberTrajectories; i++)
		{
			// Compute occupancy and the empirical policy for trajectories
			
			Map<Integer, double[][]> subsetOfTrajectoriesToAnalyze = new HashMap<Integer, double[][]> ();
			subsetOfTrajectoriesToAnalyze.put (i, trajectorySet.get (i));
			stateActionPairCountsInfoForAllTrajectories = computeOccupancy (subsetOfTrajectoriesToAnalyze, environment, 
																			stateActionPairCountsInfoForAllTrajectories, false);
			subsetOfTrajectoriesToAnalyze.clear ();
			// set the state-action counts for each trajectory; i.e. the set of state-action pairs
			// visited in this trajectory and their respective counts associated with number of visits
		}
		
		// initialize tables
		StopWatch watch = StopWatch.createStarted ();
		
		double [][] tableAssignmentMatrix = null;
		int highestTableIdxN = 0;
		// stores the weight-vector associated with each table index/value; Although the
		// numberRewardFeatures is fixed, the number of active tables at any given moment can
		// change, so we need the set of weight vectors be dynamic in size.
		Map<Integer, double[][]> tableWeightVectors = null;
		// each policy is a column matrix of dimension numStates x 1
		Map<Integer, double[][]> tablePolicyVectors = null;
		// each value is a column matrix of dimension numStates x 1 (i.e. it is NOT a row vector)
		Map<Integer, double[][]> tableValueVectors = null;		
		// each value is a column matrix of dimension (numStates * numActions) x 1  (i.e. it is NOT a row vector)
		Map<Integer, double[][]> tableQVectors = null;
		
		if (!boolStartFromScratch) // if not building from scratch,  you will be overriding any other input argument which may also specify num of tables 
		{
//			tableAssignmentMatrix = bestSampledRestaurant.getSeatingArrangement ();
			// returns a row vector containing the maximum values in each column of the
			// 'tableAssignmentMatrix'. Since this matrix is nTraj x 1. We only care about the 0th
			// element of the returned row vector
			Double highestTableIndex = MatrixUtilityJBLAS.maxPerColumn (bestSampledRestaurant.getSeatingArrangement ())[0];
			highestTableIdxN = highestTableIndex.intValue ();
			maxTables = Math.min (highestTableIdxN, numberTrajectories);
			tableAssignmentMatrix = MersenneTwisterFastIRL.RandomUniformMatrixWithIntervalMin (numberTrajectories, 1, maxTables, 1);
			
			reportValue ("Highest table index in pre-existing restaurant's table assignment", highestTableIdxN);
			
			// stores the weight-vector associated with each table index/value; Although the
			// numberRewardFeatures is fixed, the number of active tables at any given moment can
			// change, so we need the set of weight vectors be dynamic in size.
			tableWeightVectors = bestSampledRestaurant.getWeightMatrices ();
			// each policy is a column matrix of dimension numStates x 1
			tablePolicyVectors = bestSampledRestaurant.getPolicyMatrices ();
			// each value is a column matrix of dimension numStates x 1 (i.e. it is NOT a row vector)
			tableValueVectors = bestSampledRestaurant.getValueMatrices ();
			// each value is a column matrix of dimension (numStates * numActions) x 1  (i.e. it is NOT a row vector)
			tableQVectors = bestSampledRestaurant.getQMatrices ();
		}
		else
		{
			///////////////////// all the following parameters are necessary for initializeTables() internal function calls
			
			/// each trajectory is randomly assigned to a table index within the range
			/// [1,numTrajectories], (NOTE: the number of tables <= numberOfTrajetories), it is still
			/// possible that each trajectory is assigned its own unique table/reward-function (in which
			/// case the # of reward functions = # trajectories)
			// double [][] tableAssignmentMatrix = RNG.RandomUniformMatrixWithIntervalMin(numberTrajectories, 1, numberTrajectories, 1);
			/// //cl.b = nTraj x 1 matrix of pseudorandom integers drawn from the discrete uniform
			/// distribution on the interval [1,nTraj]; this integer corresponds to the 'label/index' of
			/// the table associated with the given trajectory
			
			// XXX:JK: rather than randomly assigning up to N table indices<= number of trajectories
			// double [][] tableAssignmentMatrix =
			// RNG.RandomUniformMatrixWithIntervalMin(numberTrajectories, 1, 1000, 1); //cl.b = nTraj x
			// 1 matrix of pseudorandom integers drawn from the discrete uniform distribution on the
			// interval [1,nTraj]; this integer corresponds to the 'label/index' of the table associated
			// with the given trajectory
			// cl.b = nTraj x 1 matrix of pseudorandom integers drawn from the discrete uniform distribution on the interval [1,nTraj]; 
			// this integer corresponds to the 'label/index' of the table associated with the given trajectory
//			double[][] tableAssignmentMatrix = RNG.RandomUniformMatrixWithIntervalMin (numberTrajectories, 1, 10, 1);
			tableAssignmentMatrix = MersenneTwisterFastIRL.RandomUniformMatrixWithIntervalMin (numberTrajectories, 1, 
																								Math.min (maxTables, numberTrajectories), 1);
			
			// returns a row vector containing the maximum values in each column of the 'tableAssignmentMatrix'. 
			// Since this matrix is nTraj x 1. We only care about the 0th element of the returned row vector
			Double highestTableIndex = MatrixUtilityJBLAS.maxPerColumn (tableAssignmentMatrix)[0];
			highestTableIdxN = highestTableIndex.intValue ();
			reportValue ("Highest table index created during random table assignment before intialization", highestTableIdxN);
			
			// XXX:The size of of each of these maps corresponds to the current number of ACTIVE tables
			// in the restaurant. Each element in the list is a column matrix for that table.
			// stores the weight-vector associated with each table index/value; Although the numberRewardFeatures is fixed, 
			// the number of active tables at any given moment can change, so we need the set of weight vectors be dynamic in size.
			tableWeightVectors = new HashMap<Integer, double[][]> ();
			// each policy is a column matrix of dimension numStates x 1
			tablePolicyVectors = new HashMap<Integer, double[][]> ();
			// each value is a column matrix of dimension numStates x 1 (i.e. it is NOT a row vector)
			tableValueVectors = new HashMap<Integer, double[][]> ();
			// each value is a column matrix of dimension (numStates*numActions) x 1 (i.e. it is NOT a row vector)
			tableQVectors = new HashMap<Integer, double[][]> ();
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
			// initialize table-specific parameters (weightVector, policyVector, valueVector)
			initializeTables (numThreads, numGPUs, highestTableIdxN, numberTrajectories, environment, irlAlgo, 
							  tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors);
			watch.stop ();
			reportTime ("Time required for Table Initialization: ", watch.getTime ());
			
			// after tableWeightVectors list has been computed...
			
			reportEvent ("*******Starting generateNewTableAssignmentPartition () following table intialization ");
			RestaurantMap rmap1 = generateNewTableAssignmentPartition (numThreads, numGPUs, environment.getNumStates (), environment.getNumActions (), 
																		environment.getNumRewardFeatures (), tableWeightVectors, tablePolicyVectors, 
																		tableValueVectors, tableQVectors, tableAssignmentMatrix);
			tableAssignmentMatrix = rmap1._restaurantAssignmentMatrix;
			tableWeightVectors = rmap1._restaurantTableWeightMatrices;
			tablePolicyVectors = rmap1._restaurantTablePolicyMatrices;
			tableValueVectors = rmap1._restaurantTableValueMatrices;
			tableQVectors = rmap1._restaurantTableQMatrices;
			
			// calculate scalar LOG posterior probability
			reportEvent ("*******Begin calculating INITIAL log_posteriorProbability1 via computeLogPosteriorProbabilityForDirichletProcessMixture() ");
			double log_posteriorProbability1 = 
					computeLogPosteriorProbabilityForDirichletProcessMixture (numThreads, numGPUs, trajectorySet, tableAssignmentMatrix, 
																			  tableWeightVectors, tablePolicyVectors, environment, irlAlgo, false);
			reportValue ("Initial log_posteriorProbability1 is ", log_posteriorProbability1);
//			Double	logProb = bestSampledRestaurant.getLogPosteriorProb ();
//			if ((Double.compare (log_posteriorProbability1, logProb) > 0) || Double.isInfinite (logProb))	// GTD 1/21/20 Always write if prob neg infinity
			if ((Double.compare (log_posteriorProbability1, bestSampledRestaurant.getLogPosteriorProb ()) > 0) ||
				(bestSampledRestaurant.getSeatingArrangement () == null))	// GTD must initialize bestSampledRestaurant
			{
				// if(log_posteriorProbability1 > bestSampledRestaurant.getLogPosteriorProb()) {}
				bestSampledRestaurant.setSeatingArrangement (tableAssignmentMatrix);
				bestSampledRestaurant.setWeightMatrices (tableWeightVectors);
				bestSampledRestaurant.setPolicyMatrices (tablePolicyVectors);
				bestSampledRestaurant.setValueMatrices (tableValueVectors);
				bestSampledRestaurant.setQMatrices (tableQVectors);
				
				bestSampledRestaurant.setLogPosteriorProb (log_posteriorProbability1);
			}
			// JK 6.18.2019 added if-condition to deal with possibility of initial restaurant logProb = -neg infinity
			if (Double.isFinite (bestSampledRestaurant.getLogPosteriorProb ()))
				mhSampledRestaurants.add (IRLRestaurant.clone (bestSampledRestaurant));
		}
		
		// N because we are starting table indices at 1!
		Map<Integer, Double> restaurantLikelihoods = VectorUtility.nansMap (highestTableIdxN);
		// N because we are starting table indices at 1!
		Map<Integer, Double> restaurantPriors = VectorUtility.nansMap (highestTableIdxN);
		Map<Integer, DoubleMatrix> restGradientsLLH = MatrixUtilityJBLAS.createHashMapOfRealMatricesWithNANS (numberRewardFeatures, highestTableIdxN);
		// N because we are starting table indices at 0!
		Map<Integer, DoubleMatrix> restGradientsPrior = MatrixUtilityJBLAS.createHashMapOfRealMatricesWithNANS (numberRewardFeatures, highestTableIdxN);
		
		// **************end of function calls for initializing restaurant********************
		reportEvent ("*##############Starting MH MCMC for CRP-IRL");
		// (Step 1 of Algorithm) Perform Metropolis-Hastings update for cluster assignment of i-th trajectory (aka. customer_i)
		double log_posteriorProbability2 = 0.0;
		// begin MH updates for Inference
		for (int iter = 0; iter < irlAlgo.getMaxIterations (); ++iter)
		{
			// JK 6.24.2019 doesn't this defeat the whole purpose of the Bayesian nonparametric algorithm?				 
//			if (addInferredTablesDuringRestaurantInitialization)
//			{ // use the 'K' inferred tables(and their associated weight, policy, value and
//				// q-matrices) as a substitute for 'K' randomly sampled tables in the total set of
//				// 'highestTableIdxN' tables
//				// returns a row vector containing the maximum values in each column of the 'tableAssignmentMatrix'.
//				// Since this matrix is nTraj x 1. We only care about the 0th element of the returned row vector
//				Double highestTableIndexKFromBestRestaruant = MatrixUtilityJBLAS.maxPerColumn (bestSampledRestaurant.getSeatingArrangement ())[0];
//				int numInferredTablesK = highestTableIndexKFromBestRestaruant.intValue ();
//				System.out.println ("Substituting " + numInferredTablesK + " tables during random initialization of " + highestTableIdxN + 
//									" (maximum) tables for restaurant's table assignment");
//				
//				// cl.b = nTraj x 1 matrix of pseudorandom integers drawn from the discrete uniform distribution on the interval [1,nTraj];
//				// this integer corresponds to the 'label/index' of the table associated with the given trajectory
//				tableAssignmentMatrix = MersenneTwisterFastIRL.RandomUniformMatrixWithIntervalMin (numberTrajectories, 1, 
//																									Math.min (highestTableIdxN, numberTrajectories), 1);
//				// returns a row vector containing the maximum values in each column of the 'tableAssignmentMatrix'.
//				// Since this matrix is nTraj x 1. We only care about the 0th element of the returned row vector
//				Double highestTableIndex_n = MatrixUtilityJBLAS.maxPerColumn (tableAssignmentMatrix)[0];
//				int highestTableIdx_n = highestTableIndex_n.intValue ();
//				reportValue ("Highest table index created during random table assignment (with K table substitution) before intialization", 
//							 highestTableIdxN);
//				
//				// stores the weight-vector associated with each table index/value; Although the numberRewardFeatures is fixed, 
//				// the number of active tables at any given moment can change, so we need the set of weight vectors be dynamic in size.
//				tableWeightVectors = new HashMap<Integer, double[][]> ();
//				// each policy is a column matrix of dimension numStates x 1
//				tablePolicyVectors = new HashMap<Integer, double[][]> ();
//				// each value is a column matrix of dimension numStates x 1 (i.e. it is NOT a row vector)
//				tableValueVectors = new HashMap<Integer, double[][]> ();
//				// each value is a column matrix of dimension (numStates*numActions) x 1 (i.e. it is NOT a row vector)
//				tableQVectors = new HashMap<Integer, double[][]> ();
//				
//				// Fill the restaurant with 'highestTableIdx_n' tables and all their corresponding
//				// matrices (some are obtained from K previously inferred tables)
//				initializeTablesWithSubstitution (numThreads, bestSampledRestaurant, numInferredTablesK, highestTableIdx_n, numberTrajectories, 
//												  environment, irlAlgo, tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors);
//				
//				reportEvent ("*******Starting generateNewTableAssignmentPartition() ");
//				RestaurantMap rmap1 = generateNewTableAssignmentPartition (numThreads, environment.getNumStates (), environment.getNumActions (), 
//																			environment.getNumRewardFeatures (), tableWeightVectors, tablePolicyVectors, 
//																			tableValueVectors, tableQVectors, tableAssignmentMatrix);
//				tableAssignmentMatrix = rmap1._restaurantAssignmentMatrix;
//				tableWeightVectors = rmap1._restaurantTableWeightMatrices;
//				tablePolicyVectors = rmap1._restaurantTablePolicyMatrices;
//				tableValueVectors = rmap1._restaurantTableValueMatrices;
//				tableQVectors = rmap1._restaurantTableQMatrices;
//			} // ends if() condition faddInferredTablesDuringRestaurantInitialization
			
			if (profile)
		        Controller.addBookmark ("Start iteration " + iter);
			int[] customersPermutation = VectorUtility.createPermutatedVector (numberTrajectories, 0);
			RestaurantMap rmap2 = null;
			reportEvent ("*##Cluster Assignent:Step 1 of IRL algorithm");

			System.out.print (Integer.toString (numberTrajectories));
			System.out.println (" customers are sequentially entering the restaurant...");
			
			
			rmap2 = updateTableAssignments (numThreads, numGPUs, environment, customersPermutation, irlAlgo, tableAssignmentMatrix, tableWeightVectors, 
											tablePolicyVectors, tableValueVectors, tableQVectors, restaurantLikelihoods, restaurantPriors, 
											restGradientsLLH, restGradientsPrior, stateActionPairCountsInfoForAllTrajectories);
			tableAssignmentMatrix = rmap2._restaurantAssignmentMatrix;
			tableWeightVectors = rmap2._restaurantTableWeightMatrices;
			tablePolicyVectors = rmap2._restaurantTablePolicyMatrices;
			tableValueVectors = rmap2._restaurantTableValueMatrices;
			tableQVectors = rmap2._restaurantTableQMatrices;
			
			restaurantLikelihoods = rmap2._restLikeLihoods;
			restaurantPriors = rmap2._restPriors;
			restGradientsLLH = rmap2._restGradientsFromLLH;
			restGradientsPrior = rmap2._restGradientsFromPrior;
			reportEvent ("Updated Seating arrangement :");
//			for (int customer = 0; customer < numberTrajectories; customer++)
//				System.out.println ("Customer " + customer + " sits at table : " + tableAssignmentMatrix[customer][0]);
			
			RestaurantMap rmap3 = generateNewTableAssignmentPartition (numThreads, numGPUs, environment.getNumStates (), environment.getNumActions (), 
																		environment.getNumRewardFeatures (), tableWeightVectors, tablePolicyVectors, 
																		tableValueVectors, tableQVectors, tableAssignmentMatrix);
			tableAssignmentMatrix = rmap3._restaurantAssignmentMatrix;
			tableWeightVectors = rmap3._restaurantTableWeightMatrices;
			tablePolicyVectors = rmap3._restaurantTablePolicyMatrices;
			tableValueVectors = rmap3._restaurantTableValueMatrices;
			tableQVectors = rmap3._restaurantTableQMatrices;
			
			updateRewardFunctions (numThreads, numGPUs, trajectorySet, environment, irlAlgo, tableAssignmentMatrix, tableWeightVectors, tablePolicyVectors, 
									tableValueVectors, tableQVectors, restaurantLikelihoods, restaurantPriors, restGradientsLLH, restGradientsPrior);
			
			log_posteriorProbability2 = 
					computeLogPosteriorProbabilityForDirichletProcessMixture (numThreads, numGPUs, trajectorySet, tableAssignmentMatrix, 
																			  tableWeightVectors, tablePolicyVectors, environment, irlAlgo, false);
			reportIterationValue ("logPosteriorProb", iter, log_posteriorProbability2);
			Double	logProb = bestSampledRestaurant.getLogPosteriorProb ();
			if ((Double.compare (log_posteriorProbability2, logProb) > 0) || Double.isInfinite (logProb))	// GTD 1/21/20 Always write if prob neg infinity
			{
				// if(log_posteriorProbability2 > bestSampledRestaurant.getLogPosteriorProb() )
				bestSampledRestaurant.setSeatingArrangement (tableAssignmentMatrix);
				bestSampledRestaurant.setWeightMatrices (tableWeightVectors);
				bestSampledRestaurant.setPolicyMatrices (tablePolicyVectors);
				bestSampledRestaurant.setValueMatrices (tableValueVectors);
				bestSampledRestaurant.setQMatrices (tableQVectors);
				
				bestSampledRestaurant.setLogPosteriorProb (log_posteriorProbability2);
				restaurantFactory.write (bestSampledRestaurant);
				mdpFactory.write (environment);
				
				//JK 7.31.2019 print out seating arrangement whenever a new restaurant is inferred
				double[][]	seatingArrangement = bestSampledRestaurant.getSeatingArrangement ();
				for (int customer = 0; customer < numberTrajectories; customer++)
				{
					System.out.println ("Customer " + customer + " sits at table : " + seatingArrangement[customer][0]);
				}
			}
			
			IRLRestaurant restaurantForCurrentIteration = new IRLRestaurant (tableAssignmentMatrix, tableWeightVectors, tablePolicyVectors, 
																			 tableValueVectors, tableQVectors, log_posteriorProbability2);
			mhSampledRestaurants.add (restaurantForCurrentIteration);
			reportValue ("*******Current best restaurant non-DB has logPosteriorProb", bestSampledRestaurant.getLogPosteriorProb ());
			reportValue ("*******Current best restaurant also has numTables = ", bestSampledRestaurant.getPolicyMatrices ().size ());
			if (profile)
		        Controller.saveSnapshot (new File (profileName (numThreads, numGPUs, restaurantForCurrentIteration.getPolicyMatrices ().size ()) + "." + iter + ".jps"));
//			System.out.println ("Seating arrangement :");
//			for (int customer = 0; customer < numberTrajectories; customer++)
//			{
//				System.out.println ("Customer " + customer + " sits at table :" + bestSampledRestaurant._seatingArrangement[customer][0]);
//			}
			
		} // end for-loop for MH algorithm iterations
		restaurantFactory.write (bestSampledRestaurant);
		mdpFactory.write (environment);
		reportValue ("Overall best restaurant has logPosteriorProb", bestSampledRestaurant.getLogPosteriorProb ());
		reportValue ("Overall best restaurant also has numTables = ", bestSampledRestaurant.getPolicyMatrices ().size ());

		System.out.println ("Seating arrangement :");
		double[][]	seatingArrangement = bestSampledRestaurant.getSeatingArrangement ();
		for (int customer = 0; customer < numberTrajectories; customer++)
		{
			System.out.println ("Customer " + customer + " sits at table : " + seatingArrangement[customer][0]);
		}
		
//		addInferredTablesDuringRestaurantInitialization = true;
	}
	

	/**
	 * JK 7.25.2019 mostly data validated
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param trajectorySet
	 * @param environment
	 * @param irlAlgo
	 * @param tableAssignmentMatrix
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 * @param restaurantLikelihoods
	 * @param restaurantPriors
	 * @param restGradientsLLH
	 * @param restGradientsPrior
	 */
	private static final void updateRewardFunctions (int numThreads, int numGPUs, List<double[][]> trajectorySet, MDPCancer environment, 
													 IRLAlgorithmCancer irlAlgo, double[][] tableAssignmentMatrix, 
													 Map<Integer, double[][]> tableWeightVectors, Map<Integer, double[][]> tablePolicyVectors, 
													 Map<Integer, double[][]> tableValueVectors, Map<Integer, double[][]> tableQVectors, 
													 Map<Integer, Double> restaurantLikelihoods, Map<Integer, Double> restaurantPriors, 
													 Map<Integer, DoubleMatrix> restGradientsLLH, Map<Integer, DoubleMatrix> restGradientsPrior)
	{
		Double			highestTableIndexDBL = MatrixUtilityJBLAS.matrixMaximum (tableAssignmentMatrix);
		int				highestTableIndexINT = highestTableIndexDBL.intValue ();
		int[]			tablesPermutation = VectorUtility.createPermutatedVector (highestTableIndexINT, 1);
		// Create permutated vector of length= highestTableIndexINT with starting value 1
		Map<Integer, RestaurantTable>	restResults = Collections.synchronizedMap (new HashMap<> ());
		
		System.out.print (dateTimeFormat.format (new Date ()));
		System.out.println (": *##Reward Function Updating:Step 2 of IRL algorithm");
		System.out.print ("Using ");
		System.out.print (Integer.toString (numThreads));
		System.out.println (" threads");
		
		if (numThreads <= 1)
		{
			RestaurantTable	rmap6;
			
			for (int table_i : tablesPermutation)
			{ // iterate through all tables in the restaurant in random order; for each table_i
				// update its reward-function so that its weightMatrix is a reflection of the sum of
				// counts of the s-a pairs (found in the trajectories/customers assigned to that
				// table)
				rmap6 = updateRewardFunctions (trajectorySet, environment, table_i, irlAlgo,
						tableAssignmentMatrix, tableWeightVectors.get (table_i), tablePolicyVectors.get (table_i),
						tableValueVectors.get (table_i), tableQVectors.get (table_i), restaurantLikelihoods.get (table_i), 
						restaurantPriors.get (table_i), restGradientsLLH.get (table_i), restGradientsPrior.get (table_i));
				
				restResults.put (table_i, rmap6);
			}
		}
		else
		{
			updateRewardFunctionsThreaded (numThreads, numGPUs, tablesPermutation, trajectorySet, environment, irlAlgo, tableAssignmentMatrix, 
											tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, restaurantLikelihoods, 
											restaurantPriors, restGradientsLLH, restGradientsPrior, restResults);
		}
		
		updateMaps (restResults, tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, 
					restaurantLikelihoods, restaurantPriors, restGradientsLLH, restGradientsPrior);
		reportEvent ("*##Reward Function Updating Finished");
	}
	
	
	/**
	 * 
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param tablesPermutation
	 * @param trajectorySet
	 * @param environment
	 * @param irlAlgo
	 * @param tableAssignmentMatrix
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 * @param restaurantLikelihoods
	 * @param restaurantPriors
	 * @param restGradientsLLH
	 * @param restGradientsPrior
	 */
	private static final void updateRewardFunctionsThreaded (int numThreads, int numGPUs, int[] tablesPermutation, List<double[][]> trajectorySet, 
															 MDPCancer environment, IRLAlgorithmCancer irlAlgo, double[][] tableAssignmentMatrix, 
															 Map<Integer, double[][]> tableWeightVectors, Map<Integer, double[][]> tablePolicyVectors, 
															 Map<Integer, double[][]> tableValueVectors, Map<Integer, double[][]> tableQVectors, 
															 Map<Integer, Double> restaurantLikelihoods, Map<Integer, Double> restaurantPriors, 
															 Map<Integer, DoubleMatrix> restGradientsLLH, Map<Integer, DoubleMatrix> restGradientsPrior, 
															 Map<Integer, RestaurantTable> restResults)
	{
		List<Thread>			threads = new ArrayList<> (numThreads);
		int						numTables = tablesPermutation.length;
		BlockingQueue<Integer>	tableIndexes = makeQueue (tablesPermutation);
		
		tableWeightVectors = Collections.synchronizedMap (tableWeightVectors);
		tablePolicyVectors = Collections.synchronizedMap (tablePolicyVectors);
		tableValueVectors = Collections.synchronizedMap (tableValueVectors);
		tableQVectors = Collections.synchronizedMap (tableQVectors);
		reportThreading ("updateRewardFunctionsThreaded", numThreads, numGPUs, numTables);
		if (numThreads > numTables)
			System.out.println ("updateRewardFunctionsThreaded: " + numThreads + " threads, " + numTables + " tables");
		
		for (int i = 1; (i <= numThreads) && (i <= numTables); ++i)
		{
			MDPCancer				env = new MDPCancer (environment);
			RewardFunctionsUpdater	updater = new RewardFunctionsUpdater (tableIndexes, trajectorySet, env, irlAlgo, tableAssignmentMatrix, 
																		  tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, 
																		  restaurantLikelihoods, restaurantPriors, restGradientsLLH, 
																		  restGradientsPrior, restResults);
			Thread	theThread = makeThread (updater, "updateRewardFunctions ", i, numGPUs);
			
			theThread.start ();
			threads.add (theThread);
		}
		
		// Wait for the threads to finish
		waitForThreads (threads);
	}
	
	
	/**
	 * 
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param tablesPermutation
	 * @param trajectorySet
	 * @param environment
	 * @param irlAlgo
	 * @param tableAssignmentMatrix
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 * @param restaurantLikelihoods
	 * @param restaurantPriors
	 * @param restGradientsLLH
	 * @param restGradientsPrior
	 */
	private static final void computeLogPosteriorProbabilityForDirichletProcessMixtureThreaded (
			int numThreads, int numGPUs, int numTables, List<double[][]> trajSet, double[][] tableAssignmentMatrix,
			Map<Integer, double[][]> tblWeightVectors, Map<Integer, double[][]> tblPolicyVectors, 
			MDPCancer environment, IRLAlgorithmCancer irlAlgo, boolean partOfInitialDPMPosterior, 
			double[] logLikelihoods, double[] logPriorProbs)
	{
		List<Thread>			threads = new ArrayList<> (numThreads);
		BlockingQueue<Integer>	tableIndexes = makeQueue (numTables);
		
		reportThreading ("computeLogPosteriorProbabilityForDirichletProcessMixtureThreaded", numThreads, numGPUs, numTables);
		if (numThreads > numTables)
			System.out.println ("computeLogPosteriorProbabilityForDirichlet: " + numThreads + " threads, " + numTables + " tables");
		
		for (int i = 1; (i <= numThreads) && (i <= numTables); ++i)
		{
			MDPCancer						env = new MDPCancer (environment);
			ComputeLogPosteriorProbability	updater = new ComputeLogPosteriorProbability (tableIndexes, trajSet, tableAssignmentMatrix, tblWeightVectors, 
																						  tblPolicyVectors, env, irlAlgo, partOfInitialDPMPosterior, 
																						  logLikelihoods, logPriorProbs);
			Thread	theThread = makeThread (updater, "computeLogPosteriorProbability ", i, numGPUs);
			
			theThread.start ();
			threads.add (theThread);
		}
		
		// Wait for the threads to finish
		waitForThreads (threads);
	}
	
	
	/**
	 * Iterate over the values in a {@link Map}, looking for ones that have a bad number of rows
	 * 
	 * @param theMap	{@link Map} to iterate over
	 * @param badRows	Number of rows not allows
	 * @param title		Name of {@code theMap} 
	 */
	private static final void testMapValues (Map<Integer, double[][]> theMap, int badRows, String title)
	{
		for (Entry<Integer, double[][]> theEntry : theMap.entrySet ())
		{
			double[][]	theArray = theEntry.getValue ();
			int			numRows = theArray.length;
			
			if (numRows == badRows)
			{
				Integer	key = theEntry.getKey ();
				
				System.err.print ("Entry ");
				System.err.print (key.toString ());
				System.err.print (" for Map ");
				System.err.print (title);
				System.err.print (" has value ");
				System.err.println (numRows);
			}
		}
	}


	/**
	 * Send to {@link System#out} a line specifying the routine, number of threads, and number of tables
	 * 
	 * @param title			Name of routine that's threading
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param numTables		Number of tables it can thread
	 */
	private static final void reportThreading (String title, int numThreads, int numGPUs, int numTables)
	{
		if (!kPrintDebug)
		{
			return;
		}
		System.out.print (dateTimeFormat.format (new Date ()));
		System.out.print (": In ");
		System.out.print (title);
		System.out.print (": ");
		System.out.print (Integer.toString (numTables));
		System.out.print (" tables for ");
		System.out.print (Integer.toString (numThreads));
		System.out.print (" threads and ");
		System.out.print (Integer.toString (numGPUs));
		System.out.println (" GPUs");
	}
	
	
	/**
	 * Calculate the number of items that should be processed by each thread
	 * 
	 * @param maxThreads	Number of threads the items can be spread across
	 * @param numItems		Number of items to spread across the threads
	 */
	private static final int getThreadItems (int maxThreads, int numItems)
	{
		if (numItems <= maxThreads)
			return 1;
		
		int	minDivisor = maxThreads * 5;
		int	maxDivisor = maxThreads * 20;
		
		if (numItems <= minDivisor)
			return 1;
		
		if (numItems <= (2 * maxDivisor))
			return 2;
		
		return ((numItems - 1) / maxDivisor) + 1;
	}
	
	
	/**
	 * Provide the array of items that should be processed by the specified thread
	 * 
	 * @param whichThread	Thread to get items for.  Must be {@code < numThreads}
	 * @param numThreads	Number of threads the items can be spread across
	 * @param startPos		Number of items that have already been given to other threads
	 * @param items			Items to distribute.  Length must be {@code > startPos}
	 * @return	int[] of items
	 */
	private static final int[] getThreadBlock (int whichThread, int numThreads, int startPos, int[] items)
	{
		int		remainingThreads = numThreads - whichThread;
		int		numItems = items.length - startPos;
		int		blockSize = (numItems + (remainingThreads - 1)) / remainingThreads;
		int[]	results = new int[blockSize];
		
		System.arraycopy (items, startPos, results, 0, blockSize);
		return results;
	}
	
	
	/**
	 * Check if any of the threads have finished processing
	 * 
	 * @param threads		{@link List} of {@link Thread} that are running
	 * @param maxThreads	Keep checking until fewer than maxThreads are running
	 * @param sleepTime		Time in milliseconds to wait in between checks for expired threads
	 */
	private static final void checkThreads (List<Thread> threads, int maxThreads, long sleepTime)
	{
		long	usedSleepTime = 0;
		
		do
		{
			Iterator<Thread>	iter = threads.iterator ();
			
			while (iter.hasNext ())
			{
				Thread	theThread = iter.next ();
				
				if (!theThread.isAlive ())
				{
					if (kPrintDebug)
					{
						System.out.print (dateTimeFormat.format (new Date ()));
						System.out.print (": Thread ");
						System.out.print (theThread.getName ());
						System.out.println (" has finished");
					}
					iter.remove ();
				}
			}
			
			if (threads.size () >= maxThreads)
			{
				try
				{
					if (usedSleepTime == 0)
					{
						Thread.sleep (0, kOneMicroSecond);
						usedSleepTime = 1;
					}
					else
					{
						Thread.sleep (usedSleepTime);
						usedSleepTime = sleepTime;
					}
				}
				catch (InterruptedException oops)
				{
					// Ignore
				}
			}
		}
		while (threads.size () >= maxThreads);
	}
	
	
	/**
	 * Create a thread with a name the specifies whether or not it should use a GPU when possible, and 
	 * if so, which GPU it should use
	 * 
	 * @param updater		The process to be run
	 * @param name			The starting name
	 * @param whichThread	The Thread's ID
	 * @param numGPUs		The number of GPUs available, 0 for none
	 * @return	A new {@link Thread} to be run
	 */
	private static final Thread makeThread (Runnable updater, String name, int whichThread, int numGPUs)
	{
		if (whichThread <= numGPUs)
			name = name + kGPU + whichThread;
		else
			name = name + whichThread;
		
		return new Thread (updater, name);
	}
	
	
	/**
	 * Get the name of the current thread.  If it has {@value #kGPU} in the name, get the number after 
	 * that string and return it.  Else return {@value #kInvalidGPU}
	 * 
	 * @return	Number of the GPU to use, from 0 to {@code numGPUs - 1}, or {@value #kInvalidGPU} if 
	 * not to use a GPU
	 */
	private static final int getGPUT ()
	{
		Thread	curThread = Thread.currentThread ();
		String	name = curThread.getName ();
		int		pos = name.lastIndexOf (kGPU);
		
		if (pos < 0)
		{
//			System.out.print ("No GPU: ");
//			System.out.println (name);
			
			return kInvalidGPU;
		}
		
		int	value = Integer.parseInt (name.substring (pos + kGPULen));
		
		if (value > 0)
			--value;
		
		return value;
	}
	
	
	/**
	 * Wait until all the threads have finished processing
	 * 
	 * @param threads	{@link List} of {@link Thread} that might be running
	 */
	private static final void waitForThreads (List<Thread> threads)
	{
		for (Thread theThread : threads)
		{
			if (theThread.isAlive ())
			{
				try
				{
					theThread.join ();
					if (kPrintDebug)
					{
						System.out.print (dateTimeFormat.format (new Date ()));
						System.out.print (": Joined with Thread ");
						System.out.println (theThread.getName ());
					}
				}
				catch (InterruptedException oops)
				{
					// Ignore
				}
			}
			else if (kPrintDebug)
			{
				System.out.print (dateTimeFormat.format (new Date ()));
				System.out.print (": Did not have to Join with Thread ");
				System.out.println (theThread.getName ());
			}
		}
	}
	
	
	/**
	 * Update the table maps with the new values
	 * 
	 * @param restResults
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 * @param restaurantLikelihoods
	 * @param restaurantPriors
	 * @param restGradientsLLH
	 * @param restGradientsPrior
	 */
	private static final void updateMaps (Map<Integer, RestaurantTable>	restResults, Map<Integer, double[][]> tableWeightVectors, 
										  Map<Integer, double[][]> tablePolicyVectors, Map<Integer, double[][]> tableValueVectors, 
										  Map<Integer, double[][]> tableQVectors, Map<Integer, Double> restaurantLikelihoods, 
										  Map<Integer, Double> restaurantPriors, Map<Integer, DoubleMatrix> restGradientsLLH, 
										  Map<Integer, DoubleMatrix> restGradientsPrior)
	{
//		tableAssignmentMatrix = rmap6._restaurantAssignmentMatrix;
		
		// Wipe old, bad, data
		tableWeightVectors.clear ();
		tablePolicyVectors.clear ();
		tableValueVectors.clear ();
		tableQVectors.clear ();
		restaurantLikelihoods.clear ();
		restaurantPriors.clear ();
		restGradientsLLH.clear ();
		restGradientsPrior.clear ();
		
		for (Entry<Integer, RestaurantTable> entry : restResults.entrySet ())
		{
			Integer			table = entry.getKey ();
			RestaurantTable	tableData = entry.getValue ();
			
			tableWeightVectors.put (table, tableData._restaurantTableWeightMatrices);
 			tablePolicyVectors.put (table, tableData._restaurantTablePolicyMatrices);
 			tableValueVectors.put (table, tableData._restaurantTableValueMatrices);
 			tableQVectors.put (table, tableData._restaurantTableQMatrices);
 			
 			restaurantLikelihoods.put (table, tableData._restLikeLihoods);
 			restaurantPriors.put (table, tableData._restPriors);
 			restGradientsLLH.put (table, tableData._restGradientsFromLLH);
 			restGradientsPrior.put (table, tableData._restGradientsFromPrior);
		}
	}
	
	
	/**
	 * New constructor which stores count info in cassandra table and allows for serialization/deserializatin of IRLRestaurant and MDPCancer objects
	 * 
	 * @param environment
	 * @param trajectorySet
	 * @param irlAlgo
	 * @param mhSampledRestaurants
	 * @param bestSampledRestaurant
	 * @param cassSession
	 * @param seed
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param maxTables
	 * @param boolStartFromScratch
	 * @param restaurantFactory
	 * @param mdpFactory
	 * @throws Exception
	 * @throws FileNotFoundException
	 * @throws IOException
	 * @throws ClassNotFoundException
	 * @throws InstantiationException
	 * @throws IllegalAccessException
	 * @throws NoSuchMethodException
	 * @throws IllegalArgumentException
	 * @throws InvocationTargetException
	 */
	public static void ChineseRestaurantProcessInferenceWithDatabase (MDPCancer environment, List<double[][]> trajectorySet, IRLAlgorithmCancer irlAlgo,
																	  THashSet<IRLRestaurant> mhSampledRestaurants, IRLRestaurant bestSampledRestaurant, 
																	  Session cassSession, long seed, int numThreads, int numGPUs, int maxTables, 
																	  boolean boolStartFromScratch, IRLRestaurantFactory restaurantFactory, 
																	  MDPCancerFactory mdpFactory, boolean profile) 
			throws Exception, FileNotFoundException, IOException, ClassNotFoundException, InstantiationException, IllegalAccessException, 
					NoSuchMethodException, IllegalArgumentException, InvocationTargetException
	{
		if (seed != 0)
			RNG = new MersenneTwisterFastIRL (seed);
		else
			RNG = new MersenneTwisterFastIRL ();
		_session = cassSession;
		
		// JK added 5.1.2019: we should reset countinfofortrajs_table in Cassandra every time we are
		// going to computeOccupancy from paths stored in the DB
		String truncateCountInfoForTrajsTableCQLStatement = "TRUNCATE countinfofortrajs_table";
		_session.execute (truncateCountInfoForTrajsTableCQLStatement);
		
		// XXX:Trajectory data: NOTE: each trajectory is defined by 2 sequences: 1. a sequence of
		// states, and 2. a sequence of actions.
		int numberTrajectories = trajectorySet.size (); // should be same as MDP._numberTrajectories
		System.out.println ("Total numTrajectories: " + numberTrajectories);
		
//		boolean addInferredTablesDuringRestaurantInitialization = false; //JK 6.20.2019: added
		if (maxTables == 0)
			maxTables = numberTrajectories;
		int	numberRewardFeatures = environment.getNumRewardFeatures ();
		
		// map of state-action counts for trajectories of interest
		// includes the 'count' of each 'observed' state-action pair in the subset of trajectory set.
		// Thus, if we consider all trajectories to extract 'count' information, the maximum size of
		// this trajectoryInfo dataset is: nTrajs x nSteps x 3
		// This 'map' is now found in the cassandra table 'countinfofortrajs_table'
		
		reportEvent ("Going to compute occupancy, one customer at a time before table initialization ");
		for (int i = 0; i < numberTrajectories; i++)
		{
			// Compute occupancy and the empirical policy for trajectories
			
			Map<Integer, double[][]> subsetOfTrajectoriesToAnalyze = new HashMap<Integer, double[][]> ();
			subsetOfTrajectoriesToAnalyze.put (i, trajectorySet.get (i));
			computeOccupancyFromDatabase (subsetOfTrajectoriesToAnalyze, environment, false);
			subsetOfTrajectoriesToAnalyze.clear ();
			// set the state-action counts for each trajectory; i.e. the set of state-action pairs
			// visited in this trajectory and their respective counts associated with number of visits
		}
		
		// initialize tables
		StopWatch watch = StopWatch.createStarted ();
		
		double [][] tableAssignmentMatrix = null;
		int highestTableIdxN = 0;
		// stores the weight-vector associated with each table index/value; Although the
		// numberRewardFeatures is fixed, the number of active tables at any given moment can
		// change, so we need the set of weight vectors be dynamic in size.
		Map<Integer, double[][]> tableWeightVectors = null;
		// each policy is a column matrix of dimension numStates x 1
		Map<Integer, double[][]> tablePolicyVectors = null;
		// each value is a column matrix of dimension numStates x 1 (i.e. it is NOT a row vector)
		Map<Integer, double[][]> tableValueVectors = null;		
		// each value is a column matrix of dimension (numStates * numActions) x 1  (i.e. it is NOT a row vector)
		Map<Integer, double[][]> tableQVectors = null;
		
		if (!boolStartFromScratch) // if not building from scratch,  you will be overriding any other input argument which may also specify num of tables 
		{
//			tableAssignmentMatrix = bestSampledRestaurant.getSeatingArrangement ();
			// returns a row vector containing the maximum values in each column of the
			// 'tableAssignmentMatrix'. Since this matrix is nTraj x 1. We only care about the 0th
			// element of the returned row vector
			Double highestTableIndex = MatrixUtilityJBLAS.maxPerColumn (bestSampledRestaurant.getSeatingArrangement ())[0];
			highestTableIdxN = highestTableIndex.intValue ();
			maxTables = Math.min (highestTableIdxN, numberTrajectories);
			tableAssignmentMatrix = MersenneTwisterFastIRL.RandomUniformMatrixWithIntervalMin (numberTrajectories, 1, maxTables, 1);
			
			reportValue ("Highest table index in pre-existing restaurant's table assignment", highestTableIdxN);
			
			// stores the weight-vector associated with each table index/value; Although the
			// numberRewardFeatures is fixed, the number of active tables at any given moment can
			// change, so we need the set of weight vectors be dynamic in size.
			tableWeightVectors = bestSampledRestaurant.getWeightMatrices ();
			// each policy is a column matrix of dimension numStates x 1
			tablePolicyVectors = bestSampledRestaurant.getPolicyMatrices ();
			// each value is a column matrix of dimension numStates x 1 (i.e. it is NOT a row vector)
			tableValueVectors = bestSampledRestaurant.getValueMatrices ();
			// each value is a column matrix of dimension (numStates * numActions) x 1  (i.e. it is NOT a row vector)
			tableQVectors = bestSampledRestaurant.getQMatrices ();
		}
		else
		{
			///////////////////// all the following parameters are necessary for initializeTables() internal function calls
			
			/// each trajectory is randomly assigned to a table index within the range
			/// [1,numTrajectories], (NOTE: the number of tables <= numberOfTrajetories), it is still
			/// possible that each trajectory is assigned its own unique table/reward-function (in which
			/// case the # of reward functions = # trajectories)
			// double [][] tableAssignmentMatrix = RNG.RandomUniformMatrixWithIntervalMin(numberTrajectories, 1, numberTrajectories, 1);
			/// //cl.b = nTraj x 1 matrix of pseudorandom integers drawn from the discrete uniform
			/// distribution on the interval [1,nTraj]; this integer corresponds to the 'label/index' of
			/// the table associated with the given trajectory
			
			// XXX:JK: rather than randomly assigning up to N table indices<= number of trajectories
			// double [][] tableAssignmentMatrix =
			// RNG.RandomUniformMatrixWithIntervalMin(numberTrajectories, 1, 1000, 1); //cl.b = nTraj x
			// 1 matrix of pseudorandom integers drawn from the discrete uniform distribution on the
			// interval [1,nTraj]; this integer corresponds to the 'label/index' of the table associated
			// with the given trajectory
			// cl.b = nTraj x 1 matrix of pseudorandom integers drawn from the discrete uniform distribution on the interval [1,nTraj]; 
			// this integer corresponds to the 'label/index' of the table associated with the given trajectory
//			double[][] tableAssignmentMatrix = RNG.RandomUniformMatrixWithIntervalMin (numberTrajectories, 1, 10, 1);
			tableAssignmentMatrix = MersenneTwisterFastIRL.RandomUniformMatrixWithIntervalMin (numberTrajectories, 1, 
																								Math.min (maxTables, numberTrajectories), 1);
			
			// returns a row vector containing the maximum values in each column of the 'tableAssignmentMatrix'. 
			// Since this matrix is nTraj x 1. We only care about the 0th element of the returned row vector
			Double highestTableIndex = MatrixUtilityJBLAS.maxPerColumn (tableAssignmentMatrix)[0];
			highestTableIdxN = highestTableIndex.intValue ();
			reportValue ("Highest table index created during random table assignment before intialization", highestTableIdxN);
			
			// XXX:The size of of each of these maps corresponds to the current number of ACTIVE tables
			// in the restaurant. Each element in the list is a column matrix for that table.
			// stores the weight-vector associated with each table index/value; Although the numberRewardFeatures is fixed, 
			// the number of active tables at any given moment can change, so we need the set of weight vectors be dynamic in size.
			tableWeightVectors = new HashMap<Integer, double[][]> ();
			// each policy is a column matrix of dimension numStates x 1
			tablePolicyVectors = new HashMap<Integer, double[][]> ();
			// each value is a column matrix of dimension numStates x 1 (i.e. it is NOT a row vector)
			tableValueVectors = new HashMap<Integer, double[][]> ();
			// each value is a column matrix of dimension (numStates*numActions) x 1 (i.e. it is NOT a row vector)
			tableQVectors = new HashMap<Integer, double[][]> ();
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
			// initialize table-specific parameters (weightVector, policyVector, valueVector)
			initializeTables (numThreads, numGPUs, highestTableIdxN, numberTrajectories, environment, irlAlgo, 
							  tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors);
			watch.stop ();
			reportTime ("Time required for Table Initialization: ", watch.getTime ());
			
			// after tableWeightVectors list has been computed...
			
			// GTD Not used
//			// changed data structure from double [] to Map so that we obtain likelihood by tableIndex value
//			Map<Integer, Double> restaurantLikelihoods = VectorUtility.nansMap (N); // N because we are starting table indices at 1!
//			Map<Integer, Double> restaurantPriors = VectorUtility.nansMap (N);// N because we are starting table indices at 1!
//			
//			// N because we are starting table indices at 0!
//			Map<Integer, DoubleMatrix> restGradientsLLH = MatrixUtilityJBLAS.createHashMapOfRealMatricesWithNANS (numberRewardFeatures, N);
//			Map<Integer, DoubleMatrix> restGradientsPrior = MatrixUtilityJBLAS.createHashMapOfRealMatricesWithNANS (numberRewardFeatures, N);
			
			reportEvent ("*******Starting generateNewTableAssignmentPartition () following table intialization ");
			RestaurantMap rmap1 = generateNewTableAssignmentPartition (numThreads, numGPUs, environment.getNumStates (), environment.getNumActions (), 
																		environment.getNumRewardFeatures (), tableWeightVectors, tablePolicyVectors, 
																		tableValueVectors, tableQVectors, tableAssignmentMatrix);
			tableAssignmentMatrix = rmap1._restaurantAssignmentMatrix;
			tableWeightVectors = rmap1._restaurantTableWeightMatrices;
			tablePolicyVectors = rmap1._restaurantTablePolicyMatrices;
			tableValueVectors = rmap1._restaurantTableValueMatrices;
			tableQVectors = rmap1._restaurantTableQMatrices;
			
			// calculate scalar LOG posterior probability
			reportEvent ("*******Begin calculating INITIAL log_posteriorProbability1 via computeLogPosteriorProbabilityForDirichletProcessMixture() ");
			double log_posteriorProbability1 = 
					computeLogPosteriorProbabilityForDirichletProcessMixture (numThreads, numGPUs, trajectorySet, tableAssignmentMatrix, 
																			  tableWeightVectors, tablePolicyVectors, environment, irlAlgo, false);
			reportValue ("Initial log_posteriorProbability1 is ", log_posteriorProbability1);
//			Double	logProb = bestSampledRestaurant.getLogPosteriorProb ();
//			if ((Double.compare (log_posteriorProbability1, logProb) > 0) || Double.isInfinite (logProb))	// GTD 1/21/20 Always write if prob neg infinity
			if ((Double.compare (log_posteriorProbability1, bestSampledRestaurant.getLogPosteriorProb ()) > 0) ||
				(bestSampledRestaurant.getSeatingArrangement () == null))	// GTD must initialize bestSampledRestaurant
			{
				// if(log_posteriorProbability1 > bestSampledRestaurant.getLogPosteriorProb()) {}
				bestSampledRestaurant.setSeatingArrangement (tableAssignmentMatrix);
				bestSampledRestaurant.setWeightMatrices (tableWeightVectors);
				bestSampledRestaurant.setPolicyMatrices (tablePolicyVectors);
				bestSampledRestaurant.setValueMatrices (tableValueVectors);
				bestSampledRestaurant.setQMatrices (tableQVectors);
				
				bestSampledRestaurant.setLogPosteriorProb (log_posteriorProbability1);
			}
			// JK 6.18.2019 added if-condition to deal with possibility of initial restaurant logProb = -neg infinity
			if (Double.isFinite (bestSampledRestaurant.getLogPosteriorProb ()))
				mhSampledRestaurants.add (IRLRestaurant.clone (bestSampledRestaurant));
		}
		
		// N because we are starting table indices at 1!
		Map<Integer, Double> restaurantLikelihoods = VectorUtility.nansMap (highestTableIdxN);
		// N because we are starting table indices at 1!
		Map<Integer, Double> restaurantPriors = VectorUtility.nansMap (highestTableIdxN);
		Map<Integer, DoubleMatrix> restGradientsLLH = MatrixUtilityJBLAS.createHashMapOfRealMatricesWithNANS (numberRewardFeatures, highestTableIdxN);
		// N because we are starting table indices at 0!
		Map<Integer, DoubleMatrix> restGradientsPrior = MatrixUtilityJBLAS.createHashMapOfRealMatricesWithNANS (numberRewardFeatures, highestTableIdxN);
		
		// **************end of function calls for initializing restaurant********************
		reportEvent ("*##############Starting MH MCMC for CRP-IRL");
		// (Step 1 of Algorithm) Perform Metropolis-Hastings update for cluster assignment of i-th trajectory (aka. customer_i)
		double log_posteriorProbability2 = 0.0;
		// begin MH updates for Inference
		for (int iter = 0; iter < irlAlgo.getMaxIterations (); ++iter)
		{
			// JK 6.24.2019 doesn't this defeat the whole purpose of the Bayesian nonparametric algorithm?				 
//			if (addInferredTablesDuringRestaurantInitialization)
//			{ // use the 'K' inferred tables(and their associated weight, policy, value and
//				// q-matrices) as a substitute for 'K' randomly sampled tables in the total set of
//				// 'highestTableIdxN' tables
//				// returns a row vector containing the maximum values in each column of the 'tableAssignmentMatrix'.
//				// Since this matrix is nTraj x 1. We only care about the 0th element of the returned row vector
//				Double highestTableIndexKFromBestRestaruant = MatrixUtilityJBLAS.maxPerColumn (bestSampledRestaurant.getSeatingArrangement ())[0];
//				int numInferredTablesK = highestTableIndexKFromBestRestaruant.intValue ();
//				System.out.println ("Substituting " + numInferredTablesK + " tables during random initialization of " + highestTableIdxN + 
//									" (maximum) tables for restaurant's table assignment");
//				
//				// cl.b = nTraj x 1 matrix of pseudorandom integers drawn from the discrete uniform distribution on the interval [1,nTraj];
//				// this integer corresponds to the 'label/index' of the table associated with the given trajectory
//				tableAssignmentMatrix = MersenneTwisterFastIRL.RandomUniformMatrixWithIntervalMin (numberTrajectories, 1, 
//																									Math.min (highestTableIdxN, numberTrajectories), 1);
//				// returns a row vector containing the maximum values in each column of the 'tableAssignmentMatrix'.
//				// Since this matrix is nTraj x 1. We only care about the 0th element of the returned row vector
//				Double highestTableIndex_n = MatrixUtilityJBLAS.maxPerColumn (tableAssignmentMatrix)[0];
//				int highestTableIdx_n = highestTableIndex_n.intValue ();
//				reportValue ("Highest table index created during random table assignment (with K table substitution) before intialization", 
//							 highestTableIdxN);
//				
//				// stores the weight-vector associated with each table index/value; Although the numberRewardFeatures is fixed, 
//				// the number of active tables at any given moment can change, so we need the set of weight vectors be dynamic in size.
//				tableWeightVectors = new HashMap<Integer, double[][]> ();
//				// each policy is a column matrix of dimension numStates x 1
//				tablePolicyVectors = new HashMap<Integer, double[][]> ();
//				// each value is a column matrix of dimension numStates x 1 (i.e. it is NOT a row vector)
//				tableValueVectors = new HashMap<Integer, double[][]> ();
//				// each value is a column matrix of dimension (numStates*numActions) x 1 (i.e. it is NOT a row vector)
//				tableQVectors = new HashMap<Integer, double[][]> ();
//				
//				// Fill the restaurant with 'highestTableIdx_n' tables and all their corresponding
//				// matrices (some are obtained from K previously inferred tables)
//				initializeTablesWithSubstitution (numThreads, bestSampledRestaurant, numInferredTablesK, highestTableIdx_n, numberTrajectories, 
//												  environment, irlAlgo, tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors);
//				
//				reportEvent ("*******Starting generateNewTableAssignmentPartition() ");
//				RestaurantMap rmap1 = generateNewTableAssignmentPartition (numThreads, environment.getNumStates (), environment.getNumActions (), 
//																			environment.getNumRewardFeatures (), tableWeightVectors, tablePolicyVectors, 
//																			tableValueVectors, tableQVectors, tableAssignmentMatrix);
//				tableAssignmentMatrix = rmap1._restaurantAssignmentMatrix;
//				tableWeightVectors = rmap1._restaurantTableWeightMatrices;
//				tablePolicyVectors = rmap1._restaurantTablePolicyMatrices;
//				tableValueVectors = rmap1._restaurantTableValueMatrices;
//				tableQVectors = rmap1._restaurantTableQMatrices;
//			} // ends if() condition faddInferredTablesDuringRestaurantInitialization
			
			if (profile)
		        Controller.addBookmark ("Start iteration " + iter);
			int[] customersPermutation = VectorUtility.createPermutatedVector (numberTrajectories, 0);
			RestaurantMap rmap2 = null;
			reportEvent ("*##Cluster Assignent:Step 1 of IRL algorithm");

			System.out.print (Integer.toString (numberTrajectories));
			System.out.println (" customers are sequentially entering the restaurant...");
			
			rmap2 = updateTableAssignmentsWithDatabase (numThreads, numGPUs, environment, customersPermutation, irlAlgo, tableAssignmentMatrix, 
														tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, 
														restaurantLikelihoods, restaurantPriors, restGradientsLLH, restGradientsPrior);
			tableAssignmentMatrix = rmap2._restaurantAssignmentMatrix;
			tableWeightVectors = rmap2._restaurantTableWeightMatrices;
			tablePolicyVectors = rmap2._restaurantTablePolicyMatrices;
			tableValueVectors = rmap2._restaurantTableValueMatrices;
			tableQVectors = rmap2._restaurantTableQMatrices;
			
			restaurantLikelihoods = rmap2._restLikeLihoods;
			restaurantPriors = rmap2._restPriors;
			restGradientsLLH = rmap2._restGradientsFromLLH;
			restGradientsPrior = rmap2._restGradientsFromPrior;
//			for (int customer_i : customersPermutation)
//			{ // iterate through all customers in random order and determine what table they should
//				// be assigned. They be assigned to a NEW TABLE with certain probability.
////					System.out.println ("Customer" + customer_i + " is entering the restaurant...");
//				// StopWatch updateTableAssignmentWatch = StopWatch.createStarted();
//				rmap2 = updateTableAssignmentWithDatabase (environment, customer_i, irlAlgo, tableAssignmentMatrix, tableWeightVectors, 
//															tablePolicyVectors, tableValueVectors, tableQVectors, restaurantLikelihoods, 
//															restaurantPriors, restGradientsLLH, restGradientsPrior, numThreads, numGPUs);
//				// updateTableAssignmentWatch.stop();
////				System.out.println ("Time for running updateTableAssignment for customer" + customer_i + ": " + updateTableAssignmentWatch.getTime ());
//				tableAssignmentMatrix = rmap2._restaurantAssignmentMatrix;
//				tableWeightVectors = rmap2._restaurantTableWeightMatrices;
//				tablePolicyVectors = rmap2._restaurantTablePolicyMatrices;
//				tableValueVectors = rmap2._restaurantTableValueMatrices;
//				tableQVectors = rmap2._restaurantTableQMatrices;
//				
//				restaurantLikelihoods = rmap2._restLikeLihoods;
//				restaurantPriors = rmap2._restPriors;
//				restGradientsLLH = rmap2._restGradientsFromLLH;
//				restGradientsPrior = rmap2._restGradientsFromPrior;
//			}
			reportEvent ("Updated Seating arrangement :");
//			for (int customer = 0; customer < numberTrajectories; customer++)
//				System.out.println ("Customer " + customer + " sits at table : " + tableAssignmentMatrix[customer][0]);
			
			RestaurantMap rmap3 = generateNewTableAssignmentPartition (numThreads, numGPUs, environment.getNumStates (), environment.getNumActions (), 
																		environment.getNumRewardFeatures (), tableWeightVectors, tablePolicyVectors, 
																		tableValueVectors, tableQVectors, tableAssignmentMatrix);
			tableAssignmentMatrix = rmap3._restaurantAssignmentMatrix;
			tableWeightVectors = rmap3._restaurantTableWeightMatrices;
			tablePolicyVectors = rmap3._restaurantTablePolicyMatrices;
			tableValueVectors = rmap3._restaurantTableValueMatrices;
			tableQVectors = rmap3._restaurantTableQMatrices;
			
			updateRewardFunctions (numThreads, numGPUs, trajectorySet, environment, irlAlgo, tableAssignmentMatrix, tableWeightVectors, tablePolicyVectors, 
									tableValueVectors, tableQVectors, restaurantLikelihoods, restaurantPriors, restGradientsLLH, restGradientsPrior);
			
			log_posteriorProbability2 = 
					computeLogPosteriorProbabilityForDirichletProcessMixture (numThreads, numGPUs, trajectorySet, tableAssignmentMatrix, 
																			  tableWeightVectors, tablePolicyVectors, environment, irlAlgo, false);
			reportIterationValue ("logPosteriorProb", iter, log_posteriorProbability2);
			Double	logProb = bestSampledRestaurant.getLogPosteriorProb ();
			if ((Double.compare (log_posteriorProbability2, logProb) > 0) || Double.isInfinite (logProb))	// GTD 1/21/20 Always write if prob neg infinity
			{
				// if(log_posteriorProbability2 > bestSampledRestaurant.getLogPosteriorProb() )
				bestSampledRestaurant.setSeatingArrangement (tableAssignmentMatrix);
				bestSampledRestaurant.setWeightMatrices (tableWeightVectors);
				bestSampledRestaurant.setPolicyMatrices (tablePolicyVectors);
				bestSampledRestaurant.setValueMatrices (tableValueVectors);
				bestSampledRestaurant.setQMatrices (tableQVectors);
				
				bestSampledRestaurant.setLogPosteriorProb (log_posteriorProbability2);
				restaurantFactory.write (bestSampledRestaurant);
				mdpFactory.write (environment);
			}
			
			IRLRestaurant restaurantForCurrentIteration = new IRLRestaurant (tableAssignmentMatrix, tableWeightVectors, tablePolicyVectors, 
																			 tableValueVectors, tableQVectors, log_posteriorProbability2);
			mhSampledRestaurants.add (restaurantForCurrentIteration);
			reportValue ("*******Current best restaurant DB has logPosteriorProb", bestSampledRestaurant.getLogPosteriorProb ());
			reportValue ("*******Current best restaurant also has numTables = ", bestSampledRestaurant.getPolicyMatrices ().size ());
			
			if (profile)
		        Controller.saveSnapshot (new File (profileName (numThreads, numGPUs, restaurantForCurrentIteration.getPolicyMatrices ().size ()) + "." + iter + ".jps"));
//			System.out.println ("Seating arrangement :");
//			for (int customer = 0; customer < numberTrajectories; customer++)
//			{
//				System.out.println ("Customer " + customer + " sits at table :" + bestSampledRestaurant._seatingArrangement[customer][0]);
//			}
			
		} // end for-loop for MH algorithm iterations
		restaurantFactory.write (bestSampledRestaurant);
		mdpFactory.write (environment);
		reportValue ("Overall best restaurant has logPosteriorProb", bestSampledRestaurant.getLogPosteriorProb ());
		reportValue ("Overall best restaurant also has numTables = ", bestSampledRestaurant.getPolicyMatrices ().size ());

		System.out.println ("Seating arrangement :");
		double[][]	seatingArrangement = bestSampledRestaurant.getSeatingArrangement ();
		for (int customer = 0; customer < numberTrajectories; customer++)
		{
			System.out.println ("Customer " + customer + " sits at table : " + seatingArrangement[customer][0]);
		}
		
//		addInferredTablesDuringRestaurantInitialization = true;
	}
	
	
	/**
	 * Make the start of the Profile name, that identifies what's happening
	 * 
	 * @param numThreads	Maximum number of threads that can be used at once
	 * @param numGPUs		Maximum number of GPUs that can be used at once
	 * @param numTables		Current number of tables in use
	 * @return	String with descriptor
	 */
	private static final String profileName (int numThreads, int numGPUs, int numTables)
	{
		return "Profile." + numThreads + "." + numGPUs + "." + numTables + "." + timeStampStr ();
	}
	
	
	/**
	 * 
	 * 
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param N
	 * @param nTrajectories
	 * @param env
	 * @param irlAlgo
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 */
	private static final void initializeTables (int numThreads, int numGPUs, int N, int nTrajectories, MDPCancer env, IRLAlgorithmCancer irlAlgo, 
												Map<Integer, double[][]> tableWeightVectors, Map<Integer, double[][]> tablePolicyVectors, 
												Map<Integer, double[][]> tableValueVectors, Map<Integer, double[][]> tableQVectors)
	{
		if (numThreads <= 1)
		{
			for (int table_i = 1; table_i <= N; ++table_i)
			{
//				StopWatch initializeEachTableWatch = StopWatch.createStarted ();
				
				// generate/sample new weight, policy and value vector to associate/map with each table index-value i
				generateNewWeights (table_i, env, irlAlgo, tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, true);
//				initializeEachTableWatch.stop ();
//				System.out.println ("Time required for initialize table " + table_i + " via generateNewWeights():" + 
//									initializeEachTableWatch.getTime ());
			}
		}
		else
		{
			initializeTablesThreaded (numThreads, numGPUs, 1, N, nTrajectories, env, irlAlgo, tableWeightVectors, 
									  tablePolicyVectors, tableValueVectors, tableQVectors);
		}
	}
	
	
	/**
	 * 
	 * 
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param firstTable
	 * @param numTables
	 * @param nTrajectories
	 * @param env
	 * @param irlAlgo
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 */
	private static final void initializeTablesThreaded (int numThreads, int numGPUs, int firstTable, int numTables, int nTrajectories, MDPCancer env, 
														IRLAlgorithmCancer irlAlgo, Map<Integer, double[][]> tableWeightVectors, 
														Map<Integer, double[][]> tablePolicyVectors, Map<Integer, double[][]> tableValueVectors, 
														Map<Integer, double[][]> tableQVectors)
	{
		Map<Integer, double[][]>	tableWeightVectorsT = Collections.synchronizedMap (new HashMap<> ());
		Map<Integer, double[][]>	tablePolicyVectorsT = Collections.synchronizedMap (new HashMap<> ());
		Map<Integer, double[][]>	tableValueVectorsT = Collections.synchronizedMap (new HashMap<> ());
		Map<Integer, double[][]>	tableQVectorsT = Collections.synchronizedMap (new HashMap<> ());
		List<Thread>				threads = new ArrayList<> (numThreads);
		BlockingQueue<Integer>		tableIndexes = makeQueue (firstTable, numTables);
		
		reportThreading ("initializeTablesThreaded", numThreads, numGPUs, numTables);
		int	tableCount = numTables - firstTable + 1;
		for (int i = 1; (i <= numThreads) && (i <= tableCount); ++i)
		{
			MDPCancer			environment = new MDPCancer (env);
			// generate/sample new weight, policy and value vector to associate/map with each table index-value i
			TableWeightCreator	updater = new TableWeightCreator (tableIndexes, environment, irlAlgo, tableWeightVectorsT, 
																  tablePolicyVectorsT, tableValueVectorsT, tableQVectorsT, true);
			Thread	theThread = makeThread (updater, "generateNewWeights ", i, numGPUs);
			
			theThread.start ();
			threads.add (theThread);
		}
		
		// Wait for the threads to finish
		waitForThreads (threads);
		
		tableWeightVectors.putAll (tableWeightVectorsT);
		tablePolicyVectors.putAll (tablePolicyVectorsT);
		tableValueVectors.putAll (tableValueVectorsT);
		tableQVectors.putAll (tableQVectorsT);
	}
	
	
	private static final void initializeTablesWithSubstitution (int numThreads, int numGPUs, IRLRestaurant mayorestaurant, 
																int numInferredTablesFromMayoRestaurant, int highestTblIndxForRestaurant, int nTrajectories, 
																MDPCancer env, IRLAlgorithmCancer irlAlgo, Map<Integer, double[][]> tableWeightVectors, 
																Map<Integer, double[][]> tablePolicyVectors, Map<Integer, double[][]> tableValueVectors, 
																Map<Integer, double[][]> tableQVectors)
	{
		// stores the weight-vector associated with each table index/value; Although the numberRewardFeatures is fixed, 
		// the number of active tables at any given moment can change, so we need the set of weight vectors be dynamic in size.
		Map<Integer, double[][]> inferredTableWeightVectors = mayorestaurant.getWeightMatrices ();
		// each policy is a column matrix of dimension numStates x 1
		Map<Integer, double[][]> inferredTablePolicyVectors = mayorestaurant.getPolicyMatrices ();
		// each value is a column matrix of dimension numStates x 1 (i.e. it is NOT a row vector)
		Map<Integer, double[][]> inferredTableValueVectors = mayorestaurant.getValueMatrices ();
		// each value is a column matrix of dimension (numStates*numActions) x 1 (i.e. it is NOT a row vector)
		Map<Integer, double[][]> inferredTableQVectors = mayorestaurant.getQMatrices ();
		
		for (int table_k = 1; table_k < numInferredTablesFromMayoRestaurant + 1; table_k++)
		{
			tableWeightVectors.put (table_k, inferredTableWeightVectors.get (table_k));
			tablePolicyVectors.put (table_k, inferredTablePolicyVectors.get (table_k));
			tableValueVectors.put (table_k, inferredTableValueVectors.get (table_k));
			tableQVectors.put (table_k, inferredTableQVectors.get (table_k));
			
			System.out.println ("Substituted inferred table (" + table_k + ") into restaurant table-assignment");
		}
		
		if (numThreads <= 1)
		{
			for (int table_i = numInferredTablesFromMayoRestaurant + 1; table_i <= highestTblIndxForRestaurant; table_i++)
			{
				StopWatch initializeEachTableWatch = StopWatch.createStarted ();
				
				// generate/sample new weight, policy and value vector to associate/map with each table index-value i
				generateNewWeights (table_i, env, irlAlgo, tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, true);
				initializeEachTableWatch.stop ();
				System.out.println ("Time required for initialize table " + table_i + " via generateNewWeights():" + initializeEachTableWatch.getTime ());
			}
		}
		else
		{
			initializeTablesThreaded (numThreads, numGPUs, numInferredTablesFromMayoRestaurant + 1, highestTblIndxForRestaurant, 
									  nTrajectories, env, irlAlgo, tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors);
		}
		
	}
	
	
	/**
	 * JK data validated 7.24.2019
	 * Sample new weight and compute its policy and value
	 * 
	 * @param env
	 * @param irloptions
	 * @param tblWeightVectors
	 * @param tblPlVectors
	 * @param tblVlVectors
	 * @param tblQVectors
	 * @param changeWeights
	 */
	static final Map<String, double[][]> generateNewWeights (int tableIndexValue, MDPCancer env, IRLAlgorithmCancer irloptions,  
															 Map<Integer, double[][]> tblWeightVectors,  Map<Integer, double[][]> tblPlVectors,  
															 Map<Integer, double[][]> tblVlVectors,Map<Integer, double[][]> tblQVectors, 
															 boolean changeWeights)
	{
		Map<String, double[][]> w_p_v_qHashMap = new HashMap<String, double[][]> ();
		int numFeatures = env.getNumRewardFeatures ();
		double lowerB = irloptions._lowerRewardBounds;
		double upperB = irloptions._upperRewardBounds;
		Prior pr = irloptions.getPrior ();
		// if prior is normal-gamma or beta-gamma distribution
		// calculate weight 'w' with sampleMultinomial()
		double[][] weightMatrix = null; // will be a numFeatures x 1 matrix (column vector)
		
		if (pr.get_identifier () == 1)
		{ // if using a normal-gamma prior
			// System.out.println("Using normal-gamma prior to generate new weights!");
			double[][] weights = new double[numFeatures][1];
			for (int f = 0; f < numFeatures; f++)
			{
				int index = SampleMultinomialIRL.sampleSingleStateFromMultinomial (10, irloptions.getRewardDistro (), RNG);
				weights[f] = new double[] {irloptions.getRewardArray ()[index]};
			}
			weightMatrix = weights;
			if (changeWeights)
			{
				safePut (tblWeightVectors, tableIndexValue, weightMatrix);
			}
		}
		else if (pr.get_identifier () == 3)
		{ // if using a gaussian prior
			// System.out.println("Using Gaussian prior to generate new weights");
			double mu = pr.get_mu ();
			double sigma = pr.get_sigma ();
			double[][] rndNormalWeights2DArray = MersenneTwisterFastIRL.RandomNormalMatrix (numFeatures, 1);
			DoubleMatrix rndNormalWeightsMatrix = new DoubleMatrix (rndNormalWeights2DArray);
			rndNormalWeightsMatrix = rndNormalWeightsMatrix.mul (sigma);
			rndNormalWeightsMatrix = rndNormalWeightsMatrix.add (mu);
			double[][] weightsMatrixAs2DArray = rndNormalWeightsMatrix.toArray2 ();
			// make sure that weight vector values are LESS than the UPPER bound,and then make sure
			// that weight vector values are GREATER than the LOWER BOUND
			weightMatrix = MatrixUtilityJBLAS.withinBounds (weightsMatrixAs2DArray, lowerB, upperB);
			if (changeWeights)
			{
				safePut (tblWeightVectors, tableIndexValue, weightMatrix);
			}
		}
		else if (pr.get_identifier () == 4)
		{ // if using a Uniform prior
			// System.out.println("Using UNIFORM PRIOR to generate new weights");
			double[][] rndNormalWeights2DArray = MersenneTwisterFastIRL.RandomNormalMatrix (numFeatures, 1);
			DoubleMatrix rndNormalWeightsMatrix = new DoubleMatrix (rndNormalWeights2DArray);
			rndNormalWeightsMatrix = rndNormalWeightsMatrix.mul (upperB - lowerB);
			rndNormalWeightsMatrix = rndNormalWeightsMatrix.add (lowerB);
			weightMatrix = rndNormalWeightsMatrix.toArray2 ();
			if (changeWeights)
			{
				safePut (tblWeightVectors, tableIndexValue, weightMatrix);
			}
		}
		
		// generate REWARD matrix from weightVector (convertW2R)
		// use weight vector and mdp representation of problem environment to generate the
		// corresponding reward function
//		StopWatch generateWeightedRFunctionwatch = StopWatch.createStarted ();
		
		RewardFunctionGenerationCancer.generateWeightedRewardFunction (env, weightMatrix);
//		generateWeightedRFunctionwatch.stop ();
//		reportTableTime ("generateWeightedRewardFunction()", tableIndexValue, generateWeightedRFunctionwatch.getTime ());
		
		// generate POLICY and VALUE matrix (policyIteration)
//		StopWatch runPolicyIterationnwatch = StopWatch.createStarted ();
		
		Map<String, double[][]> policy_value_h_q_Matrices = PolicySolverCancer.runPolicyIteration (env, irloptions, null);
//		runPolicyIterationnwatch.stop ();
//		reportTableTime ("runPolicyIteration()", tableIndexValue, runPolicyIterationnwatch.getTime ());
		
		if (changeWeights)
		{
			safePut (tblPlVectors, tableIndexValue, policy_value_h_q_Matrices.get ("P"));
			safePut (tblVlVectors, tableIndexValue, policy_value_h_q_Matrices.get ("V"));
			safePut (tblQVectors, tableIndexValue, policy_value_h_q_Matrices.get ("Q"));
		}
		else
		{
			w_p_v_qHashMap.put ("W", weightMatrix);
			w_p_v_qHashMap.put ("P", policy_value_h_q_Matrices.get ("P"));
			w_p_v_qHashMap.put ("V", policy_value_h_q_Matrices.get ("V"));
			w_p_v_qHashMap.put ("Q", policy_value_h_q_Matrices.get ("Q"));
		}
		return w_p_v_qHashMap;
	}
	

	// similar to getTrajInfo()
	/**
	 * Computes occupancy measure and empirical policy for trajectories. Called within
	 * dirichletProcessMHLInference()
	 * 
	 * @param trajSet
	 * @param env
	 * @param iTrajectoryInformation
	 * @param numTrajectories
	 * @param numStepsPerTrajectory
	 */
	private static final Multimap<Integer, double[]> computeOccupancy (Map<Integer, double[][]> subsetOfTrajectories, MDPCancer env, 
																		Multimap<Integer, double[]> countInfoForSubsetOfTrajs, 
																		boolean isForInitialDPMPosteriorCalculation)
	{
		int numStates = env.getNumStates ();
		int numActions = env.getNumActions ();
		// int numStepsPerTrajectory = subsetOfTrajectories.get(0)[0].length; //remember
		// subsetOfTrajectories.get(0)[0] is the state sequence, and subsetOfTrajectories.get(0)[1]
		// is the action sequence for the 0th trajectory in this subsetOfTrajectories
		Integer firstKey = getFirstKey (subsetOfTrajectories);	// subsetOfTrajectories is a Map, "first key" is random. GTD This is right
		int numStepsPerTrajectory = subsetOfTrajectories.get (firstKey)[0].length;
		Double state_ts = 0.0;
		Double action_ts = 0.0;
		
		int sBool = 0;
		int aBool = 0;
		int aBool_v2 = 0;
		int s = 0;
		int a = 0;
		
		for (Entry<Integer, double[][]> entry: subsetOfTrajectories.entrySet ())
		{
			Integer		t = entry.getKey ();
			double[][]	trajectory = entry.getValue ();
			double[]	states = trajectory[0];
			double[]	actions = trajectory[1];
			//JK 6.24.2019 create these count matrices once for each trajectory assigned to a given table
			double[][] countMatrix = new double[numStates][numActions];
			double[][] occupancyMatrix = new double[numStates][numActions]; // discounted state-action frequency
			for (int step = 0; step < numStepsPerTrajectory; ++step)
			{
				state_ts = states[step];
				action_ts = actions[step];
				sBool = Double.compare (state_ts, -1.0);
				aBool = Double.compare (action_ts, -1.0);
				if ((sBool == 0) && (aBool == 0))	//GTD changed back per John
				{ // if aBool AND sBool are TRUE (NOTE: 0 as input from Double.compare means 'TRUE')
					break;
				}
				
				// XXX:JK: 3.29.2019 added...RECALL: state int = 0 is the DEFAULT normal state and action int = 0 is the default filler action
//				sBool_v2 = Double.compare (state_ts, 0.0);	// GTD Not used
				aBool_v2 = Double.compare (action_ts, 0.0);
				
				//XXX:NOTE: JK 7.19.2019 this if-condition only applies to cancer trajectories in which action 0 is by default
				// the padding action for short trajectories
				if((!env._isGridWorldEnv) && (aBool_v2 == 0) )
				{ // So we break updating count and occupancy matrices because we Don't want to count
					// any further s-a pairs in this trajectory because we have reached an
					// actionint=0 (indicative that we are just filling in the remaining s-a pairs
					// of the maxLength=5 trajectory) (NOTE: 0 as input from Double.compare means
					// 'TRUE')
					break;
				}
				
				s = state_ts.intValue ();
				a = action_ts.intValue ();
				
				countMatrix[s][a] += 1;
				occupancyMatrix[s][a] += Math.pow (env.getDiscount (), step);
				
			}
			
			// JK 6.23.2019: modified so that we add trajectory-specific sa counts into Multimap<trajINT, double[]> stateActionCountMap
			for (int state = 0; state < numStates; ++state)
			{
				for (int action = 0; action < numActions; ++action)
				{
					if (countMatrix[state][action] > 0.0)
					{
						double[] observed_stateActionPairInfo = new double [3];
						observed_stateActionPairInfo[0] = (double) state;
						observed_stateActionPairInfo[1] = (double) action;
						observed_stateActionPairInfo[2] = countMatrix[state][action];
						countInfoForSubsetOfTrajs.put (t, observed_stateActionPairInfo);
					}
				}
			}
			// move-on to next trajectory	
			
		}
		
		return countInfoForSubsetOfTrajs;
	}
	
	
	/**
	 * Computes occupancy measure and empirical policy for trajectories. Called within
	 * dirichletProcessMHLInference()
	 * 
	 * @param trajSet
	 * @param env
	 * @param iTrajectoryInformation
	 * @param numTrajectories
	 * @param numStepsPerTrajectory
	 */
	private static final void computeOccupancyFromDatabase (Map<Integer, double[][]> subsetOfTrajectories, MDPCancer env, 
															boolean isForInitialDPMPosteriorCalculation)
	{
		int numStates = env.getNumStates ();
		int numActions = env.getNumActions ();
		// int numStepsPerTrajectory = subsetOfTrajectories.get(0)[0].length; //remember
		// subsetOfTrajectories.get(0)[0] is the state sequence, and subsetOfTrajectories.get(0)[1]
		// is the action sequence for the 0th trajectory in this subsetOfTrajectories
		Integer firstKey = getFirstKey (subsetOfTrajectories);	// subsetOfTrajectories is a Map, "first key" is random. GTD This is right
		int numStepsPerTrajectory = subsetOfTrajectories.get (firstKey)[0].length;
		Double state_ts = 0.0;
		Double action_ts = 0.0;
		
		int sBool = 0;
		int aBool = 0;
		int aBool_v2 = 0;
		int s = 0;
		int a = 0;
		
		for (Entry<Integer, double[][]> entry: subsetOfTrajectories.entrySet ())
		{
			Integer		t = entry.getKey ();
			double[][]	trajectory = entry.getValue ();
			double[]	states = trajectory[0];
			double[]	actions = trajectory[1];
			//JK 6.24.2019 create these count matrices once for each trajectory assigned to a given table
			double[][] countMatrix = new double[numStates][numActions];
			double[][] occupancyMatrix = new double[numStates][numActions]; // discounted state-action frequency
			for (int step = 0; step < numStepsPerTrajectory; ++step)
			{
				state_ts = states[step];
				action_ts = actions[step];
				sBool = Double.compare (state_ts, -1.0);
				aBool = Double.compare (action_ts, -1.0);
				if ((sBool == 0) && (aBool == 0))	//GTD changed back per John
				{ // if aBool AND sBool are TRUE (NOTE: 0 as input from Double.compare means 'TRUE')
					break;
				}
				
				// XXX:JK: 3.29.2019 added...RECALL: state int = 0 is the DEFAULT normal state and action int = 0 is the default filler action
//				sBool_v2 = Double.compare (state_ts, 0.0);	// GTD Not used
				aBool_v2 = Double.compare (action_ts, 0.0);
				if (aBool_v2 == 0)
				{ // So we break updating count and occupancy matrices because we Don't want to count
					// any further s-a pairs in this trajectory because we have reached an
					// actionint=0 (indicative that we are just filling in the remaining s-a pairs
					// of the maxLength=5 trajectory) (NOTE: 0 as input from Double.compare means
					// 'TRUE')
					break;
				}
				
				s = state_ts.intValue ();
				a = action_ts.intValue ();
				
				countMatrix[s][a] += 1;
				occupancyMatrix[s][a] += Math.pow (env.getDiscount (), step);
				
			}
			
			// JK 6.23.2019: modified so that we add trajectory-specific sa counts into Multimap<trajINT, double[]> stateActionCountMap
			for (int state = 0; state < numStates; ++state)
			{
				for (int action = 0; action < numActions; ++action)
				{
					if (countMatrix[state][action] > 0.0)
					{
//						double[] observed_stateActionPairInfo = new double [3];
//						observed_stateActionPairInfo[0] = (double) state;
//						observed_stateActionPairInfo[1] = (double) action;
//						observed_stateActionPairInfo[2] = countMatrix[state][action];
//						countInfoForSubsetOfTrajs.put (t, observed_stateActionPairInfo);
						// Store into cassandra table instead of in memory
						String cqlInsertPairCountInfoTo_countInfoTable = 
								"insert INTO countinfofortrajs_table (trajint, statedbl, actiondbl, countdbl) values(" + t + 
								"," + state + "," + action + "," + countMatrix[state][action] + ")";
						_session.execute (cqlInsertPairCountInfoTo_countInfoTable);
						
					}
				}
			}
			// move-on to next trajectory	
		}
		
	}
	
	
	/**
	 * 
	 * 
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param numStates
	 * @param numActions
	 * @param numFeatures
	 * @param tblWeightVectors	Passed in Map not modified, internally modified
	 * @param tblPlVectors		Passed in Map not modified, internally modified
	 * @param tblVlVectors	Passed in Map not modified, internally modified
	 * @param tblQVectors	Passed in Map not modified, internally modified
	 * @param tblAssignmentMatrix	Not modified
	 * @return	Updated {@link RestaurantMap}
	 */
	private static final RestaurantMap generateNewTableAssignmentPartition (int numThreads, int numGPUs, int numStates, int numActions, int numFeatures, 
																	 Map<Integer, double[][]> tblWeightVectors, Map<Integer, double[][]> tblPlVectors, 
																	 Map<Integer, double[][]> tblVlVectors, Map<Integer, double[][]> tblQVectors, 
																	 double[][] tblAssignmentMatrix)
	{
		Double	highestTableIndex = MatrixUtilityJBLAS.maxPerColumn (tblAssignmentMatrix)[0];
		int		N = highestTableIndex.intValue ();
		
		if (kPrintDebug)
		{
			System.out.println ("Largest table index = " + N);
			reportEvent ("Testing for used tables");
		}
		
		RestaurantMap rmap = null;
		
		List<double[][]> wMatrix = createListWithNulls (N);
		List<double[][]> pMatrix = createListWithNulls (N);
		List<double[][]> vMatrix = createListWithNulls (N);
		List<double[][]> qMatrix = createListWithNulls (N);
		
		List<double[]> tableLabelMatrix = createListWithNulls (N);
		
		double[][] newPartitiontblAssignmentMatrix = new double[tblAssignmentMatrix.length][1];
		
		// returns a row vector containing the maximum values in each column of the 'tableAssignmentMatrix'. 
		// Since this matrix is nTraj x 1. We only care about the 0th element of the returned row vector
		
		addUsedTables (numThreads, numGPUs, wMatrix, pMatrix, vMatrix, qMatrix, tableLabelMatrix, N, 
						tblAssignmentMatrix, tblWeightVectors, tblPlVectors, tblVlVectors, tblQVectors);
		reportValue ("Tables that are used", wMatrix.size ());
		
		int	numLabels = tableLabelMatrix.size ();
		if (numLabels > 0)
		{ // the size of tableLabelMatrix = number of distinct tables >0;
			updateTableLabels (numThreads, numGPUs, numLabels, tableLabelMatrix, tblAssignmentMatrix, newPartitiontblAssignmentMatrix);
			int fooCounter = 0;
			Set<Double> uniqueTableIndices = MatrixUtilityJBLAS.countNumberUniqueElements (newPartitiontblAssignmentMatrix);
			for (double[] nextElement : newPartitiontblAssignmentMatrix)
			{
//				System.out.println (fooCounter + ") newPartitionTblAssignmentMatrix for customer_i = " + nextElement[0]);
				if (nextElement[0] == 0.0)
				{
					System.out.println ("**********************ZERO-valued table index for row(" + fooCounter + 
										")in newPartitionTblAssignmentMatrix where customer_i table idx = " + nextElement[0]);
				}
				++fooCounter;
			}
			// for (Double unique_ti: uniqueTableIndices) {
			// System.out.println("HashSet<Double> uniqueTableIndices unique tableidx ="+unique_ti);
			//
			// }
			Double[] sortedArray = VectorUtility.sortSet (uniqueTableIndices);
//			for (Double unique_d : sortedArray)
//				System.out.println ("sortedArray unique table index = " + unique_d);
			// print statements for debugging error on mforge where wMatrix size != sortedArrayLength
//			System.out.println ("wMatrix size =" + wMatrix.size ());
//			System.out.println ("pMatrix size =" + pMatrix.size ());
//			System.out.println ("vMatrix size =" + vMatrix.size ());
//			System.out.println ("qMatrix size =" + qMatrix.size ());
//			System.out.println ("sortedArray length =" + sortedArray.length);
			// recall that wMatrix, pMatrix, and vMatrix are ArrayLists. The first element in each list 
			// corresponds to the lowest table index-value among the new table assignment index-values
			tblWeightVectors = MatrixUtilityJBLAS.convertMatrixListToMap (wMatrix, sortedArray);
			tblPlVectors = MatrixUtilityJBLAS.convertMatrixListToMap (pMatrix, sortedArray);
			tblVlVectors = MatrixUtilityJBLAS.convertMatrixListToMap (vMatrix, sortedArray);
			tblQVectors = MatrixUtilityJBLAS.convertMatrixListToMap (qMatrix, sortedArray);
			tblAssignmentMatrix = newPartitiontblAssignmentMatrix;
		}
		
		rmap = new RestaurantMap (tblWeightVectors, tblPlVectors, tblVlVectors, tblQVectors, tblAssignmentMatrix);
		return rmap;
		
	}
	
	
	/**
	 * Fill {@code wMatrix}, {@code pMatrix}, {@code vMatrix}, and {@code tableLabelMatrix} with 
	 * the information for the tables that are referenced in {@code tblAssignmentMatrix}
	 * 
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param wMatrix
	 * @param pMatrix
	 * @param vMatrix
	 * @param qMatrix
	 * @param tableLabelMatrix
	 * @param numTables
	 * @param tblAssignmentMatrix
	 * @param tblWeightVectors
	 * @param tblPlVectors
	 * @param tblVlVectors
	 * @param tblQVectors
	 */
	private static final void addUsedTables (int numThreads, int numGPUs, List<double[][]> wMatrix, List<double[][]> pMatrix, List<double[][]> vMatrix, 
											 List<double[][]> qMatrix, List<double[]> tableLabelMatrix, int numTables, 
											 double[][] tblAssignmentMatrix, Map<Integer, double[][]> tblWeightVectors, 
											 Map<Integer, double[][]> tblPlVectors, Map<Integer, double[][]> tblVlVectors, 
											 Map<Integer, double[][]> tblQVectors)
	{
		if (numThreads > 1)
		{
			addUsedTablesThreaded (numThreads, numGPUs, wMatrix, pMatrix, vMatrix, qMatrix, tableLabelMatrix, numTables, 
									tblAssignmentMatrix, tblWeightVectors, tblPlVectors, tblVlVectors, tblQVectors);
		}
		else
		{
			for (int tblIndex = 1; tblIndex <= numTables; ++tblIndex)
			{
				// System.out.println("tblIndex="+tblIndex);
				// return a logical array with elements set to logical 1 (true) where elements in the
				// assignmentMatrix equal to 'tblIndex'; i.e. which trajectories/customers were assigned
				// to table 'tblIndex'
//				DoubleMatrix logic2Darray = MatrixUtilityJBLAS.toLogicalMatrix (new DoubleMatrix (tblAssignmentMatrix), tblIndex);
//				boolean[][]	logic2Darray = toLogicalMatrix (tblAssignmentMatrix, tblIndex);
				
				// since tblAssignmentMatrix is numTraj x 1, there is only 1 column, thus we only need
				// to look at sum of 0th column which is 0th element in returned row vector
//				if (logic2Darray.sum () > 0.0)
				if (hasTrue (tblAssignmentMatrix, tblIndex))
				{
					// i.e. if the # customers assigned to table 'tblIndex' is >0
//					DoubleMatrix kthWColumnMatrix = new DoubleMatrix (tblWeightVectors.get (tblIndex));
					double[][]	kthWColumnMatrix = tblWeightVectors.get (tblIndex);
					wMatrix.set (tblIndex, kthWColumnMatrix);
					
					double[][]	kthPColumnMatrix = tblPlVectors.get (tblIndex);
					pMatrix.set (tblIndex, kthPColumnMatrix);
					
					double[][]	kthVColumnMatrix = tblVlVectors.get (tblIndex);
					vMatrix.set (tblIndex, kthVColumnMatrix);
					
					double[][]	kthQColumnMatrix = tblQVectors.get (tblIndex);
					qMatrix.set (tblIndex, kthQColumnMatrix);
					
					double[] tblLabel = new double[] {tblIndex, 0};
//					double[][] tblLabel = new double[][] {{tblIndex, 0}};
					
//					DoubleMatrix kthLabelRowMatrix = new DoubleMatrix (1, 2, tblLabel);
					tableLabelMatrix.set (tblIndex, tblLabel);
				}
			}
		}
		
		// Get rid of any unused indexes, including 0
		wMatrix.removeAll (Collections.singleton (null));
		pMatrix.removeAll (Collections.singleton (null));
		vMatrix.removeAll (Collections.singleton (null));
		qMatrix.removeAll (Collections.singleton (null));
		tableLabelMatrix.removeAll (Collections.singleton (null));
		
		relabel (tableLabelMatrix);
	}
	
	
	/**
	 * Assign labels to tableLabelMatrix entries now that it's all filled in
	 * 
	 * @param tableLabelMatrix	List to update
	 */
	private static void relabel (List<double[]> tableLabelMatrix)
	{
		int	pos = 1;
		
//		for (DoubleMatrix matrix : tableLabelMatrix)
		for (double[] matrix : tableLabelMatrix)
		{
			matrix[1] = (double) pos;
			++pos;
		}
	}
	
	
	/**
	 * 
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param wMatrix
	 * @param pMatrix
	 * @param vMatrix
	 * @param qMatrix
	 * @param tableLabelMatrix
	 * @param numTables
	 * @param tblAssignmentMatrix
	 * @param tblWeightVectors
	 * @param tblPlVectors
	 * @param tblVlVectors
	 * @param tblQVectors
	 */
	private static final void addUsedTablesThreaded (int numThreads, int numGPUs, List<double[][]> wMatrix, List<double[][]> pMatrix, 
													 List<double[][]> vMatrix, List<double[][]> qMatrix, List<double[]> tableLabelMatrix, int numTables, 
													 double[][] tblAssignmentMatrix, Map<Integer, double[][]> tblWeightVectors, 
													 Map<Integer, double[][]> tblPlVectors, Map<Integer, double[][]> tblVlVectors, 
													 Map<Integer, double[][]> tblQVectors)
	{
		List<Thread>			threads = new ArrayList<> (numThreads);
		BlockingQueue<Integer>	tableIndexes = makeQueue (numTables);
		
		reportThreading ("addUsedTablesThreaded", numThreads, numGPUs, numTables);
		if (numThreads > numTables)
			System.out.println ("addUsedTablesThreaded: " + numThreads + " threads, " + numTables + " tables");
		for (int i = 1; (i <= numThreads) && (i <= numTables); ++i)
		{
			TableUsedChecker	checker = new TableUsedChecker (tableIndexes, tblAssignmentMatrix, wMatrix, pMatrix, vMatrix, qMatrix, 
																tableLabelMatrix, tblWeightVectors, tblPlVectors, tblVlVectors, tblQVectors);
			Thread	theThread = makeThread (checker, "addUsedTables ", i, numGPUs);
			
			theThread.start ();
			threads.add (theThread);
		}
		
		// Wait for the threads to finish
		waitForThreads (threads);
	}
	
	
	/**
	 * Make a {@link BlockingQueue} that holds all the tables
	 * 
	 * @param numTables	Number of tables to add.. Will add the values 1 to {@code numTables} to the queue
	 * @return	{@link BlockingQueue} that can be used to get table indexes
	 */
	private static BlockingQueue<Integer> makeQueue (int numTables)
	{
		ArrayBlockingQueue<Integer>	theQueue = new ArrayBlockingQueue<> (numTables);
		
		for (int i = 1; i <= numTables; ++i)
			theQueue.add (Integer.valueOf (i));
		
		return theQueue;
	}
	
	
	/**
	 * Make a {@link BlockingQueue} that holds all the specified tables
	 * 
	 * @param firstTable	First table number to use
	 * @param lastTable		Last table number to use.  Will add {@code firstTable} to {@code lastTable}
	 * @return	{@link BlockingQueue} that can be used to get table indexes
	 */
	private static BlockingQueue<Integer> makeQueue (int firstTable, int lastTable)
	{
		ArrayBlockingQueue<Integer>	theQueue = new ArrayBlockingQueue<> (lastTable - firstTable + 1);
		
		for (int i = firstTable; i <= lastTable; ++i)
			theQueue.add (Integer.valueOf (i));
		
		return theQueue;
	}
	
	
	/**
	 * Make a {@link BlockingQueue} that holds all the tables in {@code tablesPermutation}, in the
	 * same order as they are in {@code tablesPermutation}
	 * 
	 * @param tablesPermutation	Indexes of the tables to add
	 * @return	{@link BlockingQueue} that can be used to get table indexes
	 */
	private static BlockingQueue<Integer> makeQueue (int[] tablesPermutation)
	{
		ArrayBlockingQueue<Integer>	theQueue = new ArrayBlockingQueue<> (tablesPermutation.length);
		
		for (int i : tablesPermutation)
			theQueue.add (Integer.valueOf (i));
		
		return theQueue;
	}
	
	
	/**
	 * 
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param numLabels
	 * @param tableLabelMatrix
	 * @param tblAssignmentMatrix
	 * @param newPartitiontblAssignmentMatrix
	 */
	private static void updateTableLabels (int numThreads, int numGPUs, int numLabels, List<double[]> tableLabelMatrix, 
											double[][] tblAssignmentMatrix, double[][] newPartitiontblAssignmentMatrix)
	{
		if (numThreads > 1)
		{
			updateTableLabelsThreaded (numThreads, numGPUs, numLabels, tableLabelMatrix, tblAssignmentMatrix, newPartitiontblAssignmentMatrix);
		}
		else
		{
			// change index-value of each distinct table index-value
			// (tableLabelMatrix.get(i).getData()[0][0]) to the new index-value
			// (tableLabelMatrix.get(i).getData()[0][1])
			for (double[] labelInfo : tableLabelMatrix)
			{
				double	oldIndex = labelInfo[0];	// GTD get first item
//				System.out.println ("old table index # " + oldIndex);
//				 // identify which tables in current assignmentMatrix have the same label of 'tableLabelMatrix.get(i).getData()[0][0]'

				int			numRows = tblAssignmentMatrix.length;
				
				for (int r = 0; r < numRows; ++r)
				{
					double[]	newRow = newPartitiontblAssignmentMatrix[r]; //JK 7.30.2019 this was missing
					double[]	theRow = tblAssignmentMatrix[r];
					int			numCols = theRow.length;
					for (int c = 0; c < numCols; ++c)
					{
						// replace the table-index associated with row/trajectory 'r' with its NEW table-index
						if (theRow[c] == oldIndex)
						{
							// replace label-index for the table assigned at position (r,c) in the new assignment matrix with new label
							// 'tableLabelMatrix.get(i).getData()[0][1]'
							
							newRow[c] = labelInfo[1];	// Get the second item 
						}
					}
				}
			}
		}
	}
	
	
	/**
	 * 
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param numLabels
	 * @param tableLabelMatrix
	 * @param tblAssignmentMatrix
	 * @param newPartitiontblAssignmentMatrix
	 */
	private static void updateTableLabelsThreaded (int numThreads, int numGPUs, int numLabels, List<double[]> tableLabelMatrix, 
													double[][] tblAssignmentMatrix, double[][] newPartitiontblAssignmentMatrix)
	{
		List<Thread>			threads = new ArrayList<> (numThreads);
		BlockingQueue<Integer>	tableIndexes = makeQueue (0, numLabels - 1);
		
		reportThreading ("updateTableLabelsThreaded", numThreads, numGPUs, numLabels);
		for (int i = 1; (i <= numThreads) && (i <= numLabels); ++i)
		{
			TableLabelUpdater	updater = new TableLabelUpdater (tableIndexes, tableLabelMatrix, tblAssignmentMatrix, 
																 newPartitiontblAssignmentMatrix);
			Thread	theThread = makeThread (updater, "updateTableLabels ", i, numGPUs);
			
			theThread.start ();
			threads.add (theThread);
		}
		
		// Wait for the threads to finish
		waitForThreads (threads);
	}
	
	
	/**
	 * JK data validated 7.24.2019
	 * Compute the gradient of the Q-function matrix
	 * 
	 * @param policyMatrix
	 * @param env
	 * @return the transpose of the computed qMatrixGradient with dimensions:
	 *         [numFeatures][numStates*numActions]
	 */
	public static final double[][] computeQMatrixGradient (double[][] policyMatrix, MDPCancer env)
	{
		Integer numStates = env.getNumStates ();
		Integer numActions = env.getNumActions ();
		Integer numFeatures = env.getNumRewardFeatures ();
		double epsilon = 1e-12;
		int maximumIterations = 10000;
		
		DoubleMatrix numStatesColMatrix = new DoubleMatrix (numStates, 1);
		for (int s = 0; s < numStates; s++)
		{
			numStatesColMatrix.put (s, 0, s);
		}
		
		// compute dQ / dw
		DoubleMatrix expectedPolicyMatrix = null;
		DoubleMatrix expectedPolicyMatrixVersion2b = new DoubleMatrix (numStates, (numStates * numActions)); //JK 7.24.2019 for debuggin purposes

		
		// If policy matrix is DETERMINISTIC (# columns ==1); initialize expectedPolicyMatrix similar
		// to PolicySolver.policyEvaluationStep()
		if (policyMatrix[0].length == 1)
		{

			
			DoubleMatrix idx = new DoubleMatrix (policyMatrix).mul (numStates);
			DoubleMatrix unitVectorTransposed = new DoubleMatrix (VectorUtility.createUnitSpaceVector (numStates, 0.0, 1.0)).transpose ();
			idx.addi (unitVectorTransposed);
			//DoubleMatrix unitVectorTransposedForLinearIdx = new DoubleMatrix (VectorUtility.createUnitSpaceVector (numStates, 1.0, 1.0)).transpose ();
			//JK 7.24.2019 start unit vector at 0 instead
			DoubleMatrix unitVectorTransposedForLinearIdx = new DoubleMatrix (VectorUtility.createUnitSpaceVector (numStates, 0.0, 1.0)).transpose (); 

			DoubleMatrix idx2b = idx.mul (numStates).add (unitVectorTransposedForLinearIdx); //JK 7.24.2019

			// idx2.subi (1); // 7.24.2019 not necessary for matrices starting at 0. 
			// JK added 6.22.2019 because this was necessary in similar calculation in PolicyEvaluation method
			

			
			int[] linearIndicesIdx2b = idx2b.toIntArray ();
			expectedPolicyMatrixVersion2b.put (linearIndicesIdx2b, 1.0);
			
			expectedPolicyMatrix = expectedPolicyMatrixVersion2b;
			
		}
		else // else if policy matrix is STOCHASTIC (only exists when we run compute logLikelihood using MLIRL which computes a stochastic policy jk)
		{
			throw new java.lang.RuntimeException("This algorithm is not yet fully configured for stochastic policies!!!");
//			expectedPolicyMatrix = expectedPolicyMatrixVersion2;
//			for (int state = 0; state < numStates; state++)
//			{
//				for (int action = 0; action < numActions; action++)
//				{ // iterate through each column/action, since in a stochastic policy more than 1 action is possible for each state
//					int columnSubscript = ((action - 1) * numStates) + state;
//					expectedPolicyMatrix.put (state, columnSubscript, policyMatrix[state][action]);
//				}
//			}
		}
		
		DoubleMatrix	qMatrixGradient = new DoubleMatrix (numStates * numActions, numFeatures);
		
		if (MatrixUtilityJBLAS.matrixMaximum (env.getDiscountedTransitionMatrix ().toArray2 ()) == 0.0)
		{
			throw new java.lang.RuntimeException ("discounted transition matrix is all zeros!!! Is that allowed???");
		}
		
//		int rowdimensionOfDTmatrix = env.getDiscountedTransitionMatrix ().getRows ();
//		int colDimensionOfDTmatrix = env.getDiscountedTransitionMatrix ().getColumns ();
//		printRowAndCols ("Current discountedTransition matrix has ", rowdimensionOfDTmatrix, colDimensionOfDTmatrix);
//		int rowDimOfExpPMatrix = expectedPolicyMatrix.getRows ();
//		int colDimofExpPMatrix = expectedPolicyMatrix.getColumns ();
//		printRowAndCols ("Expected policyMatrix has ", rowDimOfExpPMatrix, colDimofExpPMatrix);
//		System.out.println ("inside computeQMatrixGradient() Computing expected discount transition matrix by multipling " + 
//							"distcountTransitionMatrix by expectedPolicyMatrix");
		
		DoubleMatrix	expectedDiscountedTransitionMatrix = env.getDiscountedTransitionMatrix ().mmul (expectedPolicyMatrix);
		//double [][] expDiscTransMat = expectedDiscountedTransitionMatrix.toArray2();
		
		DoubleMatrix	stateFeatureREALMATRIX = MatrixUtilityJBLAS.convertMultiDimMatrixMap (env.getStateFeatureMatrixMAP ());
		
		int		rowsInQMatrixGradient = qMatrixGradient.getRows ();
		int		colsInQMatrixGradient = qMatrixGradient.getColumns ();
		
		if (kPrintDebug)
		{
			System.out.print ("Beginning calculating qmatrix gradient until convergence or maximum iterations (");
			System.out.print (Integer.toString (maximumIterations));
			System.out.println (")");
			printRowAndCols ("Starting qMatrix gradient has ", rowsInQMatrixGradient, colsInQMatrixGradient);
		}
		// GTD take this to JNI
		//JK 7.19.2019: Can't run JNI library on local laptop
		return convergeMatrix (qMatrixGradient, expectedDiscountedTransitionMatrix, stateFeatureREALMATRIX, maximumIterations, epsilon);

//UNBLOCK this code if you can't use JNI library
//		DoubleMatrix	holdMatrix = new DoubleMatrix (numStates * numActions, numFeatures);
//		boolean			isGradientConverged = false;
//		for (int i = 0; i < maximumIterations; i++)
//		{
//			DoubleMatrix qMatrixGradient_previous = qMatrixGradient;
//			// XXX: Changed matrix multiplication to mmul() not element-wise mul()
//			
//			qMatrixGradient = expectedDiscountedTransitionMatrix.mmuli (qMatrixGradient, holdMatrix);
//			matrixAdd (qMatrixGradient, stateFeatureREALMATRIX);
//			
//			double [][] qMatrixGradientDBLArray = qMatrixGradient.toArray2(); //JK 7.24.2019 for debugging contents of matrix
//			
//			isGradientConverged = compareMatrices (qMatrixGradient, qMatrixGradient_previous, epsilon);	// GTD this is faster, esp when not converged
//
//			
//			if (isGradientConverged)
//			{
//				if (kPrintDebug)
//				{
//					System.out.print ("computeQMatrixGradient Converged in ");
//					System.out.print (Integer.toString (i));
//					System.out.println (" iterations");
//				}
//				break;
//			}
//			else
//				holdMatrix = qMatrixGradient_previous;
//		}
//		DoubleMatrix qMatrixGradientTrans = qMatrixGradient.transpose ();
//		double[][] qMatrixGradientTransposed = qMatrixGradientTrans.toArray2 ();
//		return qMatrixGradientTransposed;
	}
	
	
	/**
	 * Add the contents of {@code source} to {@code target}, changing {@code target}
	 * 
	 * @param target	{@link DoubleMatrix} to add to, will get the results of the addition
	 * @param source	{@link DoubleMatrix} to add to {@code target}
	 */
	public static final void matrixAdd (DoubleMatrix target, DoubleMatrix source)
	{
		double[]	targetData = target.data;
		double[]	sourceData = source.data;
		int			len = targetData.length;
		
		for (int i = 0; i < len; ++i)
			targetData[i] += sourceData[i];
	}
	
	/**
	 * 
	 * 
	 * JK 7.26.2019 data validated
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param trajSet
	 * @param tableAssignmentMatrix
	 * @param tblWeightVectors
	 * @param tblPolicyVectors
	 * @param env
	 * @param irlAlgo
	 * @param partOfInitialDPMPosterior
	 * @return
	 */
	private static final double computeLogPosteriorProbabilityForDirichletProcessMixture (
			int numThreads, int numGPUs, List<double[][]> trajSet, double[][] tableAssignmentMatrix,
			Map<Integer, double[][]> tblWeightVectors, Map<Integer, double[][]> tblPolicyVectors, 
			MDPCancer env, IRLAlgorithmCancer irlAlgo, boolean partOfInitialDPMPosterior)
	{
		double alpha = irlAlgo.getAlpha ();
		
		// Trajectory data
		StopWatch computeTableAssignmentPriorProbWatch;
		if (kPrintDebug)
		{
			computeTableAssignmentPriorProbWatch = StopWatch.createStarted ();
		}
		
		////// ComputeTableAssignmentProbability////////Calculate prior of table assignment c, pr(c|alpha)
		double[][] z = MatrixUtilityJBLAS.tableCounter (tableAssignmentMatrix);
		double[] zRowVector = z[0];
		double[] rIndexVector = VectorUtility.createUnitSpaceVector (zRowVector.length, 1.0, 1.0);
		Double mVal = VectorUtility.multAndSum (zRowVector, rIndexVector);
		Double mValfactorial = CombinatoricsUtils.factorialDouble (mVal.intValue ());
		double[] alphaVector = VectorUtility.range (alpha, alpha + mVal.doubleValue () - 1);
		
		double[] q1 = VectorUtility.pow (rIndexVector, zRowVector);
		double[] q2 = VectorUtility.factorial (VectorUtility.cast (zRowVector));
		double[] qProduct = VectorUtility.mult (q1, q2);
		
		double scalarVal1 = 0.0;
		double alphaVecProd = VectorUtility.product (alphaVector);
		//JK added 5.29.2019 if-else condition to deal with dividing infinity/infinity
		if (!(Double.isInfinite (alphaVecProd) && Double.isInfinite (mValfactorial.doubleValue ())))
		{
			scalarVal1 = mValfactorial.doubleValue () / alphaVecProd;
		}
		else
		{
			scalarVal1 = 1.0;
		}
		
		double scalarVal2 = Math.pow (alpha, VectorUtility.sum (zRowVector)) / VectorUtility.product (qProduct);
		double probabilityVal = scalarVal1 * scalarVal2;
		///////////////////////////////////////////////////////////////////////////////////////////////////
		if (kPrintDebug)
		{
			computeTableAssignmentPriorProbWatch.stop ();
			System.out.println ("time for computing tableassignment prior prob in computeLogPosteriorProbForDirichletProcessMixture() :" + 
								computeTableAssignmentPriorProbWatch.getTime ());
		}
		
		double		logDirichletProcessPriorProb = Math.log (probabilityVal); // log prior probability of table assignment
		int			maxTableIndexInRestaurant = (int) MatrixUtilityJBLAS.matrixMaximum (tableAssignmentMatrix);
		double[]	logLikelihoods = new double[maxTableIndexInRestaurant + 1];
		double[]	logPriorProbs = new double[maxTableIndexInRestaurant + 1];
		if (numThreads > 1)
		{
			computeLogPosteriorProbabilityForDirichletProcessMixtureThreaded (numThreads, numGPUs, maxTableIndexInRestaurant, trajSet, 
																			  tableAssignmentMatrix, tblWeightVectors, tblPolicyVectors, env, 
																			  irlAlgo, partOfInitialDPMPosterior, logLikelihoods, logPriorProbs);
		}
		else
		{
			for (int t = 1; t <= maxTableIndexInRestaurant; ++t)	// Come here
			{
				computeLogPosteriorProbabilityForDirichletProcessMixture (t, trajSet, tableAssignmentMatrix, tblWeightVectors, tblPolicyVectors, 
																		  env, irlAlgo, partOfInitialDPMPosterior, logLikelihoods, logPriorProbs);
			}
			
		}
		
		double logLikelihood_total = 0.0;
		double logPriorProb_total = 0.0;
		
		for (double logLikelihood : logLikelihoods)
			logLikelihood_total += logLikelihood;
		for (double logPriorProb : logPriorProbs)
			logPriorProb_total += logPriorProb;
		
		
		// ...formulate the IRL problem into posterior optimization problem, finding reward function
		// R that maximizes the log unnormalized [posterior] = finding the reward function R that
		// maximizes [logLikelihood + logPriorProbability]
		double logPosteriorProb_total = logDirichletProcessPriorProb + logLikelihood_total + logPriorProb_total;
		//JK added 8.27.2019 to keep track of overall restaurant likelihood
		reportValue ("LogLikeLihood of current restaurant", logLikelihood_total);
		reportValue ("LogPriorProb of current restaurant", logPriorProb_total);
		reportValue ("LogDirichletProcessPriorProb of current restaurant", logDirichletProcessPriorProb);
		
		return logPosteriorProb_total;
	}
	
	
	protected static final void computeLogPosteriorProbabilityForDirichletProcessMixture (
			int table, List<double[][]> trajSet, double[][] tableAssignmentMatrix,
			Map<Integer, double[][]> tblWeightVectors, Map<Integer, double[][]> tblPolicyVectors, 
			MDPCancer env, IRLAlgorithmCancer irlAlgo, boolean partOfInitialDPMPosterior, 
			double[] logLikelihoods, double[] logPriorProbs)
	{
		// weightMatrix describing relevance of the |numFeatures| different features for table 'table'
		Integer		key = Integer.valueOf (table);
		double[][]	weightMatr = tblWeightVectors.get (key);
		double[][]	policyMatr;
		int			numberTrajectories = trajSet.size (); // should be same as MDP._numberTrajectories
		
		if (weightMatr == null)
		{
			System.out.println ("weightMatr is null for table index: " + table);
			System.err.println ("weightMatr is null for table index: " + table);
			return;
		}
		
		if (tblPolicyVectors.isEmpty ())
		{
			// generate REWARD matrix from weightVector (convertW2R)
			// use weight vector and mdp representation of problem environment to generate the
			// corresponding reward function
			StopWatch generateWeightedRewardFunctionWatch;
			if (kPrintDebug)
			{
				generateWeightedRewardFunctionWatch = StopWatch.createStarted ();
			}
			
			RewardFunctionGenerationCancer.generateWeightedRewardFunction (env, weightMatr);
			if (kPrintDebug)
			{
				generateWeightedRewardFunctionWatch.stop ();
				System.out.println ("time to generateWeightedRewardFunction() for table " + table + " in computeLogPosteriorProbForDirichletMixture: " + 
									generateWeightedRewardFunctionWatch.getTime ());
			}
			
			// generate POLICY and VALUE matrix (policyIteration)
			StopWatch runPolicyIterationWatch;
			if (kPrintDebug)
			{
				runPolicyIterationWatch = StopWatch.createStarted ();
			}
			Map<String, double[][]> P_V_H_Q_Matrices = PolicySolverCancer.runPolicyIteration (env, irlAlgo, null);
			if (kPrintDebug)
			{
				runPolicyIterationWatch.stop ();
				System.out.println ("time to runPolicyIteration() for table " + table + " in computeLogPosteriorProbForDirichletMixture:" + 
									runPolicyIterationWatch.getTime ());
			}
			policyMatr = P_V_H_Q_Matrices.get ("P");
		}
		else
		{
			policyMatr = tblPolicyVectors.get (key);
		}
		
		Multimap<Integer, double[]> stateActionPairCountsInfoForSubsetOfTrajectories = ArrayListMultimap.create ();

		Map<Integer, double[][]>	subsetOfTrajectories = getSubsetOfTrajectories (tableAssignmentMatrix, (double) table, trajSet);
		

		if (!subsetOfTrajectories.isEmpty ())
		{
			int			numTrajectoriesAssignedToCurrentTable = subsetOfTrajectories.size ();
			StopWatch	computeOccupancyWatch;
			
			if (kPrintDebug)
			{
				computeOccupancyWatch = StopWatch.createStarted ();
				System.out.println ("Going to computeOccupancy() for " + numTrajectoriesAssignedToCurrentTable + 
									" (out of " + numberTrajectories + " total)  trajectories/customers assigned table " + table + 
									" in computeLogPosteriorProbForDirichletMixture ");
			}
			
			// compute occupancy for the subset of trajectories/customers that were assigned to table 'table'
			stateActionPairCountsInfoForSubsetOfTrajectories = computeOccupancy (subsetOfTrajectories, env, 
																				 stateActionPairCountsInfoForSubsetOfTrajectories, 
																				 partOfInitialDPMPosterior);
			
			if (kPrintDebug)
			{
				computeOccupancyWatch.stop ();
				reportTime ("Time to computeOccupancy in computeLogPosteriorProbForDirichletMixture :", computeOccupancyWatch.getTime ());
			}
			
			LikelihoodFunctionCancer llhFunctIRL = irlAlgo.getLikelihood ();
			
//			StopWatch computeLLHGradientBayesianWatch = StopWatch.createStarted ();
			double likelihood_forTablec = llhFunctIRL.computeLogLikelihoodAndGradient_Bayesian (env, irlAlgo, weightMatr, 
																								stateActionPairCountsInfoForSubsetOfTrajectories, 
																								policyMatr, null, null, false).getFirst ();
//			computeLLHGradientBayesianWatch.stop ();
//			reportTableTime ("computeLogLikelihoodAndGradient_Bayesian ", table, computeLLHGradientBayesianWatch.getTime ());
			
			Prior priorIRL = irlAlgo.getPrior ();
//			StopWatch computeLogPriorAndGradientWatch = StopWatch.createStarted ();
			double priorProbability_forTablec = priorIRL.computeLogPriorAndGradient (weightMatr).getFirst ();
//			computeLogPriorAndGradientWatch.stop ();
//			reportTableTime ("computeLogPriorAndGradient in computeLogPosteriorProbForDirichletMixture", table, computeLogPriorAndGradientWatch.getTime ());
			
			logLikelihoods[table] = likelihood_forTablec;
			logPriorProbs[table] = priorProbability_forTablec;
		}
	}
	
	
	protected static final double computeLogPosteriorProbabilityForDirichletProcessMixtureWithDatabase (
			List<double[][]> trajSet, double[][] tableAssignmentMatrix,
			Map<Integer, double[][]> tblWeightVectors,
			Map<Integer, double[][]> tblPolicyVectors, MDPCancer env,
			IRLAlgorithmCancer irlAlgo, boolean partOfInitialDPMPosterior)
	{
		double logPosteriorProb_total = 0.0;
		double logLikelihood_total = 0.0;
		double logPriorProb_total = 0.0;
		double alpha = irlAlgo.getAlpha ();
		
		// Trajectory data
//		int numberTrajectories = trajSet.size (); // should be same as MDP._numberTrajectories
		// vector of state-action counts
		//// List<double[][]> trajectorySetInfo = new
		//// List<double[][]>(numberTrajectories); //intbludes 'count' with each state-action
		//// pair so maximum size is now: nTrajs x nSteps x 3
		
//		StopWatch computeTableAssignmentPriorProbWatch = StopWatch.createStarted ();
		
		////// ComputeTableAssignmentProbability////////Calculate prior of table assignment c,
		////// pr(c|alpha)
		double[][] z = MatrixUtilityJBLAS.tableCounter (tableAssignmentMatrix);
		double[] zRowVector = z[0];
		double[] rIndexVector = VectorUtility.createUnitSpaceVector (zRowVector.length, 1.0, 1.0);
		Double mVal = VectorUtility.multAndSum (zRowVector, rIndexVector);
		Double mValfactorial = CombinatoricsUtils.factorialDouble (mVal.intValue ());
		
		// JK 5.7.2019: Corrected the alpha vector construction
		double[] alphaVector = VectorUtility.range (alpha, alpha + mVal.doubleValue () - 1);
		
		double[] q1 = VectorUtility.pow (rIndexVector, zRowVector);
		double[] q2 = VectorUtility.factorial (VectorUtility.cast (zRowVector));
		double[] qProduct = VectorUtility.mult (q1, q2);
		
		double alphaVecProd = VectorUtility.product (alphaVector);
		double scalarVal1 = mValfactorial.doubleValue () / alphaVecProd;
		
		double scalarVal2 = Math.pow (alpha, VectorUtility.sum (zRowVector))
				/ VectorUtility.product (qProduct);
		double probabilityVal = scalarVal1 * scalarVal2;
		///////////////////////////////////////////////////////////////////////////////////////////////////
//		computeTableAssignmentPriorProbWatch.stop ();
//		System.out.println ("time for computing tableassignment prior prob in computeLogPosteriorProbForDirichletProcessMixture() :" + 
//							computeTableAssignmentPriorProbWatch.getTime ());
		
		double logDirichletProcessPriorProb = Math.log (probabilityVal); // log prior probability of table assignment
		
		Double maxTableIndexInRestaurant = MatrixUtilityJBLAS.matrixMaximum (tableAssignmentMatrix);
		
		for (int t = 1; t < maxTableIndexInRestaurant + 1; t++)
		{
			double[][] weightMatr = tblWeightVectors.get (t); // weightMatrix describing relevance of the |numFeatures| different features for table 't'
			double[][] policyMatr;
			if (tblPolicyVectors.isEmpty ())
			{
				// generate REWARD matrix from weightVector (convertW2R)
				// use weight vector and mdp representation of problem environment to generate the
				// corresponding reward function
				StopWatch generateWeightedRewardFunctionWatch = StopWatch.createStarted ();
				
				RewardFunctionGenerationCancer.generateWeightedRewardFunction (env, weightMatr);
				generateWeightedRewardFunctionWatch.stop ();
				System.out.println ("time to generateWeightedRewardFunction() for table " + t+ " in computeLogPosteriorProbForDirichletMixture: " + 
									generateWeightedRewardFunctionWatch.getTime ());
				
				// generate POLICY and VALUE matrix (policyIteration)
				StopWatch runPolicyIterationWatch = StopWatch.createStarted ();
				Map<String, double[][]> P_V_H_Q_Matrices = PolicySolverCancer.runPolicyIteration (env, irlAlgo, null);
				runPolicyIterationWatch.stop ();
				System.out.println ("time to runPolicyIteration() for table " + t + " in computeLogPosteriorProbForDirichletMixture:" + 
									runPolicyIterationWatch.getTime ());
				policyMatr = P_V_H_Q_Matrices.get ("P");
			}
			else
			{
				policyMatr = tblPolicyVectors.get (t);
			}
			
			Map<Integer, double[][]>	subsetOfTrajectories = getSubsetOfTrajectories (tableAssignmentMatrix, (double) t, trajSet);
			

			if (!subsetOfTrajectories.isEmpty ())
			{
//				int numTrajectoriesAssignedToCurrentTable = subsetOfTrajectories.size ();
//				System.out.println ("Going to computeOccupancy() for "+ numTrajectoriesAssignedToCurrentTable + " (out of " + numberTrajectories + 
//									" total)  trajectories/customers assigned table " + t + " in computeLogPosteriorProbForDirichletMixture ");
				
//				StopWatch computeOccupancyWatch = StopWatch.createStarted ();
				// compute occupancy for the subset of trajectories/customers that were assigned to table 't'
				computeOccupancyFromDatabase (subsetOfTrajectories, env, partOfInitialDPMPosterior);
				
//				computeOccupancyWatch.stop ();
//				reportTime ("Time to computeOccupancy in computeLogPosteriorProbForDirichletMixture :", computeOccupancyWatch.getTime ());
				
				LikelihoodFunctionCancer llhFunctIRL = irlAlgo.getLikelihood ();
				
//				StopWatch computeLLHGradientBayesianWatch = StopWatch.createStarted ();
				double likelihood_forTablec = 
						llhFunctIRL.computeLogLikelihoodAndGradient_BayesianWithDatabase (env, irlAlgo, weightMatr, policyMatr, null, null, false)
									.getFirst ();
//				computeLLHGradientBayesianWatch.stop ();
//				reportTableTime ("computeLogLikelihoodAndGradient_Bayesian ", t, computeLLHGradientBayesianWatch.getTime ());
				
				Prior priorIRL = irlAlgo.getPrior ();
//				StopWatch computeLogPriorAndGradientWatch = StopWatch.createStarted ();
				double priorProbability_forTablec = priorIRL.computeLogPriorAndGradient (weightMatr).getFirst ();
//				computeLogPriorAndGradientWatch.stop ();
//				reportTableTime ("computeLogPriorAndGradient in computeLogPosteriorProbForDirichletMixture", t, computeLogPriorAndGradientWatch.getTime ());
				
				logLikelihood_total = logLikelihood_total + likelihood_forTablec;
				logPriorProb_total = logPriorProb_total + priorProbability_forTablec;
			}
		}
		// ...formulate the IRL problem into posterior optimization problem, finding reward function
		// R that maximizes the log unnormalized [posterior] = finding the reward function R that
		// maximizes [logLikelihood + logPriorProbability]
		logPosteriorProb_total = logDirichletProcessPriorProb + logLikelihood_total + logPriorProb_total;
		
		return logPosteriorProb_total;
	}
	
	
	/**
	 * MH update for table assignment of i-th customer/trajectory | STEP 1 of Inference Algorithm
	 * 
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param env
	 * @param customer_i
	 * @param irlalgo
	 * @param tableAssignmentMatrix
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 * @param restLikeLihoods
	 * @param restPriors
	 * @param restGradientsFromLLH
	 * @param restGradientsFromPrior
	 * @param saPairCountsInfoForAllTrajectories
	 * @return	Updated {@link RestaurantMap}
	 */
	public static final RestaurantMap updateTableAssignment (int numThreads, int numGPUs, MDPCancer env, int customer_i, int maxTableIndex, 
															 IRLAlgorithmCancer irlalgo, double[][] tableAssignmentMatrix, 
															 Map<Integer, double[][]> tableWeightVectors, Map<Integer, double[][]> tablePolicyVectors, 
															 Map<Integer, double[][]> tableValueVectors, Map<Integer, double[][]> tableQVectors, 
															 Map<Integer, Double> restLikeLihoods, Map<Integer, Double> restPriors, Map<Integer, 
															 DoubleMatrix> restGradientsFromLLH, Map<Integer, DoubleMatrix> restGradientsFromPrior, 
															 Multimap<Integer, double[]> saPairCountsInfoForAllTrajectories)
	{
		RestaurantMap			rmap4;
		int						numTableAssignmentIterations = irlalgo.getTableAssignmentUpdateIterations ();
		Map<String, double[][]>	weight_policy_value_q_vectorHashMapForCustomer_iTable = null;
		
//		reportValue ("updateTableAssignment: numTableAssignmentIterations", numTableAssignmentIterations);
		for (int iter = 0; iter < numTableAssignmentIterations; ++iter)
		{ // # of iterations for MH update for the cluster assignment of a given trajectory
			// System.out.println("TableAssignmentUpdate iteration #"+iter);
			int		tblIndexI = (int) tableAssignmentMatrix[customer_i][0]; // obtain table assignment index/label for m-th trajectory according to the
																			// current tableAssignmentMatrix tblAssignMaterix
			Integer	intTableIndex = Integer.valueOf (tblIndexI);
			// obtain the weight column-matrix associated with this table index/label
			double[][] tblWeightMatrixforCustomeri_currentAssignment = tableWeightVectors.get (intTableIndex);
			// obtain the value column-matrix associated with this cluster index/label
			double[][] tblValueMatrixforCustomeri_currentAssignment = tableValueVectors.get (intTableIndex);
			
//			if (!restLikeLihoods.containsKey (intTableIndex))
//			{ /// this if condition was only added because of our original use of the RealMatrix for the gradientsLLH data structure
//				System.out.println ("******This new table index does NOT yet exist in restLikelihoods!");
//			}
			// reset likelihood of this table to NaN
			restLikeLihoods.put (intTableIndex, Double.NaN);
			
			// reset prior of this table to NaN
			restPriors.put (intTableIndex, Double.NaN);
			
//			if (!restGradientsFromLLH.containsKey (intTableIndex))
//			{ /// this if condition was only added because of our original use of the RealMatrix for the gradientsLLH data structure
//				System.out.println ("******This new table index does NOT yet exist in restGradientsFromLLLH!");
//			}
			
			int	rows = tblWeightMatrixforCustomeri_currentAssignment.length;
			// reset gradientFromLLHcomputation for this table to NaN
			restGradientsFromLLH.put (intTableIndex, MatrixUtilityJBLAS.createRealMatrixWithNANS (rows, 1));
			// reset gradientFromLLHcomputation for this table to NaN
			restGradientsFromPrior.put (intTableIndex, MatrixUtilityJBLAS.createRealMatrixWithNANS (rows, 1));
			
			double[] priorProbDistributionForCurrentSeatingArrangement = new double[maxTableIndex + 1];
			
			 // prior probability of table index/label 'table_i' = # of trajectories that have been assigned label/index 'table_i'
			computePriorProbDistributions (numThreads, numGPUs, maxTableIndex, tblIndexI, tableAssignmentMatrix, irlalgo, 
											priorProbDistributionForCurrentSeatingArrangement);
			// remove the contribution of of trajectory 'customer_i' to the prior probability of
			// table index/label being 'tblIndex'
			
			priorProbDistributionForCurrentSeatingArrangement[priorProbDistributionForCurrentSeatingArrangement.length - 1] = 
					irlalgo.getAlpha () + (maxTableIndex * irlalgo.getDiscountHyperparameter ());
			
			// Sample a NEW/ALTERNATIVE table index/value from the prior probability distribution
			// (NOTE: if this drawn table index corresponds to the index of existing table in the
			// restaurant, it will not necessarily be the same table/index-value already assigned
			// in the tableAssignmentMatrix (seatingArrangement)
			int		tblIndex2 = SampleMultinomialIRL.sampleSingleTableFromMultinomial (1000, priorProbDistributionForCurrentSeatingArrangement, RNG);
			Integer	tblIndex2V = Integer.valueOf (tblIndex2);
			if (tblIndex2 > maxTableIndex)
			{ // if the new table index/value is higher than the current (existing) largest table
				// index/value; then we need to generate NEW (weight,policy, and value) vectors
				// specifically for this table
				weight_policy_value_q_vectorHashMapForCustomer_iTable = 
						generateNewWeights (tblIndex2, env, irlalgo, tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, false);
			}
			else
			{ // OTHERWISE if the table index/value drawn for customer_i already exists in the
				// restaurant, set the weight, policy, value vectors of this table to the vectors
				// already associated with the existing table with the same index/value in the
				// restaurant.
				weight_policy_value_q_vectorHashMapForCustomer_iTable = new HashMap<String, double[][]> ();
				
				// ...you can't use .set() on an empty index, must use .add()
				// obtain the existing weight vector associated with table index-value 'tblIndex2'
				weight_policy_value_q_vectorHashMapForCustomer_iTable.put ("W", tableWeightVectors.get (tblIndex2V));
				// obtain the existing policy vector associated with table index-value 'tblIndex2'
				weight_policy_value_q_vectorHashMapForCustomer_iTable.put ("P", tablePolicyVectors.get (tblIndex2V));
				// obtain the existing value vector associated with table index-value 'tblIndex2'
				weight_policy_value_q_vectorHashMapForCustomer_iTable.put ("V", tableValueVectors.get (tblIndex2V));
				// obtain the existing q vector associated with table index-value 'tblIndex2'
				weight_policy_value_q_vectorHashMapForCustomer_iTable.put ("Q", tableQVectors.get (tblIndex2V));
			}
			
			// If tblIndex2 > tableIndex_i for all i, we need to draw a new reward function
			// r_{tblIndex2} from the reward prior P(r|?, ?). We then set tblIndex = tbLindex2
			// (table index/value of customer_i trajectory) with the acceptance probability
			double probQuotient = computeMinProbabilityQuotient (tblWeightMatrixforCustomeri_currentAssignment, 
																 tblValueMatrixforCustomeri_currentAssignment, 
																 weight_policy_value_q_vectorHashMapForCustomer_iTable.get ("W"), 
																 weight_policy_value_q_vectorHashMapForCustomer_iTable.get ("V"), env, irlalgo, 
																 saPairCountsInfoForAllTrajectories, customer_i, numThreads, numGPUs);

			
			if (!Double.isNaN (probQuotient)) //JK 5.29.2019 added if condition to check if probQuotient is NaN
			{
				double rand = RNG.nextDouble ();
				if (Double.compare (probQuotient, rand) > 0)
				{
					if (tblIndex2 > maxTableIndex)
					{
						synchronized (tableWeightVectors)
						{
							// Make sure we're adding a new value
							Integer	maxKey = getMaxKey (tableWeightVectors);
							int		maxValue;
							
							if ((maxKey != null) && ((maxValue = maxKey.intValue ()) >= tblIndex2))
								tblIndex2V = Integer.valueOf (tblIndex2 = maxValue + 1);
							
							tableWeightVectors.put (tblIndex2V, weight_policy_value_q_vectorHashMapForCustomer_iTable.get ("W"));
							tablePolicyVectors.put (tblIndex2V, weight_policy_value_q_vectorHashMapForCustomer_iTable.get ("P"));
							tableValueVectors.put (tblIndex2V, weight_policy_value_q_vectorHashMapForCustomer_iTable.get ("V"));
							tableQVectors.put (tblIndex2, weight_policy_value_q_vectorHashMapForCustomer_iTable.get ("Q"));
						}
					}
					
					tableAssignmentMatrix[customer_i][0] = tblIndex2;
				}
			}
		}
		
		rmap4 = new RestaurantMap (tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, tableAssignmentMatrix);
		rmap4._restGradientsFromLLH = restGradientsFromLLH;
		rmap4._restGradientsFromPrior = restGradientsFromPrior;
		rmap4._restLikeLihoods = restLikeLihoods;
		rmap4._restPriors = restPriors;
		return rmap4;
	}
	
	
	/**
	 * MH update for table assignment of i-th customer/trajectory | STEP 1 of Inference Algorithm
	 * 
	 * @param env
	 * @param customer_i
	 * @param irlalgo
	 * @param tableAssignmentMatrix
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 * @param restLikeLihoods
	 * @param restPriors
	 * @param restGradientsFromLLH
	 * @param restGradientsFromPrior
	 * @param numThreads				Number of threads it can use
	 * @param numGPUs					Number of GPUs it can use
	 * @return	Update {@link RestaurantMap}
	 */
	public static final RestaurantMap updateTableAssignmentWithDatabase (MDPCancer env, int customer_i, int maxTableIndex, 
			IRLAlgorithmCancer irlalgo, double[][] tableAssignmentMatrix, 
			Map<Integer, double[][]> tableWeightVectors, Map<Integer, double[][]> tablePolicyVectors,
			Map<Integer, double[][]> tableValueVectors, Map<Integer, double[][]> tableQVectors,
			Map<Integer, Double> restLikeLihoods, Map<Integer, Double> restPriors,
			Map<Integer, DoubleMatrix> restGradientsFromLLH, Map<Integer, DoubleMatrix> restGradientsFromPrior, int numThreads, int numGPUs)
	{
		
		RestaurantMap			rmap4;
		int						numTableAssignmentIterations = irlalgo.getTableAssignmentUpdateIterations ();
		Map<String, double[][]>	weight_policy_value_q_vectorHashMapForCustomer_iTable = null;
		
		for (int iter = 0; iter < numTableAssignmentIterations; ++iter)
		{ // # of iterations for MH update for the cluster assignment of a given trajectory
			// System.out.println("TableAssignmentUpdate iteration #"+iter);
			int		tblIndexI = (int) tableAssignmentMatrix[customer_i][0]; // obtain table assignment index/label for m-th trajectory according to the
																			// current tableAssignmentMatrix tblAssignMaterix
			Integer		intTableIndex = Integer.valueOf (tblIndexI);
			// obtain the weight column-matrix associated with this table index/label
			double[][]	tblWeightMatrixforCustomeri_currentAssignment = tableWeightVectors.get (intTableIndex);
			// obtain the value column-matrix associated with this cluster index/label
			double[][]	tblValueMatrixforCustomeri_currentAssignment = tableValueVectors.get (intTableIndex);
			
			/// this if condition was only added because of our original use of the RealMatrix for the gradientsLLH data structure
			if (!restLikeLihoods.containsKey (intTableIndex))
			{
				System.out.println ("******This new table index does NOT yet exist in restLikelihoods!");
			}
			// reset likelihood of this table to NaN
			restLikeLihoods.put (intTableIndex, Double.NaN);
			
			// reset prior of this table to NaN
			restPriors.put (intTableIndex, Double.NaN);
			
			/// this if condition was only added because of our original use of the RealMatrix for the gradientsLLH data structure
			if (!restGradientsFromLLH.containsKey (intTableIndex))
			{
				System.out.println ("******This new table index does NOT yet exist in restGradientsFromLLLH!");
			}
			
			int	rows = tblWeightMatrixforCustomeri_currentAssignment.length;
			restGradientsFromLLH.put (intTableIndex, MatrixUtilityJBLAS.createRealMatrixWithNANS (rows, 1));
			// reset gradientFromLLHcomputation for this table to NaN
			restGradientsFromPrior.put (intTableIndex, MatrixUtilityJBLAS.createRealMatrixWithNANS (rows, 1));
			// reset gradientFromLLHcomputation for this table to NaN
			
			double[] priorProbDistributionForCurrentSeatingArrangement = new double[maxTableIndex + 1];
			
			// prior probability of table index/label 'table_i' = # of trajectories that have been assigned label/index 'table_i'
			computePriorProbDistributions (numThreads, numGPUs, maxTableIndex, tblIndexI, tableAssignmentMatrix, irlalgo, 
											priorProbDistributionForCurrentSeatingArrangement);
			// remove the contribution of of trajectory 'customer_i' to the prior probability of
			// table index/label being 'tblIndex'
			
			priorProbDistributionForCurrentSeatingArrangement[priorProbDistributionForCurrentSeatingArrangement.length - 1] = 
					irlalgo.getAlpha () + (maxTableIndex * irlalgo.getDiscountHyperparameter ());
			
			// Sample a NEW/ALTERNATIVE table index/value from the prior probability distribution
			// (NOTE: if this drawn table index corresponds to the index of existing table in the
			// restaurant, it will not necessarily be the same table/index-value already assigned
			// in the tableAssignmentMatrix (seatingArrangement)
			int		tblIndex2 = SampleMultinomialIRL.sampleSingleTableFromMultinomial (1000, priorProbDistributionForCurrentSeatingArrangement, RNG);
			Integer	tblIndex2V = Integer.valueOf (tblIndex2);
			// JK 3.15.2019: changed variable 'n' from 10 to 100;hopefully this improves randomness of table indices that are sampled.
			if (tblIndex2 > maxTableIndex)
			{ // if the new table index/value is higher than the current (existing) largest table
				// index/value; then we need to generate NEW (weight,policy, and value) vectors
				// specifically for this table
				weight_policy_value_q_vectorHashMapForCustomer_iTable = generateNewWeights (tblIndex2, env, irlalgo, tableWeightVectors, 
																							tablePolicyVectors, tableValueVectors, tableQVectors, false);
			}
			else
			{ // OTHERWISE if the table index/value drawn for customer_i already exists in the
				// restaurant, set the weight, policy, value vectors of this table to the vectors
				// already associated with the existing table with the same index/value in the
				// restaurant.
				weight_policy_value_q_vectorHashMapForCustomer_iTable = new HashMap<String, double[][]> ();
				
				// ...you can't use .set() on an empty index, must use .add()
				// obtain the existing weight vector associated with table index-value 'tblIndex2'
				weight_policy_value_q_vectorHashMapForCustomer_iTable.put ("W", tableWeightVectors.get (tblIndex2V));
				// obtain the existing policy vector associated with table index-value 'tblIndex2'
				weight_policy_value_q_vectorHashMapForCustomer_iTable.put ("P", tablePolicyVectors.get (tblIndex2V));
				// obtain the existing value vector associated with table index-value 'tblIndex2'
				weight_policy_value_q_vectorHashMapForCustomer_iTable.put ("V", tableValueVectors.get (tblIndex2V));
				// obtain the existing q vector associated with table index-value 'tblIndex2'
				weight_policy_value_q_vectorHashMapForCustomer_iTable.put ("Q", tableQVectors.get (tblIndex2V));
			}
			
			// If tblIndex2 > tableIndex_i for all i, we need to draw a new reward function
			// r_{tblIndex2} from the reward prior P(r|?, ?). We then set tblIndex = tbLindex2
			// (table index/value of customer_i trajectory) with the acceptance probability
//			StopWatch	minProbabilityQuotientWatch = StopWatch.createStarted ();
			double probQuotient = computeMinProbabilityQuotientWithDatabase (tblWeightMatrixforCustomeri_currentAssignment, 
																			 tblValueMatrixforCustomeri_currentAssignment, 
																			 weight_policy_value_q_vectorHashMapForCustomer_iTable.get ("W"), 
																			 weight_policy_value_q_vectorHashMapForCustomer_iTable.get ("V"), 
																			 env, irlalgo, customer_i, numThreads, numGPUs);
			
//			minProbabilityQuotientWatch.stop ();
//			reportTime ("Time for computeMinProbabilityQuotientWithDatabase: ", minProbabilityQuotientWatch.getTime ());
			if (!Double.isNaN (probQuotient)) //JK 5.29.2019 added if condition to check if probQuotient is NaN
			{
				double rand = RNG.nextDouble ();
				if (Double.compare (probQuotient, rand) > 0)
				{
					if (tblIndex2 > maxTableIndex)
					{
						synchronized (tableWeightVectors)
						{
							// Make sure we're adding a new value
							Integer	maxKey = getMaxKey (tableWeightVectors);
							int		maxValue;
							
							if ((maxKey != null) && ((maxValue = maxKey.intValue ()) >= tblIndex2))
								tblIndex2V = Integer.valueOf (tblIndex2 = maxValue + 1);
							
							tableWeightVectors.put (tblIndex2V, weight_policy_value_q_vectorHashMapForCustomer_iTable.get ("W"));
							tablePolicyVectors.put (tblIndex2V, weight_policy_value_q_vectorHashMapForCustomer_iTable.get ("P"));
							tableValueVectors.put (tblIndex2V, weight_policy_value_q_vectorHashMapForCustomer_iTable.get ("V"));
							tableQVectors.put (tblIndex2, weight_policy_value_q_vectorHashMapForCustomer_iTable.get ("Q"));
						}
					}
					
					tableAssignmentMatrix[customer_i][0] = tblIndex2;
				}
			}
		}
		
		rmap4 = new RestaurantMap (tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, tableAssignmentMatrix);
		rmap4._restGradientsFromLLH = restGradientsFromLLH;
		rmap4._restGradientsFromPrior = restGradientsFromPrior;
		rmap4._restLikeLihoods = restLikeLihoods;
		rmap4._restPriors = restPriors;
		return rmap4;
	}
	
	
	/**
	 * 
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param maxIndex
	 * @param tblIndex
	 * @param tableAssignmentMatrix
	 * @param irlalgo
	 * @param priorProbDistribution
	 */
	private static final void computePriorProbDistributions (int numThreads, int numGPUs, int maxIndex, int tblIndex, double[][] tableAssignmentMatrix, 
															 IRLAlgorithmCancer irlalgo, double[] priorProbDistribution)
	{
		if (numThreads > 1)
		{
			computePriorProbDistributionsThreaded (numThreads, numGPUs, maxIndex, tblIndex, tableAssignmentMatrix, irlalgo, priorProbDistribution);
		}
		else
		{
			 // prior probability of table index/label 'table_i' = # of trajectories that have been assigned label/index 'table_i'
			for (int tableIndex_i = 1; tableIndex_i <= maxIndex; ++tableIndex_i)
			{
	   		// should return a numTraj x 1 column matrix with 1's at entries corresponding to customer assigned to table 'table_i'
				
				int	logMatrSumVal = countMatches (tableAssignmentMatrix, (double) tableIndex_i);
				if (tableIndex_i == tblIndex)
				{
					// remove the contribution of trajectory 'customer_i' to the prior
					// probability of table index/label being 'tblIndex'
					logMatrSumVal -= 1;
				}
				if (logMatrSumVal > 0)
				{
					priorProbDistribution[tableIndex_i - 1] = logMatrSumVal - irlalgo.getDiscountHyperparameter ();
					// element 0 of this double[] corresponds to table 1!!!   be careful
				}
				else
				{// need this else condition if the count for a tableIndex =0; in which case the
					// priorProb for table would become negative because we are subtracting the val
					// of discount
					priorProbDistribution[tableIndex_i - 1] = logMatrSumVal; 
					// element 0 of this double[] corresponds to table 1!!!   be careful
				}
				
			}
		}
	}
	
	
	/**
	 * 
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param maxIndex
	 * @param tblIndex
	 * @param tableAssignmentMatrix
	 * @param irlalgo
	 * @param priorProbDistribution
	 */
	private static final void computePriorProbDistributionsThreaded (int numThreads, int numGPUs, int maxIndex, int tblIndex, 
																	 double[][] tableAssignmentMatrix, IRLAlgorithmCancer irlalgo, 
																	 double[] priorProbDistribution)
	{
		List<Thread>			threads = new ArrayList<> (numThreads);
		BlockingQueue<Integer>	tableIndexes = makeQueue (maxIndex);
		
		reportThreading ("computePriorProbDistributionsThreaded", numThreads, numGPUs, maxIndex);
		for (int i = 1; (i <= numThreads) && (i <= maxIndex); ++i)
		{
			ComputePriorProb	computer = new ComputePriorProb (tableIndexes, tblIndex, tableAssignmentMatrix, irlalgo, priorProbDistribution);
			Thread	theThread = makeThread (computer, "computePriorProbDistributions ", i, numGPUs);
			
			theThread.start ();
			threads.add (theThread);
		}
		
		// Wait for the threads to finish
		waitForThreads (threads);
//		reportEvent ("computePriorProbDistributionsThreaded complete");
	}
	
	
	/**
	 * 
	 * @param numThreads
	 * @param numGPUs
	 * @param environment
	 * @param customersPermutation
	 * @param irlAlgo
	 * @param tableAssignmentMatrix
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 * @param restaurantLikelihoods
	 * @param restaurantPriors
	 * @param restGradientsLLH
	 * @param restGradientsPrior
	 * @return
	 */
	private static RestaurantMap updateTableAssignments (int numThreads, int numGPUs,
			MDPCancer environment, int[] customersPermutation, IRLAlgorithmCancer irlAlgo,
			double[][] tableAssignmentMatrix, Map<Integer, double[][]> tableWeightVectors,
			Map<Integer, double[][]> tablePolicyVectors, Map<Integer, double[][]> tableValueVectors,
			Map<Integer, double[][]> tableQVectors, Map<Integer, Double> restaurantLikelihoods,
			Map<Integer, Double> restaurantPriors, Map<Integer, DoubleMatrix> restGradientsLLH,
			Map<Integer, DoubleMatrix> restGradientsPrior, Multimap<Integer, double[]> stateActionPairCountsInfoForAllTrajectories)
	{
		RestaurantMap	rmap2 = null;
		
		if (numThreads <= 1)
		{
			int	maxTableIndex = (int) MatrixUtilityJBLAS.matrixMaximum (tableAssignmentMatrix);
			
			for (int customer_i : customersPermutation)
			{ // iterate through all customers in random order and determine what table they should
				// be assigned. They be assigned to a NEW TABLE with certain probability.
//				System.out.println ("Customer" + customer_i + " is entering the restaurant...");
//				StopWatch updateTableAssignmentWatch = StopWatch.createStarted();
				rmap2 = updateTableAssignment (numThreads, numGPUs, environment, customer_i, maxTableIndex, irlAlgo, tableAssignmentMatrix, 
												tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, restaurantLikelihoods, 
												restaurantPriors, restGradientsLLH, restGradientsPrior, stateActionPairCountsInfoForAllTrajectories);
//				updateTableAssignmentWatch.stop();
//				System.out.println ("Time for running updateTableAssignment for customer" + customer_i + ": " + updateTableAssignmentWatch.getTime ());
				tableAssignmentMatrix = rmap2._restaurantAssignmentMatrix;
				tableWeightVectors = rmap2._restaurantTableWeightMatrices;
				tablePolicyVectors = rmap2._restaurantTablePolicyMatrices;
				tableValueVectors = rmap2._restaurantTableValueMatrices;
				tableQVectors = rmap2._restaurantTableQMatrices;
				
				restaurantLikelihoods = rmap2._restLikeLihoods;
				restaurantPriors = rmap2._restPriors;
				restGradientsLLH = rmap2._restGradientsFromLLH;
				restGradientsPrior = rmap2._restGradientsFromPrior;
//				stateActionPairCountsInfoForAllTrajectories = rmap2._saPairCountsInfoForSubsetOfTrajectories;
			}
		}
		else
		{
			rmap2 = updateTableAssignmentsThreaded (numThreads, numGPUs, environment, customersPermutation, irlAlgo, tableAssignmentMatrix, 
													tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, restaurantLikelihoods, 
													restaurantPriors, restGradientsLLH, restGradientsPrior, stateActionPairCountsInfoForAllTrajectories);
		}
		
		return rmap2;
	}
	

	/**
	 * 
	 * @param numThreads
	 * @param numGPUs
	 * @param environment
	 * @param customersPermutation
	 * @param irlAlgo
	 * @param tableAssignmentMatrix
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 * @param restaurantLikelihoods
	 * @param restaurantPriors
	 * @param restGradientsLLH
	 * @param restGradientsPrior
	 * @return
	 */
	private static RestaurantMap updateTableAssignmentsWithDatabase (int numThreads, int numGPUs,
			MDPCancer environment, int[] customersPermutation, IRLAlgorithmCancer irlAlgo,
			double[][] tableAssignmentMatrix, Map<Integer, double[][]> tableWeightVectors,
			Map<Integer, double[][]> tablePolicyVectors, Map<Integer, double[][]> tableValueVectors,
			Map<Integer, double[][]> tableQVectors, Map<Integer, Double> restaurantLikelihoods,
			Map<Integer, Double> restaurantPriors, Map<Integer, DoubleMatrix> restGradientsLLH,
			Map<Integer, DoubleMatrix> restGradientsPrior)
	{
		RestaurantMap	rmap2 = null;
		
		if (numThreads <= 1)
		{
			for (int customer_i : customersPermutation)
			{ // iterate through all customers in random order and determine what table they should
				// be assigned. They be assigned to a NEW TABLE with certain probability.
//					System.out.println ("Customer" + customer_i + " is entering the restaurant...");
				// StopWatch updateTableAssignmentWatch = StopWatch.createStarted();
				int	maxTableIndex = (int) MatrixUtilityJBLAS.matrixMaximum (tableAssignmentMatrix);
				
				rmap2 = updateTableAssignmentWithDatabase (environment, customer_i, maxTableIndex, irlAlgo, tableAssignmentMatrix, tableWeightVectors, 
															tablePolicyVectors, tableValueVectors, tableQVectors, restaurantLikelihoods, 
															restaurantPriors, restGradientsLLH, restGradientsPrior, numThreads, numGPUs);
				// updateTableAssignmentWatch.stop();
//				System.out.println ("Time for running updateTableAssignment for customer" + customer_i + ": " + updateTableAssignmentWatch.getTime ());
				tableAssignmentMatrix = rmap2._restaurantAssignmentMatrix;
				tableWeightVectors = rmap2._restaurantTableWeightMatrices;
				tablePolicyVectors = rmap2._restaurantTablePolicyMatrices;
				tableValueVectors = rmap2._restaurantTableValueMatrices;
				tableQVectors = rmap2._restaurantTableQMatrices;
				
				restaurantLikelihoods = rmap2._restLikeLihoods;
				restaurantPriors = rmap2._restPriors;
				restGradientsLLH = rmap2._restGradientsFromLLH;
				restGradientsPrior = rmap2._restGradientsFromPrior;
			}
		}
		else
		{
			rmap2 = updateTableAssignmentsWithDatabaseThreaded (numThreads, numGPUs, environment, customersPermutation, irlAlgo, tableAssignmentMatrix, 
																tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, 
																restaurantLikelihoods, restaurantPriors, restGradientsLLH, restGradientsPrior);
		}
		
		return rmap2;
	}
	
	
	/**
	 * 
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param environment
	 * @param customersPermutation
	 * @param irlAlgo
	 * @param tableAssignmentMatrix
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 * @param restaurantLikelihoods
	 * @param restaurantPriors
	 * @param restGradientsLLH
	 * @param restGradientsPrior
	 * @param stateActionPairCountsInfoForAllTrajectories
	 * @return
	 */
	private static RestaurantMap updateTableAssignmentsThreaded (int numThreads, int numGPUs,
			MDPCancer environment, int[] customersPermutation, IRLAlgorithmCancer irlAlgo,
			double[][] tableAssignmentMatrix, Map<Integer, double[][]> tableWeightVectors,
			Map<Integer, double[][]> tablePolicyVectors, Map<Integer, double[][]> tableValueVectors,
			Map<Integer, double[][]> tableQVectors, Map<Integer, Double> restaurantLikelihoods,
			Map<Integer, Double> restaurantPriors, Map<Integer, DoubleMatrix> restGradientsLLH,
			Map<Integer, DoubleMatrix> restGradientsPrior,
			Multimap<Integer, double[]> stateActionPairCountsInfoForAllTrajectories)
	{
		List<Thread>			threads = new ArrayList<> (numThreads);
		int						numCustomers = customersPermutation.length;
		BlockingQueue<Integer>	customerIndexes = makeQueue (customersPermutation);
		RestaurantMap			rmap4;
		
		// Save off maps, so don't get map of map of ....
		rmap4 = new RestaurantMap (tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, tableAssignmentMatrix);
		rmap4._restGradientsFromLLH = restGradientsLLH;
		rmap4._restGradientsFromPrior = restGradientsPrior;
		rmap4._restLikeLihoods = restaurantLikelihoods;
		rmap4._restPriors = restaurantPriors;

		tableWeightVectors = Collections.synchronizedMap (tableWeightVectors);
		tablePolicyVectors = Collections.synchronizedMap (tablePolicyVectors);
		tableValueVectors = Collections.synchronizedMap (tableValueVectors);
		tableQVectors = Collections.synchronizedMap (tableQVectors);
		restaurantLikelihoods = Collections.synchronizedMap (restaurantLikelihoods);
		restaurantPriors = Collections.synchronizedMap (restaurantPriors);
		restGradientsLLH = Collections.synchronizedMap (restGradientsLLH);
		restGradientsPrior = Collections.synchronizedMap (restGradientsPrior);
		reportThreading ("updateTableAssignmentsThreaded", numThreads, numGPUs, numCustomers);
		for (int i = 1; (i <= numThreads) && (i <= numCustomers); ++i)
		{
			MDPCancer				env = new MDPCancer (environment);
			TrajectoryNoDBUpdater	updater = new TrajectoryNoDBUpdater (customerIndexes, env, irlAlgo, tableAssignmentMatrix, tableWeightVectors, 
																		 tablePolicyVectors, tableValueVectors, tableQVectors, restaurantLikelihoods, 
																		 restaurantPriors, restGradientsLLH, restGradientsPrior, 
																		 stateActionPairCountsInfoForAllTrajectories);
			Thread	theThread = makeThread (updater, "updateTableAssignments ", i, numGPUs);
			
			theThread.start ();
			threads.add (theThread);
		}
		
		// Wait for the threads to finish
		waitForThreads (threads);
		
		return rmap4;
	}

	
	/**
	 * 
	 * @param numThreads	Number of threads it can use
	 * @param numGPUs		Number of GPUs it can use
	 * @param environment
	 * @param customersPermutation
	 * @param irlAlgo
	 * @param tableAssignmentMatrix
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 * @param restaurantLikelihoods
	 * @param restaurantPriors
	 * @param restGradientsLLH
	 * @param restGradientsPrior
	 */
	private static final RestaurantMap updateTableAssignmentsWithDatabaseThreaded (int numThreads, int numGPUs,
			MDPCancer environment, int[] customersPermutation, IRLAlgorithmCancer irlAlgo,
			double[][] tableAssignmentMatrix, Map<Integer, double[][]> tableWeightVectors,
			Map<Integer, double[][]> tablePolicyVectors, Map<Integer, double[][]> tableValueVectors,
			Map<Integer, double[][]> tableQVectors, Map<Integer, Double> restaurantLikelihoods,
			Map<Integer, Double> restaurantPriors, Map<Integer, DoubleMatrix> restGradientsLLH,
			Map<Integer, DoubleMatrix> restGradientsPrior)
	{
		List<Thread>			threads = new ArrayList<> (numThreads);
		int						numCustomers = customersPermutation.length;
		BlockingQueue<Integer>	customerIndexes = makeQueue (customersPermutation);
		RestaurantMap			rmap4;
		
		// Save off maps, so don't get map of map of ....
		rmap4 = new RestaurantMap (tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, tableAssignmentMatrix);
		rmap4._restGradientsFromLLH = restGradientsLLH;
		rmap4._restGradientsFromPrior = restGradientsPrior;
		rmap4._restLikeLihoods = restaurantLikelihoods;
		rmap4._restPriors = restaurantPriors;
		
		tableWeightVectors = Collections.synchronizedMap (tableWeightVectors);
		tablePolicyVectors = Collections.synchronizedMap (tablePolicyVectors);
		tableValueVectors = Collections.synchronizedMap (tableValueVectors);
		tableQVectors = Collections.synchronizedMap (tableQVectors);
		restaurantLikelihoods = Collections.synchronizedMap (restaurantLikelihoods);
		restaurantPriors = Collections.synchronizedMap (restaurantPriors);
		restGradientsLLH = Collections.synchronizedMap (restGradientsLLH);
		restGradientsPrior = Collections.synchronizedMap (restGradientsPrior);
		reportThreading ("updateTableAssignmentsWithDatabaseThreaded", numThreads, numGPUs, numCustomers);
		for (int i = 1; (i <= numThreads) && (i <= numCustomers); ++i)
		{
			MDPCancer			env = new MDPCancer (environment);
			TrajectoryUpdater	updater = new TrajectoryUpdater (customerIndexes, env, irlAlgo, tableAssignmentMatrix, tableWeightVectors, 
																 tablePolicyVectors, tableValueVectors, tableQVectors, restaurantLikelihoods, 
																 restaurantPriors, restGradientsLLH, restGradientsPrior);
			Thread	theThread = makeThread (updater, "updateTableAssignments ", i, numGPUs);
			
			theThread.start ();
			threads.add (theThread);
		}
		
		// Wait for the threads to finish
		waitForThreads (threads);
		
		return rmap4;
	}
	
	
	/**
	 * Update the weight, policy, value of the multiple reward functions (tables) in the restaurant
	 * @param trajSet 
	 * JK 7.25.2019 data validated
	 * @param environment
	 * @param table_i
	 * @param irlAlgo
	 * @param tableAssignmentMatrix
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 * @param restaurantLikelihoods
	 * @param restaurantPriors
	 * @param restGradientsLLH
	 * @param restGradientsPrior
	 * @return	Updated {@link RestaurantTable}
	 */
	@SuppressWarnings ("null")
	public static final RestaurantTable updateRewardFunctions (List<double[][]> trajSet, MDPCancer environment, int table_i, IRLAlgorithmCancer irlAlgo, 
																double[][] tableAssignmentMatrix, double[][] tableWeightVectors, 
																double[][] tablePolicyVectors, double[][] tableValueVectors, double[][] tableQVectors, 
																Double restaurantLikelihoods, Double restaurantPriors, 
																DoubleMatrix restGradientsLLH, DoubleMatrix restGradientsPrior)
	{
		RestaurantTable rmap5;
		LikelihoodFunctionCancer llhFunctIRL = irlAlgo.getLikelihood ();
		Prior priorIRL = irlAlgo.getPrior ();
		double scalingParameter = .01; //JK 7.22.2019 this has to match the same value as specified in PRIOR!!!!
		//XXX: double scalingParameter = priorIRL.get_sigma();
		//JK 7.25.2019 NOTE: only during updateRewardFunctions() do we NOT use the sigma value preset in the Prior class variable

		Multimap<Integer, double[]> updatedSAPairCountsForSubsetOfTrajectories = ArrayListMultimap.create ();
		Map<Integer, double[][]>	subsetOfTrajectories = getSubsetOfTrajectories (tableAssignmentMatrix, (double) table_i, trajSet);
		Map<String, double[][]>		weight_policy_q_vectorMapForTable_i = new HashMap<String, double[][]> ();
		
		double			logPosteriorProbability = 0.0;
		double			logPosteriorProbability_updated = 0.0;
		double[]		gradient_updated_forTablei = null;
		int				numRows = environment.getNumRewardFeatures ();
		DoubleMatrix	gradient_forTablei = new DoubleMatrix (numRows, 1);
		double[]		scratchSpace = new double[numRows];
	
		// should not run if none of the trajectories possess a given table index-value

		if (subsetOfTrajectories.isEmpty ())
		{
			System.err.println ("None of the trajectories were assigned to table " + table_i);
		}
		else
		{
			// JK 6.23.2019 moved computeOccopuancy() outside of rewardupdateIteration loop. The count map is entirely based on the 
			// subset of trajectories currently being analyzed.  Nothing to do with the actual reward function (weights, llh, etc...)
			updatedSAPairCountsForSubsetOfTrajectories = 
					computeOccupancy (subsetOfTrajectories, environment, updatedSAPairCountsForSubsetOfTrajectories, false);
			
			for (int iter = 0; iter < irlAlgo.getRewardUpdateIterations (); ++iter)
			{
// 				System.out.println ("Iteration #" + iter + " for Updating reward function of table " + table_i);
				
			    weight_policy_q_vectorMapForTable_i.put ("W", tableWeightVectors);
			    weight_policy_q_vectorMapForTable_i.put ("P", tablePolicyVectors);
			    weight_policy_q_vectorMapForTable_i.put ("V", tableValueVectors);
			    weight_policy_q_vectorMapForTable_i.put ("Q", tableQVectors);
				
				int boolLLH = 5;
				if (restaurantLikelihoods == null)
				{
//					throw new java.lang.RuntimeException ("This table does not exist in restaurantLikelihoods");
					System.out.print (">>>>>>>>>>>>>>>>>>>>>>>>>Table ");
					System.out.print (Integer.toString (table_i));
					System.out.println (" does not exist in restaurantLikelihoods");
					// the table index doesn't exist, then restaurantLikelihoods.get(table_i) wiil
					// return NULL, which means we still need to calculate llh and gradient for this
					// new table similar to situation where the table's llh were Double.NaN
					boolLLH = 0;
				}
				// either the table-index is BRAND NEW and therefore does not have a corresponding
				// key-value entry in the restaruantLikelihoods Map, OR, if does exist, the LLH
				// value in this Map should have been RESET during updateTableAssignment() to NaN.
				else
				{
					// if aBool or sBool are TRUE (NOTE: 0 as input from Double.compare means 'TRUE')
//					boolLLH = Double.compare (restaurantLikelihoods.get (table_i), Double.NaN);
					boolLLH = Double.isNaN (restaurantLikelihoods) ? 0 : -1;	// NaN is > any number
				}
				// If table_i exists but the LLH value in restaurantLikelihoods Map is NaN (or
				// if the table was newly created), we need to calculate it
				if (boolLLH == 0)
				{ // if they are equivalent, then boolLLH should equal 0;
					// Compute both the Current_LLH and the Current_GRADIENT for the table_i
					Pair<Double, DoubleMatrix> llhAndGradientForTable_i = 
							llhFunctIRL.computeLogLikelihoodAndGradient_Bayesian (environment, irlAlgo, weight_policy_q_vectorMapForTable_i.get ("W"), 
																				  updatedSAPairCountsForSubsetOfTrajectories, 
																				  weight_policy_q_vectorMapForTable_i.get ("P"), null, null, true);
					
					restaurantLikelihoods = llhAndGradientForTable_i.getFirst ();
					restGradientsLLH = llhAndGradientForTable_i.getSecond ();
					
					// Compute both the PRIOR_PROBABILITY and the PRIOR_GRADIENT for the table_i
					Pair<Double, double[][]> priorProbAndGradientForTable_i = 
							priorIRL.computeLogPriorAndGradient (weight_policy_q_vectorMapForTable_i.get ("W"));
					
					restaurantPriors = priorProbAndGradientForTable_i.getFirst ();
					restGradientsPrior = new DoubleMatrix (priorProbAndGradientForTable_i.getSecond ());
				}
				
				// logPosteriorProb for table_i = Current_LLH + PRIOR_PROBABILITY
				logPosteriorProbability = restaurantLikelihoods.doubleValue () + restaurantPriors.doubleValue ();
				
				double randomShift[] = MersenneTwisterFastIRL.RandomNormalMatrix (1, numRows)[0];	// Swap so get single array
				double[][] boundUpdatedWeightMatrix = updateWeightMatrix (restGradientsLLH, restGradientsPrior, gradient_forTablei, scratchSpace, 
																		  randomShift, scalingParameter, numRows, 
																		  weight_policy_q_vectorMapForTable_i.get ("W"), irlAlgo);
				// automatically generates SHIFTED REWARD FUNCTION that is automatically set to the rewardFunction for the current MDP environment
				RewardFunctionGenerationCancer.generateWeightedRewardFunction (environment, boundUpdatedWeightMatrix);
				 // RE-COMPUTE POLICY and VALUE matrices
				Map<String, double[][]> newlyComputed_Policy_Value_H_Q_matrices = PolicySolverCancer.runPolicyIteration (environment, irlAlgo, null);
				
				// Use SHIFTED REWARD FUNCTION to compute both the Current_LLH_shifted and the
				// Current_GRADIENT_shifted for the table_i
				Pair<Double, DoubleMatrix> LLHAndGradientForTable_i_fromNewWeightMatrix = 
						llhFunctIRL.computeLogLikelihoodAndGradient_Bayesian (environment, irlAlgo, boundUpdatedWeightMatrix, 
																			  updatedSAPairCountsForSubsetOfTrajectories, 
																			  newlyComputed_Policy_Value_H_Q_matrices.get ("P"), null, null, true);
				
				// Use SHIFTED REWARD FUNCTION to compute both the PRIOR_PROBABILITY_shifted and the
				// PRIOR_GRADIENT_shifted for the table_i
				Pair<Double, double[][]> PriorProbAndGradientForTable_i_fromNewWeightMatrix = 
						priorIRL.computeLogPriorAndGradient (boundUpdatedWeightMatrix);
				
				// SHIFTEDlogPosteriorProb for table_i = Current_LLH_shifted + PRIOR_PROBABILITY_shifted
				// NEW log posteriorProbability of REWARD Function = NEW log likelihood + NEW priorProbability of REWARD FUNCTION
				logPosteriorProbability_updated = LLHAndGradientForTable_i_fromNewWeightMatrix.getFirst () + 
												  PriorProbAndGradientForTable_i_fromNewWeightMatrix.getFirst ();
				
				// SHIFTEDgradient for table_i = Current_GRADIENT_shifted + PRIOR_GRADIENT_shifted
				gradient_updated_forTablei = doAdd (LLHAndGradientForTable_i_fromNewWeightMatrix.getSecond (), 
															PriorProbAndGradientForTable_i_fromNewWeightMatrix.getSecond ());
				// NEW gradient = gradientBasedOnQMatrixLikelihood + gradientBasedOnPrior
				
				double[]	sumGradients = doAddinPlace (gradient_updated_forTablei, gradient_forTablei);
				double[]	prod1 = mulInPlace (sumGradients, scalingParameter * 0.5);
				double[]	sum1 = doAddinPlace (prod1, randomShift);
				double[]	g_Updated1 = squareInPlace (sum1);

				double gNumerator = getSum (g_Updated1) * -0.5;
				double fNumerator = logPosteriorProbability_updated;
				double fDenominator = logPosteriorProbability;
				double gDenominator = squareSum (randomShift) * -0.5;
				double probQuotient = Math.exp ((fNumerator + gNumerator) - (fDenominator + gDenominator));
				
				if (!Double.isNaN (probQuotient)) //JK 5.29.2019 added if condition to check if probQuotient is NaN
				{
					double rand2 = RNG.nextDouble ();
					if (probQuotient > rand2)
					{
//						System.out.print (dateTimeFormat.format (new Date ()));
//						System.out.println (": ###################Reward Function for table " + table_i + " HAS BEEN IMPROVED");
						tableWeightVectors = boundUpdatedWeightMatrix;
						tablePolicyVectors = newlyComputed_Policy_Value_H_Q_matrices.get ("P");
						tableValueVectors = newlyComputed_Policy_Value_H_Q_matrices.get ("V");
						tableQVectors = newlyComputed_Policy_Value_H_Q_matrices.get ("Q");
						
						restaurantLikelihoods = LLHAndGradientForTable_i_fromNewWeightMatrix.getFirst ();
						restaurantPriors = PriorProbAndGradientForTable_i_fromNewWeightMatrix.getFirst ();
						restGradientsLLH = LLHAndGradientForTable_i_fromNewWeightMatrix.getSecond ();
						restGradientsPrior = new DoubleMatrix (PriorProbAndGradientForTable_i_fromNewWeightMatrix.getSecond ());
					}
				}
				weight_policy_q_vectorMapForTable_i.clear (); // need to reset the weight, policy, vector for each reward update iteration
			} // end for-loop for reward update iterations
		} // end of else-condition
		rmap5 = new RestaurantTable (tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, tableAssignmentMatrix);
		rmap5._restLikeLihoods = restaurantLikelihoods;
		rmap5._restPriors = restaurantPriors;
		rmap5._restGradientsFromLLH = restGradientsLLH;
		rmap5._restGradientsFromPrior = restGradientsPrior;
		
		return rmap5;
	}
	
	
	/**
	 * Calculate and create the bounded weight matrix
	 * JK 7.25.2019 data validated
	 * @param restGradientsLLH		Matrix to add to {@code restGradientsPrior} to update {@code gradient_forTablei}
	 * @param restGradientsPrior	Matrix to add to {@code restGradientsLLH} to update {@code gradient_forTablei}
	 * @param gradient_forTablei	Matrix to get the new gradient values
	 * @param scaledGradient		Scratch space in which we'll do most calculations
	 * @param randomShift			Random numbers used to modify the gradient, will not be modified
	 * @param scalingParameter		Defined scaling parameter to use
	 * @param numRows				Number of rows in output matrix and in {@code weightMatrix}
	 * @param weightMatrix			Matrix whose values to modify, it will not be touched
	 * @param irlAlgo				Source of upper and lower bounds
	 * @return	A {@code numRows} by 1 double[][] with updated weights
	 */
	private static double[][] updateWeightMatrix (DoubleMatrix restGradientsLLH, DoubleMatrix restGradientsPrior, DoubleMatrix gradient_forTablei, 
												  double[] scaledGradient, double[] randomShift, double scalingParameter, int numRows, 
												  double[][] weightMatrix, IRLAlgorithmCancer irlAlgo)
	{
		//JK 7.19.2019 added if-conditions to find which gradient is NULL
		if(gradient_forTablei.data == null ) {
			System.err.println("gradient_forTablei.data is NULL");
		}
		if(restGradientsLLH.data ==null ) {
			System.err.println("restGradientsLLH.data is NULL");
		}
		if( restGradientsPrior.data == null) {
			System.err.println("restGradientsPrior.data is NULL");
		}
		
		doAdd (gradient_forTablei.data, restGradientsLLH.data, restGradientsPrior.data);
		
		// Use newly computed gradient for table_i to calculate a SHIFTED weight-matrix that
		// can be used to generate a SHIFTED/UPDATED REWARD FUNCTION for this table_i?
		double	scaleFactor = 0.5 * Math.pow (scalingParameter, 2);
		
		doMul (scaledGradient, gradient_forTablei.data, scaleFactor);
		
		for (int i = 0; i < numRows; ++i)
			scaledGradient[i] += (randomShift[i] * scalingParameter) + weightMatrix[i][0];
		
		double	lowerBound = irlAlgo.getRewardLowerBounds ();
		double	upperBound = irlAlgo.getRewardUpperBounds ();
		
		for (int i = 0; i < numRows; ++i)
		{
			double	value = scaledGradient[i];
			
			if (Double.compare (value, upperBound) >= 0)
				scaledGradient[i] = upperBound;
			else if (Double.compare (value, lowerBound) < 0)
				scaledGradient[i] = lowerBound;
		}
		double[][] boundUpdatedWeightMatrix = new double[numRows][1];
		
		for (int i = 0; i < numRows; ++i)
			boundUpdatedWeightMatrix[i][0] = scaledGradient[i];
		
		return boundUpdatedWeightMatrix;
	}
	
	
	/**
	 * Add the contents of two arrays and store the results in a third.<br>
	 * {@code result} can be {@code first} or {@code second}, or a completely different array
	 * 
	 * @param result	double[] to store results in
	 * @param first		double[] to get data from
	 * @param second	double[] to get data from
	 */
	private static final void doAdd (double[] result, double[] first, double[] second)
	{
		int	len = first.length;
		
		for (int i = 0; i < len; ++i)
			result[i] = first[i] + second[i];
	}
	
	
	/**
	 * Add the contents of {@code source} to the contents of {@code target}, 
	 * and store the results in {@code target}
	 * 
	 * @param target	{@link DoubleMatrix} to get data from and write results to
	 * @param source	double[] to get data from
	 */
	private static final DoubleMatrix doAddinPlace (DoubleMatrix target, double[] source)
	{
		double[]	data = target.data;
		int			len = data.length;
		
		for (int i = 0; i < len; ++i)
			data[i] += source[i];
		
		return target;
	}
	
	
	/**
	 * Add the contents of the first column of {@code source} to the contents of {@code target}, 
	 * and store the results in a new double[] that is returned
	 * 
	 * @param target	{@link DoubleMatrix} to get data from and write results to
	 * @param source	double[][] to get data from
	 */
	private static final double[] doAdd (DoubleMatrix target, double[][] source)
	{
		double[]	data = target.data;
		int			len = data.length;
		double[]	results = new double[len];
		
		for (int i = 0; i < len; ++i)
			results[i] = source[i][0] + data[i];
		
		return results;
	}
	
	
	/**
	 * Add the contents of {@code source} to the contents of {@code target}, 
	 * and store the results in {@code target}
	 * 
	 * @param target	double[] to get data from and write results to
	 * @param source	{@link DoubleMatrix} to get data from
	 * @return	{@code target}
	 */
	private static final double[] doAddinPlace (double[] target, DoubleMatrix source)
	{
		double[]	sourceData = source.data;
		int			len = target.length;
		
		for (int i = 0; i < len; ++i)
			target[i] += sourceData[i];
		
		return target;
	}
	
	
	/**
	 * Add the contents of {@code source} to the contents of {@code target}, 
	 * and store the results in {@code target}
	 * 
	 * @param target	double[] to get data from and write results to
	 * @param source	double[] to get data from
	 * @return	{@code target}
	 */
	private static final double[] doAddinPlace (double[] target, double[] source)
	{
		int	len = target.length;
		
		for (int i = 0; i < len; ++i)
			target[i] += source[i];
		
		return target;
	}
	
	
	/**
	 * Add the contents of {@code source} to the contents of {@code target}, 
	 * and store the results in {@code target}
	 * 
	 * @param target	{@link DoubleMatrix} to get data from and write results to
	 * @param source	double[][] to get data from
	 * @return	{@code target}
	 */
	private static final DoubleMatrix doAddinPlace (DoubleMatrix target, DoubleMatrix source)
	{
		double[]	data = target.data;
		double[]	sourceData = source.data;
		int			len = data.length;
		
		for (int i = 0; i < len; ++i)
			data[i] += sourceData[i];
		
		return target;
	}
	
	
	/**
	 * Multiply contents of {@code source} by {@code factor} and store the results in {@code target}
	 * 
	 * @param target	double[] to store results in
	 * @param source	double[] to get data from
	 * @param factor	double[] to multiply by
	 */
	private static final void doMul (double[] target, double[] source, double factor)
	{
		int	len = target.length;
		
		for (int i = 0; i < len; ++i)
			target[i] = source[i] * factor;
	}
	
	
	/**
	 * Multiply contents of {@code target} by {@code factor} and store the results in {@code target}
	 * 
	 * @param target	{@link DoubleMatrix} to get data from and write results to
	 * @param factor	double[] to multiply by
	 * @return	{@code target}
	 */
	private static final double[] mulInPlace (double[] target, double factor)
	{
		int	len = target.length;
		
		for (int i = 0; i < len; ++i)
			target[i] *= factor;
		
		return target;
	}
	
	
	/**
	 * Multiply contents of {@code target} by themselves and store the results in {@code target}
	 * 
	 * @param target	double[] to get data from and write results to
	 * @return	{@code target}, with data[x] = data[x] * data[x] for all elements
	 */
	private static final double[] squareInPlace (double[] target)
	{
		int	len = target.length;
		
		for (int i = 0; i < len; ++i)
			target[i] *= target[i];
		
		return target;
	}
	
	
	/**
	 * Multiply contents of {@code source} by themselves and add the results together
	 * 
	 * @param target	{@link DoubleMatrix} to get data from and write results to
	 * @return	{@code target}, with data[x] = data[x] * data[x] for all elements
	 */
	private static final double squareSum (double[] source)
	{
		int		len = source.length;
		double	square = 0.0;
		
		for (int i = 0; i < len; ++i)
		{
			double	value = source[i];
			square += value * value;
		}
		
		return square;
	}
	
	
	/**
	 * Add together contents of {@code source} are return it
	 * 
	 * @param target	double[] to store results in
	 * @param source	double[] to get data from
	 * @param factor	double[] to multiply by
	 */
	private static final double getSum (double[] source)
	{
		int		len = source.length;
		double	sum = 0.0;
		
		for (int i = 0; i < len; ++i)
			sum += source[i];
		
		return sum;
	}
	
	
	/**
	 * Update the weight, policy, value of the multiple reward functions (tables) in the restaurant
	 * @param trajSet 
	 * 
	 * @param environment
	 * @param table_i
	 * @param irlAlgo
	 * @param tableAssignmentMatrix
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 * @param restaurantLikelihoods
	 * @param restaurantPriors
	 * @param restGradientsLLH
	 * @param restGradientsPrior
	 * @return Updated {@link RestaurantMap}
	 * 
	 */
	public static final RestaurantMap updateRewardFunctionsWithDatabase (List<double[][]> trajSet, MDPCancer environment, int table_i, 
																		 IRLAlgorithmCancer irlAlgo, double[][] tableAssignmentMatrix, 
																		 Map<Integer, double[][]> tableWeightVectors, 
																		 Map<Integer, double[][]> tablePolicyVectors, 
																		 Map<Integer, double[][]> tableValueVectors, 
																		 Map<Integer, double[][]> tableQVectors, 
																		 Map<Integer, Double> restaurantLikelihoods, 
																		 Map<Integer, Double> restaurantPriors, 
																		 Map<Integer, DoubleMatrix> restGradientsLLH, 
																		 Map<Integer, DoubleMatrix> restGradientsPrior)
	{
		RestaurantMap rmap5;
		LikelihoodFunctionCancer llhFunctIRL = irlAlgo.getLikelihood ();
		Prior priorIRL = irlAlgo.getPrior ();
		double scalingParameter = .01;
		Map<Integer, double[][]>	subsetOfTrajectories = getSubsetOfTrajectories (tableAssignmentMatrix, (double) table_i, trajSet);
		Map<String, double[][]>		weight_policy_q_vectorMapForTable_i = new HashMap<String, double[][]> ();
		
		double logPosteriorProbability = 0.0;
		double logPosteriorProbability_updated = 0.0;
		double[]		gradient_updated_forTablei = null;
		int				numRows = environment.getNumRewardFeatures ();
		DoubleMatrix	gradient_forTablei = new DoubleMatrix (numRows, 1);
		double[]		scratchSpace = new double[numRows];
		
		// should not run if none of the trajectories possess a given table index-value
		if (subsetOfTrajectories.isEmpty ())
		{
			System.out.println ("None of the trajectories were assigned to table " + table_i);
		}
		else
		{
			// JK 6.23.2019 moved computeOccopuancy() outside of rewardupdateIteration loop. The count map is entirely based on the 
			// subset of trajectories currently being analyzed.  Nothing to do with the actual reward function (weights, llh, etc...)
			computeOccupancyFromDatabase (subsetOfTrajectories, environment, false);
			
			for (int iter = 0; iter < irlAlgo.getRewardUpdateIterations (); ++iter)
			{
//				System.out.println ("Iteration #" + iter + " for Updating reward function of table " + table_i);
				
			    weight_policy_q_vectorMapForTable_i.put ("W", tableWeightVectors.get (table_i));
			    weight_policy_q_vectorMapForTable_i.put ("P", tablePolicyVectors.get (table_i));
			    weight_policy_q_vectorMapForTable_i.put ("V", tableValueVectors.get (table_i));
			    weight_policy_q_vectorMapForTable_i.put ("Q", tableQVectors.get (table_i));
				
				int boolLLH = 5;
				if (!restaurantLikelihoods.containsKey (table_i))
				{
//					throw new java.lang.RuntimeException ("This table does not exist in restaurantLikelihoods");
					System.out.print (">>>>>>>>>>>>>>>>>>>>>>>>>Table ");
					System.out.print (Integer.toString (table_i));
					System.out.println (" does not exist in restaurantLikelihoods");
					// the table index doesn't exist, then restaurantLikelihoods.get(table_i) wiil
					// return NULL, which means we still need to calculate llh and gradient for this
					// new table similar to situation where the table's llh were Double.NaN
					boolLLH = 0;
				}
				// either the table-index is BRAND NEW and therefore does not have a corresponding
				// key-value entry in the restaruantLikelihoods Map, OR, if does exist, the LLH
				// value in this Map should have been RESET during updateTableAssignment() to NaN.
				else
				{
					// if aBool or sBool are TRUE (NOTE: 0 as input from Double.compare means 'TRUE')
//					boolLLH = Double.compare (restaurantLikelihoods.get (table_i), Double.NaN);
					boolLLH = Double.isNaN (restaurantLikelihoods.get (table_i)) ? 0 : -1;	// NaN is > any number
				}
				// If table_i exists but the LLH value in restaurantLikelihoods Map is NaN (or
				// if the table was newly created), we need to calculate it
				if (boolLLH == 0)
				{ // if they are equivalent, then boolLLH should equal 0;
					// Compute both the Current_LLH and the Current_GRADIENT for the table_i
					Pair<Double, DoubleMatrix> llhAndGradientForTable_i = 
							llhFunctIRL.computeLogLikelihoodAndGradient_BayesianWithDatabase (environment, irlAlgo, 
																							  weight_policy_q_vectorMapForTable_i.get ("W"), 
																							  weight_policy_q_vectorMapForTable_i.get ("P"), 
																							  null, null, true);
					
					restaurantLikelihoods.put (table_i, llhAndGradientForTable_i.getFirst ());
					restGradientsLLH.put (table_i, llhAndGradientForTable_i.getSecond ());
					
					// Compute both the PRIOR_PROBABILITY and the PRIOR_GRADIENT for the table_i
					Pair<Double, double[][]> priorProbAndGradientForTable_i =  
							priorIRL.computeLogPriorAndGradient (weight_policy_q_vectorMapForTable_i.get ("W"));
					
					restaurantPriors.put (table_i, priorProbAndGradientForTable_i.getFirst ());
					restGradientsPrior.put (table_i, new DoubleMatrix (priorProbAndGradientForTable_i.getSecond ()));
				}
				
				// logPosteriorProb for table_i = Current_LLH + PRIOR_PROBABILITY
				logPosteriorProbability = restaurantLikelihoods.get (table_i) + restaurantPriors.get (table_i);
				
				double randomShift[] = MersenneTwisterFastIRL.RandomNormalMatrix (1, numRows)[0];	// Swap so get single array
				double[][] boundUpdatedWeightMatrix = updateWeightMatrix (restGradientsLLH.get (table_i), restGradientsPrior.get (table_i), 
																		  gradient_forTablei, scratchSpace, randomShift, scalingParameter, 
																		  numRows, weight_policy_q_vectorMapForTable_i.get ("W"), irlAlgo);
				// automatically generates SHIFTED REWARD FUNCTION that is automatically set to the rewardFunction for the current MDP environment
				RewardFunctionGenerationCancer.generateWeightedRewardFunction (environment, boundUpdatedWeightMatrix);
				 // RE-COMPUTE POLICY and VALUE matrices
				Map<String, double[][]> newlyComputed_Policy_Value_H_Q_matrices = PolicySolverCancer.runPolicyIteration (environment, irlAlgo, null);
				
				// Use SHIFTED REWARD FUNCTION to compute both the Current_LLH_shifted and the
				// Current_GRADIENT_shifted for the table_i
				Pair<Double, DoubleMatrix> LLHAndGradientForTable_i_fromNewWeightMatrix = 
						llhFunctIRL.computeLogLikelihoodAndGradient_BayesianWithDatabase (environment, irlAlgo, boundUpdatedWeightMatrix, 
																						  newlyComputed_Policy_Value_H_Q_matrices.get ("P"), null, 
																						  null, true);
				
				// Use SHIFTED REWARD FUNCTION to compute both the PRIOR_PROBABILITY_shifted and the
				// PRIOR_GRADIENT_shifted for the table_i
				Pair<Double, double[][]> PriorProbAndGradientForTable_i_fromNewWeightMatrix = 
						priorIRL.computeLogPriorAndGradient (boundUpdatedWeightMatrix);
				
				// SHIFTEDlogPosteriorProb for table_i = Current_LLH_shifted + PRIOR_PROBABILITY_shifted
				// NEW log posteriorProbability of REWARD Function = NEW log likelihood + NEW priorProbability of REWARD FUNCTION
				logPosteriorProbability_updated = LLHAndGradientForTable_i_fromNewWeightMatrix.getFirst () + 
												  PriorProbAndGradientForTable_i_fromNewWeightMatrix.getFirst ();
				
				// SHIFTEDgradient for table_i = Current_GRADIENT_shifted + PRIOR_GRADIENT_shifted
				gradient_updated_forTablei = doAdd (LLHAndGradientForTable_i_fromNewWeightMatrix.getSecond (), 
													PriorProbAndGradientForTable_i_fromNewWeightMatrix.getSecond ());
				// NEW gradient = gradientBasedOnQMatrixLikelihood + gradientBasedOnPrior
				
				double[]	sumGradients = doAddinPlace (gradient_updated_forTablei, gradient_forTablei);
				double[]	prod1 = mulInPlace (sumGradients, scalingParameter * 0.5);
				double[]	sum1 = doAddinPlace (prod1, randomShift);
				double[]	g_Updated1 = squareInPlace (sum1);

				double gNumerator = getSum (g_Updated1) * -0.5;
				double fNumerator = logPosteriorProbability_updated;
				double fDenominator = logPosteriorProbability;
//				double gDenominator = MatrixUtilityJBLAS.squaredMatrix (new DoubleMatrix (randomShift)).sum () * -0.5;
				double gDenominator = squareSum (randomShift) * -0.5;
				double probQuotient = Math.exp ((fNumerator + gNumerator) - (fDenominator + gDenominator));
				
				if (!Double.isNaN (probQuotient)) //JK 5.29.2019 added if condition to check if probQuotient is NaN
				{
					double rand2 = RNG.nextDouble ();
					if (probQuotient > rand2)
					{
						tableWeightVectors.put (table_i, boundUpdatedWeightMatrix);
						tablePolicyVectors.put (table_i, newlyComputed_Policy_Value_H_Q_matrices.get ("P"));
						tableValueVectors.put (table_i, newlyComputed_Policy_Value_H_Q_matrices.get ("V"));
						tableQVectors.put (table_i, newlyComputed_Policy_Value_H_Q_matrices.get ("Q"));
						
						restaurantLikelihoods.put (table_i, LLHAndGradientForTable_i_fromNewWeightMatrix.getFirst ());
						restaurantPriors.put (table_i, PriorProbAndGradientForTable_i_fromNewWeightMatrix.getFirst ());
						restGradientsLLH.put (table_i, LLHAndGradientForTable_i_fromNewWeightMatrix.getSecond ());
						restGradientsPrior.put (table_i, new DoubleMatrix (PriorProbAndGradientForTable_i_fromNewWeightMatrix.getSecond ()));
					}
				}
				weight_policy_q_vectorMapForTable_i.clear (); // need to reset the weight, policy, vector for each reward update iteration
			} // end for-loop for reward update iterations
		} // end of else-condition
		
		rmap5 = new RestaurantMap (tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, tableAssignmentMatrix);
		rmap5._restLikeLihoods = restaurantLikelihoods;
		rmap5._restPriors = restaurantPriors;
		rmap5._restGradientsFromLLH = restGradientsLLH;
		rmap5._restGradientsFromPrior = restGradientsPrior;
		
		return rmap5;
	}
	
	
	/**
	 * JK. Validated by JK 7.24.2019
	 * 
	 * @param XcurrentTblWeightVector_customer_i
	 * @param XcurrentTblValueVector_customer_i
	 * @param YaltTblWeightVector_customer_i
	 * @param YaltTblValueVector_customer_i
	 * @param env
	 * @param irlAlgo
	 * @param stateActPairCountsInfoForSubsetOfTrajectories
	 * @param customer_i
	 * @param numThreads				Number of threads it can use
	 * @param numGPUs					Number of GPUs it can use
	 * @return	non-log based probability
	 */
	public static final double computeMinProbabilityQuotient (
			double[][] XcurrentTblWeightVector_customer_i, double[][] XcurrentTblValueVector_customer_i, 
			double[][] YaltTblWeightVector_customer_i, double[][] YaltTblValueVector_customer_i, 
			MDPCancer env, IRLAlgorithmCancer irlAlgo, Multimap<Integer, double[]> stateActPairCountsInfoForSubsetOfTrajectories,
			Integer customer_i, int numThreads, int numGPUs)
	{
		
		// Compute likelihood (according to softmax distribution) for customer_i originally assigned
		// table (based on Q matrix computed from weight and value matrices)
		// Recall:The log of a quotient is the difference of the logs;Therefore, the rule for
		// division is to subtract the logarithms
		// Since the likelihood computation involves the quotient of two log() values, we simply
		// subtract them to obtain the qBasedLikelihood
		// returns the log-based qMatrix of dimension [numStates x numActions]
		Map<String,double[][]> QVPmap = PolicySolverCancer.policyImprovementStep (env, YaltTblValueVector_customer_i, YaltTblWeightVector_customer_i);
		double[][] qMatrix1 = QVPmap.get("Q");
		
		
		double[][] qMatrixeta1 = MatrixUtilityJBLAS.scalarMultiplication (qMatrix1, irlAlgo.getLikelihood ().getEta ());
		DoubleMatrix qMatrixeta1RealMat = new DoubleMatrix (qMatrixeta1);
		
		double[][] expQLLH1 = MatrixUtilityJBLAS.exp (qMatrixeta1);
		// returns column matrix, with sum across columns per row; thus for each trajectory/customer we obtain sum of q-values
		double[][] sumeQLLH1 = MatrixUtilityJBLAS.sumPerRow (expQLLH1);
		double[][] logSumQLLH1 = MatrixUtilityJBLAS.log (sumeQLLH1); // returns log of each element in column matrix
		DoubleMatrix logSumMatrix1 = new DoubleMatrix (logSumQLLH1);
		
		
		// =loglikelihood of original restaurant table assigned to trajectory 'customer_i' ; [numStates x numActions] matrix
		DoubleMatrix qBasedLikelihood1 = MatrixUtilityJBLAS.elementwiseSubtractionByColumnVector (qMatrixeta1RealMat, logSumMatrix1);
		
		// Compute likelihood (according to softmax distribution) for customer_i alternative/new
		// table (based on Q matrix computed from weight and value matrices) ; same as above
		Map<String,double[][]> QVPmap2 = 
				PolicySolverCancer.policyImprovementStep (env, XcurrentTblValueVector_customer_i, XcurrentTblWeightVector_customer_i);
		double[][] qMatrix2 = QVPmap2.get ("Q");

		
		double[][] qMatrixeta2 = MatrixUtilityJBLAS.scalarMultiplication (qMatrix2, irlAlgo.getLikelihood ().getEta ());
		
		DoubleMatrix qMatrixeta2realMat = new DoubleMatrix (qMatrixeta2);
		
		double[][] expQLLH2 = MatrixUtilityJBLAS.exp (qMatrixeta2);
		double[][] sumeQLLH2 = MatrixUtilityJBLAS.sumPerRow (expQLLH2); // returns column matrix, with sum across columns per row
		double[][] logSumQLLH2 = MatrixUtilityJBLAS.log (sumeQLLH2); // returns log of each element in column matrix
		DoubleMatrix logSumMatrix2 = new DoubleMatrix (logSumQLLH2);

		
		// loglikelihood of alternative/new restaurant table assigned to trajectory 'customer_i'; [numStates x numActions] matrix
		DoubleMatrix qBasedLikelihood2 = MatrixUtilityJBLAS.elementwiseSubtractionByColumnVector (qMatrixeta2realMat, logSumMatrix2);
		
		// Compute quotient of likelihoods ( between original table likelihood and alternative/new table likelihood)
		double logLLHQuotient = 0;
		
		if (customer_i == null)
		{ // calculate likelihood quotients based on ALL customers/trajectories (this is only used for DPM_MH without Langevin diffusion)
			if (!stateActPairCountsInfoForSubsetOfTrajectories.isEmpty ())
			{
				for (int traj_j = 1; traj_j < stateActPairCountsInfoForSubsetOfTrajectories.keySet ().size () + 1; traj_j++)
				{ // for each sub-trajectory listed in map;recall that this Map of <Integer,
					// stateActionPairCounts> begins at integer =1; (since this map can be
					// representative of a subset/whole of original trajectory dataset, the first
					// sub-trajectory in this map is not necessarily customer 1.
					// System.out.println("stateActPairCountsInfoForSubsetOfTrajectories keyset size
					// = "+stateActPairCountsInfoForSubsetOfTrajectories.keySet().size());
					List<double[]> saPairCountsForTrajJ = (List<double[]>) stateActPairCountsInfoForSubsetOfTrajectories.get (traj_j);
					
					for (int observedSA = 0; observedSA < saPairCountsForTrajJ.size (); observedSA++)
					{ // for each OBSERVED state-action pair (with count >1) in trajectory j
						Double state = saPairCountsForTrajJ.get (observedSA)[0]; // get state for that observed sa-pair
						Double action = saPairCountsForTrajJ.get (observedSA)[1]; // get action
						Double count = saPairCountsForTrajJ.get (observedSA)[2]; // get count ; number of times s,a was observed in trajectory j
						logLLHQuotient = logLLHQuotient + ((qBasedLikelihood1.get (state.intValue (), action.intValue ()) - 
															qBasedLikelihood2.get (state.intValue (), action.intValue ())) * count);
					}
				}
			}
		}
		else
		{
			if (!stateActPairCountsInfoForSubsetOfTrajectories.isEmpty ())
			{
				List<double[]> saPairCountsForTrajJ = (List<double[]>) stateActPairCountsInfoForSubsetOfTrajectories.get (customer_i);
				
				for (int observedSA = 0; observedSA < saPairCountsForTrajJ.size (); observedSA++)
				{ // for each OBSERVED state-action pair (with count >1) in trajectory j
					Double state = saPairCountsForTrajJ.get (observedSA)[0]; // get state for that observed sa-pair
					Double action = saPairCountsForTrajJ.get (observedSA)[1]; // get action
					Double count = saPairCountsForTrajJ.get (observedSA)[2]; // get count ; number of times s,a was observed in trajectory j
					logLLHQuotient = logLLHQuotient + ((qBasedLikelihood1.get (state.intValue (), action.intValue ()) - 
														qBasedLikelihood2.get (state.intValue (), action.intValue ())) * count);
				}
				
			}
			
		}
		
		// we want to return the non-log based probability
		double minProbability = Math.exp (logLLHQuotient);
		return minProbability;
	}
	
	
	/**
	 * 
	 * 
	 * @param currentTblWeightVector_customer_i
	 * @param currentTblValueVector_customer_i
	 * @param altTblWeightVector_customer_i
	 * @param altTblValueVector_customer_i
	 * @param env
	 * @param irlAlgo
	 * @param customer_i
	 * @param numThreads				Number of threads it can use
	 * @param numGPUs					Number of GPUs it can use
	 * @return	non-log based probability
	 */
	public static final double computeMinProbabilityQuotientWithDatabase (
			double[][] currentTblWeightVector_customer_i, double[][] currentTblValueVector_customer_i, 
			double[][] altTblWeightVector_customer_i, double[][] altTblValueVector_customer_i, 
			MDPCancer env, IRLAlgorithmCancer irlAlgo, Integer customer_i, int numThreads, int numGPUs)
	{
		// Compute likelihood (according to softmax distribution) for customer_i originally assigned
		// table (based on Q matrix computed from weight and value matrices)
		// Recall:The log of a quotient is the difference of the logs;Therefore, the rule for
		// division is to subtract the logarithms
		// Since the likelihood computation involves the quotient of two log() values, we simply
		// subtract them to obtain the qBasedLikelihood
		// returns  the  log-based  qMatrix  of  dimension  [  numStates  x  numActions]
//		double[][] qMatrix1 = PolicySolverCancer.policyImprovementStep (env, currentTblValueVector_customer_i, currentTblWeightVector_customer_i).get ("Q");
		Map<String, double[][]> QVPmap1 = 
				PolicySolverCancer.policyImprovementStep (env, currentTblValueVector_customer_i, currentTblWeightVector_customer_i);
		double[][] qMatrix1 = QVPmap1.get("Q");

		double[][] qMatrixeta1 = MatrixUtilityJBLAS.scalarMultiplication (qMatrix1, irlAlgo.getLikelihood ().getEta ());
		DoubleMatrix qMatrixeta1RealMat = new DoubleMatrix (qMatrixeta1);
		
		double[][] expQLLH1 = MatrixUtilityJBLAS.exp (qMatrixeta1);
		// returns column matrix,  with sum across columns  per row; thus for each  trajectory/customer we  obtain sum of q-values
		double[][] sumeQLLH1 = MatrixUtilityJBLAS.sumPerRow (expQLLH1);
		// returns log of each element  in column matrix
		double[][] logSumQLLH1 = MatrixUtilityJBLAS.log (sumeQLLH1);
		DoubleMatrix logSumMatrix1 = new DoubleMatrix (logSumQLLH1);
		// JK 3.25.2019 losumMatrix1 should already be a column vector right? RealVector
		// logSumRealVec = logSumMatrix1.getColumnVector(0);
		
		// =log likelihood  of  original  restaurant  table  assigned  to  trajectory  'customer_i'  ;  [numStates  x  numActions]  matrix
		DoubleMatrix qBasedLikelihood1 = MatrixUtilityJBLAS.elementwiseSubtractionByColumnVector (qMatrixeta1RealMat, logSumMatrix1);
		
		// Compute likelihood (according to softmax distribution) for customer_i alternative/new
		// table (based on Q matrix computed from weight and value matrices) ; same as above
		double[][] qMatrix2 = PolicySolverCancer.policyImprovementStep (env, altTblValueVector_customer_i, altTblWeightVector_customer_i).get ("Q");
		
		double[][] qMatrixeta2 = MatrixUtilityJBLAS.scalarMultiplication (qMatrix2, irlAlgo.getLikelihood ().getEta ());
		
		DoubleMatrix qMatrixeta2realMat = new DoubleMatrix (qMatrixeta2);
		
		double[][] expQLLH2 = MatrixUtilityJBLAS.exp (qMatrixeta2);
		double[][] sumeQLLH2 = MatrixUtilityJBLAS.sumPerRow (expQLLH2); // returns column matrix, with sum across columns per row
		double[][] logSumQLLH2 = MatrixUtilityJBLAS.log (sumeQLLH2); // returns log of each element in column matrix
		DoubleMatrix logSumMatrix2 = new DoubleMatrix (logSumQLLH2);
		/// JK 3.25.2019 logsummatrix2 should already be a column matrix 

		
		// log likelihood  of  alternative/new  restaurant  table  assigned  to  trajectory  'customer_i';  [numStates  x  numActions]  matrix
		DoubleMatrix qBasedLikelihood2 = MatrixUtilityJBLAS.elementwiseSubtractionByColumnVector (qMatrixeta2realMat, logSumMatrix2);
		
		// Compute quotient of likelihoods ( between original table likelihood and alternative/new
		// table likelihood)
		double logLLHQuotient = 0;
		Double state = 0.0;
		Double action = 0.0;
		Double count = 0.0;
		
		if (customer_i == null)
		{ // calculate likelihood quotients based on ALL customers/trajectories (this is only used
			// for DPM_MH without Langevin diffusion)
			String cqlSelectPairCountInfofrom_countInfoTable = "select * from countinfofortrajs_table";
			for (Row cinfo_row : _session.execute (cqlSelectPairCountInfofrom_countInfoTable))
			{
				state = cinfo_row.getDouble ("statedbl");
				action = cinfo_row.getDouble ("actiondbl");
				count = cinfo_row.getDouble ("countdbl");
				logLLHQuotient = logLLHQuotient
						+ (qBasedLikelihood1.get (state.intValue (), action.intValue ())
								- qBasedLikelihood2.get (state.intValue (), action.intValue ()))
								* count;
			}
		}
		else
		{
			// for each OBSERVED state-action pair (with count >1) in trajectory j(customer j)
			String cqlSelectPairCountInfoForSingleCustomerfrom_countInfoTable = "select * from countinfofortrajs_table where trajint="
					+ customer_i;
			for (Row cinfo_row : _session
					.execute (cqlSelectPairCountInfoForSingleCustomerfrom_countInfoTable))
			{
				state = cinfo_row.getDouble ("statedbl");
				action = cinfo_row.getDouble ("actiondbl");
				count = cinfo_row.getDouble ("countdbl");
				logLLHQuotient = logLLHQuotient
						+ (qBasedLikelihood1.get (state.intValue (), action.intValue ())
								- qBasedLikelihood2.get (state.intValue (), action.intValue ()))
								* count;
			}
		}
		
		// we want to return the non-log based probability
		double minProbability = Math.exp (logLLHQuotient);
		return minProbability;
	}
	
	
	/**
	 * Print out a matrix's number of rows and columns to {@link System#out}
	 * 
	 * @param title	Title to print
	 * @param rows	Number of rows
	 * @param cols	Number of cols
	 */
	protected static final void printRowAndCols (String title, int rows, int cols)
	{
		System.out.print (dateTimeFormat.format (new Date ()));
		System.out.print (": ");
		System.out.print (title);
		System.out.print (rows);
		System.out.print (" rows and ");
		System.out.print (cols);
		System.out.println (" columns.");
	}
	
	
	/**
	 * Report the start / end of an event, with timestamp  
	 * 
	 * @param title	Title of event that ran
	 */
	private static void reportEvent (String title)
	{
		System.out.print (dateTimeFormat.format (new Date ()));
		System.out.print (": ");
		System.out.println (title);
	}
	
	
	/**
	 * @return	A {@link String} holding the year-month-day=hour-minute-seconds
	 */
	protected static String timeStampStr ()
	{
		return timeStampFormat.format (new Date ());
	}
	
	
	/**
	 * Report how long an event took.  
	 * 
	 * @param title	Title of event that ran
	 * @param time	Milliseconds it took
	 */
	private static void reportTime (String title, long time)
	{
		System.out.print (dateTimeFormat.format (new Date ()));
		System.out.print (": ");
		System.out.print (title);
		if (time <= 0)
			time = 1;
		System.out.println (timeFormat.format (time / 1000.0));
//		System.out.println (Long.toString (time));
	}
	
	
	/**
	 * Report time taken to run a process
	 * 
	 * @param title		Title of routine that ran
	 * @param tableID	Table ID
	 * @param time		Milliseconds it took
	 */
	private static final void reportTableTime (String title, int tableID, long time)
	{
		System.out.print (dateTimeFormat.format (new Date ()));
		System.out.print (": Time required for ");
		System.out.print (title);
		System.out.print (" for table ");
		System.out.print (numberFormat.format (tableID));
//		System.out.print (Integer.toString (tableID));
		System.out.print (": ");
		System.out.println (timeFormat.format (time / 1000.0));
//		System.out.println (Long.toString (time));
	}
	
	
	/**
	 * Report value
	 * 
	 * @param title	Title of value
	 * @param value	Value to report
	 */
	private static final void reportValue (String title, double value)
	{
		System.out.print (dateTimeFormat.format (new Date ()));
		System.out.print (": ");
		System.out.print (title);
		System.out.print (": ");
		System.out.println (timeFormat.format (value));
//		System.out.println (Double.toString (value));
	}
	
	
	/**
	 * Report value
	 * 
	 * @param title	Title of value
	 * @param value	Value to report
	 */
	private static final void reportValue (String title, int value)
	{
		System.out.print (dateTimeFormat.format (new Date ()));
		System.out.print (": ");
		System.out.print (title);
		System.out.print (": ");
		System.out.println (numberFormat.format (value));
//		System.out.println (Integer.toString (value));
	}
	
	
	/**
	 * Report value for a given iteration
	 * 
	 * @param title	Title of value
	 * @param iter	Which iteration
	 * @param value	Value to report
	 */
	private static final void reportIterationValue (String title, int iter, double value)
	{
		System.out.print (dateTimeFormat.format (new Date ()));
		System.out.print (": Iteration #");
		System.out.print (numberFormat.format (iter));
//		System.out.print (Integer.toString (iter));
		System.out.print (" ");
		System.out.print (title);
		System.out.print (": ");
		System.out.println (timeFormat.format (value));
//		System.out.println (Double.toString (value));
	}
	
	
	/**
	 * Quickly get the "first" key from a {@link Map}
	 * 
	 * @param theMap	{@link Map} of interest
	 * @return	Null if {@code theMap} is null or empty, else the first element of {@code theMap} 
	 * returned by the iterator of its keySet
	 */
	public static final <T, V> T getFirstKey (Map<T, V> theMap)
	{
		if ((theMap == null) || theMap.isEmpty ())
			return null;
		
		return getFirstKey (theMap.keySet ());
	}
	
	
	/**
	 * Quickly get the "first" key from a {@link Set}
	 * 
	 * @param theSet	{@link Set} of interest
	 * @return	Null if {@code theSet} is null or empty, else the first element of {@code theSet} 
	 * returned by its iterator
	 */
	public static final <T> T getFirstKey (Set<T> theSet)
	{
		if (theSet == null)
			return null;
		
		Iterator<T>	iter = theSet.iterator ();
		
		if (iter.hasNext ())
			return iter.next ();
		
		return null;
	}
	
	
	/**
	 * Quickly get the "largest" key from a {@link Map}
	 * 
	 * @param theMap	{@link Map} of interest
	 * @return	Null if {@code theMap} is null or empty, else the largest element of {@code theMap} 
	 * returned by the iterator of its keySet
	 */
	public static final <T extends Comparable<T>, V> T getMaxKey (Map<T, V> theMap)
	{
		if ((theMap == null) || theMap.isEmpty ())
			return null;
		
		return getMaxKey (theMap.keySet ());
	}
	
	
	/**
	 * Quickly get the "largest" key from a {@link Set}
	 * 
	 * @param theSet	{@link Set} of interest
	 * @return	Null if {@code theSet} is null or empty, else the largest element of {@code theSet} 
	 * returned by its iterator
	 */
	public static final <T extends Comparable<T>, V> T getMaxKey (Set<T> theSet)
	{
		if ((theSet == null) || theSet.isEmpty ())
			return null;
		
		SortedSet<T>	sortedSet = new TreeSet<> (theSet);
		
		return sortedSet.last ();
	}
	
	
	/**
	 * Quickly get the first key from a {@link Set} of elements that are {@link Comparable}
	 * 
	 * @param theSet	{@link Set} of interest
	 * @return	Null if {@code theSet} is null or empty, else the first element of {@code theSet} 
	 * returned by a {@link TreeSet} iterator
	 */
	public static final <T extends Comparable<T>> T getSortedFirstKey (Set<T> theSet)
	{
		if (theSet == null)
			return null;
		
		Iterator<T>	iter = new TreeSet<T> (theSet).iterator ();
		
		if (iter.hasNext ())
			return iter.next ();
		
		return null;
	}
	
	
	/**
	 * Scan over a {@link DoubleMatrix} with a comparisonVal, returning true if any value in 
	 * {@code dblMatrix} exactly matches {@code comparisonVal}
	 * 
	 * @param dblMatrix		{@link DoubleMatrix} to scan
	 * @param comparisonVal	Value to test for equality
	 * @return	True if find a match, false if find no matches
	 */
	public static final boolean hasTrue (DoubleMatrix dblMatrix, double comparisonVal)
	{
		int			length = dblMatrix.length;
		double[]	data = dblMatrix.data;
		
		for (int i = 0; i < length; i++)
		{
			if (data[i] == comparisonVal)
				return true;
		}
		
		return false;
	}
	
	
	/**
	 * Determine if at least one element in {@code dblMatrix} exactly matches {@code comparisonVal}
	 * 
	 * @param dblMatrix		{@link DoubleMatrix} to search through
	 * @param comparisonVal	Value to test for equality
	 * @return	True if have at least one match, else false
	 */
	public static final boolean hasTrue (double[][] dblMatrix, double comparisonVal)
	{
		int	rows = dblMatrix.length;
		int	columns = dblMatrix[0].length;
		
		for (int row = 0; row < rows; ++row)
		{
			double[]	data = dblMatrix[row];
			
			for (int col = 0; col < columns; ++col)
			{
				if (data[col] == comparisonVal)
					return true;
			}
		}
		
		return false;
	}
	
	
	/**
	 * Make a logical matrix from a {@link DoubleMatrix} and a comparisonVal, setting an element 
	 * to true if the value in {@code dblMatrix} exactly matches {@code comparisonVal}
	 * 
	 * @param dblMatrix		{@link DoubleMatrix} to build from
	 * @param comparisonVal	Value to test for equality
	 * @return	boolean[][], same dimensions as {@code dblMatrix}
	 */
	public static final boolean[][] toLogicalMatrix (DoubleMatrix dblMatrix, double comparisonVal)
	{
		int			rows = dblMatrix.rows;
		int			columns = dblMatrix.columns;
		int			length = dblMatrix.length;
		boolean[][] result = new boolean[rows][columns];
		double[]	data = dblMatrix.data;
		int			row = 0;
		int			col = 0;
		
		for (int i = 0; i < length; i++)
		{
			result[row][col] = data[i] == comparisonVal;
			++row;
			
			if (row >= rows)
			{
				row = 0;
				++col;
			}
		}
		
		return result;
	}
	
	
	/**
	 * Make a logical matrix from a {@code double[][]} and a comparisonVal, setting an element 
	 * to true if the value in {@code dblMatrix} exactly matches {@code comparisonVal}
	 * 
	 * @param dblMatrix		{@code double[][]} to build from
	 * @param comparisonVal	Value to test for equality
	 * @return	boolean[][], same dimensions as {@code dblMatrix}
	 */
	public static final boolean[][] toLogicalMatrix (double[][] dblMatrix, double comparisonVal)
	{
		int			rows = dblMatrix.length;
		int			columns = dblMatrix[0].length;
		boolean[][] result = new boolean[rows][columns];
		
		for (int row = 0; row < rows; ++row)
		{
			double[]	data = dblMatrix[row];
			boolean[]	results = result[row];
			
			for (int col = 0; col < columns; ++col)
			{
				results[col] = data[col] == comparisonVal;
			}
		}
		
		return result;
	}
	
	
	/**
	 * Report the number of times a value in a {@code double[][]} exactly matches {@code comparisonVal}
	 * 
	 * @param dblMatrix		{@code double[][]} to process
	 * @param comparisonVal	Value to test for match
	 * @return	int, number of matches.  Therefore {@code >= 0}
	 */
	public static final int countMatches (double[][] dblMatrix, double comparisonVal)
	{
		int	rows = dblMatrix.length;
		int	columns = dblMatrix[0].length;
		int	count = 0;
		
		for (int row = 0; row < rows; ++row)
		{
			double[]	data = dblMatrix[row];
			
			for (int col = 0; col < columns; ++col)
			{
				if (data[col] == comparisonVal)
					++count;
			}
		}
		
		return count;
	}
	
	
	/**
	 * Return true if any element in {@code testMatrix} is true
	 * 
	 * @param testMatrix	boolean[][] to iterate over
	 * @return	True if at least one element is true
	 */
	public static final boolean hasTrue (boolean [][] testMatrix)
	{
		int	rows = testMatrix.length;
		if (rows == 0)
			return false;
		
		int	columns = testMatrix[0].length;
		if (columns == 0)
			return false;
		
		for (int row = 0; row < rows; ++row)
		{
			boolean[]	testArray = testMatrix[row];
			
			for (int col = 0; col < columns; ++col)
			{
				if (testArray[col])
					return true;
			}
		}
		
		return false;
	}
	
	
	/**
	 * Generate a SubsetOfTrajectories {@link Map} from the input data
	 * 
	 * @param logicalMatrixOfSubsetOfTrajectories	Matrix of trajectories and their test values
	 * @param testValue	Value to test against
	 * @param trajSet	{@link List} of trajectory info
	 * @return	{@link Map} from "trajectory number" to trajectory values
	 */
	private static final Map<Integer, double[][]> getSubsetOfTrajectories (double[][] logicalMatrixOfSubsetOfTrajectories, double testValue, 
																			List<double[][]> trajSet)
	{
		Map<Integer, double[][]>	subsetOfTrajectories = new HashMap<Integer, double[][]> ();
		int							numRows = logicalMatrixOfSubsetOfTrajectories.length;
		
		for (int row = 0; row < numRows; ++row)
		{
			double[]	theRow = logicalMatrixOfSubsetOfTrajectories[row];
			int			numCols = theRow.length;
			
			for (int col = 0; col < numCols; ++col)
			{
				if (theRow[col] == testValue)
				{
					int	traj = (col * numRows) + row;
					subsetOfTrajectories.put (traj, trajSet.get (traj));
				}
			}
		}
		return subsetOfTrajectories;
	}
	
	
	/**
	 * 
	 * @param logicalMatrixOfSubsetOfTrajectories	Matrix of which trajectories matched the test value
	 * @param trajSet	{@link List} of trajectory info
	 * @return	{@link Map} from "trajectory number" to trajectory values
	 */
	private static final Map<Integer, double[][]> getSubsetOfTrajectories (boolean[][] logicalMatrixOfSubsetOfTrajectories, List<double[][]> trajSet)
	{
		Map<Integer, double[][]>	subsetOfTrajectories = new HashMap<Integer, double[][]> ();
		int							numRows = logicalMatrixOfSubsetOfTrajectories.length;
		
		for (int row = 0; row < numRows; ++row)
		{
			boolean[]	theRow = logicalMatrixOfSubsetOfTrajectories[row];
			int			numCols = theRow.length;
			
			for (int col = 0; col < numCols; ++col)
			{
				if (theRow[col])
				{
					int	traj = (col * numRows) + row;
					subsetOfTrajectories.put (traj, trajSet.get (traj));
				}
			}

		}
		return subsetOfTrajectories;
	}
	
	
	/**
	 * Do a synchronized put into a {@link Map}
	 * 
	 * @param theMap
	 * @param key
	 * @param value
	 */
	public static final <T, U> void safePut (Map<T, U> theMap, T key, U value)
	{
		synchronized (theMap)
		{
			theMap.put (key, value);
		}
	}
	
	
	/**
	 * Place data into {@link List} in thread safe manner
	 * 
	 * @param theList
	 * @param pos
	 * @param theValue
	 */
	public static final <T> void safeSet (List<T> theList, int pos, T theValue)
	{
		synchronized (theList)
		{
			theList.set (pos, theValue);
		}
	}
	
	
	/**
	 * Create a thread safe {@link List} that contains {@code numNulls + 1} null values
	 * 
	 * @param numNulls	Size + 1 of the list when done
	 * @return	{@link List}, never null, empty only if {@code numNulls} < 0
	 */
	public static final <T> List<T> createListWithNulls (int numNulls)
	{
		List<T>	results = Collections.synchronizedList (new ArrayList<> (numNulls + 1));
		
		for (int i = 0; i <= numNulls; ++i)
			results.add (null);
		
		return results;
	}
	
}
