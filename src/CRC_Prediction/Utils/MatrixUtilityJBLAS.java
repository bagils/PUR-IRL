
package CRC_Prediction.Utils;


import java.util.*;
import java.util.Map.Entry;
import java.util.stream.IntStream;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * 
 * @author John Kalantari
 * Copyright 2019, Mayo Foundation for Medical Education and Research
 */
public class MatrixUtilityJBLAS
{
	
	public static final double[][] minComparisonX (double[][] dbl2dArray, double minVal)
	{
		double minimalValued2DArray[][] = new double[dbl2dArray.length][dbl2dArray[0].length];
		double actualMin;
		for (int i = 0; i < dbl2dArray.length; i++)
		{
			for (int j = 0; j < dbl2dArray[i].length; j++)
			{
				actualMin = Math.min (minVal, dbl2dArray[i][j]);
				minimalValued2DArray[i][j] = actualMin;
			}
			
		}
		return minimalValued2DArray;
	}
	
	
	public static final double[][] maxComparisonX (double[][] dbl2dArray, double maxVal)
	{
		double maximalValued2DArray[][] = new double[dbl2dArray.length][dbl2dArray[0].length];
		double actualMax;
		for (int i = 0; i < dbl2dArray.length; i++)
		{
			for (int j = 0; j < dbl2dArray[i].length; j++)
			{
				actualMax = Math.max (maxVal, dbl2dArray[i][j]);
				maximalValued2DArray[i][j] = actualMax;
			}
		}
		return maximalValued2DArray;
	}
	
	
	/**
	 * Ensure 2d array element values are within specified lower and upper bounds
	 * 
	 * @param dbl2dArray
	 * @param lowerBound
	 * @param upperBound
	 * @return
	 */
	public static final double[][] withinBounds (double[][] dbl2dArray, double lowerBound,
			double upperBound)
	{
		double withinBounds2DArray[][] = new double[dbl2dArray.length][dbl2dArray[0].length];
		for (int i = 0; i < dbl2dArray.length; i++)
		{
			for (int j = 0; j < dbl2dArray[i].length; j++)
			{
				
				if (Double.compare (upperBound, dbl2dArray[i][j]) < 0)
				{
					withinBounds2DArray[i][j] = upperBound;
					if (Double.compare (lowerBound, withinBounds2DArray[i][j]) > 0)
					{
						withinBounds2DArray[i][j] = lowerBound;
					}
				}
				else
				{
					withinBounds2DArray[i][j] = dbl2dArray[i][j];
					if (Double.compare (lowerBound, withinBounds2DArray[i][j]) > 0)
					{
						withinBounds2DArray[i][j] = lowerBound;
					}
				}
				
			}
		}
		return withinBounds2DArray;
	}

	
	public static final DoubleMatrix toLogicalMatrix (DoubleMatrix dblMatrix, double comparisonVal)
	{
		int rows = dblMatrix.rows;
		int columns = dblMatrix.columns;
		int length = dblMatrix.length;
		DoubleMatrix result = new DoubleMatrix (rows, columns);
		
		for (int i = 0; i < length; i++)
			result.put (i, dblMatrix.get (i) == comparisonVal ? 1.0 : 0.0);
		
		return result;
		
	}
	
	
	public static final int[][] equalityComparisonINT (int[][] int2dArray, int comparisonVal)
	{
		int logical2DArray[][] = new int[int2dArray.length][int2dArray[0].length];
		for (int i = 0; i < int2dArray.length; i++)
		{
			for (int j = 0; j < int2dArray[i].length; j++)
			{
				if (comparisonVal == int2dArray[i][j])
				{
					logical2DArray[i][j] = 1;
				}
				else
				{
					logical2DArray[i][j] = 0;
				}
			}
		}
		return logical2DArray;
	}
	
	
	/**
	 * Returns logical matrix indicating whether each element is LESS than specified value.
	 * 
	 * @param dbl2dArray
	 * @param comparisonVal
	 * @return
	 */
	public static final double[][] lessEqualityComparison (double[][] dbl2dArray, double comparisonVal)
	{
		// double logical2DArray [][] = new double [dbl2dArray.length][dbl2dArray[0].length];
		// for (int i = 0; i < dbl2dArray.length; i++) {
		// for(int j=0; j<dbl2dArray[i].length; j++) {
		// if(Double.compare(dbl2dArray[i][j], comparisonVal) <0 ) {
		// //if(dbl2dArray[i][j] < comparisonVal) {
		// logical2DArray[i][j]= 1.0;
		// }
		// else {
		// logical2DArray[i][j]=0.0;
		// }
		//
		// }
		// }
		DoubleMatrix dblMatrix = new DoubleMatrix (dbl2dArray);
		int rows = dblMatrix.rows;
		int cols = dblMatrix.columns;
		int length = dblMatrix.length;
		DoubleMatrix logicalMatrix = new DoubleMatrix (rows, cols);
		for (int i = 0; i < length; i++)
			logicalMatrix.put (i, dblMatrix.get (i) < comparisonVal ? 1.0 : 0.0);
		
		return logicalMatrix.toArray2 ();
		// return logical2DArray;
	}
	
	
	/**
	 * Returns logical matrix indicating whether each element is GREATER than specified value.
	 * 
	 * @param dbl2dArray
	 * @param comparisonVal
	 * @return
	 */
	public static final double[][] greaterEqualityComparison (double[][] dbl2dArray, double comparisonVal)
	{
		
		//
		// double logical2DArray [][] = new double [dbl2dArray.length][dbl2dArray[0].length];
		// for (int i = 0; i < dbl2dArray.length; i++) {
		// for(int j=0; j<dbl2dArray[i].length; j++) {
		// if(Double.compare(dbl2dArray[i][j], comparisonVal) >0) {
		// // if(dbl2dArray[i][j] > comparisonVal) {
		// logical2DArray[i][j]= 1.0;
		// }
		// else {
		// logical2DArray[i][j]=0.0;
		// }
		//
		// }
		// }
		//
		DoubleMatrix dblMatrix = new DoubleMatrix (dbl2dArray);
		int rows = dblMatrix.rows;
		int cols = dblMatrix.columns;
		int length = dblMatrix.length;
		DoubleMatrix logicalMatrix = new DoubleMatrix (rows, cols);
		for (int i = 0; i < length; i++)
			logicalMatrix.put (i, dblMatrix.get (i) > comparisonVal ? 1.0 : 0.0);
		
		return logicalMatrix.toArray2 ();
		// return logical2DArray;
	}
	
	
	/**
	 * Returns the sum along rows.
	 * Compute the sum for each column across all rows.
	 * 
	 * @param mat
	 * @return double[] containing sum for each column
	 */
	public static final double[] sumPerColumn (double[][] mat)
	{
		// double [] sumValsAcrossColumns = new double [mat[0].length]; //a row is of length =
		// mat[0].length= num columns
		// for (int c=0; c< mat[0].length ; c++) { //iterate by column
		// double sum = mat[0][c];
		// for (int i = 1; i < mat.length; i++) { //for each column iterate through all rows to
		// calculate sum; start at i=1 since we begin with sum=mat[i=0][c]
		// sum = sum+ mat[i][c];
		// }
		// sumValsAcrossColumns[c]= sum;
		// }
		DoubleMatrix dblmatrix = new DoubleMatrix (mat);
		DoubleMatrix columnSums = dblmatrix.columnSums ();
		return columnSums.toArray ();
		// return sumValsAcrossColumns;
	}
	
	
	public static final int[] sumPerColumnINT (int[][] mat)
	{
		int[] sumValsAcrossColumns = new int[mat[0].length]; // a row is of length = mat[0].length=
																// num columns
		for (int c = 0; c < mat[0].length; c++)
		{ // iterate by column
			int sum = mat[0][c];
			for (int i = 1; i < mat.length; i++)
			{ // for each column iterate through all rows to calculate sum
				sum = sum + mat[i][c];
			}
			sumValsAcrossColumns[c] = sum;
		}
		return sumValsAcrossColumns;
	}
	
	
	/**
	 * returns the sum along columns (sum for each row along all columns)
	 * 
	 * @param mat
	 * @return double[][] (column matrix) containing sum for each row
	 */
	public static final double[][] sumPerRow (double[][] mat)
	{
		// double [][] sumValForEachRow = new double [mat.length][1]; //num of rows
		// for (int r=0; r< mat.length ; r++) { //iterate by row
		// double sum = mat[r][0];
		// for (int i = 1; i < mat[0].length; i++) { //for each row iterate through all cols to
		// calculate sum
		// sum = sum+ mat[r][i];
		// }
		// sumValForEachRow[r][0]= sum;
		// }
		DoubleMatrix sumValsPerRow = new DoubleMatrix (mat).rowSums ();
		return sumValsPerRow.toArray2 ();
		// return sumValForEachRow;
	}
	
	
	/**
	 * maximum value across columns in matrix
	 * 
	 * @param mat
	 * @return
	 */
	public static final double[] maxPerColumn (double[][] mat)
	{
		// double [] maxValsAcrossColumns = new double [mat[0].length]; //a row is of length =
		// mat[0].length= num columns
		// for (int c=0; c< mat[0].length ; c++) { //iterate by column
		// double max = mat[0][c];
		// for (int i = 1; i < mat.length; i++) { //for each column iterate through all rows to find
		// highest element value
		// if(Double.compare(mat[i][c], max) >0) {
		// //if (mat[i][c] > max)
		// max = mat[i][c];
		// }
		// }
		// maxValsAcrossColumns[c]= max;
		// }
		
		DoubleMatrix dblMatrix = new DoubleMatrix (mat);
		DoubleMatrix colMaxVals = dblMatrix.columnMaxs ();
		return colMaxVals.toArray ();
		// return maxValsAcrossColumns;
	}
	
	
	public static final double matrixMaximum (double[][] mat)
	{ // GTD, get each row once
// 		StopWatch darrayWatch = StopWatch.createStarted ();
// 		
// 		double max = 0.0;
// 		for (int c = 0; c < mat[0].length; c++)
// 		{ // iterate by column
// 			for (int i = 0; i < mat.length; i++)
// 			{ // for each column iterate through all rows to find highest element value
// 				if (Double.compare (mat[i][c], max) > 0)
// 				{
// 					// if (mat[i][c] > max)
// 					max = mat[i][c];
// 				}
// 			}
// 		}
// 		darrayWatch.stop ();
// 		System.out.println ("Array max time = " + darrayWatch.getTime () + ", value = " + max);
// 		
// 		StopWatch dRowWatch = StopWatch.createStarted ();
// 		
		double	rMax = 0.0;
		int		rows = mat.length;
		int		cols = mat[0].length;
		
		// iterate by row
		for (int i = 0; i < rows; ++i)
		{
			double[]	row = mat[i];
			
			 // for each row iterate through all columns to find highest element value
			for (int j = 0; j < cols; ++j)
			{
				double	value = row[j];
				if (Double.compare (value, rMax) > 0)
					rMax = value;
			}
		}
// 		dRowWatch.stop ();
// 		System.out.println ("Row max time = " + dRowWatch.getTime () + ", value = " + rMax);
// 		
// 		StopWatch dmatrixWatch = StopWatch.createStarted ();
// 		DoubleMatrix dblMat = new DoubleMatrix (mat);
// 		double dMax = dblMat.max ();
// 		// return dMax;
// 		dmatrixWatch.stop ();
// 		System.out.println ("DBlmatrix max time = " + dmatrixWatch.getTime () + ", value = " + dMax);
		return rMax;
	}
	
	
	/**
	 * Multiplies the given vector and matrix using Java 8 streams.
	 *
	 * @param matrix the matrix
	 * @param vector the vector to multiply
	 *
	 * @return result after multiplication.
	 */
	public static final double[] multiplyMatrixByVectorWithStreams (final double[][] matrix, final double[] vector)
	{
		final int rows = matrix.length;
		final int columns = matrix[0].length;
		
		return IntStream.range (0, rows).mapToDouble (row -> IntStream.range (0, columns)
				.mapToDouble (col -> matrix[row][col] * vector[col]).sum ()).toArray ();
	}
	
	
	/**
	 * 
	 * 
	 * @param m1
	 * @param m2
	 * @return result after multiplication.
	 */
	public static final double[][] multiplyMatricesWithStreamsZ (double[][] m1, double[][] m2)
	{
		
		double[][] result = Arrays.stream (m1)
				.map (r -> IntStream.range (0, m2[0].length)
						.mapToDouble (i -> IntStream.range (0, m2.length)
								.mapToDouble (j -> r[j] * m2[j][i]).sum ())
						.toArray ())
				.toArray (double[][]::new);
		return result;
	}
	
	
	/**
	 * 
	 * 
	 * @param m1
	 * @param m2
	 * @return result after multiplication.
	 */
	public static final DoubleMatrix multiplyMatricesWithMatrixUtils (double[][] m1, double[][] m2)
	{
		// RealMatrix m1Asrealmatrix = MatrixUtils.createRealMatrix(m1);
		// RealMatrix m2Asrealmatrix= MatrixUtils.createRealMatrix(m2);
		//
		// RealMatrix product = m1Asrealmatrix.multiply(m2Asrealmatrix);
		// return product;
		DoubleMatrix m1asDblMatrix = new DoubleMatrix (m1);
		DoubleMatrix m2asDblMatrix = new DoubleMatrix (m2);
		
		DoubleMatrix product = m1asDblMatrix.mmul (m2asDblMatrix);
//		DoubleMatrix productVersionB = m1asDblMatrix.mul (m2asDblMatrix);	// GTD Not used
		return product;
	}
	
	
	/**
	 * Multiplies the given vector and matrix using vanilla for loops.
	 *
	 * @param matrix the matrix
	 * @param vector the vector to multiply
	 *
	 * @return result after multiplication.
	 */
	public static final double[] multiplyMatrixByVectorWithForLoops (double[][] matrix, double[] vector)
	{
		int rows = matrix.length;
		int columns = matrix[0].length;
		
		double[] result = new double[rows];
		
		for (int row = 0; row < rows; ++row)
		{
			double[]	theRow = matrix[row];
			double		sum = 0;
			for (int column = 0; column < columns; column++)
			{
				sum += theRow[column] * vector[column];
			}
			result[row] = sum;
		}
		return result;
	}
	
	
	/**
	 * Create a matrix with every cell initialized to 1.0
	 * 
	 * @param nRows
	 * @param nCols
	 * @return	Matrix
	 */
	public static final double[][] createMatrixWithOnes (int nRows, int nCols)
	{
		double[][] onesMatrix = new double[nRows][nCols];
		
		for (int i = 0; i < nRows; i++)
		{
			double[]	theRow = onesMatrix[i];
			
			for (int j = 0; j < nCols; j++)
			{
				theRow[j] = 1.0;
			}
		}
		
		return onesMatrix;
	}
	
	
	/**
	 * Create a matrix with every cell initialized to 0.0
	 * 
	 * @param nRows
	 * @param nCols
	 * @return	Matrix
	 */
	public static final double[][] createMatrixWithZeros (int nRows, int nCols)
	{
		double[][] onesMatrix = new double[nRows][nCols];
		
		for (int i = 0; i < nRows; i++)
		{
			double[]	theRow = onesMatrix[i];
			
			for (int j = 0; j < nCols; j++)
			{
				theRow[j] = 0.0;
			}
		}
		
		return onesMatrix;
	}
	
	
	/**
	 * Create a matrix with every cell initialized to {@link Double#NaN}
	 * 
	 * @param nRows
	 * @param nCols
	 * @return	Matrix
	 */
	public static final double[][] createMatrixWithNANS (int nRows, int nCols)
	{
		double[][] nansMatrix = new double[nRows][nCols];
		
		for (int i = 0; i < nRows; i++)
		{
			double[]	theRow = nansMatrix[i];
			
			for (int j = 0; j < nCols; j++)
			{
				theRow[j] = Double.NaN;
			}
		}
		
		return nansMatrix;
	}
	
	
	/**
	 * Create a matrix with every cell initialized to {@link Double#NaN}
	 * 
	 * @param nRows
	 * @param nCols
	 * @return	Matrix
	 */
	public static final DoubleMatrix createRealMatrixWithNANS (int nRows, int nCols)
	{
		// RealMatrix nansMatrix = MatrixUtils.createRealMatrix(nRows,nCols);
		// for (int i = 0; i < nRows; i++) {
		// for (int j = 0; j < nCols; j++) {
		// nansMatrix.setEntry(i,j,Double.NaN);
		// }
		// }
		// return nansMatrix;
		DoubleMatrix nansMatrix = DoubleMatrix.zeros (nRows, nCols);
		for (int i = 0; i < nRows; i++)
		{
			for (int j = 0; j < nCols; j++)
			{
				nansMatrix.put (i, j, Double.NaN);
			}
		}
		return nansMatrix;
	}
	
	
	/**
	 * Creates Map containing RealMatrix elements consisting of NaNs. NOTE: The set of keys for
	 * this Map begins at "1"!! Not 0 on purpose!
	 * 
	 * @param nRows
	 * @param nCols
	 * @return Map<Integer, RealMatrix>, each key corresponds to a tableIndex and its value is a
	 *         column matrix
	 */
	public static final Map<Integer, DoubleMatrix> createHashMapOfRealMatricesWithNANS (int nRows, int nCols)
	{
		Map<Integer, DoubleMatrix> nansMap = new HashMap<Integer, DoubleMatrix> ();
		for (int j = 1; j < nCols + 1; j++)
		{
			DoubleMatrix colMatrix_j = new DoubleMatrix (nRows, 1);
			for (int i = 0; i < nRows; i++)
			{
				colMatrix_j.put (i, 0, Double.NaN);
			}
			nansMap.put (j, colMatrix_j);
		}
		
		return nansMap;
	}
	
	
	/**
	 * Return matrix indices of elements with matching values, where the {i} element of the
	 * list is the row for the ith match,
	 * 
	 * @param matrix : each column of the matrix could represent a different policy; each row is one
	 *            of (numStates); a matrix element represents the action executed for a given state,
	 *            policy (pi(state_j)=action_i))
	 * @param item : specifies the action which are you looking to find a match for within the
	 *            policyMatrix
	 * @return	{@link List} holding results
	 */
	public static final List<Integer> findMatches (double[][] matrix, double item)
	{
		if (matrix == null)
		{
			System.out.println ("matrix is null!!!");
			return new ArrayList<Integer> ();
		}
		
		List<Integer> indexList = new ArrayList<Integer> ();
		// Iterate through each row of a column vector corresponding to a single policy (where the
		// element-value (at a given matrix row=state, col=policy) indicates with an integer the
		// action to be executed according to the given policy
		for (int i = 0; i < matrix.length; i++)
		{
			double[] rowArray = matrix[i];
			int colsPerRow = rowArray.length;
			for (int j = 0; j < colsPerRow; j++)
			{
				double elementVal = rowArray[j];
				if (Double.compare (elementVal, item) == 0)
				{
					// if (elementVal == item) {
					indexList.add (i); // row=state
					// indexList.add(j); //col = policy ; will keep this just in case we ever have
					// multiple policies in the policyMatrix and if we want to select only those
					// state,action pairs of a specific policy. (You will need to increment your
					// counter to reflect this if you do in fact add state,policy PAIRS in the
					// indexList)
					
					break;	// GTD: no point in continuing if found a hit for the row
				}
			}
		}
		return indexList;
	}
	
	
	/**
	 * 
	 * 
	 * @param numRows
	 * @param r
	 * @param c
	 * @return	(c - 1) * numRows + r
	 */
	public static final int getLinearIndex (int numRows, int r, int c)
	{
		int index = (c - 1) * numRows + r;
		return index;
	}
	
	
	/**
	 * 
	 * 
	 * @param numRows
	 * @param index
	 * @return	Results
	 */
	public static final int[] getMatrixIndices (int numRows, int index)
	{
		int r = ((index - 1) % numRows);
		Double cDbl = Math.floor ((index - 1) / numRows) + 1;
		int c = cDbl.intValue ();
		int[] matrixIndices = new int[] {r, c};
		return matrixIndices;
	}
	
	
	/**
	 * 
	 * 
	 * @param m1
	 * @param m2
	 * @param epsilon
	 * @return	True if every element in m1 is within epsilon of its matching element in m2
	 */
	public static final boolean compareREALMatrices (DoubleMatrix m1, DoubleMatrix m2, double epsilon)
	{
		
		boolean normIsBelowEpsilon = false;
		
		// RealMatrix m1ColMatrix = MatrixUtility.convertRealMatrixIntoColMatrix(m1);
		// RealMatrix m2ColMatrix = MatrixUtility.convertRealMatrixIntoColMatrix(m2);
		// RealMatrix diffMatrix = m1ColMatrix.subtract(m2ColMatrix);
		// //RealMatrix differenceMatrix = m1.subtract(m2);
		// //double normVal = differenceMatrix.getNorm();
		// //double normVal = diffMatrix.getNorm();
		// double normVal = MatrixUtils.createRealVector(diffMatrix.getColumn(0)).getLInfNorm();
		//
		// //System.out.println("L_normval = "+normVal+" epsilon ="+epsilon);
		// //System.out.println("norm val: "+normVal);
		// if(Double.compare(normVal, epsilon)<0) {
		// //if(normVal < epsilon) {
		// normIsBelowEpsilon = true;
		// //System.out.println(" norm is BELOW epsilon at :"+normVal);
		// }
		// return normIsBelowEpsilon;
		
		//NOTE: infinity norm = norm(A,infinity) = max(sum(abs(A')))
		//...however if we convert each input matrix (m1 and m2) into column matrices, all we need is convert difference matrix into absolute values and obtain maximum element.
		//We then compare this maximum element to the epsilon. If the maximal element in the entire matrix is less than epsilon, we are guaranteed that the rest of the matrix is also less than epsilon.
		
		DoubleMatrix m1ColMatrix = convertRealMatrixIntoColMatrix (m1);
		DoubleMatrix m2ColMatrix = convertRealMatrixIntoColMatrix (m2);
		DoubleMatrix diffMatrix = m1ColMatrix.sub (m2ColMatrix);
		
		double normVal = diffMatrix.normmax ();
		
		if (Double.compare (normVal, epsilon) < 0)
		{
			normIsBelowEpsilon = true;
			// System.out.println(" norm is BELOW epsilon at :"+normVal);
		}
		
//		boolean comparisonBoolVersionB = m1.compare (m2, epsilon);
		
		return normIsBelowEpsilon;
		
	}
	
	
	/**
	 * 
	 * 
	 * @param m1
	 * @param m2
	 * @param epsilon
	 * @return	True if every element in m1 is within epsilon of its matching element in m2
	 */
	public static final boolean compareREALMatricesAndOutputNorm (DoubleMatrix m1, DoubleMatrix m2, double epsilon)
	{
		
		boolean normIsBelowEpsilon = false;
		
		DoubleMatrix m1ColMatrix = convertRealMatrixIntoColMatrix (m1);
		DoubleMatrix m2ColMatrix = convertRealMatrixIntoColMatrix (m2);
		DoubleMatrix diffMatrix = m1ColMatrix.sub (m2ColMatrix);
		
		double normVal = diffMatrix.normmax ();
		
		if (Double.compare (normVal, epsilon) < 0)
		{
			normIsBelowEpsilon = true;
			// System.out.println(" norm is BELOW epsilon at :"+normVal);
		}
		else
		{
			System.out.println (normVal + " = norm is ABOVE epsilon =" + epsilon);
		}
		
		// boolean comparisonBoolVersionB = m1.compare(m2, epsilon);
		
		return normIsBelowEpsilon;
		
	}
	
	
	/**
	 * 
	 * 
	 * @param m1
	 * @param m2
	 * @return	True if every element in m1 is within 0.0001 of its matching element in m2
	 */
	public static final boolean areRealMatricesEqual (double[][] m1, double[][] m2)
	{
		
		// boolean normIsZero= false;
		// RealMatrix m1Asrealmatrix = MatrixUtils.createRealMatrix(m1);
		// RealMatrix m2Asrealmatrix = MatrixUtils.createRealMatrix(m2);
		//
		// RealMatrix m1ColMatrix = MatrixUtility.convertRealMatrixIntoColMatrix(m1Asrealmatrix);
		// RealMatrix m2ColMatrix = MatrixUtility.convertRealMatrixIntoColMatrix(m2Asrealmatrix);
		// RealMatrix diffMatrix = m1ColMatrix.subtract(m2ColMatrix);
		//
		// double normVal = MatrixUtils.createRealVector(diffMatrix.getColumn(0)).getLInfNorm();
		//
		// //System.out.println("norm val: "+normVal);
		// if(Double.compare(normVal, 0.0)==0) {
		// //if(normVal < epsilon) {
		// normIsZero = true;
		// }
		// return normIsZero;
		
		boolean areEqualBool = false;
		DoubleMatrix m1AsDblMatrix = new DoubleMatrix (m1);
		DoubleMatrix m2AsDblMatrix = new DoubleMatrix (m2);
		
		areEqualBool = m1AsDblMatrix.compare (m2AsDblMatrix, 0.0001);
		return areEqualBool;
		
	}
	
	
	/**
	 * 
	 * 
	 * @param m1
	 * @param m2
	 * @param epsilon
	 * @return	True if every element in m1 is within epsilon of its matching element in m2
	 */
	public static final boolean compareMatrices (double[][] m1, double[][] m2, double epsilon)
	{
		// RealMatrix m1Asrealmatrix = MatrixUtils.createRealMatrix(m1);
		// RealMatrix m2Asrealmatrix = MatrixUtils.createRealMatrix(m2);
		// boolean isNormBelowEpsilon=false;
		//
		// isNormBelowEpsilon = compareREALMatrices(m1Asrealmatrix, m2Asrealmatrix, epsilon);
		// return isNormBelowEpsilon;
		DoubleMatrix m1AsDblMatrix = new DoubleMatrix (m1);
		DoubleMatrix m2AsDblMatrix = new DoubleMatrix (m2);
		boolean isNormBelowEpsilon = false;
		isNormBelowEpsilon = compareREALMatrices (m1AsDblMatrix, m2AsDblMatrix, epsilon);
		return isNormBelowEpsilon;
	}
	
	
	/**
	 * Make a {@link List} of double[][] from a List of {@link DoubleMatrix}[][]
	 * 
	 * @param realMatrixList
	 * @return {@link List} of double[][], possibly empty, never null
	 */
	public static final List<double[][]> convertMatrixList (List<DoubleMatrix> realMatrixList)
	{
		List<double[][]> dbl2dArrayList = new ArrayList<double[][]> ();
		for (int i = 0; i < realMatrixList.size (); i++)
		{
			double[][] dbl2Darray = realMatrixList.get (i).toArray2 (); // Done: GTD : Need to make sure
																		// that we copy the object
																		// and not the pointer
			dbl2dArrayList.add (dbl2Darray);
		}
		return dbl2dArrayList;
	}
	
	
	/**
	 * Modified by GTD to just take a {@link List} of {@code double[][]}, save two array creates per element
	 * 
	 * @param realMatrixList
	 * @param sortedIndexValuesArr
	 * @return	{@link Map} from {@code sortedIndexValuesArr} value as int to matching {@code realMatrixList} double[][]
	 */
//	public static final Map<Integer, double[][]> convertMatrixListToMap (List<DoubleMatrix> realMatrixList, Double[] sortedIndexValuesArr)
	public static final Map<Integer, double[][]> convertMatrixListToMap (List<double[][]> realMatrixList, Double[] sortedIndexValuesArr)
	{
		int							numValues = sortedIndexValuesArr.length;
		Map<Integer, double[][]>	dbl2dArrayMap = new HashMap<Integer, double[][]> (numValues);
		
		System.out.println ("# of table indices in new restaurant = " + numValues);
		System.out.println ("number of table vectors in new restaurant = " + realMatrixList.size ());
		if (realMatrixList.size () != numValues)
		{
			throw new java.lang.RuntimeException ("number of new table indices (" + numValues + ") should = number of new table vectors(" + 
												  realMatrixList.size () + "!!!");
		}
		
		for (int i = 0; i < numValues; ++i)
		{
			Double		tblIndexValDBL = sortedIndexValuesArr[i];
//			double[][]	dbl2Darray = realMatrixList.get (i).toArray2 ();
			double[][]	dbl2Darray = realMatrixList.get (i);
			
			dbl2dArrayMap.put (Integer.valueOf (tblIndexValDBL.intValue ()), dbl2Darray);
		}
		
		return dbl2dArrayMap;
	}
	
	
	public static Map<Integer, double[][]> reorderHashMap (Map<Integer, DoubleMatrix> mapOfMatrices)
	{
		Map<Integer, double[][]> dbl2dArrayMap = new HashMap<Integer, double[][]> (mapOfMatrices.size ());
		System.out.println ("number of table vectors in new restaurant = " + mapOfMatrices.size ());
		
		for (Entry<Integer, DoubleMatrix> entry : mapOfMatrices.entrySet ())
		{
			dbl2dArrayMap.put (entry.getKey (), entry.getValue ().toArray2 ());
		}
		
		return dbl2dArrayMap;
		
	}
	
	
	/**
	 * JK 7.26.2019 data validated
	 * @param assgnmentMat	assigns each trajectory to a given table/reward-function<br>
	 *  tblSizesMat: 1 x numTables matrix, indicating # of trajectories assigned to each table
	 *            index-value<br>
	 *  numTablesOfSameSize : 1 x numTrajectories, indicating how many tables consists of the
	 *            same number of trajectories. i.e. numTablesOfSameSize[0][1]= number of tables
	 *            which contain 1 trajectory, numTablesOfSameSize[0][5]= number of tables which
	 *            contain 5 trajectories<br>
	 *  sizeToTableMap: 1 x numTrajectories, maps each possible table size to the set of
	 *            tables that contribute to that number.
	 * @return 
	 */
	public static final double[][] tableCounter (double[][] assgnmentMat)
	{
		// numTablesOfSameSize[0][i]= number of tables which contain i trajectories
		// tblSizesMat[0][i]: size of the ith table
		int			numTrajs = assgnmentMat.length; // # of trajectories we are modeling
		Set<Double>	uniqueTablesInRestaurant = MatrixUtilityJBLAS.countNumberUniqueElements (assgnmentMat);
		Double[]	sortedSetOfUniqueTableIndicesInRest = VectorUtility.sortSet (uniqueTablesInRestaurant);
		int			maxTableIndexVal = sortedSetOfUniqueTableIndicesInRest[sortedSetOfUniqueTableIndicesInRest.length - 1].intValue ();
		
		// double [][] tblSizesMat = MatrixUtils.createRealMatrix(1, maxTableIndexVal+1).getData();
		// double [][] tblSizesMat = MatrixUtils.createRealMatrix(1, maxTableIndexVal).getData();
		Map<Integer, Double> tblSizesMat = new HashMap<Integer, Double> ();
		
//		double[][] numTablesOfSameSizeMat = MatrixUtils.createRealMatrix (1, numTrajs + 1).getData ();
		double[][] numTablesOfSameSizeMat = MatrixUtils.createRealMatrix (1, numTrajs).getData ();
		
		Map<Integer, double[]> sizeToTableMap = new HashMap<Integer, double[]> ();
		
		for (Double tblVal : sortedSetOfUniqueTableIndicesInRest)
		{
			
			double tblIndexDBLVal = tblVal;
//			double[][] logicMatrix = MatrixUtilityJBLAS.equalityComparisonDBL (assgnmentMat, tblIndexDBLVal);
			DoubleMatrix logicMatrix = MatrixUtilityJBLAS.toLogicalMatrix (new DoubleMatrix (assgnmentMat), tblIndexDBLVal);
			
			// tblSizesMat[0][tblVal.intValue()]= MatrixUtility.sumPerColumn(logicMatrix)[0]; //the
			// total number of trajectories that have been assigned to the table with index-value = 'tblVal'
			// tblSizesMat.put(tblVal.intValue(), MatrixUtilityJBLAS.sumPerColumn(logicMatrix)[0]);
			// //the total number of trajectories that have been assigned to the table with
			// index-value = 'tblVal'
			tblSizesMat.put (tblVal.intValue (), logicMatrix.sum ()); // the total number of trajectories that have been assigned to the
																		// table with index-value = 'tblVal'
			
		}
		
		// iterate through all the table-index values (in tblSizesMat) to which  > 0 trajectories have been attributed
		// for(int tableIndexj=0; tableIndexj< maxTableIndexVal+1; tableIndexj++) {
		for (int tableIndexj = 1; tableIndexj <= maxTableIndexVal; ++tableIndexj)
		{
			// if(tblSizesMat[0][tableIndexj]>0.0) {
			if (Double.compare (tblSizesMat.get (Integer.valueOf (tableIndexj)), 0.0) > 0)
			{
				// if(tblSizesMat.get(tableIndexj)>0.0) {
				
				// for tableIndexj
				// Double sizeVal = tblSizesMat[0][tableIndexj]; //retrieve number of trajectories assigned to the table with index-value 'tableIndexj'
				Double	sizeVal = tblSizesMat.get (tableIndexj); // retrieve number of trajectories assigned to the table with index-value 'tableIndexj'
				int		sizeIntVal = sizeVal.intValue ();
				
				// increment the column in 'numTablesOfSameSizeMat' which records the number of
				// OTHER tables that also have |sizeIntVal| trajectories assigned to their table-index value
				// since we are using a double array in which element indices matter, we need to
				// ensure that numTablesOfSameSizeMat[0][0] corresponds to sizeIntVal = 1
				/*
				 * numTablesOfSameSizeMat[0][sizeIntVal] = numTablesOfSameSizeMat[0][sizeIntVal] + 1;	//increment
				 * Double numTablesOfGivenSizeDblVal = numTablesOfSameSizeMat[0][sizeIntVal];
				 */
				numTablesOfSameSizeMat[0][sizeIntVal - 1] += 1;// increment
//				Double	numTablesOfGivenSizeDblVal = numTablesOfSameSizeMat[0][sizeIntVal - 1];	// GTD Not used
				// get the CURRENT integer count of the number of tables that also have the same size
//				int		numTablesOfGivenSizeIntVal = numTablesOfGivenSizeDblVal.intValue ();	// GTD Not used
				
				// expand the size of the double[] that will store table indices
				int			lengthOfarrayOfTableIndices = 0;
				double[]	tablesOfGivenSizeArray = null;
				double[]	mapTable = sizeToTableMap.get (sizeIntVal);
				// if(sizeToTableMap.get(numTablesOfGivenSizeIntVal)!=null) {
				if (mapTable != null)
				{
					// lengthOfarrayOfTableIndices =
					// sizeToTableMap.get(numTablesOfGivenSizeIntVal).length;
					// tablesOfGivenSizeArray = new double [lengthOfarrayOfTableIndices+1];
					lengthOfarrayOfTableIndices = mapTable.length;
					tablesOfGivenSizeArray = new double[lengthOfarrayOfTableIndices + 1];
					
					System.arraycopy (mapTable, 0, tablesOfGivenSizeArray, 0, lengthOfarrayOfTableIndices);	// GTD use faster fxn
					tablesOfGivenSizeArray[lengthOfarrayOfTableIndices] = tableIndexj; // concatenate/append as additional column to row matrix
				}
				else
				{
					tablesOfGivenSizeArray = new double[1];
					tablesOfGivenSizeArray[0] = tableIndexj; // concatenate/append as additional column to row matrix
				}
				
				// sizeToTableMap.put(numTablesOfGivenSizeIntVal, tablesOfGivenSizeArray); // set
				// this row vector for tables of size = numTablesOfGivenSizeIntVal
				sizeToTableMap.put (sizeIntVal, tablesOfGivenSizeArray); // set this row vector for tables of size = numTablesOfGivenSizeIntVal
			}
		}
		return numTablesOfSameSizeMat;
		
	}
	
	
	/**
	 * 
	 * 
	 * @param src
	 * @return	A brand new double[][], whose values all match {@code src}'s values, but with whom 
	 * no arrays are shared
	 */
	public static final double[][] deepCopy (double[][] src)
	{
		int			numRows = src.length;
		double[][]	results =  new double[numRows][];
		
		for (int i = 0; i < numRows; ++i)
		{
			double[]	row = src[i];
			
			results[i] = Arrays.copyOf (row, row.length);
		}
		
		return results;
//		double[][] copiedMatrix = new DoubleMatrix (src).dup ().toArray2 ();
//		return copiedMatrix;
		
		// double [][] dest = new double [src.length][src[0].length];
		// for (int i = 0; i < src.length; i++) {
		// System.arraycopy(src[i], 0, dest[i], 0, src[0].length);
		// }
		// return dest;
	}
	
	
	/**
	 * 
	 * 
	 * @param rMatrix
	 * @return	A column matrix version of {@code rMatrix}
	 */
	public static final DoubleMatrix convertRealMatrixIntoColMatrix (DoubleMatrix rMatrix)
	{
		// int numRows = rMatrix.getRowDimension();
		// int numCols = rMatrix.getColumnDimension();
		// RealMatrix colMatrix = MatrixUtils.createRealMatrix(numRows*numCols, 1);
		// for (int i=0; i< numCols; i++) {
		// colMatrix.setSubMatrix(rMatrix.getColumnMatrix(i).getData(), i*numRows, 0);
		// }
		// return colMatrix;
		double[] colArray = rMatrix.toArray ();
		DoubleMatrix colMatrix = new DoubleMatrix (colArray);
		return colMatrix;
	}
	
	
	/**
	 * Assuming all the double[][] in {@code matrixList} are the same size, build a single 
	 * {@link DoubleMatrix} with the same number of columns as the arrays in {@code matrixList}, 
	 * and fill it in from first to last 
	 * 
	 * @param matrixList
	 * @return	{@link DoubleMatrix} holding contents of {@code matrixList}
	 */
	public static final DoubleMatrix convertMultiDimMatrixList (List<double[][]> matrixList)
	{
		int numSubMatrices = matrixList.size ();
		double[][]	first = matrixList.get (0);
		int numRowsPerSubmatrix = first.length;
		int numColsPerSubmatrix = first[0].length; // # columns in 1st row of 1st sub-matrix in matrixList
		int totalNumRows = numSubMatrices * numRowsPerSubmatrix; // assumes each double[][] in
																	// matrixList is a row/block
																	// matrix duplicate of same
																	// dimension, stacked vertically
																	// ( i.e. if matrixList is of
																	// size=4, then total num rows
																	// is 4*numRowsPerSubmatrix,
																	// total num cols is same as
																	// numColsPerSubmatrix
		RealMatrix multiDimMatrix = MatrixUtils.createRealMatrix (totalNumRows, numColsPerSubmatrix);
		for (int i = 0; i < numSubMatrices; i++)
		{
			multiDimMatrix.setSubMatrix (matrixList.get (i), i * numRowsPerSubmatrix, 0);
		}
		DoubleMatrix multiDimDblMatrix = new DoubleMatrix (multiDimMatrix.getData ());
		return multiDimDblMatrix;
	}
	
	
	/**
	 * Given a {@link Map} of size {@code x}, whose keys range for 0 to {@code x - 1}, build a 
	 * {@link DoubleMatrix} with all the elements in key increasing order
	 * 
	 * @param matrixMap
	 * @return	{@link DoubleMatrix} with elements from {@code matrixMap}
	 */
	public static final DoubleMatrix convertMultiDimMatrixMap (Map<Integer, double[][]> matrixMap)
	{
		int numSubMatrices = matrixMap.size ();
		double[][]	first = matrixMap.get (Integer.valueOf (0));
		int numRowsPerSubmatrix = first.length;
		int numColsPerSubmatrix = first[0].length; // # columns in 1st row of 1st sub-matrix in matrixList
		int totalNumRows = numSubMatrices * numRowsPerSubmatrix; // assumes each double[][] in
																	// Map is a row/block
																	// matrix duplicate of same
																	// dimension, stacked vertically
																	// ( i.e. if matrixList is of
																	// size=4, then total num rows
																	// is 4*numRowsPerSubmatrix,
																	// total num cols is same as
																	// numColsPerSubmatrix
		RealMatrix multiDimMatrix = MatrixUtils.createRealMatrix (totalNumRows, numColsPerSubmatrix);
		for (int i = 0; i < numSubMatrices; i++)
		{
			Integer iAsInteger = Integer.valueOf (i);
			multiDimMatrix.setSubMatrix (matrixMap.get (iAsInteger), i * numRowsPerSubmatrix, 0);
		}
		DoubleMatrix multiDimDblMatrix = new DoubleMatrix (multiDimMatrix.getData ());
		return multiDimDblMatrix;
	}
	
	
	/**
	 * Given a {@link Map}, build a {@link DoubleMatrix} with all the elements in key increasing order, 
	 * With all columns the length of an entry in {@code mapOf2dArrays}
	 * 
	 * @param mapOf2dArrays
	 * @return	{@link DoubleMatrix} with elements from {@code mapOf2dArrays}
	 */
	public static final DoubleMatrix convertHashMapToRealMatrixRows (Map<Integer, double[][]> mapOf2dArrays)
	{
//		if (mapOf2dArrays.containsKey (0))	// GTD just get first key
//		{
			SortedMap<Integer, double[][]>	sorted = new TreeMap<> (mapOf2dArrays);
			double[][]	first = sorted.get (sorted.keySet ().iterator ().next ());
			int numRowsPerSubmatrix = first.length;
			int numColsPerSubmatrix = first[0].length; // # columns in 1st row of 1st sub-matrix in matrixList
			int totalNumRows = mapOf2dArrays.size () * numRowsPerSubmatrix; // assumes each
																			// double[][] in Map
																			// is a row/block matrix
																			// duplicate of same
																			// dimension, stacked
																			// vertically ( i.e. if
																			// Map is of size=4,
																			// then total num rows
																			// is
																			// 4*numRowsPerSubmatrix,
																			// total num cols is
																			// same as
																			// numColsPerSubmatrix
			RealMatrix	multiDimMatrix = MatrixUtils.createRealMatrix (totalNumRows, numColsPerSubmatrix);
			int			pos = 0;
			
			for (Entry<Integer, double[][]> entry : sorted.entrySet ())
			{
				multiDimMatrix.setSubMatrix (entry.getValue (), pos, 0);
				pos += numRowsPerSubmatrix;
			}
			
			DoubleMatrix multiDimDBLMatrix = new DoubleMatrix (multiDimMatrix.getData ());
			return multiDimDBLMatrix;
//		}
//		else
//		{
//			throw new java.lang.RuntimeException ("This Map does not contain the Key used for subsequent initialization");
//			
//		}
	}
	
	
	/**
	 * Given a {@link Map}, build a {@link DoubleMatrix} with all the elements in key increasing order, 
	 * With all rows the length of an entry in {@code mapOf2dArrays}
	 * 
	 * @param mapOf2dArrays
	 * @return	{@link DoubleMatrix} with elements from {@code mapOf2dArrays}
	 */
	public static final DoubleMatrix convertHashMapToRealMatrixCols (Map<Integer, double[][]> mapOf2dArrays)
	{
//		if (mapOf2dArrays.containsKey (0))	// GTD just get first key
//		{
			SortedMap<Integer, double[][]>	sorted = new TreeMap<> (mapOf2dArrays);
			double[][]	first = sorted.get (sorted.keySet ().iterator ().next ());
			int numRowsPerSubmatrix = first.length;
			int numColsPerSubmatrix = first[0].length; // # columns in 1st row of 1st sub-matrix in matrixList
			int totalNumCols = mapOf2dArrays.size () * numColsPerSubmatrix;
			RealMatrix	multiDimMatrix = MatrixUtils.createRealMatrix (numRowsPerSubmatrix, totalNumCols);
			int			pos = 0;
			
			for (Entry<Integer, double[][]> entry : sorted.entrySet ())
			{
				multiDimMatrix.setSubMatrix (entry.getValue (), 0, pos);
				pos += numColsPerSubmatrix;
			}
			
			DoubleMatrix multiDimDBLMatrix = new DoubleMatrix (multiDimMatrix.getData ());
			return multiDimDBLMatrix;
//		}
//		else
//		{
//			throw new java.lang.RuntimeException (
//					"This Map does not contain the Key used for subsequent initialization");
//			
//		}
	}
	
	
	/**
	 * transpose the matrix
	 * 
	 * @param mat
	 * @return	New transposed matrix
	 */
	public static final double[][] transpose (double[][] mat)
	{
		int	numCols = mat.length;
		int	numRows = mat[0].length;
		
		double[][]	a = new double[numRows][];
		double[]	theRow;
		
		for (int i = 0; i < numRows; ++i)
		{
			a[i] = theRow = new double[numCols];
			for (int j = 0; j < numCols; j++)
			{
				theRow[j] = mat[j][i];
			}
		}
		
		return a;
	}
	
	
	/**
	 * transpose the matrix
	 * 
	 * @param mat
	 * @return	New transposed matrix
	 */
	public static final int[][] transpose (int[][] mat)
	{
		int	numCols = mat.length;
		int	numRows = mat[0].length;
		
		int[][]	a = new int[numRows][];
		int[]	theRow;
		
		for (int i = 0; i < numRows; ++i)
		{
			a[i] = theRow = new int[numCols];
			for (int j = 0; j < numCols; j++)
			{
				theRow[j] = mat[j][i];
			}
		}
		
		return a;
	}
	
	
	/**
	 * Returns the exponential for each element in provided double[][]
	 * 
	 * @param dbl2dArray
	 * @return	double[][] with values being the exp of the values in {@code dbl2dArray}
	 */
	public static final double[][] exp (double[][] dbl2dArray)
	{
// 		StopWatch dblWatch = StopWatch.createStarted ();
		DoubleMatrix	dblMatrix = new DoubleMatrix (dbl2dArray);
		double[][]		expMatrix = MatrixFunctions.expi (dblMatrix).toArray2 ();
		return expMatrix;
// 		dblWatch.stop ();
// 		System.out.println ("Time to exp with dblmatrix = " + dblWatch.getTime ());
// 		
// 		StopWatch arrayWatch = StopWatch.createStarted ();	// GTD Re-do test
// 		
// 		int			numRows = dbl2dArray.length;
// 		int			numCols = dbl2dArray[0].length;
// 		double[][]	exp2DArray = new double[numRows][];
// 		double[]	row, copyRow;
// 		
// 		for (int i = 0; i < numRows; ++i)
// 		{
// 			exp2DArray[i] = row = new double[numCols];
// 			copyRow = dbl2dArray[i];
// 			
// 			for (int j = 0; j < numCols; ++j)
// 			{
// 				row[j] = Math.exp (copyRow[j]);
// 			}
// 		}
// 		arrayWatch.stop ();
// 		System.out.println ("Time to exp with arraymat=" + arrayWatch.getTime ());
// 		return exp2DArray;
	}
	
	
	/**
	 * Returns the natural logarithm for each element in inputted double[][]
	 * 
	 * @param dbl2dArray
	 * @return
	 */
	public static final double[][] log (double[][] dbl2dArray)
	{
		
		// StopWatch dblWatch = StopWatch.createStarted();
		DoubleMatrix dblMatrix = new DoubleMatrix (dbl2dArray);
		double[][] logMatr = MatrixFunctions.logi (dblMatrix).toArray2 ();
		return logMatr;
		// dblWatch.stop();
		// System.out.println("Time to log with dblmatrix ="+dblWatch.getTime());
		
		// StopWatch arrayWatch = StopWatch.createStarted();
		// double log2DArray [][] = new double [dbl2dArray.length][dbl2dArray[0].length];
		// for (int i = 0; i < dbl2dArray.length; i++) {
		// for(int j=0; j<dbl2dArray[i].length; j++) {
		// log2DArray[i][j]= Math.log(dbl2dArray[i][j]);
		// }
		// }
		// arrayWatch.stop();
		// System.out.println("Time to log with arraymat="+arrayWatch.getTime());
		//
		// return log2DArray;
	}
	
	
	public static final double[][] scalarMultiplication (double[][] dbl2dArray, double scalar)
	{
		
		// StopWatch dblMatrixWatch = StopWatch.createStarted();
		DoubleMatrix origMatrix = new DoubleMatrix (dbl2dArray);
		DoubleMatrix prodMatrix = origMatrix.mul (scalar);
		return prodMatrix.toArray2 ();
		// dblMatrixWatch.stop();
		// System.out.println("dblMatrix scalar mult="+dblMatrixWatch.getTime());
		
		// StopWatch arrayWatch = StopWatch.createStarted();
		// double result2DArray [][] = new double [dbl2dArray.length][dbl2dArray[0].length];
		// for (int i = 0; i < dbl2dArray.length; i++) {
		// for(int j=0; j<dbl2dArray[i].length; j++) {
		// result2DArray[i][j]= dbl2dArray[i][j]*scalar;
		// }
		// }
		// arrayWatch.stop();
		// System.out.println("array scalr multip="+arrayWatch.getTime());
		// return result2DArray;
	}
	
	
	public static final DoubleMatrix expandColumnMatrix (DoubleMatrix colMatrix, int numFinalRows,
			int numFinalCols)
	{
		
		// RealMatrix expandedMatrix = MatrixUtils.createRealMatrix(numFinalRows, numFinalCols);
		// for(int c=0; c< numFinalCols; c++) {
		// expandedMatrix.setColumnMatrix(c, colMatrix);
		// }
		// return expandedMatrix;
		DoubleMatrix expandedMatrix = new DoubleMatrix (numFinalRows, numFinalCols);
		for (int c = 0; c < numFinalCols; c++)
		{
			expandedMatrix.putColumn (c, colMatrix);
		}
		return expandedMatrix;
	}
	
	
	public static final DoubleMatrix expandRowMatrix (DoubleMatrix rowMatrix, int numFinalRows,
			int numFinalCols)
	{
		
		// RealMatrix expandedMatrix = MatrixUtils.createRealMatrix(numFinalRows, numFinalCols);
		// for(int r=0; r< numFinalRows; r++) {
		// expandedMatrix.setRowMatrix(r, rowMatrix);
		// }
		// return expandedMatrix;
		DoubleMatrix expandedMatrix = new DoubleMatrix (numFinalRows, numFinalCols);
		for (int r = 0; r < numFinalRows; r++)
		{
			expandedMatrix.putRow (r, rowMatrix);
		}
		return expandedMatrix;
	}
	
	
	/*
	 * public static final <T> T[][] deepCopyMatrixX(T[][] matrix) {
	 * return java.util.Arrays.stream(matrix).map(el -> el.clone()).toArray($ -> matrix.clone());
	 * }
	 */
	/*
	 * public static final <T> ArrayList<T[][]> deepCopyArrayListY(ArrayList<T[][]> arrListToCopy){
	 * ArrayList<T[][]> clonedList = new ArrayList<T[][]>(arrListToCopy.size());
	 * for(int i=0; i< arrListToCopy.size(); i++) {
	 * clonedList.set(i, deepCopyMatrix(arrListToCopy.get(i)));
	 * }
	 * return clonedList;
	 * }
	 */
	public static final ArrayList<double[][]> deepCopyDBLArrayList (ArrayList<double[][]> arrListToCopy)
	{
		ArrayList<double[][]> clonedList = new ArrayList<double[][]> (arrListToCopy.size ());
		for (int i = 0; i < arrListToCopy.size (); i++)
		{
			clonedList.set (i, deepCopy (arrListToCopy.get (i)));
		}
		return clonedList;
	}
	
	
	public static final Set<Double> countNumberUniqueElements (double[][] dbl2darray)
	{
//		System.out.print ("countNumberUniqueElements called with ");
//		System.out.print (Integer.toString (dbl2darray.length));
//		System.out.print (" by ");
//		System.out.print (Integer.toString (dbl2darray[0].length));
//		System.out.println (" array");
		Set<Double> alreadyPresent = new HashSet<Double> ();
		
		for (double[] nextElem : dbl2darray)
		{
			alreadyPresent.add (nextElem[0]);
		}
		
//		System.out.print ("countNumberUniqueElements array had ");
//		System.out.print (Integer.toString (alreadyPresent.size ()));
//		System.out.println (" elements");
		
		return alreadyPresent;
	}
	
	
	public static final DoubleMatrix elementwiseMultiply (DoubleMatrix a, DoubleMatrix b)
	{
		
		// RealVector rowProduct;
		// RealMatrix matrixProduct = MatrixUtils.createRealMatrix(a.getRowDimension(),
		// a.getColumnDimension());
		// for (int i =0; i< a.getRowDimension(); i++) {
		// rowProduct = a.getRowVector(i).ebeMultiply(b.getRowVector(i));
		// matrixProduct.setRowVector(i, rowProduct);
		// }
		// return matrixProduct;
		DoubleMatrix matrixProduct = a.mul (b); // element-wise multiplication
		return matrixProduct;
		
	}
	
	
	public static final DoubleMatrix elementwiseMultiplicationByColumnVector (DoubleMatrix a,
			DoubleMatrix b)
	{
		// RealVector colProduct;
		// RealMatrix matrixProduct = MatrixUtils.createRealMatrix(a.getRowDimension(),
		// a.getColumnDimension());
		// for (int i =0; i< a.getColumnDimension(); i++) {
		// colProduct = a.getColumnVector(i).ebeMultiply(b);
		// matrixProduct.setColumnVector(i, colProduct);
		// }
		// return matrixProduct;
		DoubleMatrix matrixProduct = a.mulColumnVector (b);
		return matrixProduct;
		
	}
	
	
	public static final DoubleMatrix elementwiseDivisionByColumnVector (DoubleMatrix a, DoubleMatrix b)
	{
		// RealVector colQuotient;
		// RealMatrix matrixQuotient = MatrixUtils.createRealMatrix(a.getRowDimension(),
		// a.getColumnDimension());
		// for (int i =0; i< a.getColumnDimension(); i++) {
		// colQuotient = a.getColumnVector(i).ebeDivide(b);
		// matrixQuotient.setColumnVector(i, colQuotient);
		// }
		// return matrixQuotient;
		DoubleMatrix matrixQuotient = a.divColumnVector (b);
		
		return matrixQuotient;
		
	}
	
	
	public static final DoubleMatrix elementwiseSubtractionByColumnVector (DoubleMatrix a, DoubleMatrix b)
	{
		// RealVector colDifference;
		// RealMatrix matrixDifference = MatrixUtils.createRealMatrix(a.getRowDimension(),
		// a.getColumnDimension());
		// for (int i =0; i< a.getColumnDimension(); i++) {
		// colDifference = a.getColumnVector(i).subtract(b);
		// matrixDifference.setColumnVector(i, colDifference);
		// }
		// return matrixDifference;
		DoubleMatrix matrixDifference = a.subColumnVector (b);
		return matrixDifference;
		
	}
	
	
	public static final DoubleMatrix elementwiseAdditionByColumnVector (DoubleMatrix a, DoubleMatrix b)
	{
		// RealVector colSum;
		// RealMatrix matrixSum = MatrixUtils.createRealMatrix(a.getRowDimension(),
		// a.getColumnDimension());
		// for (int i =0; i< a.getColumnDimension(); i++) {
		// colSum = a.getColumnVector(i).add(b);
		// matrixSum.setColumnVector(i, colSum);
		// }
		// return matrixSum;
		DoubleMatrix matrixSum = a.add (b);
		return matrixSum;
		
	}
	
	
	public static final DoubleMatrix squaredMatrix (DoubleMatrix a)
	{
		// RealVector colProduct;
		// RealMatrix matrixProduct =
		// MatrixUtils.createRealMatrix(a.getRowDimension(),a.getColumnDimension());
		// for (int i =0; i< a.getColumnDimension(); i++) {
		// colProduct = a.getColumnVector(i).ebeMultiply(a.getColumnVector(i));
		// matrixProduct.setColumnVector(i, colProduct);
		// }
		// return matrixProduct;
		DoubleMatrix matrixProductVersionB = a.mul (a); // since a is multiplied by itself, we can
														// use .mul() which is designed for
														// element-wise multipliation of a matrix of
														// identical dimensions
		return matrixProductVersionB;
		
	}
	
}
