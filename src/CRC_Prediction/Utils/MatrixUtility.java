
package CRC_Prediction.Utils;


import java.util.*;
import java.util.stream.IntStream;
import org.apache.commons.math3.linear.*;


public class MatrixUtility
{
	
	public static double[][] minComparisonX (double[][] dbl2dArray, double minVal)
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
	
	
	public static double[][] maxComparisonX (double[][] dbl2dArray, double maxVal)
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
	public static double[][] withinBounds (double[][] dbl2dArray, double lowerBound,
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
	
	
	public static double[][] equalityComparisonDBL (double[][] dbl2dArray, double comparisonVal)
	{
		double logical2DArray[][] = new double[dbl2dArray.length][dbl2dArray[0].length];
		for (int i = 0; i < dbl2dArray.length; i++)
		{
			for (int j = 0; j < dbl2dArray[i].length; j++)
			{
				if (Double.compare (comparisonVal, dbl2dArray[i][j]) == 0)
				{
					// if(comparisonVal == dbl2dArray[i][j]) {
					logical2DArray[i][j] = 1.0;
				}
				else
				{
					logical2DArray[i][j] = 0.0;
				}
				
			}
		}
		return logical2DArray;
	}
	
	
	public static int[][] equalityComparisonINT (int[][] int2dArray, int comparisonVal)
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
	public static double[][] lessEqualityComparison (double[][] dbl2dArray, double comparisonVal)
	{
		double logical2DArray[][] = new double[dbl2dArray.length][dbl2dArray[0].length];
		for (int i = 0; i < dbl2dArray.length; i++)
		{
			for (int j = 0; j < dbl2dArray[i].length; j++)
			{
				if (Double.compare (dbl2dArray[i][j], comparisonVal) < 0)
				{
					// if(dbl2dArray[i][j] < comparisonVal) {
					logical2DArray[i][j] = 1.0;
				}
				else
				{
					logical2DArray[i][j] = 0.0;
				}
				
			}
		}
		return logical2DArray;
	}
	
	
	/**
	 * Returns logical matrix indicating whether each element is GREATER than specified value.
	 * 
	 * @param dbl2dArray
	 * @param comparisonVal
	 * @return
	 */
	public static double[][] greaterEqualityComparison (double[][] dbl2dArray, double comparisonVal)
	{
		double logical2DArray[][] = new double[dbl2dArray.length][dbl2dArray[0].length];
		for (int i = 0; i < dbl2dArray.length; i++)
		{
			for (int j = 0; j < dbl2dArray[i].length; j++)
			{
				if (Double.compare (dbl2dArray[i][j], comparisonVal) > 0)
				{
					// if(dbl2dArray[i][j] > comparisonVal) {
					logical2DArray[i][j] = 1.0;
				}
				else
				{
					logical2DArray[i][j] = 0.0;
				}
				
			}
		}
		return logical2DArray;
	}
	
	
	/**
	 * Returns the sum along rows.
	 * Compute the sum for each column across all rows.
	 * 
	 * @param mat
	 * @return double[] containing sum for each column
	 */
	public static double[] sumPerColumn (double[][] mat)
	{
		double[] sumValsAcrossColumns = new double[mat[0].length]; // a row is of length =
																	// mat[0].length= num columns
		for (int c = 0; c < mat[0].length; c++)
		{ // iterate by column
			double sum = mat[0][c];
			for (int i = 1; i < mat.length; i++)
			{ // for each column iterate through all rows to calculate sum; start at i=1 since we
				// begin with sum=mat[i=0][c]
				sum = sum + mat[i][c];
			}
			sumValsAcrossColumns[c] = sum;
		}
		return sumValsAcrossColumns;
	}
	
	
	public static int[] sumPerColumnINT (int[][] mat)
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
	public static double[][] sumPerRow (double[][] mat)
	{
		double[][] sumValForEachRow = new double[mat.length][1]; // num of rows
		for (int r = 0; r < mat.length; r++)
		{ // iterate by row
			double sum = mat[r][0];
			for (int i = 1; i < mat[0].length; i++)
			{ // for each row iterate through all cols to calculate sum
				sum = sum + mat[r][i];
			}
			sumValForEachRow[r][0] = sum;
		}
		return sumValForEachRow;
	}
	
	
	/**
	 * maximum value across columns in matrix
	 * 
	 * @param mat
	 * @return
	 */
	public static double[] maxPerColumn (double[][] mat)
	{
		double[] maxValsAcrossColumns = new double[mat[0].length]; // a row is of length =
																	// mat[0].length= num columns
		for (int c = 0; c < mat[0].length; c++)
		{ // iterate by column
			double max = mat[0][c];
			for (int i = 1; i < mat.length; i++)
			{ // for each column iterate through all rows to find highest element value
				if (Double.compare (mat[i][c], max) > 0)
				{
					// if (mat[i][c] > max)
					max = mat[i][c];
				}
			}
			maxValsAcrossColumns[c] = max;
		}
		return maxValsAcrossColumns;
	}
	
	
	public static double matrixMaximum (double[][] mat)
	{
		double max = 0.0;
		for (int c = 0; c < mat[0].length; c++)
		{ // iterate by column
			for (int i = 0; i < mat.length; i++)
			{ // for each column iterate through all rows to find highest element value
				if (Double.compare (mat[i][c], max) > 0)
				{
					// if (mat[i][c] > max)
					max = mat[i][c];
				}
			}
		}
		return max;
	}
	
	
	/**
	 * Multiplies the given vector and matrix using Java 8 streams.
	 *
	 * @param matrix the matrix
	 * @param vector the vector to multiply
	 *
	 * @return result after multiplication.
	 */
	public static double[] multiplyMatrixByVectorWithStreams (final double[][] matrix,
			final double[] vector)
	{
		final int rows = matrix.length;
		final int columns = matrix[0].length;
		
		return IntStream.range (0, rows).mapToDouble (row -> IntStream.range (0, columns)
				.mapToDouble (col -> matrix[row][col] * vector[col]).sum ()).toArray ();
	}
	
	
	public static double[][] multiplyMatricesWithStreamsZ (double[][] m1, double[][] m2)
	{
		
		double[][] result = Arrays.stream (m1)
				.map (r -> IntStream.range (0, m2[0].length)
						.mapToDouble (i -> IntStream.range (0, m2.length)
								.mapToDouble (j -> r[j] * m2[j][i]).sum ())
						.toArray ())
				.toArray (double[][]::new);
		return result;
	}
	
	
	public static RealMatrix multiplyMatricesWithMatrixUtils (double[][] m1, double[][] m2)
	{
		RealMatrix m1Asrealmatrix = MatrixUtils.createRealMatrix (m1);
		RealMatrix m2Asrealmatrix = MatrixUtils.createRealMatrix (m2);
		
		RealMatrix product = m1Asrealmatrix.multiply (m2Asrealmatrix);
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
	public static double[] multiplyMatrixByVectorWithForLoops (double[][] matrix, double[] vector)
	{
		int rows = matrix.length;
		int columns = matrix[0].length;
		
		double[] result = new double[rows];
		
		for (int row = 0; row < rows; row++)
		{
			double sum = 0;
			for (int column = 0; column < columns; column++)
			{
				sum += matrix[row][column] * vector[column];
			}
			result[row] = sum;
		}
		return result;
	}
	
	
	public static double[][] createMatrixWithOnes (int nRows, int nCols)
	{
		double[][] onesMatrix = new double[nRows][nCols];
		
		for (int i = 0; i < nRows; i++)
		{
			
			for (int j = 0; j < nCols; j++)
			{
				
				onesMatrix[i][j] = 1.0;
			}
			
		}
		
		return onesMatrix;
	}
	
	
	public static double[][] createMatrixWithZeros (int nRows, int nCols)
	{
		double[][] onesMatrix = new double[nRows][nCols];
		
		for (int i = 0; i < nRows; i++)
		{
			
			for (int j = 0; j < nCols; j++)
			{
				
				onesMatrix[i][j] = 0.0;
			}
			
		}
		
		return onesMatrix;
	}
	
	
	public static double[][] createMatrixWithNANS (int nRows, int nCols)
	{
		double[][] nansMatrix = new double[nRows][nCols];
		
		for (int i = 0; i < nRows; i++)
		{
			
			for (int j = 0; j < nCols; j++)
			{
				
				nansMatrix[i][j] = Double.NaN;
			}
			
		}
		
		return nansMatrix;
	}
	
	
	public static RealMatrix createRealMatrixWithNANS (int nRows, int nCols)
	{
		RealMatrix nansMatrix = MatrixUtils.createRealMatrix (nRows, nCols);
		
		for (int i = 0; i < nRows; i++)
		{
			
			for (int j = 0; j < nCols; j++)
			{
				
				nansMatrix.setEntry (i, j, Double.NaN);
			}
			
		}
		
		return nansMatrix;
	}
	
	
	/**
	 * Creates HashMap containing RealMatrix elements consisting of NaNs. NOTE: The set of keys for
	 * this HashMap begins at "1"!! Not 0 on purpose!
	 * 
	 * @param nRows
	 * @param nCols
	 * @return HashMap<Integer, RealMatrix>, each key corresponds to a tableIndex and its value is a
	 *         column matrix
	 */
	public static HashMap<Integer, RealMatrix> createHashMapOfRealMatricesWithNANS (int nRows,
			int nCols)
	{
		HashMap<Integer, RealMatrix> nansMap = new HashMap<Integer, RealMatrix> ();
		for (int j = 1; j < nCols + 1; j++)
		{
			RealMatrix colMatrix_j = MatrixUtils.createRealMatrix (nRows, 1);
			for (int i = 0; i < nRows; i++)
			{
				colMatrix_j.setEntry (i, 0, Double.NaN);
			}
			nansMap.put (j, colMatrix_j);
		}
		
		return nansMap;
	}
	
	
	/**
	 * Return matrix indices of elements with matching values, where the {i,i+1} elements of the
	 * list are the row,col for the ith match,
	 * 
	 * @param matrix : each column of the matrix could represent a different policy; each row is one
	 *            of (numStates); a matrix element represents the action executed for a given state,
	 *            policy (pi(state_j)=action_i))
	 * @param item : specifies the action which are you looking to find a match for within the
	 *            policyMatrix
	 * @return
	 */
	public static ArrayList<Integer> findMatches (double[][] matrix, double item)
	{
		if (matrix == null)
		{
			System.out.println ("matrix is null!!!");
			return null;
		}
		ArrayList<Integer> indexList = new ArrayList<Integer> ();
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
				}
			}
		}
		return indexList;
	}
	
	
	public static int getLinearIndex (int numRows, int r, int c)
	{
		int index = (c - 1) * numRows + r;
		return index;
	}
	
	
	public static int[] getMatrixIndices (int numRows, int index)
	{
		int r = ((index - 1) % numRows);
		Double cDbl = Math.floor ((index - 1) / numRows) + 1;
		int c = cDbl.intValue ();
		int[] matrixIndices = new int[] {r, c};
		return matrixIndices;
	}
	
	
	public static boolean compareREALMatrices (RealMatrix m1, RealMatrix m2, double epsilon)
	{
		
		boolean normIsBelowEpsilon = false;
		
		RealMatrix m1ColMatrix = MatrixUtility.convertRealMatrixIntoColMatrix (m1);
		RealMatrix m2ColMatrix = MatrixUtility.convertRealMatrixIntoColMatrix (m2);
		RealMatrix diffMatrix = m1ColMatrix.subtract (m2ColMatrix);
		// RealMatrix differenceMatrix = m1.subtract(m2);
		// double normVal = differenceMatrix.getNorm();
		// double normVal = diffMatrix.getNorm();
		double normVal = MatrixUtils.createRealVector (diffMatrix.getColumn (0)).getLInfNorm ();
		
		// System.out.println("L_normval = "+normVal+" epsilon ="+epsilon);
		// System.out.println("norm val: "+normVal);
		if (Double.compare (normVal, epsilon) < 0)
		{
			// if(normVal < epsilon) {
			normIsBelowEpsilon = true;
			// System.out.println(" norm is BELOW epsilon at :"+normVal);
		}
		
		return normIsBelowEpsilon;
	}
	
	
	public static boolean areRealMatricesEqual (double[][] m1, double[][] m2)
	{
		
		boolean normIsZero = false;
		RealMatrix m1Asrealmatrix = MatrixUtils.createRealMatrix (m1);
		RealMatrix m2Asrealmatrix = MatrixUtils.createRealMatrix (m2);
		
		RealMatrix m1ColMatrix = MatrixUtility.convertRealMatrixIntoColMatrix (m1Asrealmatrix);
		RealMatrix m2ColMatrix = MatrixUtility.convertRealMatrixIntoColMatrix (m2Asrealmatrix);
		RealMatrix diffMatrix = m1ColMatrix.subtract (m2ColMatrix);
		
		double normVal = MatrixUtils.createRealVector (diffMatrix.getColumn (0)).getLInfNorm ();
		
		// System.out.println("norm val: "+normVal);
		if (Double.compare (normVal, 0.0) == 0)
		{
			// if(normVal < epsilon) {
			normIsZero = true;
		}
		
		return normIsZero;
	}
	
	
	public static boolean compareMatrices (double[][] m1, double[][] m2, double epsilon)
	{
		RealMatrix m1Asrealmatrix = MatrixUtils.createRealMatrix (m1);
		RealMatrix m2Asrealmatrix = MatrixUtils.createRealMatrix (m2);
		boolean isNormBelowEpsilon = false;
		
		isNormBelowEpsilon = compareREALMatrices (m1Asrealmatrix, m2Asrealmatrix, epsilon);
		return isNormBelowEpsilon;
	}
	
	
	public static ArrayList<double[][]> convertMatrixList (ArrayList<RealMatrix> realMatrixList)
	{
		ArrayList<double[][]> dbl2dArrayList = new ArrayList<double[][]> ();
		for (int i = 0; i < realMatrixList.size (); i++)
		{
			double[][] dbl2Darray = realMatrixList.get (i).getData (); // XXX: Need to make sure
																		// that we copy the object
																		// and not the pointer
			dbl2dArrayList.add (dbl2Darray);
		}
		return dbl2dArrayList;
	}
	
	
	public static HashMap<Integer, double[][]> convertMatrixListToMap (
			ArrayList<RealMatrix> realMatrixList, Double[] sortedIndexValuesArr)
	{
		HashMap<Integer, double[][]> dbl2dArrayMap = new HashMap<Integer, double[][]> ();
		System.out.println ("# of new table indices =" + sortedIndexValuesArr.length);
		System.out.println ("number of new table vectors = " + realMatrixList.size ());
		if (realMatrixList.size () != sortedIndexValuesArr.length)
		{
			throw new java.lang.RuntimeException ("number of new table indices ("
					+ sortedIndexValuesArr.length + ") should = number of new table vectors("
					+ realMatrixList.size () + "!!!");
		}
		else
		{
			for (int i = 0; i < sortedIndexValuesArr.length; i++)
			{
				Double tblIndexValDBL = sortedIndexValuesArr[i];
				double[][] dbl2Darray = realMatrixList.get (i).getData ();
				dbl2dArrayMap.put (tblIndexValDBL.intValue (), dbl2Darray);
			}
			return dbl2dArrayMap;
		}
		
	}
	
	
	/**
	 * 
	 * @param assgnmentMat : assigns each trajectory to a given table/reward-function
	 *            tblSizesMat: 1 x numTables matrix, indicating # of trajectories assigned to each
	 *            table index-value
	 *            numTablesOfSameSize : 1 x numTrajectories, indicating how many tables consists of
	 *            the same number of trajectories. i.e. numTablesOfSameSize[0][1]= number of tables
	 *            which contain 1 trajectory, numTablesOfSameSize[0][5]= number of tables which
	 *            contain 5 trajectories
	 *            sizeToTableMap: 1 x numTrajectories, maps each possible table size to the set of
	 *            tables that contribute to that number.
	 */
	public static double[][] tableCounter (double[][] assgnmentMat)
	{
		// numTablesOfSameSize[0][i]= number of tables which contain i trajectories
		// tblSizesMat[0][i]: size of the ith table
		int numTrajs = assgnmentMat.length; // # of trajectories we are modeling
		HashSet<Double> uniqueTablesInRestaurant = MatrixUtility
				.countNumberUniqueElements (assgnmentMat);
		Double[] sortedSetOfUniqueTableIndicesInRest = VectorUtility
				.sortSet (uniqueTablesInRestaurant);
		int maxTableIndexVal = sortedSetOfUniqueTableIndicesInRest[sortedSetOfUniqueTableIndicesInRest.length
				- 1].intValue ();
		
		// double [][] tblSizesMat = MatrixUtils.createRealMatrix(1, maxTableIndexVal+1).getData();
		// double [][] tblSizesMat = MatrixUtils.createRealMatrix(1, maxTableIndexVal).getData();
		HashMap<Integer, Double> tblSizesMat = new HashMap<Integer, Double> ();
		
		// double [][] numTablesOfSameSizeMat = MatrixUtils.createRealMatrix(1,
		// numTrajs+1).getData();
		double[][] numTablesOfSameSizeMat = MatrixUtils.createRealMatrix (1, numTrajs).getData ();
		
		HashMap<Integer, double[]> sizeToTableMap = new HashMap<Integer, double[]> ();
		
		for (Double tblVal : sortedSetOfUniqueTableIndicesInRest)
		{
			
			double tblIndexDBLVal = tblVal;
			double[][] logicMatrix = MatrixUtility.equalityComparisonDBL (assgnmentMat,
					tblIndexDBLVal);
			
			// tblSizesMat[0][tblVal.intValue()]= MatrixUtility.sumPerColumn(logicMatrix)[0]; //the
			// total number of trajectories that have been assigned to the table with index-value =
			// 'tblVal'
			tblSizesMat.put (tblVal.intValue (), MatrixUtility.sumPerColumn (logicMatrix)[0]); // the
																								// total
																								// number
																								// of
																								// trajectories
																								// that
																								// have
																								// been
																								// assigned
																								// to
																								// the
																								// table
																								// with
																								// index-value
																								// =
																								// 'tblVal'
		}
		
		// iterate through all the table-index values (in tblSizesMat) to which >0 trajectories have
		// been attributed
		// for(int tableIndexj=0; tableIndexj< maxTableIndexVal+1; tableIndexj++) {
		for (int tableIndexj = 1; tableIndexj < maxTableIndexVal + 1; tableIndexj++)
		{
			// if(tblSizesMat[0][tableIndexj]>0.0) {
			if (Double.compare (tblSizesMat.get (tableIndexj), 0.0) > 0)
			{
				// if(tblSizesMat.get(tableIndexj)>0.0) {
				
				// for tableIndexj
				// Double sizeVal =tblSizesMat[0][tableIndexj]; //retrieve number of trajectories
				// assigned to the table with index-value 'tableIndexj'
				Double sizeVal = tblSizesMat.get (tableIndexj); // retrieve number of trajectories
																// assigned to the table with
																// index-value 'tableIndexj'
				
				int sizeIntVal = sizeVal.intValue ();
				
				// increment the column in 'numTablesOfSameSizeMat' which records the number of
				// OTHER tables that also have |sizeIntVal| trajectories assigned to their
				// table-index value
				// since we are using a double array in which element indices matter, we need to
				// ensure that numTablesOfSameSizeMat[0][0] corresponds to sizeIntVal=1
				/*
				 * numTablesOfSameSizeMat[0][sizeIntVal] = numTablesOfSameSizeMat[0][sizeIntVal]
				 * +1;//increment
				 * Double numTablesOfGivenSizeDblVal = numTablesOfSameSizeMat[0][sizeIntVal];
				 */
				numTablesOfSameSizeMat[0][sizeIntVal
						- 1] = numTablesOfSameSizeMat[0][sizeIntVal - 1] + 1;// increment
//				Double numTablesOfGivenSizeDblVal = numTablesOfSameSizeMat[0][sizeIntVal - 1];	// GTD Not used
//				int numTablesOfGivenSizeIntVal = numTablesOfGivenSizeDblVal.intValue (); // get the	// GTD Not used
																							// CURRENT
																							// integer
																							// count
																							// of
																							// the
																							// number
																							// of
																							// tables
																							// that
																							// also
																							// have
																							// the
																							// same
																							// size
				
				// expand the size of the double[] that will store table indices
				int lengthOfarrayOfTableIndices = 0;
				double[] tablesOfGivenSizeArray = null;
				// if(sizeToTableMap.get(numTablesOfGivenSizeIntVal)!=null) {
				if (sizeToTableMap.get (sizeIntVal) != null)
				{
					// lengthOfarrayOfTableIndices =
					// sizeToTableMap.get(numTablesOfGivenSizeIntVal).length;
					// tablesOfGivenSizeArray = new double [lengthOfarrayOfTableIndices+1];
					lengthOfarrayOfTableIndices = sizeToTableMap.get (sizeIntVal).length;
					tablesOfGivenSizeArray = new double[lengthOfarrayOfTableIndices + 1];
					
					for (int l = 0; l < lengthOfarrayOfTableIndices; l++)
					{
						// tablesOfGivenSizeArray[l]=
						// sizeToTableMap.get(numTablesOfGivenSizeIntVal)[l];
						tablesOfGivenSizeArray[l] = sizeToTableMap.get (sizeIntVal)[l];
						
					}
					tablesOfGivenSizeArray[lengthOfarrayOfTableIndices] = tableIndexj; // concatenate/append
																						// as
																						// additional
																						// column to
																						// row
																						// matrix
				}
				else
				{
					tablesOfGivenSizeArray = new double[1];
					tablesOfGivenSizeArray[0] = tableIndexj; // concatenate/append as additional
																// column to row matrix
				}
				
				// sizeToTableMap.put(numTablesOfGivenSizeIntVal, tablesOfGivenSizeArray); // set
				// this row vector for tables of size = numTablesOfGivenSizeIntVal
				sizeToTableMap.put (sizeIntVal, tablesOfGivenSizeArray); // set this row vector for
																			// tables of size =
																			// numTablesOfGivenSizeIntVal
				
			}
		}
		return numTablesOfSameSizeMat;
		
	}
	
	
	public static double[][] deepCopy (double[][] src)
	{
		double[][] dest = new double[src.length][src[0].length];
		for (int i = 0; i < src.length; i++)
		{
			System.arraycopy (src[i], 0, dest[i], 0, src[0].length);
		}
		return dest;
	}
	
	
	public static RealMatrix convertRealMatrixIntoColMatrix (RealMatrix rMatrix)
	{
		int numRows = rMatrix.getRowDimension ();
		int numCols = rMatrix.getColumnDimension ();
		RealMatrix colMatrix = MatrixUtils.createRealMatrix (numRows * numCols, 1);
		for (int i = 0; i < numCols; i++)
		{
			colMatrix.setSubMatrix (rMatrix.getColumnMatrix (i).getData (), i * numRows, 0);
		}
		return colMatrix;
	}
	
	
	public static RealMatrix convertMultiDimMatrixList (ArrayList<double[][]> matrixList)
	{
		int numSubMatrices = matrixList.size ();
		int numRowsPerSubmatrix = matrixList.get (0).length;
		int numColsPerSubmatrix = matrixList.get (0)[0].length; // # columns in 1st row of 1st
																// submatrix in matrixList
		int totalNumRows = numSubMatrices * numRowsPerSubmatrix; // assumes each double[][] in
																	// ArrayList is a row/block
																	// matrix duplicate of same
																	// dimension, stacked vertically
																	// ( i.e. if matrixList is of
																	// size=4, then total num rows
																	// is 4*numRowsPerSubmatrix,
																	// total num cols is same as
																	// numColsPerSubmatrix
		RealMatrix multiDimMatrix = MatrixUtils.createRealMatrix (totalNumRows,
				numColsPerSubmatrix);
		for (int i = 0; i < numSubMatrices; i++)
		{
			multiDimMatrix.setSubMatrix (matrixList.get (i), i * numRowsPerSubmatrix, 0);
		}
		return multiDimMatrix;
	}
	
	
	public static RealMatrix convertMultiDimMatrixMap (HashMap<Integer, double[][]> matrixMap)
	{
		int numSubMatrices = matrixMap.size ();
		int numRowsPerSubmatrix = matrixMap.get (0).length;
		int numColsPerSubmatrix = matrixMap.get (0)[0].length; // # columns in 1st row of 1st
																// submatrix in matrixList
		int totalNumRows = numSubMatrices * numRowsPerSubmatrix; // assumes each double[][] in
																	// ArrayList is a row/block
																	// matrix duplicate of same
																	// dimension, stacked vertically
																	// ( i.e. if matrixList is of
																	// size=4, then total num rows
																	// is 4*numRowsPerSubmatrix,
																	// total num cols is same as
																	// numColsPerSubmatrix
		RealMatrix multiDimMatrix = MatrixUtils.createRealMatrix (totalNumRows,
				numColsPerSubmatrix);
		for (int i = 0; i < numSubMatrices; i++)
		{
			Integer iAsInteger = Integer.valueOf (i);
			multiDimMatrix.setSubMatrix (matrixMap.get (iAsInteger), i * numRowsPerSubmatrix, 0);
		}
		return multiDimMatrix;
	}
	
	
	public static RealMatrix convertHashMapToRealMatrixRows (
			HashMap<Integer, double[][]> mapOf2dArrays)
	{
		int i = 0;
		if (mapOf2dArrays.containsKey (0))
		{
			int numRowsPerSubmatrix = mapOf2dArrays.get (0).length;
			int numColsPerSubmatrix = mapOf2dArrays.get (0)[0].length; // # columns in 1st row of
																		// 1st submatrix
			int totalNumRows = mapOf2dArrays.size () * numRowsPerSubmatrix; // assumes each
																			// double[][] in HashMap
																			// is a row/block matrix
																			// duplicate of same
																			// dimension, stacked
																			// vertically ( i.e. if
																			// HashMap is of size=4,
																			// then total num rows
																			// is
																			// 4*numRowsPerSubmatrix,
																			// total num cols is
																			// same as
																			// numColsPerSubmatrix
			RealMatrix multiDimMatrix = MatrixUtils.createRealMatrix (totalNumRows,
					numColsPerSubmatrix);
			
			for (int k : mapOf2dArrays.keySet ())
			{
				multiDimMatrix.setSubMatrix (mapOf2dArrays.get (k), i * numRowsPerSubmatrix, 0);
				i++;
				
			}
			
			return multiDimMatrix;
			
		}
		else
		{
			throw new java.lang.RuntimeException (
					"This HashMap does not contain the Key used for subsequent initialization");
			
		}
	}
	
	
	public static RealMatrix convertHashMapToRealMatrixCols (
			HashMap<Integer, double[][]> mapOf2dArrays)
	{
		int i = 0;
		if (mapOf2dArrays.containsKey (0))
		{
			int numRowsPerSubmatrix = mapOf2dArrays.get (0).length;
			int numColsPerSubmatrix = mapOf2dArrays.get (0)[0].length; // # columns in 1st row of
																		// 1st submatrix
			int totalNumCols = mapOf2dArrays.size () * numColsPerSubmatrix;
			RealMatrix multiDimMatrix = MatrixUtils.createRealMatrix (numRowsPerSubmatrix,
					totalNumCols);
			
			for (int k : mapOf2dArrays.keySet ())
			{
				multiDimMatrix.setSubMatrix (mapOf2dArrays.get (k), 0, i * numColsPerSubmatrix);
				i++;
				
			}
			
			return multiDimMatrix;
			
		}
		else
		{
			throw new java.lang.RuntimeException (
					"This HashMap does not contain the Key used for subsequent initialization");
			
		}
	}
	
	
	/**
	 * transpose the matrix
	 * 
	 * @param mat
	 * @return
	 */
	public static double[][] transpose (double[][] mat)
	{
		double[][] a = new double[mat[0].length][mat.length];
		for (int i = 0; i < mat[0].length; i++)
		{
			for (int j = 0; j < mat.length; j++)
			{
				a[i][j] = mat[j][i];
			}
		}
		return a;
	}
	
	
	/**
	 * transpose the matrix
	 * 
	 * @param mat
	 * @return
	 */
	public static int[][] transposeINT (int[][] mat)
	{
		int[][] a = new int[mat[0].length][mat.length];
		for (int i = 0; i < mat[0].length; i++)
		{
			for (int j = 0; j < mat.length; j++)
			{
				a[i][j] = mat[j][i];
			}
		}
		return a;
	}
	
	
	/**
	 * Returns the exponential for each element in inputted double[][]
	 * 
	 * @param dbl2dArray
	 * @return
	 */
	public static double[][] exp (double[][] dbl2dArray)
	{
		double exp2DArray[][] = new double[dbl2dArray.length][dbl2dArray[0].length];
		for (int i = 0; i < dbl2dArray.length; i++)
		{
			for (int j = 0; j < dbl2dArray[i].length; j++)
			{
				exp2DArray[i][j] = Math.exp (dbl2dArray[i][j]);
			}
		}
		return exp2DArray;
	}
	
	
	/**
	 * Returns the natural logarithm for each element in inputted double[][]
	 * 
	 * @param dbl2dArray
	 * @return
	 */
	public static double[][] log (double[][] dbl2dArray)
	{
		double log2DArray[][] = new double[dbl2dArray.length][dbl2dArray[0].length];
		for (int i = 0; i < dbl2dArray.length; i++)
		{
			for (int j = 0; j < dbl2dArray[i].length; j++)
			{
				log2DArray[i][j] = Math.log (dbl2dArray[i][j]);
			}
		}
		return log2DArray;
	}
	
	
	public static double[][] scalarMultiplication (double[][] dbl2dArray, double scalar)
	{
		double result2DArray[][] = new double[dbl2dArray.length][dbl2dArray[0].length];
		for (int i = 0; i < dbl2dArray.length; i++)
		{
			for (int j = 0; j < dbl2dArray[i].length; j++)
			{
				result2DArray[i][j] = dbl2dArray[i][j] * scalar;
			}
		}
		return result2DArray;
	}
	
	
	public static RealMatrix expandColumnMatrix (RealMatrix colMatrix, int numFinalRows,
			int numFinalCols)
	{
		
		RealMatrix expandedMatrix = MatrixUtils.createRealMatrix (numFinalRows, numFinalCols);
		for (int c = 0; c < numFinalCols; c++)
		{
			expandedMatrix.setColumnMatrix (c, colMatrix);
		}
		return expandedMatrix;
	}
	
	
	public static RealMatrix expandRowMatrix (RealMatrix rowMatrix, int numFinalRows,
			int numFinalCols)
	{
		
		RealMatrix expandedMatrix = MatrixUtils.createRealMatrix (numFinalRows, numFinalCols);
		for (int r = 0; r < numFinalRows; r++)
		{
			expandedMatrix.setRowMatrix (r, rowMatrix);
		}
		return expandedMatrix;
	}
	
	
	/*
	 * public static <T> T[][] deepCopyMatrixX(T[][] matrix) {
	 * return java.util.Arrays.stream(matrix).map(el -> el.clone()).toArray($ -> matrix.clone());
	 * }
	 */
	/*
	 * public static <T> ArrayList<T[][]> deepCopyArrayListY(ArrayList<T[][]> arrListToCopy){
	 * ArrayList<T[][]> clonedList = new ArrayList<T[][]>(arrListToCopy.size());
	 * for(int i=0; i< arrListToCopy.size(); i++) {
	 * clonedList.set(i, deepCopyMatrix(arrListToCopy.get(i)));
	 * }
	 * return clonedList;
	 * }
	 */
	public static ArrayList<double[][]> deepCopyDBLArrayList (ArrayList<double[][]> arrListToCopy)
	{
		ArrayList<double[][]> clonedList = new ArrayList<double[][]> (arrListToCopy.size ());
		for (int i = 0; i < arrListToCopy.size (); i++)
		{
			clonedList.set (i, deepCopy (arrListToCopy.get (i)));
		}
		return clonedList;
	}
	
	
	public static HashSet<Double> countNumberUniqueElements (double[][] dbl2darray)
	{
		
		HashSet<Double> alreadyPresent = new HashSet<Double> ();
		
		for (double[] nextElem : dbl2darray)
		{
			alreadyPresent.add (nextElem[0]);
		}
		
		return alreadyPresent;
		
	}
	
	
	public static RealMatrix elementwiseMultiply (RealMatrix a, RealMatrix b)
	{
		RealVector rowProduct;
		RealMatrix matrixProduct = MatrixUtils.createRealMatrix (a.getRowDimension (),
				a.getColumnDimension ());
		for (int i = 0; i < a.getRowDimension (); i++)
		{
			rowProduct = a.getRowVector (i).ebeMultiply (b.getRowVector (i));
			matrixProduct.setRowVector (i, rowProduct);
		}
		return matrixProduct;
		
	}
	
	
	public static RealMatrix elementwiseMultiplicationByColumnVector (RealMatrix a, RealVector b)
	{
		RealVector colProduct;
		RealMatrix matrixProduct = MatrixUtils.createRealMatrix (a.getRowDimension (),
				a.getColumnDimension ());
		for (int i = 0; i < a.getColumnDimension (); i++)
		{
			colProduct = a.getColumnVector (i).ebeMultiply (b);
			matrixProduct.setColumnVector (i, colProduct);
		}
		return matrixProduct;
		
	}
	
	
	public static RealMatrix elementwiseDivisionByColumnVector (RealMatrix a, RealVector b)
	{
		RealVector colQuotient;
		RealMatrix matrixQuotient = MatrixUtils.createRealMatrix (a.getRowDimension (),
				a.getColumnDimension ());
		for (int i = 0; i < a.getColumnDimension (); i++)
		{
			colQuotient = a.getColumnVector (i).ebeDivide (b);
			matrixQuotient.setColumnVector (i, colQuotient);
		}
		return matrixQuotient;
		
	}
	
	
	public static RealMatrix elementwiseSubtractionByColumnVector (RealMatrix a, RealVector b)
	{
		RealVector colDifference;
		RealMatrix matrixDifference = MatrixUtils.createRealMatrix (a.getRowDimension (),
				a.getColumnDimension ());
		for (int i = 0; i < a.getColumnDimension (); i++)
		{
			colDifference = a.getColumnVector (i).subtract (b);
			matrixDifference.setColumnVector (i, colDifference);
		}
		return matrixDifference;
		
	}
	
	
	public static RealMatrix elementwiseAdditionByColumnVector (RealMatrix a, RealVector b)
	{
		RealVector colSum;
		RealMatrix matrixSum = MatrixUtils.createRealMatrix (a.getRowDimension (),
				a.getColumnDimension ());
		for (int i = 0; i < a.getColumnDimension (); i++)
		{
			colSum = a.getColumnVector (i).add (b);
			matrixSum.setColumnVector (i, colSum);
		}
		return matrixSum;
		
	}
	
	
	public static RealMatrix squaredMatrix (RealMatrix a)
	{
		RealVector colProduct;
		RealMatrix matrixProduct = MatrixUtils.createRealMatrix (a.getRowDimension (),
				a.getColumnDimension ());
		for (int i = 0; i < a.getColumnDimension (); i++)
		{
			colProduct = a.getColumnVector (i).ebeMultiply (a.getColumnVector (i));
			matrixProduct.setColumnVector (i, colProduct);
		}
		return matrixProduct;
		
	}
	
}
