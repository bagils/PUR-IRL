
package CRC_Prediction;


import CRC_Prediction.Utils.MatrixUtility;

import java.util.Map;
import java.util.Map.Entry;
import org.apache.commons.math3.linear.*;

/**
 * 
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
public class RewardFunctionGenerationCancer
{
	
	/**
	 * Generate a {@code weightMatrix.length} x {@code env.getStateFeatureMatrixMAP ().size ()} 
	 * matrix
	 * 
	 * @param env			{@link MDPCancer} to get map from
	 * @param weightMatrix	
	 */
	public static void generateWeightedRewardFunction (MDPCancer env, double[][] weightMatrix)
	{
		if (weightMatrix == null)
		{
			System.out.println ("weightMatrix is null for generateWeightedRewardFunction");
			System.err.println ("weightMatrix is null for generateWeightedRewardFunction");
			return;
		}
		
		env.setWeight (weightMatrix);
		
		int			numStates = env.getNumStates ();
		int			numActions = env.getNumActions ();
		double[][]	results = new double[numStates][];
		double[]	mulResults = new double[numStates];
		
		for (int i = 0; i < numStates; ++i)
			results[i] = new double[numActions];
		
		Map<Integer, double[][]> safmatrixMAP = env.getStateFeatureMatrixMAP ();
		
		for (Entry<Integer, double[][]> entry : safmatrixMAP.entrySet ())
		{
			Integer action_iINTEGER = entry.getKey ();
			int action_i = action_iINTEGER.intValue ();
			double[][] sfmatrixBLOCK = entry.getValue ();	// retrieves the 64x16 matrix indicating which features are
															// pertinent for each state when executed with action_i;
			if (sfmatrixBLOCK == null)
			{
				System.out.println ("sfmatrixBLOCK is null for action_i: " + action_i);
				System.err.println ("sfmatrixBLOCK is null for action_i: " + action_i);
				continue;
			}
			try
			{
				InferenceAlgoCancer.multiplyMatrix1 (sfmatrixBLOCK, weightMatrix, mulResults); 
				// a 64x1 vector/matrix ( 64x16 matrix * 16x1 matrix) should become a 64x1 vector/matrix after multiplying 64x16 matrix (indicating
				// pertinence/non-pertinence (0/1) of each feature) * 16x1 matrix indicating the weight of each feature for the
				setColumnOfMatrix (results, action_i, mulResults); // set the reward function for all state-action_i
			}
			catch (RuntimeException oops)
			{
				StringBuilder	out = new StringBuilder (200);
				int				rows = sfmatrixBLOCK.length;
				double[]		row = sfmatrixBLOCK[0];
				int				cols = row.length;
				
				out.append ("sfmatrixBLOCK: action: ");
				out.append (action_i);
				out.append (": ");
				out.append (cols);
				out.append (" x ");
				out.append (rows);
				out.append ("\n");
				rows = weightMatrix.length;
				row = weightMatrix[0];
				cols = row.length;
				out.append ("weightMatrix: ");
				out.append (cols);
				out.append (" x ");
				out.append (rows);
				System.err.println (out.toString ());
				oops.printStackTrace ();
				throw oops;
			}
		}
		
//		double[][] rewardMatrix2DArray = rewardMatrix.getData ();
		env.setRewardFunction (results); // this should be a 64 x 4 matrix ....i.e. each state-action pair has a reward-value
														// associated with it that is equal to the SUM of weights corresponding to only a
														// SPECIFIC SUBSET of features pertinent to each state-action pair.
	}
	
	
	/**
	 * 
	 * @param match
	 * @param results
	 */
	protected static final boolean compareArrays (double[][] match, double[][] results)
	{
		int	len = match.length;
		
		if (len != results.length)
		{
			String	message = "match.length (" + len + ") != results.length (" + results.length + ")";
			System.out.println (message);
			System.err.println (message);
			return false;
		}
		
		int	numCols = match[0].length;
		
		if (numCols != results[0].length)
		{
			String	message = "match[0].length (" + numCols + ") != results[0].length (" + results[0].length + ")";
			System.out.println (message);
			System.err.println (message);
			return false;
		}
		
		for (int i = 0; i < len; ++i)
		{
			double[]	mRow = match[i];
			double[]	rRow = results[i];
			
			for (int j = 0; j < numCols; ++j)
			{
				double	mValue, rValue;
				
				if ((mValue = mRow[j]) != (rValue = rRow[j]))
				{
					String	message = "match[" + i + "][" + j + "] (" + mValue + ") != results[" + i + "][" + j + "] (" + rValue + ")";
					System.out.println (message);
					System.err.println (message);
					return false;
				}
			}
		}
		
		return true;
	}
	
	
	/**
	 * Set the contents of {@code whichCol} of {@code results} to the contents of {@code newCol}<br>
	 * Requires that {@code results.length == newCol.length}
	 * 
	 * @param results	double[][] to write to.  Length must equal {@code newCol.length}
	 * @param whichCol	Which column of {@code results} to fill in.  0 based
	 * @param newCol	double[] to copy from.  Length must equal {@code results.length}
	 */
	private static final void setColumnOfMatrix (double[][] results, int whichCol, double[] newCol)
	{
		int	len = results.length;
		
		if (len != newCol.length)
		{
			String	message = "results.length != newCol.length for action: " + whichCol;
			System.out.println (message);
			System.err.println (message);
			return;
		}
		
		int	numCols = results[0].length;
		
		if (numCols <= whichCol)
		{
			String	message = "results[0].length = " + numCols + ", whichCol = " + whichCol;
			System.out.println (message);
			System.err.println (message);
			return;
		}
		
		for (int i = 0; i < len; ++i)
			results[i][whichCol] = newCol[i];
		
	}



	public static void generateWeightedRewardFunctionRM (MDPCancer env, double[][] weightMatrix)
	{
		if (weightMatrix == null)
		{
			System.out.println ("weightMatrix is null for generateWeightedRewardFunction");
			System.err.println ("weightMatrix is null for generateWeightedRewardFunction");
			return;
		}
		
		env.setWeight (weightMatrix);
		
//		int numFeatures = env.getNumRewardFeatures ();
//		int numStates = env.getNumStates ();
//		int numActions = env.getNumActions ();
		RealMatrix rewardMatrix = MatrixUtils.createRealMatrix (env.getNumStates (), env.getNumActions ());
		RealMatrix weightedFMatrixBlock = null;
		Map<Integer, double[][]> safmatrixMAP = env.getStateFeatureMatrixMAP ();
		
//		DoubleMatrix Fmatrix = MatrixUtilityJBLAS.convertMultiDimMatrixMap(safmatrixMAP);
//		DoubleMatrix fwMatrix = Fmatrix.mmul(new DoubleMatrix(weightMatrix));
//		DoubleMatrix fwREWARDmatrix = fwMatrix.reshape(env.getNumStates(), env.getNumActions());
//		double [][] fwRewardDBLArray = fwREWARDmatrix.toArray2();
				
		for (Entry<Integer, double[][]> entry : safmatrixMAP.entrySet ())
		{
			Integer action_iINTEGER = entry.getKey ();
			int action_i = action_iINTEGER.intValue ();
			double[][] sfmatrixBLOCK = entry.getValue (); // retrieves the 64x16 matrix indicating which features are
														 // pertinent for each state when executed with action_i;
			if (sfmatrixBLOCK == null)
			{
				System.out.println ("sfmatrixBLOCK is null for action_i: " + action_i);
				System.err.println ("sfmatrixBLOCK is null for action_i: " + action_i);
				continue;
			}
			try
			{
				weightedFMatrixBlock = MatrixUtility.multiplyMatricesWithMatrixUtils (sfmatrixBLOCK, weightMatrix); 
				// a 64x1 vector/matrix ( 64x16 matrix * 16x1 matrix) should become a 64x1 vector/matrix after multiplying 64x16 matrix (indicating
				// pertinence/non-pertinence (0/1) of each feature) * 16x1 matrix indicating the weight of each feature for the
				rewardMatrix.setColumnMatrix (action_i, weightedFMatrixBlock); // set the reward function for all state-action_i
			}
			catch (RuntimeException oops)
			{
				StringBuilder	out = new StringBuilder (100);
				int				rows = sfmatrixBLOCK.length;
				double[]		row = sfmatrixBLOCK[0];
				int				cols = row.length;
				
				out.append ("sfmatrixBLOCK: action: ");
				out.append (action_i);
				out.append (": ");
				out.append (cols);
				out.append (" x ");
				out.append (rows);
				out.append ("\n");
				rows = weightMatrix.length;
				row = weightMatrix[0];
				cols = row.length;
				out.append ("weightMatrix: ");
				out.append (cols);
				out.append (" x ");
				out.append (rows);
				System.err.println (out.toString ());
				oops.printStackTrace ();
				throw oops;
			}
		}
		double[][] rewardMatrix2DArray = rewardMatrix.getData ();
		env.setRewardFunction (rewardMatrix2DArray); // this should be a 64 x 4 matrix ....i.e. each state-action pair has a reward-value
														// associated with it that is equal to the SUM of weights corresponding to only a
														// SPECIFIC SUBSET of features pertinent to each state-action pair.
	}
	
}
