
package CRC_Prediction;


import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;
import org.jblas.DoubleMatrix;
import CRC_Prediction.Utils.MatrixUtilityJBLAS;
import CRC_Prediction.Utils.VectorUtility;
import java.util.*;


public class PolicySolverCancer
{
	/**
	 * JK data validated 7.24.2019
	 * @param env : MDP representation of a specific problem environment
	 * @param irloptions
	 * @param policyMatrix : numStates x 1 matrix
	 * @return newPolicyMatrix
	 *  newValueMatrix : numStates x 1 matrix
	 *  newHmatrix
	 *  newQmatrix
	 *  Map<String, double[][]> newPolicy_Value_H_Q_matrices
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
	@SuppressWarnings ("null")
	public static Map<String, double[][]> runPolicyIteration (MDPCancer env, IRLAlgorithmCancer irloptions, double[][] policyMatrix)
	{
		
		// int maxIterations = irloptions.getMaxIterations(); JK: 5.13.2019 this should never have
			// been used. The policy and/or value matrix update iterations will have difficulty
			// converging if maxIterations is too low
		int maxIterations = 10000;
		double epsilon = 1E-12;
		int numStates = env.getNumStates ();
		double[][] previousPolicyMatrix = null;
//		double[][] newPolicyMatrix = null;	// GTD Not used
		double[][] previousValueMatrix = null;
		double[][] newValueMatrix = null;
		double[][] newHMatrix = null;
//		double[][] newQMatrix = null;	// GTD Not used
		
		Map<String, double[][]> newQVPmatrices = null;
		
		if (policyMatrix == null)
		{	// this is true when being called by generateNewWeight <--- initializeTable() within InferenceAlgo class
			// create matrix with 0's as elements (policy specifies actions to take for each given state; recall that actions begin at '0'
			previousPolicyMatrix = MatrixUtilityJBLAS.createMatrixWithZeros (numStates, 1); 
		}
		else
		{
			previousPolicyMatrix = MatrixUtilityJBLAS.deepCopy (policyMatrix);
		}
		previousValueMatrix = DoubleMatrix.zeros (numStates).toArray2 (); // create matrix with 0's as elements
		
		for (int iteration = 0; iteration < maxIterations; iteration++)
		{ // iterate until convergence!!!
			Pair<double[][], double[][]> newHandValMatrices = policyEvaluationStep (env, previousPolicyMatrix, null);
			newHMatrix = newHandValMatrices.getFirst ();
			newValueMatrix = newHandValMatrices.getSecond ();
			
			newQVPmatrices = policyImprovementStep (env, newValueMatrix, null);
			
			if (MatrixUtilityJBLAS.areRealMatricesEqual (previousPolicyMatrix, newQVPmatrices.get ("P")))
			{
				break;
			}
			else if (MatrixUtilityJBLAS.compareMatrices (newQVPmatrices.get ("V"), previousValueMatrix, epsilon))
			{
				break;
			}
			else
			{
				previousPolicyMatrix = newQVPmatrices.get ("P");
				previousValueMatrix = newQVPmatrices.get ("V");
			}
			
		}
		
		Map<String, double[][]> newPolicy_Value_H_Q_matrices = new HashMap<String, double[][]> ();
		newPolicy_Value_H_Q_matrices.put ("P", newQVPmatrices.get ("P"));
		newPolicy_Value_H_Q_matrices.put ("V", newQVPmatrices.get ("V"));
		newPolicy_Value_H_Q_matrices.put ("H", newHMatrix);
		newPolicy_Value_H_Q_matrices.put ("Q", newQVPmatrices.get ("Q"));
		return newPolicy_Value_H_Q_matrices;
	}
	
	
	/**
	 * JK data validated 7.24.2019
	 * Start with a random policy, then find the value function of that policy.Given a policy, its
	 * value function can be obtained using the Bellman operator.
	 * Evaluate the previousPolicyMatrix according to the MDP weightVector. Returns new Hmatrix and
	 * ValueMatrix
	 * 
	 * @param env
	 * @param previousPolicyMatr
	 * @param weightArray
	 * @return Pair<double [][],double[][]> newHAndValueMatrices
	 */
	public static Pair<double[][], double[][]> policyEvaluationStep (MDPCancer env, double[][] previousPolicyMatr, double[][] weightArray)
	{
		double[][] weight2DArray = weightArray;
		DoubleMatrix weight;
		Map<Integer, double[][]> _transitionMatrix = env.getTransitionMatrix ();
		Integer numStates = env.getNumStates ();
		Integer numActions = env.getNumActions ();
		double epsilon = 1e-12;
		int maximumIterations = 10000;
		
		if (weight2DArray == null)
		{
			weight = new DoubleMatrix (env.getWeight ());
		}
		else
		{
			weight = new DoubleMatrix (weight2DArray);
		}
		
		// DoubleMatrix identityMatrix = DoubleMatrix.eye(numStates);	// GTD Not used

		
		DoubleMatrix transitionMatrixForPolicy = new DoubleMatrix (numStates, numStates); // create nSxnS matrix initialized with all 0's
		
		
		for (int a = 0; a < numActions; a++)
		{
			Integer	a_intVal = a;
			Double	a_dblVal = a_intVal.doubleValue ();
			// if runPolicyEvaluationStep() has been called by initializeTables()--> generateNewWeights(), previousPolicyMatrix will be a 64x1 matrix
			int[] logicalIndexArry = MatrixUtilityJBLAS.toLogicalMatrix (new DoubleMatrix (previousPolicyMatr), a_dblVal).findIndices ();
			double[] doubleLogicalIndexArray = new double[logicalIndexArry.length];
			for (int i = 0; i < logicalIndexArry.length; i++)
			{
				doubleLogicalIndexArray[i] = logicalIndexArry[i];
			}
			
			if (logicalIndexArry.length > 0)
			{ // check if any of the states in the currentPolicy uses action 'a' (assuming that
				// previousPolicyMatr is a numStates x 1 matrix)
				double[][] actionSpecificTransitionMatrix = _transitionMatrix.get (a_intVal);
				
				for (int i = 0; i < logicalIndexArry.length; i++)
				{
					DoubleMatrix colVectorTransposed = new DoubleMatrix (actionSpecificTransitionMatrix).getColumn (logicalIndexArry[i]).transpose ();
					transitionMatrixForPolicy.putRow (logicalIndexArry[i], colVectorTransposed); // set the transitionMatrix for this SPECIFIC POLICY
				}
			}
		}
		// JK 6.22.2019: replaced linear indexing operation
		DoubleMatrix expectedPolicyMatrixVersion2 = new DoubleMatrix (numStates, (numStates * numActions));
		
		
		DoubleMatrix idx = new DoubleMatrix (previousPolicyMatr).mul (numStates);
		DoubleMatrix unitVectorTransposed = new DoubleMatrix (VectorUtility.createUnitSpaceVector (numStates, 0.0, 1.0)).transpose ();
		idx.addi (unitVectorTransposed);
		DoubleMatrix unitVectorTransposedForLinearIdx = new DoubleMatrix (VectorUtility.createUnitSpaceVector (numStates, 1.0, 1.0)).transpose ();	

		
		
		DoubleMatrix idx2 = idx.mul (numStates).add (unitVectorTransposedForLinearIdx);

		idx2.subi (1); // JK added 6.22.2019 necessary to create correct linear indices for creating correct expectedPolicyMatrixVersion2
		
		
		int[] linearIndicesIdx2 = idx2.toIntArray ();
		expectedPolicyMatrixVersion2.put (linearIndicesIdx2, 1.0);
		

		
		Map<Integer, double[][]> stateFeatureMatrixMAP = env.getStateFeatureMatrixMAP ();
		//uses same functionality logic as below for-loop to convert Map into DoubleMatrix
		//DoubleMatrix FMatrix = MatrixUtilityJBLAS.convertMultiDimMatrixMap(stateFeatureMatrixMAP);
		
		// convert _stateFeatureMatrix Map into 2DArray RealMatrix
		double[][]	matrix = stateFeatureMatrixMAP.get (Integer.valueOf (0));
		RealMatrix	sfRealMatrix = MatrixUtils.createRealMatrix ((stateFeatureMatrixMAP.size () * matrix.length), matrix[0].length);
		double[][]	subMatrix_i;
		for (int i = 0; i < stateFeatureMatrixMAP.size (); i++)
		{
			subMatrix_i = stateFeatureMatrixMAP.get (Integer.valueOf (i));
			
			sfRealMatrix.setSubMatrix (subMatrix_i, subMatrix_i.length * i, 0);
		}
		DoubleMatrix sfDblMatrix = new DoubleMatrix (sfRealMatrix.getData ());
		DoubleMatrix expectedFeaturesMatrix = expectedPolicyMatrixVersion2.mmul (sfDblMatrix); //

		DoubleMatrix	H_new = expectedFeaturesMatrix.dup ();
		DoubleMatrix	H_old;
		DoubleMatrix	holdMatrix = new DoubleMatrix (expectedFeaturesMatrix.rows, expectedFeaturesMatrix.columns);
		boolean			hasConverged = false;
		double			discount = env.getDiscount ();
		
		
		
		for (int i = 0; i < maximumIterations; i++)
		{
			// System.out.println("iteration "+i);
			H_old = H_new;
			H_new = transitionMatrixForPolicy.mmuli (H_new, holdMatrix).muli (discount, holdMatrix);
			
			InferenceAlgoCancer.matrixAdd (H_new, expectedFeaturesMatrix);
//			H_new = expectedFeaturesMatrix.add (featureProbabilityMatrix);
//			hasConverged = MatrixUtilityJBLAS.compareREALMatrices (H_new, H_old, epsilon);
			hasConverged = InferenceAlgoCancer.compareMatrices (H_new, H_old, epsilon);	// GTD this is faster, esp when not converged
			if (hasConverged)
			{
				//JK 7.19.2019 commented out print statement
//				System.out.print ("policyEvaluationStep Converged in ");
//				System.out.print (Integer.toString (i));
//				System.out.println (" iterations");
				break;
			}
			else
				holdMatrix = H_old;
		}
		
		double[][] HMatrix = H_new.toArray2 ();
		DoubleMatrix valMatrix = H_new.mmul (weight);
		double[][] valueMatrix = valMatrix.toArray2 ();
		
		Pair<double[][], double[][]> newHAndValueMatrices = new Pair<double[][], double[][]> (HMatrix, valueMatrix);
		return newHAndValueMatrices;
	}
	
	
	/**
	 * JK data validated 7.24.2019
	 * Find a new (improved) policy based on the previous value function. Return qMatrix
	 * In the policy improvement step, the policy-iteration method updates possibly
	 * improved actions for every state in one iteration. If the the current policy is improved for
	 * at most one state in one iteration, then it is called simple policy-iteration.
	 * 
	 * @param env
	 * @param valueMatr
	 * @param weightArray
	 * @return Map<String, double[][]> "Q":newQmatrix, "V":newValueMatrix, "P":newPolicyMatrix
	 */
	public static Map<String, double[][]> policyImprovementStep (MDPCancer env, double[][] valueMatr, double[][] weightArray)
	{
		int			numStates = env.getNumStates ();
		int			numActions = env.getNumActions ();
		double[][]	weight2DArray = weightArray;
//		DoubleMatrix weight;	// GTD Not used
		
		if (weight2DArray == null)
		{
//			weight = new DoubleMatrix (env.getWeight ());	// GTD Not used
			//JK 7.20.2019 overide
			weightArray = env.getWeight ();

		}
		else
		{
			RewardFunctionGenerationCancer.generateWeightedRewardFunction (env, weight2DArray);
		}
		
		DoubleMatrix qMatrix = new DoubleMatrix (numStates, numActions);
		DoubleMatrix rewFunction = new DoubleMatrix (env.getRewardFunction ());
		DoubleMatrix vMatrix = new DoubleMatrix (valueMatr);
		
		// convert _transitionMatrix ArrayList into 2DArray RealMatrix 'transMatrix'
		Map<Integer, double[][]> transitionMatrix = env.getTransitionMatrix ();
		
		for (int a = 0; a < numActions; a++)
		{
			// Q for policy \pi at state s_i, a_j = R(s_i,a_j) + discount*transitionMatrix(s_i, a_j, s') * valueMatrix(s')
			Integer			key = Integer.valueOf (a);
			DoubleMatrix	tMatrixForAction_ai = new DoubleMatrix (transitionMatrix.get (key));
			DoubleMatrix	intermediate = tMatrixForAction_ai.transpose ().mmul (vMatrix).mul (env.getDiscount ());
			DoubleMatrix	qColMatrix = rewFunction.getColumn (key).add (intermediate);

			qMatrix.putColumn (key, qColMatrix);
		}
		
		
		DoubleMatrix improvedValueMatrix = new DoubleMatrix (qMatrix.getRows (), 1);
		DoubleMatrix improvedPolicyMatrix = new DoubleMatrix (qMatrix.getRows (), 1);
		
		Pair<Integer, Double> positionAndValue;
		for (int r = 0; r < qMatrix.getRows (); r++)
		{
			positionAndValue = VectorUtility.maxPositionAndVal (qMatrix.getRow (r).toArray ());
			improvedPolicyMatrix.put (r, 0, positionAndValue.getFirst ());
			improvedValueMatrix.put (r, 0, positionAndValue.getSecond ());
		}
		
		double[][] newQmat = qMatrix.toArray2 ();
		double[][] newVmat = improvedValueMatrix.toArray2 ();
		double[][] newPmat = improvedPolicyMatrix.toArray2 ();
		
		Map<String, double[][]> QVP_matrices = new HashMap<String, double[][]> ();
		QVP_matrices.put ("Q", newQmat);
		QVP_matrices.put ("V", newVmat);
		QVP_matrices.put ("P", newPmat);
		
		return QVP_matrices;
		
	}
	

	
}
