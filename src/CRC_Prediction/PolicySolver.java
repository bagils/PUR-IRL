
package CRC_Prediction;


import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;
import org.jblas.DoubleMatrix;
import CRC_Prediction.Utils.MatrixUtilityJBLAS;
import CRC_Prediction.Utils.VectorUtility;
import java.util.*;


/**
 * 
 *
 */
public class PolicySolver
{
	/**
	 * 
	 * @param env : MDP representation of a specific problem environment
	 * @param irloptions
	 * @param policyMatrix : numStates x 1 matrix
	 * @return newValueMatrix : numStates x 1 matrix
	 * @return newHMatrix
	 * @return HashMap<String, double[][]> newPolicy_Value_H_Q_matrices
	 */
	@SuppressWarnings ({"null", "javadoc"})	// Compiler is wrong here
	public static HashMap<String, double[][]> runPolicyIteration (MDP env, IRLAlgorithm irloptions, double[][] policyMatrix)
	{
		
		// int maxIterations = irloptions.getMaxIterations();
		// JK: 5.13.2019 this should never have been used. The policy and/or value matrix update iterations
		// will have difficulty converging if maxIterations is too low
		int maxIterations = 10000;
		
		double epsilon = 1E-12;
		int numStates = env.getNumStates ();
		double[][] previousPolicyMatrix = null;
//		double[][] newPolicyMatrix = null;
		double[][] previousValueMatrix = null;
		double[][] newValueMatrix = null;
		double[][] newHMatrix = null;
//		double[][] newQMatrix =null;
		
		HashMap<String, double[][]> newQVPmatrices = null;
		
		if (policyMatrix == null)
		{
			// this is true when being called by generateNewWeight <--- initializeTable() within
			// InferenceAlgo class
			// previousPolicyMatrix = MatrixUtility.createMatrixWithOnes(numStates, 1); //create
			// matrix with 1's as elements
			previousPolicyMatrix = MatrixUtilityJBLAS.createMatrixWithZeros (numStates, 1); // create matrix with 0's as elements
		}
		else
		{
			previousPolicyMatrix = MatrixUtilityJBLAS.deepCopy (policyMatrix);
		}
		previousValueMatrix = new DoubleMatrix (numStates, 1).toArray2 (); // create matrix with 0's
																			// as elements
		 // iterate until convergence!!!
		for (int iteration = 0; iteration < maxIterations; iteration++)
		{
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
		
		HashMap<String, double[][]> newPolicy_Value_H_Q_matrices = new HashMap<String, double[][]> ();
		newPolicy_Value_H_Q_matrices.put ("P", newQVPmatrices.get ("P"));
		newPolicy_Value_H_Q_matrices.put ("V", newQVPmatrices.get ("V"));
		newPolicy_Value_H_Q_matrices.put ("H", newHMatrix);
		newPolicy_Value_H_Q_matrices.put ("Q", newQVPmatrices.get ("Q"));
		return newPolicy_Value_H_Q_matrices;
		
	}
	
	
	/**
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
	public static Pair<double[][], double[][]> policyEvaluationStep (MDP env, double[][] previousPolicyMatr, double[][] weightArray)
	{
		double[][] weight2DArray = weightArray;
		DoubleMatrix weight;
		HashMap<Integer, double[][]> _transitionMatrix = env.getTransitionMatrix ();
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
		
//		DoubleMatrix identityMatrix = DoubleMatrix.eye (numStates);	// GTD Not used
		
		DoubleMatrix transitionMatrixForPolicy = new DoubleMatrix (numStates, numStates); // create nSxnS matrix initialized with all 0's
		
//		List<Integer> indexList;	// GTD Not used
		for (int a = 0; a < numActions; a++)
		{
			Integer a_intVal = a;
			Double a_dblVal = a_intVal.doubleValue ();
//			indexList = MatrixUtilityJBLAS.findMatches (previousPolicyMatr, a_dblVal); // if runPolicyEvaluationStep () has been called by	// GTD Not used
																						// initializeTables ()--> generateNewWeights (),
																						// previousPolicyMatrix will be a 64x1 matrix
			int[] logicalIndexArry = MatrixUtilityJBLAS.toLogicalMatrix(new DoubleMatrix(previousPolicyMatr), a_dblVal).findIndices();
			
			double[] doubleLogicalIndexArray = new double[logicalIndexArry.length];
			for (int i = 0; i < logicalIndexArry.length; i++)
			{
				doubleLogicalIndexArray[i] = logicalIndexArry[i];
			}
//			DoubleMatrix logicalIndexDBLMatrix = new DoubleMatrix (doubleLogicalIndexArray);
			
//			int[] indexArray = indexList.stream ().mapToInt (i -> i).toArray ();
//			double[] indexDblArray = indexList.stream ().mapToDouble (i -> i).toArray ();
			
//			int[] colRange = new DoubleMatrix (VectorUtility.createUnitSpaceVector (numStates, 0.0, 1.0)).toIntArray ();
//			int[] rowRange = new DoubleMatrix (VectorUtility.createUnitSpaceVector (numStates, 0.0, 1.0)).toIntArray ();
//			DoubleMatrix indexAsMat = new DoubleMatrix (indexDblArray);
			// int foo=1;
			if (logicalIndexArry.length > 0)
			{ // check if any of the states in the currentPolicy uses action 'a' (assuming that
				// previousPolicyMatr is a numStates x 1 matrix)
				// if (!indexList.isEmpty()) { //check if any of the states in the currentPolicy
				// uses action 'a' (assuming that previousPolicyMatr is a numStates x 1 matrix)
				double[][] actionSpecificTransitionMatrix = _transitionMatrix.get (a_intVal);
//				DoubleMatrix action_a_SpecificTransitionMatrix = new DoubleMatrix (_transitionMatrix.get (a_intVal));
				
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
//		double[][] expmv2dblarray = expectedPolicyMatrixVersion2.toArray2 ();
//		double[][] tMatrixPolicyDBlarray = transitionMatrixForPolicy.toArray2 ();
		
		List<double[][]> stateFeatureMatrix = env.getStateFeatureMatrix ();
//		DoubleMatrix stateFeatureDBLmatrixV2 = MatrixUtilityJBLAS.convertMultiDimMatrixList (stateFeatureMatrix);	// GTD Not used
		
		// convert _stateFeatureMatrix ArrayList into 2DArray RealMatrix
		RealMatrix sfRealMatrix = MatrixUtils.createRealMatrix ((stateFeatureMatrix.size () * stateFeatureMatrix.get (0).length), 
																stateFeatureMatrix.get (0)[0].length);
		double[][] subMatrix_i;
		for (int i = 0; i < stateFeatureMatrix.size (); i++)
		{
			subMatrix_i = stateFeatureMatrix.get (i);
			
			sfRealMatrix.setSubMatrix (subMatrix_i, subMatrix_i.length * i, 0);
		}
		
		DoubleMatrix sfDblMatrix = new DoubleMatrix (sfRealMatrix.getData ());
		DoubleMatrix expectedFeaturesMatrix = expectedPolicyMatrixVersion2.mmul (sfDblMatrix); //JK 6.17.2019 trying alternative version with linear indices
//		DoubleMatrix expectedFeaturesMatrixV2 = expectedPolicyMatrixVersion2.mmul (stateFeatureDBLmatrixV2);	// GTD Not used
//		double[][] efMatV2 = expectedFeaturesMatrixV2.toArray2 ();	// GTD Not used
//		double[][] efMat = expectedFeaturesMatrix.toArray2 ();	// GTD Not used
		
		DoubleMatrix H_new = expectedFeaturesMatrix.dup ();
		DoubleMatrix H_old;
		DoubleMatrix featureProbabilityMatrix;
		boolean hasConverged = false;
		
		for (int i = 0; i < maximumIterations; i++)
		{
//			System.out.println("iteration "+i);
			H_old = H_new;
			featureProbabilityMatrix = transitionMatrixForPolicy.mmul (H_new).mul (env.getDiscount ());
			
			H_new = expectedFeaturesMatrix.add (featureProbabilityMatrix);
//			double[][] hNewMat = H_new.toArray2 ();	// GTD Not used
			hasConverged = MatrixUtilityJBLAS.compareREALMatrices (H_new, H_old, epsilon);
			if (hasConverged)
			{
				break;
			}
		}
		
		double[][] HMatrix = H_new.toArray2 ();
		DoubleMatrix valMatrix = H_new.mmul (weight);
		double[][] valueMatrix = valMatrix.toArray2 ();
		
		Pair<double[][], double[][]> newHAndValueMatrices = new Pair<double[][], double[][]> (HMatrix, valueMatrix);
		return newHAndValueMatrices;
	}
	
	
	/**
	 * Find a new (improved) policy based on the previous value function. Return qMatrix
	 * In the policy improvement step, the policy-iteration method updates possibly
	 * improved actions for every state in one iteration. If the the current policy is improved for
	 * at most one state in one iteration, then it is called simple policy-iteration.
	 * 
	 * @param env
	 * @param valueMatr
	 * @param weightArray
	 * @return HashMap<String, double[][]> "Q":newQmatrix, "V":newValueMatrix, "P":newPolicyMatrix
	 */
	public static HashMap<String, double[][]> policyImprovementStep (MDP env, double[][] valueMatr, double[][] weightArray)
	{
		int numStates = env.getNumStates ();
		int numActions = env.getNumActions ();
		double[][] weight2DArray = weightArray;
//		DoubleMatrix weight;	// GTD not used
		
		if (weight2DArray == null)
		{
//			weight = new DoubleMatrix (env.getWeight ());	// GTD not used
		}
		else
		{
			RewardFunctionGeneration.generateWeightedRewardFunction (env, weight2DArray);
		}
		
		DoubleMatrix qMatrix = new DoubleMatrix (numStates, numActions);
		DoubleMatrix rewFunction = new DoubleMatrix (env.getRewardFunction ());
		DoubleMatrix vMatrix = new DoubleMatrix (valueMatr);
		
		// convert _transitionMatrix ArrayList into 2DArray RealMatrix 'transMatrix'
		HashMap<Integer, double[][]> transitionMatrix = env.getTransitionMatrix ();
		
		for (int a = 0; a < numActions; a++)
		{
			// Q for policy \pi at state s_i, a_j = R(s_i,a_j) + discount*transitionMatrix(s_i, a_j,
			// s')*valueMatrix(s')
			DoubleMatrix tMatrixForAction_ai = new DoubleMatrix (transitionMatrix.get (a)); // XXX: do we need to convert int to Integer?
			DoubleMatrix intermediate = tMatrixForAction_ai.transpose ().mmul (vMatrix).mul (env.getDiscount ());
			DoubleMatrix qColMatrix = rewFunction.getColumn (a).add (intermediate);
			qMatrix.putColumn (a, qColMatrix);
		}
		
//		DoubleMatrix tMatrix_a0 = new DoubleMatrix (transitionMatrix.get (0)).transpose ();	// GTD not used
//		DoubleMatrix tMatrix_a1 = new DoubleMatrix (transitionMatrix.get (1)).transpose ();	// GTD not used
//		DoubleMatrix tMatrix_a2 = new DoubleMatrix (transitionMatrix.get (2)).transpose ();	// GTD not used
//		DoubleMatrix tMatrix_a3 = new DoubleMatrix (transitionMatrix.get (3)).transpose ();	// GTD not used
		
		DoubleMatrix improvedValueMatrix = new DoubleMatrix (qMatrix.getRows (), 1);
		DoubleMatrix improvedPolicyMatrix = new DoubleMatrix (qMatrix.getRows (), 1);
		;
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
		
		HashMap<String, double[][]> QVP_matrices = new HashMap<String, double[][]> ();
		QVP_matrices.put ("Q", newQmat);
		QVP_matrices.put ("V", newVmat);
		QVP_matrices.put ("P", newPmat);
		
		return QVP_matrices;
		
	}
	
	
	/**
	 * What does this do? GTD
	 */
	public static void runValueIteration ()
	{
		// GTD Does nothing
	}
	
}
