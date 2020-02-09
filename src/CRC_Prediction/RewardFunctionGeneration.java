package CRC_Prediction;

import java.util.ArrayList;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import CRC_Prediction.Utils.MatrixUtility;

public class RewardFunctionGeneration {
	

	
	
	public static void generateWeightedRewardFunction(MDP env, double[][] weightMatrix ) {
		
		env.setWeight(weightMatrix);

//		int numFeatures = env.getNumRewardFeatures();	// GTD Not used
//		int numStates = env.getNumStates();				// GTD Not used
//		int numActions = env.getNumActions();			// GTD Not used
		RealMatrix rewardMatrix = MatrixUtils.createRealMatrix(env.getNumStates(), env.getNumActions());
				
		RealMatrix weightedFMatrixBlock = null; 

		
		ArrayList<double[][]> sfmatrix = env.getStateFeatureMatrix();  //(numActions) replicate blocks of [numStates x numFeatures] matrices ; i.e. certain features are pertinent/non-pertinent for a given state. This matrix extends this binary matrix, so that we know which features are pertinent for each state-action pair. 
		
		
		for (int action_i =0; action_i<sfmatrix.size(); action_i++) {  //JK 2.13.2019 Here we are multiplying the newly sampled weight matrix (which indicates what reward features are important in the given reward function)
			double [][] sfmatrixBLOCK = sfmatrix.get(action_i); //retrieves the 64x16 matrix indicating which features are pertinent for each state when executed with action_i; 
			weightedFMatrixBlock = MatrixUtility.multiplyMatricesWithMatrixUtils(sfmatrixBLOCK, weightMatrix); //a 64x1 vector/matrix ( 64x16 matrix * 16x1 matrix) 
			//should become a 64x1 vector/matrix after multiplying 64x16 matrix(indicating pertinence/non-pertinece (0/1) of each feature) * 16x1 matrix indicating the weight of each feature for the 
			rewardMatrix.setColumnMatrix(action_i, weightedFMatrixBlock); //set the reward function for all state-action_i
		}
		double [][] rewardMatrix2DArray = rewardMatrix.getData();
		env.setRewardFunction(rewardMatrix2DArray);  //this should be a 64 x 4 matrix ....i.e. each state-action pair has a reward-value associated with it that is equal to the SUM of weights corresponding to only a SPECIFIC SUBSET of features pertinent to each state-action pair.
	}

}
