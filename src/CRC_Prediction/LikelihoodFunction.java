package CRC_Prediction;

import java.util.HashMap;

import org.apache.commons.math3.util.Pair;
import org.jblas.DoubleMatrix;

import com.google.common.collect.Multimap;

import CRC_Prediction.Utils.MatrixUtilityJBLAS;


/**
 * Create LikelihoodFunction class object and set its parameters
 * @author m186806
 *
 */
public class LikelihoodFunction {
	
	public    double _eta;
	public    double _slackPenalty;
	public    boolean _useNaturalGradient;
	
	/**
	 * Constructor for Likelihood function for Bayesian IRL or Maximum Likelihood IRL
	 * @param e
	 */
	public LikelihoodFunction(double e) {
		setEta(e);
	}
	
	public LikelihoodFunction() {
		//Use MaxEnt IRL
	}
	
	public void setEta(double e) {
		_eta = e;
	}
	
	public double getEta() {
		return _eta;
	}
	

	/**
	 * 
	 * Compute log LIKELIHOOD and GRADIENT of trajectories given reward function w and inverse temperature eta, log p(X | w, opts.eta)
	 * with Bayesian IRL method
	 * @param env
	 * @param irlalgo
	 * @param weightMatr
	 * @param stateActionPairCountsInfoForSubsetOfTrajs
	 * @param policyMatrix
	 * @param Hmatrix
	 * @param qMatrixGrad
	 * @param computeGradient
	 * @return Pair<Double, double[][]> logLikelihoodAndRewardGradient
	 */
	protected  Pair<Double, DoubleMatrix> computeLogLikelihoodAndGradient_Bayesian(MDP env, IRLAlgorithm irlalgo, double [][] weightMatr, Multimap<Integer,double[]> stateActionPairCountsInfoForSubsetOfTrajs, double [][] policyMatrix, double [][] Hmatrix, double [][] qMatrixGrad, boolean computeGradient ) {
	
		int numFeatures = env.getNumRewardFeatures();
		int numStates = env.getNumStates();
		int numActions = env.getNumActions();
		
		double [][] qMatrixGradient= qMatrixGrad;
		Double likelihood =0.0;
		double [][] qMatrix_Optimal;
		DoubleMatrix  rewardGradient = null;
		
		RewardFunctionGeneration.generateWeightedRewardFunction(env, weightMatr);
		if(policyMatrix ==null || Hmatrix==null) {
			HashMap<String, double[][]> policy_Value_H_Q_Matrices = PolicySolver.runPolicyIteration(env, irlalgo, null); //this call to policyIteration needs policy to be null no matter what.

			if(computeGradient && qMatrixGradient==null) {
				qMatrixGradient = InferenceAlgo.computeQMatrixGradient(policy_Value_H_Q_Matrices.get("P"), env);
			}
			qMatrix_Optimal = policy_Value_H_Q_Matrices.get("Q");
			
		}
		else { //never called since CRPIRL uses BIRL which always has policyMatrix ==null
			double [][] valueMatrixLikeLihood =  new DoubleMatrix(Hmatrix).mmul(new DoubleMatrix(weightMatr)).toArray2();
			double [][] optimalQMatrix = PolicySolver.policyImprovementStep(env, valueMatrixLikeLihood, null).get("Q");
			qMatrix_Optimal = optimalQMatrix;
		}
		
		//Compute Likelihood JK 6.22.2019
		double [][] policySM1 =  new DoubleMatrix(qMatrix_Optimal).mul(_eta).toArray2();
//		double [][] sum_policySM1 = MatrixUtilityJBLAS.sumPerRow(policySM1);	// GTD Not used
//		DoubleMatrix sum_policySM1DBLMatrix = new DoubleMatrix(sum_policySM1);	// GTD Not used
		double [][] policySM = MatrixUtilityJBLAS.exp(policySM1);
		double [][] sum_policySM = MatrixUtilityJBLAS.sumPerRow(policySM);
		double[][] log_sumPolicySM = MatrixUtilityJBLAS.log(sum_policySM);
		DoubleMatrix log_sumPolicySMDBLMatrix = new DoubleMatrix(log_sumPolicySM);
		DoubleMatrix policySM1DBLMatrix =new DoubleMatrix(policySM1);
//		DoubleMatrix policyQuotient = MatrixUtilityJBLAS.elementwiseDivisionByColumnVector(policySM1DBLMatrix, sum_policySM1DBLMatrix);	// GTD Not used

		DoubleMatrix BQ = policySM1DBLMatrix;
		DoubleMatrix BQsum = log_sumPolicySMDBLMatrix;
		DoubleMatrix NBQ = MatrixUtilityJBLAS.elementwiseSubtractionByColumnVector(BQ, BQsum);
		
		for (double[] observedSAPair_i : stateActionPairCountsInfoForSubsetOfTrajs.values()) { 
			Double state_i = observedSAPair_i[0];
			Double action_i = observedSAPair_i[1];
			Double count_i = observedSAPair_i[2];
			likelihood = likelihood + NBQ.get(state_i.intValue(),action_i.intValue())*count_i;
		}
		
		
		//Compute Reward Gradient...
		if(!computeGradient) {
			rewardGradient = null;
		}
		else { //else if boolean indicates that gradient of reward function needs to be computed
			
			//compute soft-max policy
			
			double [][] pi_sto = MatrixUtilityJBLAS.exp(policySM1);
			double [][] sum_pi_sto = MatrixUtilityJBLAS.sumPerRow(pi_sto);
			DoubleMatrix pi_stoDBLMat = new DoubleMatrix(pi_sto);
			DoubleMatrix sum_pi_stoDBLMat = new DoubleMatrix(sum_pi_sto);

			DoubleMatrix pi_stoDivDBLMatrix = MatrixUtilityJBLAS.elementwiseDivisionByColumnVector(pi_stoDBLMat, sum_pi_stoDBLMat);
			
			
			
			//Compute gradient of policy with respect to the weightMatrix which is based on the numFeatures)
			DoubleMatrix logPolicyGradientWRTfeatures = new DoubleMatrix(numFeatures,numStates*numActions);

			DoubleMatrix xMatr = new DoubleMatrix(numStates, numActions);
			for (int f=0; f<numFeatures; f++) {
				//need to convert row vector for feature f in dQ into [numStates x numActions] matrix
				double [] featureRowVector = qMatrixGradient[f];  //[1 x (numStates*numActions)] matrix
				DoubleMatrix fRowMatrix = new DoubleMatrix(featureRowVector);

				xMatr = fRowMatrix.reshape(numStates, numActions);
				DoubleMatrix yMatrRowSumVec = MatrixUtilityJBLAS.elementwiseMultiply(pi_stoDivDBLMatrix,xMatr).rowSums();
				
				DoubleMatrix zMatr = MatrixUtilityJBLAS.elementwiseSubtractionByColumnVector(xMatr,yMatrRowSumVec).mul(_eta);
				
				DoubleMatrix zFeatureRowMatrix = zMatr.reshape(1, numStates*numActions);

				logPolicyGradientWRTfeatures.putRow(f, zFeatureRowMatrix);
			}
			
			//Compute GRADIENT of reward function (rewardGradient) using policyGradient and s-a pair counts
			for (double[] observedSAPair_i : stateActionPairCountsInfoForSubsetOfTrajs.values()) { 
				Double state_i = observedSAPair_i[0];
				Double action_i = observedSAPair_i[1];
				Double count_i = observedSAPair_i[2];
				//Double columnSubScriptDblVal =  (action_i-1.0)*numStates+state_i; //this formula would work if our actions start at 1 and states start at 1
				Double columnSubScriptDblVal =  (action_i)*numStates+state_i;  //get subscript of column of interest
				int columnSubScriptIntVal = columnSubScriptDblVal.intValue();
				if(rewardGradient ==null) {
					rewardGradient = logPolicyGradientWRTfeatures.getColumn(columnSubScriptIntVal).mul(count_i);
				}
				else {
					// need to actually reset rewardGradient to updated value!!
					rewardGradient = rewardGradient.add(logPolicyGradientWRTfeatures.getColumn(columnSubScriptIntVal).mul(count_i));

				}
			}//end for-loop computing reward function gradient
			
		
		}
		

		Pair<Double, DoubleMatrix> logLikelihoodAndRewardGradient = new Pair<Double, DoubleMatrix>( likelihood, rewardGradient);

		return logLikelihoodAndRewardGradient;
		
	}
	
	/**
	 * Compute log likelihood and gradient of trajectories given reward function w and inverse temperature eta, log p(X | w, opts.eta)
	 * with Maximum Likehood IRL method
	 */
	public static void computeLogLkelihoodAndGradient_MaximumLikelihood() {
		// Not used
	}

}
