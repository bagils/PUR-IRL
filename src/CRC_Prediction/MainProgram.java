package CRC_Prediction;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.*;
import org.apache.commons.math3.util.Pair;
import org.jblas.DoubleMatrix;

import CRC_Prediction.Utils.MatrixUtilityJBLAS;
import CRC_Prediction.Utils.PowerLaw;
import CRC_Prediction.Utils.VectorUtility;
import com.jprofiler.api.controller.Controller;
import gnu.trove.set.hash.THashSet;

/**
 * 7.19.2019 Run MDP Gridworld environment within existing cancer java classes
 * @author John Kalantari
 * 
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


public class MainProgram {


	
	protected IRLAlgorithmCancer _irlAlgo;
	protected MDPCancer _mdp;
	protected IRLRestaurant _bestRestaurant;
	protected ArrayList<IRLRestaurant> _restaurantIterations;
	
	
	static
	{
//		System.out.println (System.getProperty ("jni.library.path"));
//		System.out.println (System.getProperty ("java.library.path"));
//		System.load ("/Users/m082166/Documents/workspace/irl/IRLJK_TFS2_MAVEN/libjniconverge.so");
		String		path = System.getProperty ("jni.library");
//		boolean		loaded = false;
		Set<String>	libNames = new HashSet<> ();
		
		if ((path != null) && !path.isEmpty ())
		{
			String[]	paths = path.split (":");
			
			for (String libPath : paths)
			{
				File	libFile = new File (libPath);
				if (libFile.exists ())
				{
					try
					{
						System.out.print ("Loading ");
						System.out.print (libPath);
						System.load (libPath);
						System.out.println (": Success");
						libNames.add (libFile.getName ());
					}
					catch (Exception | UnsatisfiedLinkError oops)
					{
						System.out.println (": Failed");
						oops.printStackTrace ();
					}
				}
			}
		}
		if (!libNames.contains ("libcudaconverge.so"))
		{
			try
			{
  			String userPath = System.getProperty("user.dir");
				System.out.print ("Loading libcudaconverge.so from "+ userPath);
				System.load (userPath + "/libcudaconverge.so");
				System.out.println (": Success");
			}
			catch (Exception | UnsatisfiedLinkError oops)
			{
				System.out.println (": Failed");
				oops.printStackTrace ();
			}
		}
		
//		System.loadLibrary ("libjniconverge");
    }
	
	
	
	
	public static ArrayList<double[][]> generateDemoData (MDPCancer env, IRLAlgorithmCancer irlAlgo, double[][] tblAssignmentMatrix, 
														  Map<Integer, double[][]> tblWeightVectorsMap, Map<Integer, double[][]> tblPolicyVectorsMap)
	{
		ArrayList<double[][]> dataTrajectories = new ArrayList<double[][]>();
		MersenneTwisterFastIRL RNG = new MersenneTwisterFastIRL(1);

		int numDemoTrajPerTable = env.getNumDemoTrajectoriesPerTable();
		int numFeatures = env.getNumRewardFeatures();
		int numStepsPerTraj= env.getNumStepsPerTrajectory();
		double discountFactor = env.getDiscount();
		
		
		// XXX:The size of of each of these lists corresponds to the number of ground-truth tables specified by env.getTrueTables in the restaurant. 
		// Each element in the list is a column matrix for that table.
		// stores the weight-vector associated with each table index/value; 
	//		 HashMap<Integer, double[][]> tableWeightVectors = new ArrayList<double[][]>();
		// each policy is a column matrix of dimension numStates x 1
	//		 HashMap<Integer, double[][]> tablePolicyVectors = new ArrayList<double[][]>();
		// each value is a column matrix of dimension numStates x 1  (i.e. it is NOT a row vector)
	//		 HashMap<Integer, double[][]> tableValueVectors = new ArrayList<double[][]>();


		
		
		for(int table=1; table< env.getNumTrueTables()+1; table++) {
			
			//sample a weightMatrix for each unique table/reward-function
			double [][] weightMatr = new double [numFeatures][1];
			int [] randomPermutation1 = VectorUtility.createPermutatedVector(numFeatures,0);
			Double k = Math.ceil(0.3*numFeatures);
			//for each ground-truth table/expert randomly select a subset of features that are indicative/relevant for that table 
			int [] subVectorOfRndPermutation = VectorUtility.rangeOfVector(randomPermutation1, 0, k.intValue()-1, 1);
			double [][] randomWeights = MersenneTwisterFastIRL.RandomUniformMatrix(numFeatures, 1, 1);

			//randomly generate weights for each feature of relevance for the given ground-truth table/expert
			for (int r: subVectorOfRndPermutation) {
				weightMatr[r][0]= (randomWeights[r][0]*2)-1; 
			}
			


			RewardFunctionGenerationCancer.generateWeightedRewardFunction(env, weightMatr);

			
			
			//begin generating demo trajectories for current ground truth table/expert
			Map<String, double[][]> policy_Value_H_Q_matricesForDemoData = PolicySolverCancer.runPolicyIteration(env, irlAlgo, null);
			double [][] policyUsedtoCreateDemoTrajs = policy_Value_H_Q_matricesForDemoData.get("P");
			System.out.println("Ground truth policy of table "+table+":");
			for (int s=0; s<policyUsedtoCreateDemoTrajs.length; s++) {
				System.out.println("state "+s+" : action "+policyUsedtoCreateDemoTrajs[s][0]);
			}
			
			//JK added 4.26.2019 so we can compare ground-truth vs. inferred reward functions
			tblWeightVectorsMap.put(table, weightMatr);
			tblPolicyVectorsMap.put(table, policyUsedtoCreateDemoTrajs);
		
//			Sample trajectories by executing policy piL
			double [] valueFunctionsArray = new double [numDemoTrajPerTable];
			for(int traj_i=0; traj_i< numDemoTrajPerTable; traj_i++) { //for each demo trajectory to be created (of a given reward-function/table type)
				//NOTE: each trajectory consists of a 1. state-sequence and 2. action-sequence, each with same number of steps
				double [][] trajectory_i = new double [2][numStepsPerTraj];
				
				int sampleState = SampleMultinomialIRL.sampleSingleStateFromMultinomial(100, env.getStartDistribution(), RNG);
				double value_i = 0.0; //initialize the value function of policy for current trajectory 
				for(int step=0; step< numStepsPerTraj; step++) { //for each step of the trajectory

					//retrieve the action to be executed at the current state according to the policy used to generate the demo trajectories
					Double action = policyUsedtoCreateDemoTrajs[sampleState][0];
					//retrieve the reward value for the state-action pair according to the reward function stored in the MDP env.rewardFunction
					Double rewardValue = env.getRewardFunction()[sampleState][action.intValue()];
					
					// update the value function with the discounted reward; XXX: Does the exponent need to be ^(step#-1)? or just ^step
					value_i = value_i +(rewardValue*Math.pow(discountFactor, (step)));
					trajectory_i[0][step]= (double) sampleState; //set the state for the current step in this trajectory
					trajectory_i[1][step] = action; //ste the action for the current step in this trajectory
					
					DoubleMatrix transitionMatrixForAction_i = new DoubleMatrix(env.getTransitionMatrix().get(action.intValue()));
					//sample the NEXT STATE according to the transition matrix for current state 'sampleState' and action 'a'
					sampleState = SampleMultinomialIRL.sampleSingleStateFromMultinomial (10, 
																						 transitionMatrixForAction_i.getColumn (sampleState).toArray(), 
																						 RNG);
				}
				valueFunctionsArray[traj_i]= value_i;
				dataTrajectories.add(traj_i, trajectory_i); //add trajectory to trajectory dataset
				
				tblAssignmentMatrix[((table-1)*numDemoTrajPerTable)+traj_i] = new double [] {(double) table};
				
			}
			//double valueFunctionMean = VectorUtility.getMean(valueFunctionsArray); //compute the mean of value functions for all trajectories
			//compute the variance of value functions for all trajectories
			//double valueFunctionVariance = VectorUtility.getVariance(valueFunctionsArray);
		}	
		return dataTrajectories;
	}
	
	
	
	public static ArrayList<double[][]> generateNonUniformDistributedDemoData (MDPCancer env, IRLAlgorithmCancer irlAlgo, 
																				double[][] tblAssignmentMatrix, 
																				Map<Integer, double[][]> tblWeightVectorsMap, 
																				Map<Integer, double[][]> tblPolicyVectorsMap, 
																				int exptNumber, int totalPathsInt)
	{
		ArrayList<double[][]> dataTrajectories = new ArrayList<double[][]>();
		MersenneTwisterFastIRL RNG = new MersenneTwisterFastIRL(1);

		//int numDemoTrajPerTable = env.getNumDemoTrajectoriesPerTable();  
		//The number of demos attributed to each table will follow a power-law distribution instead of being uniformly distributed.
    	int[] tableSizings = PowerLaw.generatePowerLawDistributedDatasetSizings(env.getNumTrueTables(), totalPathsInt, exptNumber);
    	for (int j=0; j< tableSizings.length; j++) {
    		System.out.println("Table #"+j+" will have "+tableSizings[j]+" paths!");
    	}
    	
		int numFeatures = env.getNumRewardFeatures();
		int numStepsPerTraj= env.getNumStepsPerTrajectory();
		double discountFactor = env.getDiscount();
		
		// XXX:The size of of each of these lists corresponds to the number of ground-truth tables specified by env.getTrueTables in the restaurant. 
		// Each element in the list is a column matrix for that table.
		//stores the weight-vector associated with each table index/value; 
	//		 HashMap<Integer, double[][]> tableWeightVectors = new ArrayList<double[][]>();  
		//each policy is a column matrix of dimension numStates x 1
	//		 HashMap<Integer, double[][]> tablePolicyVectors = new ArrayList<double[][]>();
		//each value is a column matrix of dimension numStates x 1  (i.e. it is NOT a row vector)
	//		 HashMap<Integer, double[][]> tableValueVectors = new ArrayList<double[][]>();

		int numTrajectoriesFromPreviousTables =0;

		for(int tIndex=0; tIndex< tableSizings.length; tIndex++) {
			
			int table = tIndex+1;
			int numDemoTrajectoriesForTableT = tableSizings[tIndex];
			
			
			//sample a weightMatrix for each unique table/reward-function
			double [][] weightMatr = new double [numFeatures][1];
			int [] randomPermutation1 = VectorUtility.createPermutatedVector(numFeatures,0);
			Double k = Math.ceil(0.3*numFeatures);
			//for each ground-truth table/expert randomly select a subset of features that are indicative/relevant for that table 
			int [] subVectorOfRndPermutation = VectorUtility.rangeOfVector(randomPermutation1, 0, k.intValue()-1, 1);
			//TODO: JK 11.1.2019: Why is our ground-truth reward function using a uniform distribution???
			double [][] randomWeights = MersenneTwisterFastIRL.RandomUniformMatrix(numFeatures, 1, 1);

			//randomly generate weights for each feature of relevance for the given ground-truth table/expert
			for (int r: subVectorOfRndPermutation) {
				weightMatr[r][0]= (randomWeights[r][0]*2)-1; 

			}
			


			RewardFunctionGenerationCancer.generateWeightedRewardFunction(env, weightMatr);

			
			
			//begin generating demo trajectories for current ground truth table/expert
			Map<String, double[][]> policy_Value_H_Q_matricesForDemoData = PolicySolverCancer.runPolicyIteration(env, irlAlgo, null);
			double [][] policyUsedtoCreateDemoTrajs = policy_Value_H_Q_matricesForDemoData.get("P");
			System.out.println("Ground truth policy of table "+table+":");
			for (int s=0; s<policyUsedtoCreateDemoTrajs.length; s++) {
				System.out.println("state "+s+" : action "+policyUsedtoCreateDemoTrajs[s][0]);
			}
			
			//JK added 4.26.2019 so we can compare ground-truth vs. inferred reward functions
			tblWeightVectorsMap.put(table, weightMatr);
			tblPolicyVectorsMap.put(table, policyUsedtoCreateDemoTrajs);
		
//			Sample trajectories by executing policy piL
			double [] valueFunctionsArray = new double [numDemoTrajectoriesForTableT];
			//for each demo trajectory to be created (of a given reward-function/table type)
			for (int traj_i = 0; traj_i < numDemoTrajectoriesForTableT; traj_i++)
			{
				//NOTE: each trajectory consists of a 1. state-sequence and 2. action-sequence, each with same number of steps
				double [][] trajectory_i = new double [2][numStepsPerTraj];
				
				int sampleState = SampleMultinomialIRL.sampleSingleStateFromMultinomial(100, env.getStartDistribution(), RNG);
				double value_i = 0.0; //initialize the value function of policy for current trajectory 
				for (int step = 0; step < numStepsPerTraj; step++)
				{ //for each step of the trajectory

					//retrieve the action to be executed at the current state according to the policy used to generate the demo trajectories
					Double action = policyUsedtoCreateDemoTrajs[sampleState][0];
					//retrieve the reward value for the state-action pair according to the reward function stored in the MDP env.rewardFunction
					Double rewardValue = env.getRewardFunction()[sampleState][action.intValue()];
					
					// update the value function with the discounted reward; XXX: Does the exponent need to be ^(step#-1)? or just ^step
					value_i = value_i +(rewardValue*Math.pow(discountFactor, (step)));
					trajectory_i[0][step]= (double) sampleState; //set the state for the current step in this trajectory
					trajectory_i[1][step] = action; //ste the action for the current step in this trajectory
					
					DoubleMatrix transitionMatrixForAction_i = new DoubleMatrix (env.getTransitionMatrix ().get (action.intValue ()));
					//sample the NEXT STATE according to the transition matrix for current state 'sampleState' and action 'a'
					sampleState = SampleMultinomialIRL.sampleSingleStateFromMultinomial (10, 
																						 transitionMatrixForAction_i.getColumn (sampleState).toArray (),
																						 RNG); 
				}
				valueFunctionsArray[traj_i]= value_i;
				int trueTrajPosition =numTrajectoriesFromPreviousTables+traj_i;
				dataTrajectories.add(trueTrajPosition, trajectory_i); //add trajectory to trajectory dataset JK modified 9.27.2019 
				// JK 9.27.2019 NOTE: Using .add() to ArrayList() pushes an existing element at a given position to the right.
				// Thus the trajectories for the 1st table (index 0) are at the bottom of this list since they were among the first added.
				
				//tblAssignmentMatrix[((table-1)*numDemoTrajectoriesForTableT)+traj_i] = new double [] {(double) table};
				tblAssignmentMatrix[trueTrajPosition] = new double [] {(double) table};

			}
			numTrajectoriesFromPreviousTables += numDemoTrajectoriesForTableT;
			//double valueFunctionMean = VectorUtility.getMean(valueFunctionsArray); //compute the mean of value functions for all trajectories
			//compute the variance of value functions for all trajectories
			//double valueFunctionVariance = VectorUtility.getVariance(valueFunctionsArray);
			
		}	
		return dataTrajectories;
	}
	
	
	public static double expectedValDifference (List<double[][]> demoTrajectories, Map<Integer, double[][]> groundTruthTableWeightVectorsMap, 
												Map<Integer, double[][]> groundTruthTablePolicyVectorsMap, double[][] groundTruthTableAssignmentMatrix,
												IRLRestaurant bestMHSampledRestaurant, double mdpDiscountVal, int numSimulMDPStates, 
												int numSimulMDPActions, int numSimulMDPRewardFeatures, int numSimulExperts, int numDemosPerExpert)
	{
		
		double EVD = 0.0;
		DoubleMatrix experimentValueDifference = new DoubleMatrix(demoTrajectories.size()*demoTrajectories.size(), 1);
		DoubleMatrix trueValueFunction = new DoubleMatrix(demoTrajectories.size()*demoTrajectories.size(), 1);

		
		int trueTableIDkFor_customerj =0;
		double[][] trueWeightVectorkFor_customerj=null;
		double[][] truePolicyVectorkFor_customerj=null;
		//MDPCancer mdpForResultsCalculation = new MDPCancer("microbial-influence", 1, 0.9, 64, 4, 16, false);
//		double discountVal = 0.9;
//		int numStatesInMDP = 64;
//		int numActionsInMDP = 4;
//		int numberOfRewardFeatures = 16;

		MDPCancer mdpForResultsCalculation = new MDPCancer ("microbial-influence", 1, mdpDiscountVal, numSimulMDPStates, numSimulMDPActions, 
															numSimulMDPRewardFeatures, false, true,  numSimulExperts,  numDemosPerExpert);
		
		
		
		Pair<double[][], double [][]> HAndValueMatrices = null;
		DoubleMatrix hMatrix = null;
//		DoubleMatrix vMatrix = null;	// GTD 12/24/19 Not used
		DoubleMatrix hMatrix_transposed = null;
		DoubleMatrix weightVector_transposed = null;
		DoubleMatrix initialStartProbDistribution = new DoubleMatrix(mdpForResultsCalculation._startDistribution);

		DoubleMatrix valueForCustomerj = null;
		double trueValueForCustomerj=0.0;
		double learnedValueForCustomerj= 0.0;
		double valueDifferenceForCustomerj = 0.0;
		
		for (int c_j =0 ; c_j < demoTrajectories.size(); c_j++) {
			trueTableIDkFor_customerj = (int) groundTruthTableAssignmentMatrix[c_j][0];
			trueWeightVectorkFor_customerj = groundTruthTableWeightVectorsMap.get(trueTableIDkFor_customerj);
			truePolicyVectorkFor_customerj = groundTruthTablePolicyVectorsMap.get(trueTableIDkFor_customerj);
			HAndValueMatrices = PolicySolverCancer.policyEvaluationStep (mdpForResultsCalculation, truePolicyVectorkFor_customerj, 
																		 trueWeightVectorkFor_customerj);
			hMatrix = new DoubleMatrix(HAndValueMatrices.getFirst());
//			vMatrix = new DoubleMatrix(HAndValueMatrices.getSecond());
			hMatrix_transposed = hMatrix.transpose();
			weightVector_transposed = new DoubleMatrix(trueWeightVectorkFor_customerj).transpose();
			valueForCustomerj = weightVector_transposed.mmul(hMatrix_transposed.mmul(initialStartProbDistribution));
			trueValueForCustomerj = valueForCustomerj.sum();
			trueValueFunction.put(c_j, 0, trueValueForCustomerj);
		}
		
		
		DoubleMatrix learnedValueFunction = new DoubleMatrix(demoTrajectories.size()*demoTrajectories.size(), 1);
		int learnedTableIDkFor_customerj =0;
		double[][] learnedWeightVectorkFor_customerj=null;
		double[][] learnedPolicyVectorkFor_customerj=null;
		double[][] learnedTableAssignmentMatrix = bestMHSampledRestaurant.getSeatingArrangement();
		Map<Integer, double[][]> learnedTablePolicyVectorsMap = bestMHSampledRestaurant.getPolicyMatrices();
		Map<Integer, double[][]> learnedTableWeightVectorsMap = bestMHSampledRestaurant.getWeightMatrices();

		for (int c_j =0 ; c_j < demoTrajectories.size(); c_j++) {
			learnedTableIDkFor_customerj = (int) learnedTableAssignmentMatrix[c_j][0];
			learnedWeightVectorkFor_customerj = learnedTableWeightVectorsMap.get(learnedTableIDkFor_customerj);
			learnedPolicyVectorkFor_customerj = learnedTablePolicyVectorsMap.get(learnedTableIDkFor_customerj);
			HAndValueMatrices = PolicySolverCancer.policyEvaluationStep (mdpForResultsCalculation, learnedPolicyVectorkFor_customerj, 
																		 learnedWeightVectorkFor_customerj);
			hMatrix = new DoubleMatrix(HAndValueMatrices.getFirst());
//			vMatrix = new DoubleMatrix(HAndValueMatrices.getSecond());
			hMatrix_transposed = hMatrix.transpose();
			weightVector_transposed = new DoubleMatrix(learnedWeightVectorkFor_customerj).transpose();
			valueForCustomerj = weightVector_transposed.mmul(hMatrix_transposed.mmul(initialStartProbDistribution));
			learnedValueForCustomerj = valueForCustomerj.sum();
			learnedValueFunction.put(c_j, 0, learnedValueForCustomerj);
			valueDifferenceForCustomerj = trueValueFunction.get(c_j, 0) - learnedValueForCustomerj;
			experimentValueDifference.put(c_j, 0, valueDifferenceForCustomerj);
		}
		EVD = experimentValueDifference.mean();
		
		return EVD;
		
	}
	
public static double computeFScore(double[][] groundTruthTableAssignmentMat, IRLRestaurant mayoRestaurant) {
	
		double fscore = 0.0;
		double[][] experimentalTableAssignmentMat = mayoRestaurant.getSeatingArrangement();
		DoubleMatrix grndTruthTblAssignmentMatrix = new DoubleMatrix(groundTruthTableAssignmentMat);
		DoubleMatrix expTblAssignmentMatrix = new DoubleMatrix(experimentalTableAssignmentMat);

		//JK 7.29.2019 this is incorrect since this is a column matrix. int numTrueTables = grndTruthTblAssignmentMatrix.columns;
		Set<Double> uniqTrueTables = MatrixUtilityJBLAS.countNumberUniqueElements(groundTruthTableAssignmentMat);
		int numTrueTables = uniqTrueTables.size();
		
		Set<Double> uniqexpTables = MatrixUtilityJBLAS.countNumberUniqueElements(experimentalTableAssignmentMat);
		int numExpTables = uniqexpTables.size();
		
		int numTrajectories = expTblAssignmentMatrix.rows;
		
		DoubleMatrix truePositiveMatrix = new DoubleMatrix(numTrueTables,1);
		DoubleMatrix falsePositiveMatrix = new DoubleMatrix(numTrueTables,1);
		DoubleMatrix falseNegativeMatrix = new DoubleMatrix(numTrueTables,1);
		DoubleMatrix trueNegativeMatrix = new DoubleMatrix(numTrueTables,1);

		for (int trueTableJ=1; trueTableJ < numTrueTables+1 ; trueTableJ++) {
			DoubleMatrix boolRelevantTrajs = MatrixUtilityJBLAS.toLogicalMatrix(grndTruthTblAssignmentMatrix, (double)trueTableJ);
			//shows only trajectories that actually belong trueTableJ with their experimental tableidx
			DoubleMatrix tmp = expTblAssignmentMatrix.mul(boolRelevantTrajs);
			DoubleMatrix countMatrix = new DoubleMatrix(numExpTables,1);
			
			for (int expTblk=1; expTblk < numExpTables+1; expTblk++) {
				//for all trjaectories assigned to trueTableJ (which may have also been assigned more than one experimental tblIdx) 
				DoubleMatrix trajMatchingMatrix = MatrixUtilityJBLAS.toLogicalMatrix(tmp, (double)expTblk);
				//get a count of how many of the trajs belonging to trueTablej were assigned to each experimentaltblIdx
				double countNumCorrectTrajs = trajMatchingMatrix.sum();
					
				countMatrix.put(expTblk-1, countNumCorrectTrajs);  //store the number of true trajs assigned to the kth experimental table
			}
			double trajCountOfHighestTrueCountTblIdx = countMatrix.max(); //get the highest trajcount associated with any expeirmental table id
			int highestTrueCountTblIdx = countMatrix.argmax()+1; //get the experimental tableidx associated with this high trajectory count

			truePositiveMatrix.put(trueTableJ-1, trajCountOfHighestTrueCountTblIdx); 
			
			DoubleMatrix logicalMatrixOfHighestCountTblIdx = MatrixUtilityJBLAS.toLogicalMatrix(expTblAssignmentMatrix, highestTrueCountTblIdx);
			//count total number of trajecories assigned to this table
			double countTotalNumTrajAssignedToHighestExpTblIdx = logicalMatrixOfHighestCountTblIdx.sum();
			double numFalsePositives = countTotalNumTrajAssignedToHighestExpTblIdx-trajCountOfHighestTrueCountTblIdx;
			falsePositiveMatrix.put(trueTableJ-1, numFalsePositives);
			
			double countNumTrueTrajs = boolRelevantTrajs.sum();
			double numFalseNegatives = countNumTrueTrajs-trajCountOfHighestTrueCountTblIdx;
			falseNegativeMatrix.put(trueTableJ-1, numFalseNegatives);
			
			double numTrueNegatives = numTrajectories - (truePositiveMatrix.get (trueTableJ - 1) + falsePositiveMatrix.get (trueTableJ - 1) + 
									  falseNegativeMatrix.get (trueTableJ - 1));
			trueNegativeMatrix.put (trueTableJ - 1, numTrueNegatives);
		}
		
		double precision = truePositiveMatrix.sum()/(truePositiveMatrix.sum()+falsePositiveMatrix.sum());
		System.out.println("Precision = "+precision);
		
		double recall = truePositiveMatrix.sum()/(truePositiveMatrix.sum()+ falseNegativeMatrix.sum());
		System.out.println("Recall = "+recall);

		fscore = (2.0*precision*recall)/(precision+recall);
		return fscore;
		
	}

	public static double computeNMI(double[][] grndTruthTblAssignmentMat, IRLRestaurant mayoRestaurant) {
		
		double[][] experimentalTableAssignmentMat = mayoRestaurant.getSeatingArrangement();
		DoubleMatrix grndTruthTblAssignmentMatrix = new DoubleMatrix(grndTruthTblAssignmentMat);
		DoubleMatrix expTblAssignmentMatrix = new DoubleMatrix(experimentalTableAssignmentMat);

		//JK 7.29.2019 incorrect. This is a column matrix with only 1 column. int numTrueTables = grndTruthTblAssignmentMatrix.columns;
		Set<Double> uniqTrueTables = MatrixUtilityJBLAS.countNumberUniqueElements(grndTruthTblAssignmentMat);
		int numTrueTables = uniqTrueTables.size();

		Set<Double> uniqexpTables = MatrixUtilityJBLAS.countNumberUniqueElements(experimentalTableAssignmentMat);
		int numExpTables = uniqexpTables.size();
		int numTrajectories = expTblAssignmentMatrix.rows;

		
		double NMI = 0.0;
		DoubleMatrix trueClassMatrix = new DoubleMatrix (numTrueTables, 1);
		DoubleMatrix expClusterMatrix = new DoubleMatrix (numExpTables, 1);
		double[][] conditionalMatrix = new double[numExpTables][numTrueTables];
		
		for (int trueTblj = 1; trueTblj < numTrueTables + 1; trueTblj++)
		{
			DoubleMatrix trueBoolRelevantTrajsA = MatrixUtilityJBLAS.toLogicalMatrix (grndTruthTblAssignmentMatrix, trueTblj);
			trueClassMatrix.put (trueTblj - 1, trueBoolRelevantTrajsA.sum ());
		}
		
		for (int expTblk = 1; expTblk < numExpTables + 1; expTblk++)
		{
			DoubleMatrix expBoolRelevantTrajsB = MatrixUtilityJBLAS.toLogicalMatrix (expTblAssignmentMatrix, expTblk);
			expClusterMatrix.put (expTblk - 1, expBoolRelevantTrajsB.sum ());
		}
		
		for (int exptbl_k = 1; exptbl_k < numExpTables + 1; exptbl_k++)
		{
			DoubleMatrix expBoolRelevantTrajsC = MatrixUtilityJBLAS.toLogicalMatrix (expTblAssignmentMatrix, exptbl_k);
			
			for (int trueTbl_j = 1; trueTbl_j < numTrueTables + 1; trueTbl_j++)
			{
				DoubleMatrix trueBoolRelevantTrajsD = 
						MatrixUtilityJBLAS.toLogicalMatrix (grndTruthTblAssignmentMatrix.mul (expBoolRelevantTrajsC), trueTbl_j);
				
				double countFractionTrueTrajsWithinExpTrajs = trueBoolRelevantTrajsD.sum ();
				if (countFractionTrueTrajsWithinExpTrajs > 0.0)
				{
					conditionalMatrix[exptbl_k - 1][trueTbl_j - 1] = (countFractionTrueTrajsWithinExpTrajs / numTrajectories) * 
																	 Math.log ((numTrajectories * countFractionTrueTrajsWithinExpTrajs) / 
																			   (expClusterMatrix.get (exptbl_k - 1)) / 
																			   trueClassMatrix.get (trueTbl_j - 1));
				}
			}
		}
		
		DoubleMatrix countFractionB = expClusterMatrix.div (numTrajectories);
		DoubleMatrix boolClusteringWithTrajs = expClusterMatrix.ge (0.0);
		DoubleMatrix countFractionC = countFractionB.mul (boolClusteringWithTrajs);
		
		double[][] logProduct = MatrixUtilityJBLAS.log (countFractionC.toArray2 ());
		DoubleMatrix lp = new DoubleMatrix (logProduct);
		DoubleMatrix matrproduct = countFractionC.mul (lp);
		double sumMat = matrproduct.sum ();
		double clusterLabelEntropy = sumMat * -1.0;
		//double clusterLabelEntropy = countFractionC.mul().sum()*(-1.0);
		
		DoubleMatrix countFractionD = trueClassMatrix.div (numTrajectories);
		double classLabelEntropy = countFractionD.mul (new DoubleMatrix (MatrixUtilityJBLAS.log (countFractionD.toArray2 ()))).sum () * (-1.0);
		
		NMI = (2.0 * new DoubleMatrix (conditionalMatrix).sum ()) / (clusterLabelEntropy + classLabelEntropy);
		
		return NMI;
	}
	
	
	public static void main (String[] args) throws Exception, FileNotFoundException, IOException, ClassNotFoundException, InstantiationException, 
													IllegalAccessException, NoSuchMethodException, IllegalArgumentException, InvocationTargetException
	{
		CRC_Prediction.CommandLineIRLOptions.ParseReturn parseReturn = CommandLineIRLOptions.parse (args);		

		long	seed = parseReturn._seed;
		int		numThreads = parseReturn._numThreads;
		int		numGPUs = parseReturn._numCuda;
		boolean maxTablesBool = parseReturn._maxTablesDoExist;
		boolean profile = parseReturn._profile;
		int		numTables = maxTablesBool ? parseReturn._maxTablesInRestaurant : 0;
		
		double inverseTemperatureEta = parseReturn._inverseTemperatureEta; // 10.0; //eta parameter for bayesian IRL
		double rewardFuction_mean = parseReturn._rewardFunctionMean; // 0.0; //gaussian prior mu parameter
		double rewardFunction_stDev = parseReturn._rewardFunctionStDev; // 0.1; //gaussian prior sigma parameter
		double alpha = parseReturn._alphaConcentration; // 1.0; //concentration parameter
		double discountHyperParam = parseReturn._discount; // 0.0; //discount hyperparameter
		
		int maxMHIterations = parseReturn._maxMHIterations; // 1000;
		int iterationsForTableAssignmentUpdate = parseReturn._iterationsForTableAssignmentUpdate; // 2; default
		int iterationsForRewardFuctionUpdate = parseReturn._iterationsForRewardFunctionUpdate;// 10; default
		int iterationsForTransferLearning = parseReturn._iterationsForTransferLearning;// 100;
		double discountVal = parseReturn._discountValForMDP;
		
		int numExperiments = parseReturn._numExperiments;
		
		int numGridWorldStates = parseReturn._numMDPstates;
		int numGridWorldActions = parseReturn._numMDPactions;
		int numGridWorldExperts = parseReturn._numSimulatedExperts;
		
		int numGridWorldRewardFeatures = numGridWorldStates/4;
		
		int numTrajsPerSimulExpert = parseReturn._numDemonstrationsPerExpert;

		THashSet<IRLRestaurant> MHRestarurantSamplesSet = new THashSet<IRLRestaurant>();
		IRLRestaurant bestMHSampledRestaurant_DPMIRL = new IRLRestaurant();
		
		IRLRestaurant bestMHSampledRestaurant_PURIRL = new IRLRestaurant();

		
		
		LikelihoodFunctionCancer llhfunction = new LikelihoodFunctionCancer(inverseTemperatureEta);
		Prior prior = new Prior("gaussian", rewardFuction_mean, rewardFunction_stDev, 3 );
		//Prior prior = new Prior("uniform", rewardFuction_mean, rewardFunction_stDev, 4 );

//		IRLAlgorithmCancer irlalgo = new IRLAlgorithmCancer ("CRCIRL", llhfunction, prior, 1, alpha,discountHyperParam, maxMHIterations, 
//															 iterationsForTableAssignmentUpdate, iterationsForRewardFuctionUpdate, 
//															 iterationsForTransferLearning);
		//JK creating individual IRLAlgorithmCancerObjects for DPM and PUR-IRL
		IRLAlgorithmCancer irlalgo_dpmIRL = new IRLAlgorithmCancer ("CRCIRL", llhfunction, prior, 1, alpha, 0.0, maxMHIterations, 
																	iterationsForTableAssignmentUpdate, iterationsForRewardFuctionUpdate, 
																	iterationsForTransferLearning); 
		//JK creating individual IRLAlgorithmCancerObjects for DPM and PUR-IRL
		IRLAlgorithmCancer irlalgo_PURIRL = new IRLAlgorithmCancer ("CRCIRL", llhfunction, prior, 1, alpha,discountHyperParam , maxMHIterations, 
																	iterationsForTableAssignmentUpdate, iterationsForRewardFuctionUpdate, 
																	iterationsForTransferLearning);

		
		//MDPCancer reinforcementLearningEnvironment = new MDPCancer("microbial-influence", 1, 0.9, 64, 4, 16, false, true);
		MDPCancer reinforcementLearningEnvironment = new MDPCancer ("microbial-influence", 1, discountVal, numGridWorldStates, numGridWorldActions, 
																	numGridWorldRewardFeatures, false, true,numGridWorldExperts, numTrajsPerSimulExpert);

//		int numUninformlyDistributedPaths = reinforcementLearningEnvironment.getNumTrueTables () * 
//				reinforcementLearningEnvironment._numberDemoTrajectoriesPerTable;
//		int numNonUniformDistributedPaths = 30; 
		int numNonUniformDistributedPaths = 100; 

		
		//DEPRECATE intializing groundTruthTableAssignmentMatrix using env.numDemoTrajectories per table; instead use int values
//		double[][] groundTruthTableAssignmentMatrix = new double[reinforcementLearningEnvironment.getNumTrueTables () * 
//		                                                         reinforcementLearningEnvironment._numberDemoTrajectoriesPerTable][1];
		
		//Called for UNIFORM distributed trajectories
		//double [][] groundTruthTableAssignmentMatrix = new double [numUninformlyDistributedPaths][1];


		//Called for NON-UNIFORM (Power-law) distributed trajectories
		double [][] groundTruthTableAssignmentMatrix = new double [numNonUniformDistributedPaths][1];

		
		//XXX:The size of of each of these maps corresponds to the current number of TRUE tables in the restaurant. 
		// Each element in the list is a column matrix for that table.
		// stores the weight-vector associated with each table index/value; Although the numberRewardFeatures is fixed, 
		// the number of active tables at any given moment can change, so we need the set of weight vectors be dynamic in size.
		Map<Integer, double[][]> groundTruthTableWeightVectorsMap = new HashMap<Integer,double[][]>();
		// each policy is a column matrix of dimension numStates x 1
		Map<Integer, double[][]> groundTruthTablePolicyVectorsMap = new HashMap<Integer,double[][]>();
		
//		HashMap<Integer, Double> EVD_map = new HashMap<Integer, Double>();
//		HashMap<Integer, Double> F1SCORE_map = new HashMap<Integer, Double>();
//		HashMap<Integer, Double> NMI_map = new HashMap<Integer, Double>();
		double[] EVD_vector = new double[numExperiments];
		double[] F1SCORE_vector = new double[numExperiments];
		double[] NMI_vector = new double[numExperiments];
		double[] numInfTables_vector = new double[numExperiments];
		
		double[] EVD_PUR_vector = new double[numExperiments];
		double[] F1SCORE_PUR_vector = new double[numExperiments];
		double[] NMI_PUR_vector = new double[numExperiments];
		double[] numInfTables_PUR_vector = new double[numExperiments];
		
		if (numGPUs > 0)
		{
			Thread	curThread = Thread.currentThread ();
			String	name = curThread.getName ();
			
			name = name + InferenceAlgoCancer.kGPU + (numGPUs / 2);
			curThread.setName (name);
			InferenceAlgoCancer.initGPUs (numGPUs);
		}
		
		if (profile)
		{
			Controller.startCPURecording (true);
			Controller.startThreadProfiling ();
		}

		for(int e=0; e< numExperiments; e++) {
			//Called for uniform distributed trajectories
//			List<double[][]> demoTrajectories = generateDemoData (reinforcementLearningEnvironment, irlalgo, groundTruthTableAssignmentMatrix, 
//																	groundTruthTableWeightVectorsMap,groundTruthTablePolicyVectorsMap);
			
			//Called for NON-UNIFORM (Power-law) distributed trajectories
//			List<double[][]> demoTrajectories = generateNonUniformDistributedDemoData (reinforcementLearningEnvironment, irlalgo, 
//																						groundTruthTableAssignmentMatrix, 
//																						groundTruthTableWeightVectorsMap, 
//																						groundTruthTablePolicyVectorsMap, e, 
//																						numNonUniformDistributedPaths);
			//Doesn't matter which IRLAlgorithmCancer object is used as input arg
			List<double[][]> demoTrajectories = generateNonUniformDistributedDemoData (reinforcementLearningEnvironment, irlalgo_dpmIRL, 
																						groundTruthTableAssignmentMatrix, 
																						groundTruthTableWeightVectorsMap, 
																						groundTruthTablePolicyVectorsMap, e, 
																						numNonUniformDistributedPaths);

			
			IRLRestaurantFactory irlfactory = new IRLRestaurantFactory(parseReturn._outputDir);
			MDPCancerFactory mdpfactory = new MDPCancerFactory(parseReturn._outputDir);
	//		File preexistingIRLRestaurantFile = new File("/Users/m186806/BestIRLRestaurantModelGridWorld.serialized");
	//    	IRLRestaurant bestMHSampledRestaurantFromSerialFile = irlfactory.get(preexistingIRLRestaurantFile);
	//		double f1scoreA = computeFScore(groundTruthTableAssignmentMatrix, bestMHSampledRestaurantFromSerialFile.getSeatingArrangement()  );
	//		System.out.println("F1-score : "+f1scoreA);
	    	
	    	
//			InferenceAlgoCancer.ChineseRestaurantProcessInference (reinforcementLearningEnvironment, demoTrajectories, irlalgo, MHRestarurantSamplesSet, 
//																	bestMHSampledRestaurant, seed, numThreads, numTables, parseReturn._startFromScratch, 
//																	irlfactory, mdpfactory);
			//JK 11.2.2019 added to allow _DPMIRL for distinction during simultaneous comparison of DPM-IRL vs PUR-IRL on same demo trajectory dataset
			InferenceAlgoCancer.ChineseRestaurantProcessInference (reinforcementLearningEnvironment, demoTrajectories, irlalgo_dpmIRL, 
																	MHRestarurantSamplesSet, bestMHSampledRestaurant_DPMIRL, seed, numThreads, numGPUs, 
																	numTables, parseReturn._startFromScratch, irlfactory, mdpfactory, profile);
			//JK 11.2.2019 added to allow _DPMIRL for distinction during simultaneous comparison of DPM-IRL vs PUR-IRL on same demo trajectory dataset
			InferenceAlgoCancer.ChineseRestaurantProcessInference (reinforcementLearningEnvironment, demoTrajectories, irlalgo_PURIRL, 
																	MHRestarurantSamplesSet, bestMHSampledRestaurant_PURIRL, seed, numThreads, numGPUs, 
																	numTables, parseReturn._startFromScratch, irlfactory, mdpfactory, profile);

			
			
			//Serialize best IRL restaurant as well as MDPCancer object used during inference 
	
	        try {
	        	//Store the best IRLRestaurant model irlfactory.write(bestMHSampledRestaurant);
				// JK 11.2.2019 added to allow _DPMIRL for distinction during simultaneous comparison of DPM-IRL vs PUR-IRL 
	        	// on same demo trajectory dataset
				irlfactory.write (bestMHSampledRestaurant_DPMIRL);

	            System.out.println ("Stored the best IRLRestaurant model : " + "/Users/m186806/BestIRLRestaurantGridWorld.serialized");
			}
			catch (IOException ioe)
			{
				System.err.println ("Error writing IRLRestaurant model file -- probably corrupt -- try again");
				throw ioe;
			}
			try
			{
				mdpfactory.write (reinforcementLearningEnvironment);
				System.out.println ("Stored the MDPCancer object : " + "/Users/m186806/MDPGridWorld.serialized");
				
			}
			catch (IOException ioe)
			{
				System.err.println ("Error writing MDP GridWorld object file -- probably corrupt -- try again");
				throw ioe;
			}
			
			double EVD=0.0;
			double EVD_PUR = 0.0;
//			EVD = expectedValDifference (demoTrajectories, groundTruthTableWeightVectorsMap, groundTruthTablePolicyVectorsMap, 
//										 groundTruthTableAssignmentMatrix, bestMHSampledRestaurant, discountVal, numGridWorldStates, 
//										 numGridWorldActions, numGridWorldRewardFeatures, numGridWorldExperts, numTrajsPerSimulExpert);
			//// JK 11.2.2019 added to allow _DPMIRL for distinction during simultaneous comparison of 
			// DPM-IRL vs PUR-IRL on same demo trajectory dataset
			EVD = expectedValDifference (demoTrajectories, groundTruthTableWeightVectorsMap, groundTruthTablePolicyVectorsMap, 
										 groundTruthTableAssignmentMatrix, bestMHSampledRestaurant_DPMIRL, discountVal, numGridWorldStates, 
										 numGridWorldActions, numGridWorldRewardFeatures, numGridWorldExperts, numTrajsPerSimulExpert);
			//// JK 11.2.2019 added to allow _DPMIRL for Distinction during simultaneous comparison of
			//// DPM-IRL vs PUR-IRL on same demo trajectory dataset
			EVD_PUR = expectedValDifference (demoTrajectories, groundTruthTableWeightVectorsMap, groundTruthTablePolicyVectorsMap, 
											 groundTruthTableAssignmentMatrix, bestMHSampledRestaurant_PURIRL, discountVal, numGridWorldStates, 
											 numGridWorldActions, numGridWorldRewardFeatures, numGridWorldExperts, numTrajsPerSimulExpert);
			
			System.out.println("Overall EVD for this experiment is :"+EVD);
			System.out.println("Overall PUR-IRL EVD for this experiment is :"+EVD_PUR);

			///EVD_map.put(e, EVD);
			EVD_vector[e]=EVD;
			EVD_PUR_vector[e]=EVD_PUR;

			
			//double f1score = computeFScore(groundTruthTableAssignmentMatrix, bestMHSampledRestaurant  );
			////JK 11.2.2019 added to allow _DPMIRL for distinction during simultaneous comparison of 
			// DPM-IRL vs PUR-IRL on same demo trajectory dataset
			double f1score = computeFScore (groundTruthTableAssignmentMatrix, bestMHSampledRestaurant_DPMIRL);
			////JK 11.2.2019 added to allow _DPMIRL for distinction during simultaneous comparison of 
			// DPM-IRL vs PUR-IRL on same demo trajectory dataset
			double f1score_PUR = computeFScore (groundTruthTableAssignmentMatrix, bestMHSampledRestaurant_PURIRL);
			
			System.out.println ("F1-score : " + f1score);
			System.out.println ("PUR F1-score : " + f1score_PUR);
			
			/// F1SCORE_map.put(e, f1score);
			F1SCORE_vector[e] = f1score;
			F1SCORE_PUR_vector[e] = f1score_PUR;
			
			//double nmi = computeNMI(groundTruthTableAssignmentMatrix, bestMHSampledRestaurant);
			////JK 11.2.2019 added to allow _DPMIRL for distinction during simultaneous comparison of 
			// DPM-IRL vs PUR-IRL on same demo trajectory dataset
			double nmi = computeNMI(groundTruthTableAssignmentMatrix, bestMHSampledRestaurant_DPMIRL);
			////JK 11.2.2019 added to allow _DPMIRL for distinction during simultaneous comparison of 
			// DPM-IRL vs PUR-IRL on same demo trajectory dataset
			double nmi_PUR = computeNMI(groundTruthTableAssignmentMatrix, bestMHSampledRestaurant_PURIRL);

			System.out.println("Normalized Mutual information : "+nmi);
			System.out.println("PUR Normalized Mutual information : "+nmi_PUR);

			///NMI_map.put(e, nmi);
			NMI_vector[e]= nmi;
			NMI_PUR_vector[e]= nmi_PUR;

			
			//int numInferredTables = bestMHSampledRestaurant.getWeightMatrices().size();
			////JK 11.2.2019 added to allow _DPMIRL for distinction during simultaneous comparison of 
			// DPM-IRL vs PUR-IRL on same demo trajectory dataset
			int numInferredTables = bestMHSampledRestaurant_DPMIRL.getWeightMatrices().size();
			////JK 11.2.2019 added to allow _DPMIRL for distinction during simultaneous comparison of 
			// DPM-IRL vs PUR-IRL on same demo trajectory dataset
			int numInferredTables_PUR = bestMHSampledRestaurant_PURIRL.getWeightMatrices().size();

			numInfTables_vector[e] = numInferredTables;
			numInfTables_PUR_vector[e] = numInferredTables_PUR;

			
			
			//JK 11.2.2019 Visualize Reward-Function HEATMAP/////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////
			//HashMap<Integer, double[][]> learnedWeightVectorsMap = (HashMap) bestMHSampledRestaurant.getWeightMatrices();
			////JK 11.2.2019 added to allow _DPMIRL for distinction during simultaneous comparison of DPM-IRL vs PUR-IRL
			// on same demo trajectory dataset
			Map<Integer, double[][]> learned_DPM_WeightVectorsMap = bestMHSampledRestaurant_DPMIRL.getWeightMatrices();
			////JK 11.2.2019 added to allow _DPMIRL for distinction during simultaneous comparison of DPM-IRL vs PUR-IRL
			// on same demo trajectory dataset
			Map<Integer, double[][]> learned_PURIRL_WeightVectorsMap = bestMHSampledRestaurant_PURIRL.getWeightMatrices();

			FileWriter csvWriter = null;

			double [][] groundTruthRewardMatrix = null;
			double [][] learned_DPM_RewardMatrix = null;
			double [][] learned_PURIRL_RewardMatrix = null;


			int numStates =0;
//			int numActions =0;	// Not used. GTD 12/24/19
			double [] row = null;
			
			Set<Integer> trueTablesKeySet = groundTruthTableWeightVectorsMap.keySet();
			for (int tablek: trueTablesKeySet) {
				
				//Save ground-truth matrix .csv
				csvWriter = new FileWriter("exp_"+e+"_table_"+tablek+"GroundTruthRewardMatrix.csv");
				RewardFunctionGenerationCancer.generateWeightedRewardFunction (reinforcementLearningEnvironment, 
																				groundTruthTableWeightVectorsMap.get (tablek));
				groundTruthRewardMatrix = reinforcementLearningEnvironment.getRewardFunction();
				numStates = groundTruthRewardMatrix.length;
//				numActions = groundTruthRewardMatrix[0].length;
				
//				//Specify row and column labels if printing out s,a reward matrix. But we are commenting this out since we want 
				// 8x8 grid representation of our reward matrix
//				//set column-labels for matrix
//				csvWriter.append("states");
//				for (int a=0; a<numActions; a++) {
//					csvWriter.append(",");
//					csvWriter.append("A"+Integer.toString(a));					
//				}
//				csvWriter.append("\n");
//				
//				row = null;
//				for (int s=0; s< numStates; s++) {
//					csvWriter.append("S"+Integer.toString(s));
//					row = groundTruthRewardMatrix[s];
//					for (int rVal =0; rVal < row.length; rVal++) {
//						csvWriter.append(",");
//						csvWriter.append(Double.toString(row[rVal]));
//					}
//					csvWriter.append("\n");
//				}
//				csvWriter.flush();
//				csvWriter.close();
				
				//printing reward values for 8x8 grid of states
				row = null;
				int c=1;
				for (int s=0; s< numStates; s++) {
					row = groundTruthRewardMatrix[s];
					csvWriter.append(Double.toString(row[0]));
					if(c % 8 == 0) {
						csvWriter.append("\n");
					}
					else {
						csvWriter.append(",");
					}
					c++;
				}
				csvWriter.flush();
				csvWriter.close();
				
				//Save ****LEARNED DPM-IRL**** reward-matrix .csv
				//////////////////////////////////////////////////////////////////////////

				csvWriter = new FileWriter("exp_"+e+"_table_"+tablek+"Learned_DPMIRL_TruthrewardMatrix.csv");
				RewardFunctionGenerationCancer.generateWeightedRewardFunction (reinforcementLearningEnvironment, 
																				learned_DPM_WeightVectorsMap.get (tablek));
				learned_DPM_RewardMatrix = reinforcementLearningEnvironment.getRewardFunction();
				numStates = learned_DPM_RewardMatrix.length;
//				numActions = learned_DPM_RewardMatrix[0].length;
				
//				//set column-labels for matrix
//				csvWriter.append("states");
//				for (int a=0; a<numActions; a++) {
//					csvWriter.append(",");
//					csvWriter.append("A"+Integer.toString(a));					
//				}
//				csvWriter.append("\n");
//				
//				row = null;
//				for (int s=0; s< numStates; s++) {
//					csvWriter.append("S"+Integer.toString(s));
//					row = learned_DPM_RewardMatrix[s];
//					for (int rVal =0; rVal < row.length; rVal++) {
//						csvWriter.append(",");
//						csvWriter.append(Double.toString(row[rVal]));
//					}
//					csvWriter.append("\n");
//				}
//				csvWriter.flush();
//				csvWriter.close();
			
				//Printing reward values for 8x8 grid of states
				row = null;
				c=1;
				for (int s=0; s< numStates; s++) {
					row = learned_DPM_RewardMatrix[s];
					csvWriter.append(Double.toString(row[0]));
					if(c % 8 == 0) {
						csvWriter.append("\n");
					}
					else {
						csvWriter.append(",");
					}
					c++;
				}
				csvWriter.flush();
				csvWriter.close();
			
			
				//Save ****LEARNED PUR-IRL**** reward-matrix .csv
				//////////////////////////////////////////////////////////////////////////
	
				csvWriter = new FileWriter("exp_"+e+"_table_"+tablek+"Learned_PURIRL_TruthrewardMatrix.csv");
				RewardFunctionGenerationCancer.generateWeightedRewardFunction (reinforcementLearningEnvironment, 
																				learned_PURIRL_WeightVectorsMap.get (tablek));
				learned_PURIRL_RewardMatrix = reinforcementLearningEnvironment.getRewardFunction();
				numStates = learned_PURIRL_RewardMatrix.length;
//				numActions = learned_PURIRL_RewardMatrix[0].length;
				
//				//set column-labels for matrix
//				csvWriter.append("states");
//				for (int a=0; a<numActions; a++) {
//					csvWriter.append(",");
//					csvWriter.append("A"+Integer.toString(a));					
//				}
//				csvWriter.append("\n");
//				
//				row = null;
//				for (int s=0; s< numStates; s++) {
//					csvWriter.append("S"+Integer.toString(s));
//					row = learned_PURIRL_RewardMatrix[s];
//					for (int rVal =0; rVal < row.length; rVal++) {
//						csvWriter.append(",");
//						csvWriter.append(Double.toString(row[rVal]));
//					}
//					csvWriter.append("\n");
//				}
//				csvWriter.flush();
//				csvWriter.close();
				
				//Printing reward values for 8x8 grid of states
				row = null;
				c=1;
				for (int s=0; s< numStates; s++) {
					row = learned_PURIRL_RewardMatrix[s];
					csvWriter.append(Double.toString(row[0]));
					if(c % 8 == 0) {
						csvWriter.append("\n");
					}
					else {
						csvWriter.append(",");
					}
					c++;
				}
				csvWriter.flush();
				csvWriter.close();
			}
	    }
		double evd_mu = VectorUtility.getMean (EVD_vector);
		double evd_se = VectorUtility.getStdDev (EVD_vector);
		double f1score_mu = VectorUtility.getMean (F1SCORE_vector);
		double f1score_se = VectorUtility.getStdDev (F1SCORE_vector);
		double nmi_mu = VectorUtility.getMean (NMI_vector);
		double nmi_se = VectorUtility.getStdDev (NMI_vector);
		double nrTables_mu = VectorUtility.getMean (numInfTables_vector);
		double nrTables_se = VectorUtility.getStdDev (numInfTables_vector);
		
		double evd_mu_PUR = VectorUtility.getMean (EVD_PUR_vector);
		double evd_se_PUR = VectorUtility.getStdDev (EVD_PUR_vector);
		double f1score_mu_PUR = VectorUtility.getMean (F1SCORE_PUR_vector);
		double f1score_se_PUR = VectorUtility.getStdDev (F1SCORE_PUR_vector);
		double nmi_mu_PUR = VectorUtility.getMean (NMI_PUR_vector);
		double nmi_se_PUR = VectorUtility.getStdDev (NMI_PUR_vector);
		double nrTables_mu_PUR = VectorUtility.getMean (numInfTables_PUR_vector);
		double nrTables_se_PUR = VectorUtility.getStdDev (numInfTables_PUR_vector);
		
		System.out.println ("evd_mu =" + evd_mu);
		System.out.println ("evd_se =" + evd_se);
		System.out.println ("fsc_mu =" + f1score_mu);
		System.out.println ("fsc_se =" + f1score_se);
		System.out.println ("nmi_mu =" + nmi_mu);
		System.out.println ("nmi_se =" + nmi_se);
		System.out.println ("nrCl_mu =" + nrTables_mu);
		System.out.println ("nrCl_se =" + nrTables_se);
		
		System.out.println ("evd_mu_PUR =" + evd_mu_PUR);
		System.out.println ("evd_se_PUR =" + evd_se_PUR);
		System.out.println ("fsc_mu_PUR =" + f1score_mu_PUR);
		System.out.println ("fsc_se_PUR =" + f1score_se_PUR);
		System.out.println ("nmi_mu_PUR =" + nmi_mu_PUR);
		System.out.println ("nmi_se_PUR =" + nmi_se_PUR);
		System.out.println ("nrCl_mu_PUR =" + nrTables_mu_PUR);
		System.out.println ("nrCl_se_PUR =" + nrTables_se_PUR);
		
	}
}

