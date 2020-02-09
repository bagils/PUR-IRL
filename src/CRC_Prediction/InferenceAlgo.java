package CRC_Prediction;

import java.util.HashMap;
import java.util.Map;
import java.util.List;

import org.apache.commons.math3.util.CombinatoricsUtils;
import org.apache.commons.math3.util.Pair;
import org.jblas.DoubleMatrix;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

import org.apache.commons.lang3.time.StopWatch;


import CRC_Prediction.Utils.MatrixUtilityJBLAS;
import CRC_Prediction.Utils.VectorUtility;
import gnu.trove.set.hash.THashSet;
import CRC_Prediction.LikelihoodFunction;

public class InferenceAlgo {
	
	public static MersenneTwisterFastIRL RNG;
	
	 //trajectorySet is matrix of dimension : nTrajs x nSteps x 2; each trajectory is of length (nSteps), where each step consists of the PAIR state and action
	
	
	public static void ChineseRestaurantProcessInference (MDP environment,
			List<double[][]> trajectorySet, IRLAlgorithm irlAlgo,
			THashSet<IRLRestaurant> mhSampledRestaurants, IRLRestaurant bestSampledRestaurant)
	{
		RNG = new MersenneTwisterFastIRL (1);
		
		int numberRewardFeatures = environment.getNumRewardFeatures ();
		
		//XXX:Trajectory data: NOTE: each trajectory is defined by 2 sequences:  1. a sequence of states, and 2. a sequence of actions. 
		int numberTrajectories = trajectorySet.size();  //should be same as MDP._numberTrajectories
//		int trajectoryLength = trajectorySet.get(0)[0].length; //should be same as MDP._numberStepsPerTrajectory	// GTD Not used
		
		//map of state-action counts for trajectories of interest
		//includes the 'count' of each 'observed' state-action pair in the subset of trajectoryset.  Thus, if we consider all trajectories to extract 'count' information, the maximum size of this trajectoryInfo dataset is: nTrajs x nSteps x 3 
	    Multimap<Integer,double[]> stateActionPairCountsInfoForAllTrajectories = ArrayListMultimap.create();

		for (int i=0 ; i< numberTrajectories; i++) {
			//Compute occupancy and the empirical policy for trajectories
			
			Map<Integer, double[][]> subsetOfTrajectoriesToAnalyze = new HashMap<Integer, double[][]>();
			subsetOfTrajectoriesToAnalyze.put(i,trajectorySet.get(i));
			stateActionPairCountsInfoForAllTrajectories = computeOccupancy(subsetOfTrajectoriesToAnalyze, environment, stateActionPairCountsInfoForAllTrajectories, false); 
			subsetOfTrajectoriesToAnalyze.clear();
			//set the state-action counts for each trajectory; i.e. the set of state-action pairs visited in this trajectory and their respective counts associated with number of visits 
		}
		
		//initialize tables
		StopWatch watch = StopWatch.createStarted();
		
		/////////////////////all the following parameters are necessary for initializeTables() internal function calls
		
		///each trajectory is randomly assigned to a table index within the range [1,numTrajectories], (NOTE: the number of tables <= numberOfTrajetories), it is still possible that each trajectory is assigned its own unique table/reward-function (in which case the # of reward functions = # trajectories) 
		double [][] tableAssignmentMatrix = MersenneTwisterFastIRL.RandomUniformMatrixWithIntervalMin(numberTrajectories, 1, numberTrajectories, 1); //cl.b = nTraj x 1 matrix of pseudorandom integers drawn from the discrete uniform distribution on the interval [1,nTraj]; this integer corresponds to the 'label/index' of the table associated with the given trajectory

		Double highestTableIndex = MatrixUtilityJBLAS.maxPerColumn(tableAssignmentMatrix)[0];  //returns a row vector containing the maximum values in each column of the 'tableAssignmentMatrix'. Since this matrix is nTraj x 1. We only care about the 0th element of the returned row vector
		int N = highestTableIndex.intValue();
		
		//XXX:The size of of each of these maps corresponds to the current number of ACTIVE tables in the restaurant. Each element in the list is a column matrix for that table.
		Map<Integer, double[][]> tableWeightVectors = new HashMap<Integer,double[][]>();  //stores the weight-vector associated with each table index/value; Although the numberRewardFeatures is fixed, the number of active tables at any given moment can change, so we need the set of weight vectors be dynamic in size.
		Map<Integer, double[][]> tablePolicyVectors = new HashMap<Integer,double[][]>();  //each policy is a column matrix of dimension numStates x 1
		Map<Integer,double[][]> tableValueVectors = new HashMap<Integer,double[][]>();	//each value is a column matrix of dimension numStates x 1  (i.e. it is NOT a row vector)
		Map<Integer,double[][]> tableQVectors = new HashMap<Integer,double[][]>();	//each value is a column matrix of dimension (numStates*numActions) x 1  (i.e. it is NOT a row vector)

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		//initialize table-specific parameters (weightVector, policyVector, valueVector) 
		initializeTables(N, numberTrajectories, environment, irlAlgo, tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors);
		watch.stop();
		System.out.println("Time required for Table Initialization: "+watch.getTime());
		
		//after tableWeightVectors list has been computed...
		
		//changed datastructure from double [] to Map so that we obtain likelihood by tableIndex value
		Map<Integer, Double> restaurantLikelihoods = VectorUtility.nansMap(N); //N becasuse we are starting table indices at 1!
		Map<Integer, Double> restaurantPriors = VectorUtility.nansMap(N);//N becasuse we are starting table indices at 1!
		
		Map<Integer, DoubleMatrix> restGradientsLLH = MatrixUtilityJBLAS.createHashMapOfRealMatricesWithNANS(numberRewardFeatures, N);
		Map<Integer, DoubleMatrix> restGradientsPrior = MatrixUtilityJBLAS.createHashMapOfRealMatricesWithNANS(numberRewardFeatures, N);//N becasuse we are starting table indices at 0!

		
		RestaurantMap rmap1 =generateNewTableAssignmentPartition(environment.getNumStates(), environment.getNumActions(), environment.getNumRewardFeatures(), tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, tableAssignmentMatrix);
		tableAssignmentMatrix = rmap1._restaurantAssignmentMatrix;
		tableWeightVectors = rmap1._restaurantTableWeightMatrices;
		tablePolicyVectors = rmap1._restaurantTablePolicyMatrices;
		tableValueVectors = rmap1._restaurantTableValueMatrices;
		tableQVectors = rmap1._restaurantTableQMatrices;
		
		
		 //calculate scalar LOG posterior probability
		 double  log_posteriorProbability1 = computeLogPosteriorProbabilityForDirichletProcessMixture(trajectorySet, tableAssignmentMatrix, tableWeightVectors, tablePolicyVectors, environment, irlAlgo, false);
		 if(Double.compare(log_posteriorProbability1, bestSampledRestaurant.getLogPosteriorProb())>0) {
			 bestSampledRestaurant.setSeatingArrangement(tableAssignmentMatrix);
			 bestSampledRestaurant.setWeightMatrices(tableWeightVectors);
			 bestSampledRestaurant.setPolicyMatrices(tablePolicyVectors);
			 bestSampledRestaurant.setValueMatrices(tableValueVectors);
			 bestSampledRestaurant.setQMatrices(tableQVectors);

			 bestSampledRestaurant.setLogPosteriorProb(log_posteriorProbability1);
		 }
		 mhSampledRestaurants.add(IRLRestaurant.clone(bestSampledRestaurant));
		
		//***end of function calls for initializing restaurant
		 double log_posteriorProbability2 = 0.0;
		 //begin MH updates for Inference
		 for (int iter=0; iter< irlAlgo.getMaxIterations(); iter++) {

			int [] customersPermutation = VectorUtility.createPermutatedVector(numberTrajectories,0);
			RestaurantMap rmap2= null;
			for (int customer_i : customersPermutation){ //iterate through all customers in random order and determine what table they should be assigned. They be assigned to a NEW TABLE with certain probability.
				 rmap2 = updateTableAssignment(environment, customer_i, irlAlgo, tableAssignmentMatrix, tableWeightVectors, tablePolicyVectors, tableValueVectors,tableQVectors, restaurantLikelihoods, restaurantPriors, restGradientsLLH, restGradientsPrior, stateActionPairCountsInfoForAllTrajectories  );
				 tableAssignmentMatrix = rmap2._restaurantAssignmentMatrix;
				 tableWeightVectors= rmap2._restaurantTableWeightMatrices;
				 tablePolicyVectors = rmap2._restaurantTablePolicyMatrices;
				 tableValueVectors = rmap2._restaurantTableValueMatrices;
				 tableQVectors = rmap2._restaurantTableQMatrices;

				 restaurantLikelihoods = rmap2._restLikeLihoods;
				 restaurantPriors = rmap2._restPriors;
				 restGradientsLLH = rmap2._restGradientsFromLLH;
				 restGradientsPrior = rmap2._restGradientsFromPrior;
//				 stateActionPairCountsInfoForAllTrajectories = rmap2._saPairCountsInfoForSubsetOfTrajectories;
			}
// 			System.out.println("Updated Seating arrangement :");
// 			for (int customer =0; customer < numberTrajectories ; customer++) {
// 				System.out.println("Customer "+customer+" sits at table : "+tableAssignmentMatrix[customer][0]);
// 			}
			
			
			RestaurantMap rmap3 = generateNewTableAssignmentPartition (environment.getNumStates (), environment.getNumActions (), 
																		environment.getNumRewardFeatures (), tableWeightVectors, tablePolicyVectors, 
																		tableValueVectors, tableQVectors, tableAssignmentMatrix);
			tableAssignmentMatrix = rmap3._restaurantAssignmentMatrix;
			tableWeightVectors = rmap3._restaurantTableWeightMatrices;
			tablePolicyVectors = rmap3._restaurantTablePolicyMatrices;
			tableValueVectors = rmap3._restaurantTableValueMatrices;
			tableQVectors = rmap3._restaurantTableQMatrices;
			
			Double	highestTableIndexDBL = MatrixUtilityJBLAS.matrixMaximum (tableAssignmentMatrix);
			int		highestTableIndexINT = highestTableIndexDBL.intValue ();
			// Create permutated vector of length= highestTableIndexINT with starting value 1
			int[]	tablesPermutation = VectorUtility.createPermutatedVector (highestTableIndexINT, 1);
			
			RestaurantMap	rmap6;
			for (int table_i : tablesPermutation)
			{ // iterate through all tables in the restaurant in random order; for each table_i
				// update its reward-function so that its weightMatrix is a reflection of the sum of
				// counts of the s-a pairs (found in the trajectories/customers assigned to that
				// table)
				rmap6 = updateRewardFunctions (trajectorySet, environment, table_i, irlAlgo, tableAssignmentMatrix, 
												tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, 
												restaurantLikelihoods, restaurantPriors, restGradientsLLH, restGradientsPrior);
				
				tableAssignmentMatrix = rmap6._restaurantAssignmentMatrix;
				tableWeightVectors = rmap6._restaurantTableWeightMatrices;
				tablePolicyVectors = rmap6._restaurantTablePolicyMatrices;
				tableValueVectors = rmap6._restaurantTableValueMatrices;
				tableQVectors = rmap6._restaurantTableQMatrices;
				
				restaurantLikelihoods = rmap6._restLikeLihoods;
				restaurantPriors = rmap6._restPriors;
				restGradientsLLH = rmap6._restGradientsFromLLH;
				restGradientsPrior = rmap6._restGradientsFromPrior;
				
				// JK 6.23.2019: setting global stateActioPairCounts map to updated one??
				// stateActionPairCountsInfoForSubsetOfTrajectories =
				// rmap6._saPairCountsInfoForSubsetOfTrajectories;
				
			}
			log_posteriorProbability2 = 
					computeLogPosteriorProbabilityForDirichletProcessMixture (trajectorySet, tableAssignmentMatrix, tableWeightVectors, 
																			  tablePolicyVectors, environment, irlAlgo, false);
			System.out.println ("Iteration #" + iter + " logPosteriorProb = " + log_posteriorProbability2);
			if (Double.compare (log_posteriorProbability2, bestSampledRestaurant.getLogPosteriorProb ()) > 0)
			{
				bestSampledRestaurant.setSeatingArrangement (tableAssignmentMatrix);
				bestSampledRestaurant.setWeightMatrices (tableWeightVectors);
				bestSampledRestaurant.setPolicyMatrices (tablePolicyVectors);
				bestSampledRestaurant.setValueMatrices (tableValueVectors);
				bestSampledRestaurant.setQMatrices (tableQVectors);
				
				bestSampledRestaurant.setLogPosteriorProb (log_posteriorProbability2);
			}
			
			IRLRestaurant	restaurantForCurrentIteration = new IRLRestaurant (tableAssignmentMatrix, tableWeightVectors, tablePolicyVectors, 
																				tableValueVectors, tableQVectors, log_posteriorProbability2);
			mhSampledRestaurants.add (restaurantForCurrentIteration);
			System.out.println ("*******Current best restaurant has logPosteriorProb = " +  bestSampledRestaurant.getLogPosteriorProb ());
			 
		}//end for-loop for MH algorithm iterations
		System.out.println ("Overall best restaurant has logPosteriorProb = " +  bestSampledRestaurant.getLogPosteriorProb ());
		System.out.println ("Seating arrangement :");
		double[][]	seatingArrangement = bestSampledRestaurant.getSeatingArrangement ();
		for (int customer = 0; customer < numberTrajectories; ++customer)
		{
			System.out.println ("Customer " + customer + " sits at table : " + seatingArrangement[customer][0]);
		}		 
		 
	}
	
	
	// this function call is kind of redundant...we can just call the for-loop within
	// 'dirichelProcessMHLInference()' method
	private static void initializeTables (int N, int nTrajectories, MDP env, IRLAlgorithm irlAlgo, Map<Integer, double[][]> tableWeightVectors, 
										  Map<Integer, double[][]> tablePolicyVectors, Map<Integer, double[][]> tableValueVectors,
			Map<Integer, double[][]> tableQVectors)
	{
		
		for (int table_i = 1; table_i < N + 1; table_i++)
		{
			
			// generate/sample new weight, policy and value vector to associate/map with each table index-value i
			generateNewWeights (table_i, env, irlAlgo, tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, true);
		}
		
	}
	
	/**
	 * Sample new weight and compute its policy and value
	 * @param env
	 * @param irloptions
	 * @param tblWeightVectors
	 * @param tblPlVectors
	 * @param tblVlVectors
	 */
	// private static List<double[][]> generateNewWeights(int tableIndexValue, MDP env, IRLAlgorithm
	// irloptions, Map<Integer, double[][]> tblWeightVectors, Map<Integer, double[][]> tblPlVectors,
	// Map<Integer, double[][]> tblVlVectors,Map<Integer, double[][]> tblQVectors, boolean
	// changeWeights) {
	private static Map<String, double[][]> generateNewWeights (int tableIndexValue, MDP env,
			IRLAlgorithm irloptions, Map<Integer, double[][]> tblWeightVectors,
			Map<Integer, double[][]> tblPlVectors, Map<Integer, double[][]> tblVlVectors,
			Map<Integer, double[][]> tblQVectors, boolean changeWeights)
	{
		
		Map<String, double[][]> w_p_v_qHashMap = new HashMap<String, double[][]> ();
		
		int numFeatures = env.getNumRewardFeatures();
		double lowerB = irloptions._lowerRewardBounds;
		double upperB = irloptions._upperRewardBounds;
		Prior pr = irloptions.getPrior();
		//if prior is normal-gamma or beta-gamma distribuition
		// calculate weight 'w' with sampleMultinomial()
		double [][] weightMatrix=null;  //will be a numFeatures x 1 matrix (column vector)

		
		if(pr.get_identifier() == 1) { //if using a normal-gamma prior
			double [][] weights = new double [numFeatures][1];
			for(int f =0; f<numFeatures; f++) {
				int index = SampleMultinomialIRL.sampleSingleStateFromMultinomial(10,irloptions.getRewardDistro(), RNG);
				weights[f]= new double [] {irloptions.getRewardArray()[index]};		
			}
			weightMatrix = weights;
			if(changeWeights) {
				tblWeightVectors.put(tableIndexValue, weightMatrix);
			}
		}
		else if(pr.get_identifier() ==3) { //if using a gaussian prior
			double mu = pr.get_mu();
			double sigma = pr.get_sigma();
			double[][] rndNormalWeights2DArray = MersenneTwisterFastIRL.RandomNormalMatrix(numFeatures, 1);
			DoubleMatrix rndNormalWeightsMatrix = new DoubleMatrix(rndNormalWeights2DArray);
			rndNormalWeightsMatrix= rndNormalWeightsMatrix.mul(sigma);
			rndNormalWeightsMatrix= rndNormalWeightsMatrix.add(mu);
			double [][] weightsMatrixAs2DArray = rndNormalWeightsMatrix.toArray2();
		    //make sure that weight vector values are LESS than the UPPER bound,and then make sure that weight vector values are GREATER than the LOWER BOUND
			weightMatrix = MatrixUtilityJBLAS.withinBounds(weightsMatrixAs2DArray, lowerB, upperB);
			if(changeWeights) {
				tblWeightVectors.put(tableIndexValue, weightMatrix);
			}
		}
		else if(pr.get_identifier()==4) { //if using a Uniform prior
			double[][] rndNormalWeights2DArray = MersenneTwisterFastIRL.RandomNormalMatrix(numFeatures, 1);
			DoubleMatrix rndNormalWeightsMatrix = new DoubleMatrix(rndNormalWeights2DArray);
			rndNormalWeightsMatrix= rndNormalWeightsMatrix.mul(upperB-lowerB);
			rndNormalWeightsMatrix= rndNormalWeightsMatrix.add(lowerB);
			weightMatrix = rndNormalWeightsMatrix.toArray2();
			if(changeWeights) {
				tblWeightVectors.put(tableIndexValue, weightMatrix);
			}
		}
		
		//generate REWARD matrix from weightVector (convertW2R)
		// use weight vector and mdp representation of problem environment to generate the corresponding reward function
		RewardFunctionGeneration.generateWeightedRewardFunction(env, weightMatrix);
		
		//generate POLICY and VALUE matrix (policyIteration)
		Map<String, double[][]> policy_value_h_q_Matrices = PolicySolver.runPolicyIteration(env, irloptions, null);
		if(changeWeights) {
			tblPlVectors.put(tableIndexValue, policy_value_h_q_Matrices.get("P"));
			tblVlVectors.put(tableIndexValue, policy_value_h_q_Matrices.get("V"));
			tblQVectors.put(tableIndexValue, policy_value_h_q_Matrices.get("Q"));
		}
		if(!changeWeights) {

			w_p_v_qHashMap.put("W", weightMatrix);
			w_p_v_qHashMap.put("P", policy_value_h_q_Matrices.get("P"));
			w_p_v_qHashMap.put("V", policy_value_h_q_Matrices.get("V"));
			w_p_v_qHashMap.put("Q", policy_value_h_q_Matrices.get("Q"));

		}
		return w_p_v_qHashMap;
	}
	
	// similar to getTrajInfo() 
	/**
	 * Computes occupancy measure and empirical policy for trajectories. Called within dirichletProcessMHLInference()
	 * @param trajSet
	 * @param env
	 * @param iTrajectoryInformation
	 * @param numTrajectories
	 * @param numStepsPerTrajectory
	 */
	private static Multimap<Integer,double[]> computeOccupancy(Map<Integer, double[][]>subsetOfTrajectories, MDP env,  Multimap<Integer,double[]> countInfoForSubsetOfTrajs, boolean isForInitialDPMPosteriorCalculation) {
	//private static Multimap<Integer,double[]> computeOccupancy(List<double[][]>subsetOfTrajectories, MDP env,  Multimap<Integer,double[]> countInfoForSubsetOfTrajs, boolean isForInitialDPMPosteriorCalculation) {
		int numStates = env.getNumStates();
		int numActions = env.getNumActions();
		//int numStepsPerTrajectory = subsetOfTrajectories.get(0)[0].length; //remember subsetOfTrajectories.get(0)[0] is the state sequence, and subsetOfTrajectories.get(0)[1] is the action sequence for the 0th trajectory in this subsetOfTrajectories
		Integer firstKey = (Integer) subsetOfTrajectories.keySet().toArray()[0]; //key set contains the list of  trajectory index #'s for trajectories assigned to the given table
		int numStepsPerTrajectory = subsetOfTrajectories.get(firstKey)[0].length;
		Double state_ts = 0.0;
		Double action_ts =0.0;
//		int indexKeyForSubsetOfTrajectoriesAnalyzed;	// GTD Not used
		
//JK 6.23.2019 create these 2 matrices for each trajectory		
//		double [][] countMatrix = new double[numStates][numActions]; 
//		double [][] occupancyMatrix = new double[numStates][numActions];  //discounted state-action frequency
//		int stepsTaken =0;	// GTD Not used
		
		for(int t: subsetOfTrajectories.keySet()) {
			//JK 6.23.2019 creating these count matrices once, for each trajectory assigned to the current table
		double [][] countMatrix = new double[numStates][numActions]; 
		double [][] occupancyMatrix = new double[numStates][numActions];  //discounted state-action frequency
		
			for (int step=0; step< numStepsPerTrajectory; step++) {

				state_ts = subsetOfTrajectories.get(t)[0][step];
				action_ts = subsetOfTrajectories.get(t)[1][step];
				int sBool =Double.compare(state_ts, -1.0);
				int aBool = Double.compare(action_ts, -1.0);
				if (sBool==0 && aBool==0){ //if aBool or sBool are TRUE (NOTE: 0 as input from Double.compare means 'TRUE')
					break;
				}
				int s = state_ts.intValue();
				int a = action_ts.intValue();
				
				countMatrix[s][a] = countMatrix[s][a]+1;
				occupancyMatrix[s][a] = occupancyMatrix[s][a] + Math.pow(env.getDiscount(),(step));	

//				++stepsTaken;	// GTD Not used
		}
			//JK 6.23.2019: modified so that we add trajectory-specifc sa counts into Multimap<trajINT, double[]> stateActionCountMap
			for(int state=0; state< numStates; state++) {
				for (int action=0; action<numActions; action++) {
					if(Double.compare(countMatrix[state][action], 0.0)>0) {
						double [] observed_stateActionPairInfo = new double [3];
						observed_stateActionPairInfo[0]= (double) state;
						observed_stateActionPairInfo[1] = (double) action;
						observed_stateActionPairInfo [2] = countMatrix[state][action];
						countInfoForSubsetOfTrajs.put(t, observed_stateActionPairInfo);
					}
				}
			}
		//move-on to next trajectory	
		}
						

		
	return countInfoForSubsetOfTrajs;	
	}
	
	public static RestaurantMap generateNewTableAssignmentPartition(int numStates, int numActions, int numFeatures, Map<Integer, double[][]> tblWeightVectors,  Map<Integer, double[][]> tblPlVectors,  Map<Integer, double[][]> tblVlVectors, Map<Integer, double[][]> tblQVectors, double[][] tblAssignmentMatrix ) {

		RestaurantMap rmap = null;
//		List<double[]> temp = new ArrayList<double []>();	// GTD Not used


		//JK 6.23.2019 switched from List to Map to avoid mistakes in insertion order
		Map<Integer, DoubleMatrix> wMatrix = new HashMap<Integer,DoubleMatrix>();
		Map<Integer,DoubleMatrix> pMatrix = new HashMap<Integer,DoubleMatrix>();
		Map<Integer,DoubleMatrix> vMatrix = new HashMap<Integer,DoubleMatrix>();
		Map<Integer,DoubleMatrix> qMatrix = new HashMap<Integer,DoubleMatrix>();

		Map<Integer, DoubleMatrix> tableLabelMatrix = new HashMap<Integer, DoubleMatrix>();
		
		DoubleMatrix newPartitiontblAssignmentMatrix = new DoubleMatrix(tblAssignmentMatrix.length, 1);
		
		Double highestTableIndex = MatrixUtilityJBLAS.maxPerColumn(tblAssignmentMatrix)[0];  //returns a row vector containing the maximum values in each column of the 'tableAssignmentMatrix'. Since this matrix is nTraj x 1. We only care about the 0th element of the returned row vector
		int N = highestTableIndex.intValue();
		
//		boolean generateNewPartition = false;	// GTD Not used
		
		for (int tblIndex=1; tblIndex< N+1; tblIndex++) {

			//return a logical array with elements set to logical 1 (true) where elements in the assignmentMatrix equal to 'tblIndex'; i.e. which trajectories/customers were assigned to table 'tblIndex'
			DoubleMatrix logic2Darray = MatrixUtilityJBLAS.toLogicalMatrix(new DoubleMatrix(tblAssignmentMatrix), tblIndex);

			//since tblAssignmentMatrix is numTraj x 1, there is only 1 column, thus we only need to look at sum of 0th column which is 0th element in returned row vector 
			if(logic2Darray.sum() > 0.0) {	
			 //i.e. if the # customers assigned to table 'tblIndex' is >0
				DoubleMatrix kthWColumnMatrix = new DoubleMatrix(tblWeightVectors.get(tblIndex));
				wMatrix.put(wMatrix.size()+1, kthWColumnMatrix); //wMatrix.size because we start our table indices at 1, not at 0!
				
				DoubleMatrix kthPColumnMatrix = new DoubleMatrix(tblPlVectors.get(tblIndex));
				pMatrix.put(pMatrix.size()+1, kthPColumnMatrix);
				
				DoubleMatrix kthVColumnMatrix =new DoubleMatrix(tblVlVectors.get(tblIndex));
				vMatrix.put(vMatrix.size()+1, kthVColumnMatrix);
				
				DoubleMatrix kthQColumnMatrix =new DoubleMatrix(tblQVectors.get(tblIndex));
				qMatrix.put(qMatrix.size()+1, kthQColumnMatrix);
				
				double [] tblLabel = new double [] {tblIndex, wMatrix.size()}; //wMatrix already has a table added to it. So we are matchign the size of wMatrix as opposed to wMatrix+1
				//we want table indices to start at 1!

				DoubleMatrix kthLabelRowMatrix = new DoubleMatrix(1,2,tblLabel); //JK: need to make sure that this stored as a ROW vector and not a column matrix
				tableLabelMatrix.put(tableLabelMatrix.size()+1, kthLabelRowMatrix);
			}
		}
		if(tableLabelMatrix.size()>0) { // the size of tableLabelMatrix = number of distinct tables >0;
			// change index-value of each distinct table index-value (tableLabelMatrix.get(i).getData()[0][0]) to the new index-value (tableLabelMatrix.get(i).getData()[0][1])
			for (int i=1; i<= tableLabelMatrix.size(); i++) {

				DoubleMatrix tablesToReassign = MatrixUtilityJBLAS.toLogicalMatrix(new DoubleMatrix(tblAssignmentMatrix), tableLabelMatrix.get(i).toArray2()[0][0]); //identify which tables in current assignmentMatrix have the same label of 'tableLabelMatrix.get(i).getData()[0][0]'

				for (int r = 0; r < tablesToReassign.rows; r++) {
    				for(int c=0; c<tablesToReassign.columns; c++) {
    					if(tablesToReassign.get(r, c) == 1.0) { //replace the table-index associated with row/trajectory 'r' with its NEW table-index
		  		    	  		newPartitiontblAssignmentMatrix.put(r,c, tableLabelMatrix.get(i).toArray2()[0][1]); //replace label-index for the table assigned at position (r,c) in the new assignment matrix with new label 'tableLabelMatrix.get(i).getData()[0][1]'
	    				}     
		    		}
				}
			}
			
//			Set<Double> uniqueTableIndices = MatrixUtilityJBLAS.countNumberUniqueElements(newPartitiontblAssignmentMatrix.toArray2());	// GTD Not used
//			Double [] sortedArray = VectorUtility.sortSet(uniqueTableIndices);	// GTD Not used

			tblWeightVectors = MatrixUtilityJBLAS.reorderHashMap(wMatrix ); //recall that wMatrix, pMatrix, and vMatrix are ArrayLists. The first element in each list corresponds to the lowest table index-value among the new table assignment index-values
			tblPlVectors = MatrixUtilityJBLAS.reorderHashMap(pMatrix);
			tblVlVectors = MatrixUtilityJBLAS.reorderHashMap(vMatrix);
			tblQVectors = MatrixUtilityJBLAS.reorderHashMap(qMatrix);

			
			

			tblAssignmentMatrix = newPartitiontblAssignmentMatrix.toArray2();
			
		}

		rmap = new RestaurantMap(tblWeightVectors, tblPlVectors, tblVlVectors, tblQVectors, tblAssignmentMatrix);
		return rmap;

	}
	
	/**
	 * JK data validated 7.24.2019
	 * Compute the gradient of the Q-function matrix
	 * @param policyMatrix
	 * @param env
	 * @return the transpose of the computed qMatrixGradient with dimensions: [numFeatures][numStates*numActions]
	 */
	public static double [][] computeQMatrixGradient(double [][] policyMatrix, MDP env){
		Integer numStates = env.getNumStates();
		Integer numActions = env.getNumActions();
		Integer numFeatures = env.getNumRewardFeatures();
		double epsilon       = 1e-12;
	    int maximumIterations = 10000;
		
	    DoubleMatrix numStatesColMatrix= new DoubleMatrix(numStates, 1);
		for (int s=0; s<numStates; s++) {
			numStatesColMatrix.put(s, 0, s);
		}
		
		//compute dQ/ dw
		DoubleMatrix expectedPolicyMatrix = null;
		DoubleMatrix expectedPolicyMatrixVersion2 = new DoubleMatrix (numStates, (numStates * numActions));
		
		//If policy matrix is DETERMINISTIC  (# columns ==1); initialze expectedPolicyMatrix similar to PolicySolver.policyEvaluationStep()
		if(policyMatrix[0].length ==1) {
			//JK. 6.22.2019 replaced this linear indexing operation 
			DoubleMatrix idx = new DoubleMatrix(policyMatrix).mul(numStates);
			DoubleMatrix unitVectorTransposed = new DoubleMatrix(VectorUtility.createUnitSpaceVector(numStates, 0.0, 1.0)).transpose();
			idx.addi(unitVectorTransposed);
			DoubleMatrix unitVectorTransposedForLinearIdx = new DoubleMatrix(VectorUtility.createUnitSpaceVector(numStates, 1.0, 1.0)).transpose();
			DoubleMatrix idx2 = idx.mul(numStates).add(unitVectorTransposedForLinearIdx);
			idx2.subi(1); //JK added 6.22.2019 because this was necessary in similar calculation in PolicyEvaluation method
			
			int [] linearIndicesIdx2 = idx2.toIntArray();
			expectedPolicyMatrixVersion2.put(linearIndicesIdx2, 1.0);
//			double[][] expmv2dblarray = expectedPolicyMatrixVersion2.toArray2();	// GTD Not used
			expectedPolicyMatrix = expectedPolicyMatrixVersion2;
				


		}
		else { //else if policy matrix is STOCHASTIC (only exists when we run compute logLikelihood using MLIRL which computes a stochastic policy jk)
			expectedPolicyMatrix = expectedPolicyMatrixVersion2;
			for(int state=0; state< numStates; state++) {
				for (int action=0; action<numActions; action++) { // iterate through each column/action, since in a stochastic policy more than 1 action is possible for each state
					int columnSubscript = ((action-1)*numStates) + state;
					expectedPolicyMatrix.put(state, columnSubscript, policyMatrix[state][action]);
				}
			}
			
		}
		
//		double [][] expecPolMat2dArray = expectedPolicyMatrix.toArray2();	// GTD Not used
//		double [] sumsCountForJK = MatrixUtilityJBLAS.sumPerColumn(expecPolMat2dArray);	// GTD Not used
		
		DoubleMatrix qMatrixGradient =new DoubleMatrix(numStates*numActions,numFeatures);
		if(MatrixUtilityJBLAS.matrixMaximum(env.getDiscountedTransitionMatrix().toArray2())==0.0) {
		   	throw new java.lang.RuntimeException("discounted transition matrix is all zeros!!! Is that allowed???");

		}
		DoubleMatrix expectedDiscountedTransitionMatrix = env.getDiscountedTransitionMatrix().mmul(expectedPolicyMatrix);

		
		
		
		DoubleMatrix stateFeatureREALMATRIX = MatrixUtilityJBLAS.convertMultiDimMatrixList(env.getStateFeatureMatrix());
//		double [][] exDiscountedTMatrix = expectedDiscountedTransitionMatrix.toArray2();	// GTD Not used
		boolean isGradientConverged= false;
		for (int i=0; i< maximumIterations; i++) {

			DoubleMatrix qMatrixGradient_previous = qMatrixGradient;

			qMatrixGradient = stateFeatureREALMATRIX.add(expectedDiscountedTransitionMatrix.mmul(qMatrixGradient));
			
//			double [][] qGrad_old = qMatrixGradient_previous.toArray2();	// GTD Not used
//			double [][] qGrad = qMatrixGradient.toArray2();	// GTD Not used
			
			isGradientConverged = MatrixUtilityJBLAS.compareREALMatrices(qMatrixGradient, qMatrixGradient_previous, epsilon);
			if(isGradientConverged) {
				if(i > 263) {
					System.out.println("****Q matrix gradient has converged at a value > 263!!! = "+i+" iterations");
				}
				break;
			}
		}
		DoubleMatrix qMatrixGradientTrans = qMatrixGradient.transpose();
		double [][] qMatrixGradientTransposed = qMatrixGradientTrans.toArray2();
		return qMatrixGradientTransposed;
		
		
	}
	
	private static double computeLogPosteriorProbabilityForDirichletProcessMixture(List<double[][]> trajSet, double [][] tableAssignmentMatrix, Map<Integer, double[][]> tblWeightVectors,  Map<Integer, double[][]> tblPolicyVectors, MDP env, IRLAlgorithm irlAlgo, boolean partOfInitialDPMPosterior) {
		double logPosteriorProb_total= 0.0;
		double logLikelihood_total =0.0;
		double logPriorProb_total = 0.0;
		double alpha = irlAlgo.getAlpha();
		
		//Trajectory data
//		int numberTrajectories = trajSet.size();  //should be same as MDP._numberTrajectories	// GTD Not used
//		int trajectoryLength = trajSet.get(0)[0].length; //should be same as MDP._numberStepsPerTrajectory	// GTD Not used
		//vector of state-action counts
//		List<double[][]> trajectorySetInfo = new ArrayList<double[][]>(numberTrajectories); 	// GTD Not used
		//intbludes 'count' with each state-action pair so maximum size is now: nTrajs x nSteps x 3 		
		
		
		
		//////ComputeTableAssignmentProbability////////Calculate prior of table assignment c, pr(c|alpha)
		double [][] z = MatrixUtilityJBLAS.tableCounter(tableAssignmentMatrix);
		double [] zRowVector = z[0];
		double [] rIndexVector = VectorUtility.createUnitSpaceVector(zRowVector.length, 1.0, 1.0);

		Double  mVal = VectorUtility.multAndSum(zRowVector, rIndexVector);
		Double mValfactorial = CombinatoricsUtils.factorialDouble(mVal.intValue());

		//JK 5.7.2019: Corrected the alpha vector construction
		double [] alphaVector = VectorUtility.range(alpha, alpha+mVal.doubleValue()-1);
		
		double [] q1 = VectorUtility.pow(rIndexVector, zRowVector);
		double [] q2 = VectorUtility.factorial(VectorUtility.cast(zRowVector));
		double [] qProduct = VectorUtility.mult(q1, q2);
		
		double alphaVecProd = VectorUtility.product(alphaVector);
		double scalarVal1 = mValfactorial.doubleValue()/ alphaVecProd;

		double scalarVal2 =  Math.pow(alpha, VectorUtility.sum(zRowVector))/ VectorUtility.product(qProduct);
		double probabilityVal = scalarVal1*scalarVal2;
		///////////////////////////
		
		double logDirichletProcessPriorProb = Math.log(probabilityVal); //log prior probability of table assignment
		
		Double maxTableIndexInRestaurant = MatrixUtilityJBLAS.matrixMaximum(tableAssignmentMatrix);
		
		for(int t=1; t< maxTableIndexInRestaurant+1; t++) {
			double [][] weightMatr = tblWeightVectors.get(t);  //weightMatrix describing relevance of the |numFeatures| different features for table 't'
			double [][] policyMatr;
			if(tblPolicyVectors.isEmpty()) {
				
				//generate REWARD matrix from weightVector (convertW2R)
				// use weight vector and mdp representation of problem environment to generate the corresponding reward function
				RewardFunctionGeneration.generateWeightedRewardFunction(env, weightMatr);
				
				//generate POLICY and VALUE matrix (policyIteration)
				Map<String, double[][]> P_V_H_Q_Matrices=PolicySolver.runPolicyIteration(env, irlAlgo,  null);
				policyMatr = P_V_H_Q_Matrices.get("P");
			}
			else {
				policyMatr = tblPolicyVectors.get(t);
			}
			
		    Multimap<Integer,double[]> stateActionPairCountsInfoForSubsetOfTrajectories = ArrayListMultimap.create();
		    DoubleMatrix logicalMatrixOfSubsetOfTrajectories = MatrixUtilityJBLAS.toLogicalMatrix(new DoubleMatrix(tableAssignmentMatrix), (double)t); //make sure this double cast works!!
		    if(logicalMatrixOfSubsetOfTrajectories.sum() > 0.0) {

		    	Map<Integer, double [][]> subsetOfTrajectories = new HashMap<Integer, double[][]>();
			    
			    for(int traj=0; traj< logicalMatrixOfSubsetOfTrajectories.rows; traj++) {
					if(logicalMatrixOfSubsetOfTrajectories.get(traj) == 1.0) {
						//assuming tableAssignmentMatrix is a single column matrix; if this trajectory was assigned to table 't'
						//add this trajectory's state and action sequences to subsetOfTrajectories dataset
						subsetOfTrajectories.put(traj, trajSet.get(traj));
						
					}
				}
			    
			    stateActionPairCountsInfoForSubsetOfTrajectories =computeOccupancy(subsetOfTrajectories, env, stateActionPairCountsInfoForSubsetOfTrajectories, partOfInitialDPMPosterior);
			    
			    
			    LikelihoodFunction llhFunctIRL = irlAlgo.getLikelihood();
			    double likelihood_forTablec = llhFunctIRL.computeLogLikelihoodAndGradient_Bayesian(env, irlAlgo, weightMatr, stateActionPairCountsInfoForSubsetOfTrajectories, policyMatr, null, null, false).getFirst();
			    
			    Prior priorIRL = irlAlgo.getPrior();
			    double priorProbability_forTablec = priorIRL.computeLogPriorAndGradient(weightMatr).getFirst();
			    logLikelihood_total   = logLikelihood_total + likelihood_forTablec;
			    logPriorProb_total = logPriorProb_total + priorProbability_forTablec;  
		    }	
		}
		//...formulate the IRL problem into posterior optimization problem, finding reward function R that maximizes the log unnormalized [posterior] = finding the reward function R that maximizes [logLikelihood + logPriorProbability]
		logPosteriorProb_total = logDirichletProcessPriorProb + logLikelihood_total + logPriorProb_total;
		
		return logPosteriorProb_total;
	}
	
	/**
	 * MH update for table assignment of i-th customer/trajectory | STEP 1 of Inference Algorithm
	 * @param customer_i
	 * @param irlalgo
	 */
	public static RestaurantMap updateTableAssignment(MDP env, int customer_i, IRLAlgorithm irlalgo, double [][] tableAssignmentMatrix,  Map<Integer, double[][]> tableWeightVectors,  Map<Integer, double[][]> tablePolicyVectors,  Map<Integer, double[][]> tableValueVectors, Map<Integer, double[][]> tableQVectors, Map<Integer, Double> restLikeLihoods, Map<Integer, Double> restPriors, Map<Integer, DoubleMatrix> restGradientsFromLLH, Map<Integer, DoubleMatrix> restGradientsFromPrior, Multimap<Integer,double[]> saPairCountsInfoForAllTrajectories) {

		RestaurantMap rmap4;
		int numTableAssignmentIterations= irlalgo.getTableAssignmentUpdateIterations();
		Map<String, double[][]> weight_policy_value_q_vectorHashMapForCustomer_iTable = null;
		
		for(int iter=0; iter< numTableAssignmentIterations; iter++) {
		    Double tblIndex = tableAssignmentMatrix[customer_i][0];  //obtain table assignment index/label for m-th trajectory according to the current tableAssignmentMatrix tblAssignMaterix
		    double [][] tblWeightMatrixforCustomeri_currentAssignment = tableWeightVectors.get(tblIndex.intValue()); //obtain the weight column-matrix  associated with this table index/label
//		    double [][] tblPolicyMatrixforCustomeri_currentAssignment = tablePolicyVectors.get(tblIndex.intValue());	// GTD Not used
		    //obtain the policy column-matrix associated with this table index/label;
		    double [][] tblValueMatrixforCustomeri_currentAssignment = tableValueVectors.get(tblIndex.intValue());//obtain the value column-matrix associated with this cluster index/label

		    if(!restLikeLihoods.containsKey(tblIndex.intValue())) { ///this if condition was only added because of our original use of the RealMatrix for the gradientsLLH datastructure
			   	System.out.println("******This new table index does NOT yet exist in restLikelihoods!");

		   }
		    //reset likelihood of this table to NaN
		    restLikeLihoods.put(tblIndex.intValue(), Double.NaN);

		    //reset prior of this table to NaN
		    restPriors.put(tblIndex.intValue(), Double.NaN);
		    
		    if(!restGradientsFromLLH.containsKey(tblIndex.intValue())) { ///this if condition was only added because of our original use of the RealMatrix for the gradientsLLH datastructure
			   	System.out.println("******This new table index does NOT yet exist in restGradientsFromLLLH!");
		    }
		    

		    restGradientsFromLLH.put(tblIndex.intValue(), MatrixUtilityJBLAS.createRealMatrixWithNANS(tblWeightMatrixforCustomeri_currentAssignment.length, 1)); //reset gradientFromLLHcomputation for this table to NaN
		    restGradientsFromPrior.put(tblIndex.intValue(), MatrixUtilityJBLAS.createRealMatrixWithNANS(tblWeightMatrixforCustomeri_currentAssignment.length, 1)); //reset gradientFromPriorcomputaiton for this table to NaN

		    
		    Double N = MatrixUtilityJBLAS.matrixMaximum(tableAssignmentMatrix);
		    
		    double [] priorProbDistributionForCurrentSeatingArrangement= new double[(N.intValue()+1)]; 
		    
		    for (int tableIndex_i=1; tableIndex_i< N.intValue()+1;tableIndex_i++ ) {		//prior probability of table index/label 'table_i' = # of trajectories that have been assigned label/index 'table_i'

		    		 //should return a numTraj x 1 column matrix with 1's at entries corresponding to customer assigned to table 'table_i'
				DoubleMatrix logMatrB = MatrixUtilityJBLAS.toLogicalMatrix(new DoubleMatrix(tableAssignmentMatrix), (double) tableIndex_i);

		    		 //element 0 of this doulbe[] corresponds to table 1!!! be careful
	    		double logMatrSumVal = logMatrB.sum();
	    		if(tableIndex_i ==tblIndex) {
	    		    //remove the contribution of of trajectory 'customer_i' to the prior probability of table index/label being 'tblIndex'
	    			logMatrSumVal = logMatrSumVal -1;
	    		}
	    		if(logMatrSumVal >0 ) {
	    			priorProbDistributionForCurrentSeatingArrangement[tableIndex_i-1]= logMatrSumVal - irlalgo.getDiscountHyperparameter(); //element 0 of this doulbe[] corresponds to table 1!!! be careful
	    		}
	    		else {//need this else condition if the count for a tableIndex =0; in which case the priorProb for table would becoe negative because we are subtracting the val of discount
	    			priorProbDistributionForCurrentSeatingArrangement[tableIndex_i-1]= logMatrSumVal; //element 0 of this doulbe[] corresponds to table 1!!! be careful

	    		}
			}
		    //remove the contribution of of trajectory 'customer_i' to the prior probability of table index/label being 'tblIndex'

		    priorProbDistributionForCurrentSeatingArrangement[priorProbDistributionForCurrentSeatingArrangement.length-1] = irlalgo.getAlpha()+(N*irlalgo.getDiscountHyperparameter());
		    	
		    	
		    	//Sample a NEW/ALTERNATIVE table index/value from the prior probability distribution  (NOTE: if this drawn table index corresponds to the index of existing table in the restaurant, it will not necessarilly be the same table/index-value already assigned in the tableAssignmentMatrix (seatingArrangement)
		    int tblIndex2 = SampleMultinomialIRL.sampleSingleTableFromMultinomial(10000, priorProbDistributionForCurrentSeatingArrangement, RNG);
		    if(tblIndex2 > N.intValue()) {  //if the new table index/value is higher than the current (existing) largest table index/value; then we need to generate NEW (weight,policy, and value) vectors specifically for this table
		    	weight_policy_value_q_vectorHashMapForCustomer_iTable = generateNewWeights(0, env, irlalgo, tableWeightVectors, tablePolicyVectors, tableValueVectors,tableQVectors, false);
		    }
		    else { //OTHERWISE if the table index/value drawn for customer_i  already exists in the restaurant, set the weight, policy, value vectors of this table to the vectors already associated with the existing table with the same index/value in the restaurant.
		    	weight_policy_value_q_vectorHashMapForCustomer_iTable = new HashMap<String, double[][]>();
		    		
		    		//...you can't use .set() on an empty index, must use .add()
		    	weight_policy_value_q_vectorHashMapForCustomer_iTable.put("W", tableWeightVectors.get(tblIndex2)); //obtain the existing weight vector associated with table index-value 'tblIndex2'
		    	weight_policy_value_q_vectorHashMapForCustomer_iTable.put("P", tablePolicyVectors.get(tblIndex2));   //obtain the existing policy vector assoicated with table index-value 'tblIndex2'
		    	weight_policy_value_q_vectorHashMapForCustomer_iTable.put("V", tableValueVectors.get(tblIndex2));  //obtain the existing value vector associated with table index-value 'tblIndex2'
		    	weight_policy_value_q_vectorHashMapForCustomer_iTable.put("Q", tableQVectors.get(tblIndex2));  //obtain the existing q vector associated with table index-value 'tblIndex2'

		    }
		    
		    //If tblIndex2 > tableIndex_i for all i, we need to draw a new reward function r_{tblIndex2} from the reward prior P(r|?, ?). We then set tblIndex = tbLindex2 (table index/value of customer_i trajectory) with the acceptance probability
		    double probQuotient = computeMinProbabilityQuotient(tblWeightMatrixforCustomeri_currentAssignment, tblValueMatrixforCustomeri_currentAssignment, weight_policy_value_q_vectorHashMapForCustomer_iTable.get("W"), weight_policy_value_q_vectorHashMapForCustomer_iTable.get("V"), env, irlalgo, saPairCountsInfoForAllTrajectories, customer_i);

		    double rand = RNG.nextDouble();
		    if(Double.compare(probQuotient, rand)>0) {
		    		tableAssignmentMatrix[customer_i][0]= tblIndex2;
		    		
		    		if(Double.compare(tblIndex2, N.intValue())>0) {
		    			tableWeightVectors.put(tblIndex2, weight_policy_value_q_vectorHashMapForCustomer_iTable.get("W"));
		    			tablePolicyVectors.put(tblIndex2, weight_policy_value_q_vectorHashMapForCustomer_iTable.get("P"));
		    			tableValueVectors.put(tblIndex2, weight_policy_value_q_vectorHashMapForCustomer_iTable.get("V"));
		    			tableQVectors.put(tblIndex2, weight_policy_value_q_vectorHashMapForCustomer_iTable.get("Q"));

		    		}
		    }
		 	
		}
		rmap4 = new RestaurantMap(tableWeightVectors, tablePolicyVectors, tableValueVectors,tableQVectors, tableAssignmentMatrix);
		rmap4._restGradientsFromLLH = restGradientsFromLLH;
		rmap4._restGradientsFromPrior =restGradientsFromPrior;
		rmap4._restLikeLihoods = restLikeLihoods;
		rmap4._restPriors = restPriors;
		//rmap4._saPairCountsInfoForSubsetOfTrajectories = saPairCountsInfoForAllTrajectories;
		return rmap4;
	}
	
	/**
	 * Update the weight, policy, value of the multiple reward functions (tables) in the restaurant
	 * @param environment
	 * @param table_i
	 * @param irlAlgo
	 * @param tableAssignmentMatrix
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param restaurantLikelihoods
	 * @param restaurantPriors
	 * @param restGradientsLLH
	 * @param restGradientsPrior
	 * 
	 */
	public static RestaurantMap updateRewardFunctions(List<double[][]> trajSet, MDP environment, int table_i, IRLAlgorithm irlAlgo, double[][] tableAssignmentMatrix,  Map<Integer, double[][]> tableWeightVectors,  Map<Integer, double[][]> tablePolicyVectors,  Map<Integer, double[][]> tableValueVectors, Map<Integer, double[][]> tableQVectors, Map<Integer, Double> restaurantLikelihoods, Map<Integer, Double> restaurantPriors, Map<Integer, DoubleMatrix> restGradientsLLH, Map<Integer, DoubleMatrix> restGradientsPrior){
	//do we ever use stateActionPairCountsInfoForSubsetOfTrajectories that we pass into function
	 	RestaurantMap rmap5;
		LikelihoodFunction llhFunctIRL = irlAlgo.getLikelihood();
	    Prior priorIRL = irlAlgo.getPrior();
		double scalingParameter = .01;
		Multimap<Integer, double[]> updatedSAPairCountsForSubsetOfTrajectories = ArrayListMultimap.create();
	    DoubleMatrix logicalMatrixOfSubsetOfTrajectories = MatrixUtilityJBLAS.toLogicalMatrix(new DoubleMatrix(tableAssignmentMatrix), table_i); //identify which trajectories(rows) were assigned to table_i(column_i)

	    Map<Integer, double[][]> subsetOfTrajectories = new HashMap<Integer, double[][]>();
	    Map<String, double[][]> weight_policy_q_vectorMapForTable_i = new HashMap<String, double[][]>();

		double logPosteriorProbability = 0.0;
		double logPosteriorProbability_updated = 0.0;
		DoubleMatrix gradient_forTablei = null;
		DoubleMatrix gradient_updated_forTablei =null;
		
		//TODO should not run if none of the trajectories possess a given table index-value
	    if(logicalMatrixOfSubsetOfTrajectories.sum() == 0.0) {

	    		System.out.println("None of the trajectories were assigned to table "+table_i);
	    }    
	    else {
		    
		    for(int traj=0; traj< logicalMatrixOfSubsetOfTrajectories.rows; traj++) {
				if (logicalMatrixOfSubsetOfTrajectories.get(traj)==1.0 ) { //assuming tableAssignmentMatrix is a single column matrix
					
					//obtain and store the subset of trajectories that were found to be assigned to table_i
					subsetOfTrajectories.put(traj, trajSet.get(traj));
				}
			}
		    //JK 6.23.2019 moved computeOccopuancy() outside of rewardupdateIteration loop. The count map is entirely based on the subset of trajectories currently being analyzed. Nothing to do with the actual reward function (weights, llh, etc...)
	    	updatedSAPairCountsForSubsetOfTrajectories = computeOccupancy(subsetOfTrajectories, environment, updatedSAPairCountsForSubsetOfTrajectories, false);

			for (int iter=0; iter < irlAlgo.getRewardUpdateIterations(); iter++) {
			    //JK 6.23.2019 moved computeOccupancy() outside of for loop;	
			
			    	weight_policy_q_vectorMapForTable_i.put("W", tableWeightVectors.get(table_i));
			    	weight_policy_q_vectorMapForTable_i.put("P", tablePolicyVectors.get(table_i));
			    	weight_policy_q_vectorMapForTable_i.put("V", tableValueVectors.get(table_i));
			    	weight_policy_q_vectorMapForTable_i.put("Q", tableQVectors.get(table_i));

				
				int boolLLH=5;
				if(!restaurantLikelihoods.containsKey(table_i)) {
					System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>This table does not exist in in restaurantLikelihoods");
					// the table index doesn't exist, then restaurantLikelihoods.get(table_i) wiil return NULL, which means we still need to calculate llh and gradient for this new table similar to situation where the table's llh were Double.NaN
					boolLLH =0;
				}

				else {
					boolLLH = Double.compare(restaurantLikelihoods.get(table_i), Double.NaN);//if aBool or sBool are TRUE (NOTE: 0 as input from Double.compare means 'TRUE')
				}
				if(  boolLLH ==0) { //if they are equivalent, then boolLLH should equal 0
	
				    Pair<Double, DoubleMatrix> llhAndGradientForTable_i = llhFunctIRL.computeLogLikelihoodAndGradient_Bayesian(environment, irlAlgo, weight_policy_q_vectorMapForTable_i.get("W"), updatedSAPairCountsForSubsetOfTrajectories, weight_policy_q_vectorMapForTable_i.get("P"), null, null, true);
		
					restaurantLikelihoods.put(table_i, llhAndGradientForTable_i.getFirst());
	
					restGradientsLLH.put(table_i, llhAndGradientForTable_i.getSecond());
				    
				    Pair<Double, double[][]> priorProbAndGradientForTable_i= priorIRL.computeLogPriorAndGradient(weight_policy_q_vectorMapForTable_i.get("W"));
					
					restaurantPriors.put(table_i, priorProbAndGradientForTable_i.getFirst());
	
					restGradientsPrior.put(table_i, new DoubleMatrix(priorProbAndGradientForTable_i.getSecond()));
	
				}
				
				logPosteriorProbability = restaurantLikelihoods.get(table_i) + restaurantPriors.get(table_i);
	
				
				gradient_forTablei = restGradientsLLH.get(table_i).add(restGradientsPrior.get(table_i));
	
				
				DoubleMatrix scaledGradient = gradient_forTablei.mul((0.5*Math.pow(scalingParameter,2)));
				double randomShift [][] = MersenneTwisterFastIRL.RandomNormalMatrix(environment.getNumRewardFeatures(), 1);
				DoubleMatrix rshift = new DoubleMatrix(randomShift).mul(scalingParameter);
				DoubleMatrix updatedWeightMatrixForTable_i = new DoubleMatrix(weight_policy_q_vectorMapForTable_i.get("W")).add(scaledGradient).add(rshift); 
				double [][] boundUpdatedWeightMatrix = MatrixUtilityJBLAS.withinBounds(updatedWeightMatrixForTable_i.toArray2(), irlAlgo.getRewardLowerBounds(), irlAlgo.getRewardUpperBounds());
				RewardFunctionGeneration.generateWeightedRewardFunction(environment, boundUpdatedWeightMatrix); //automatically generates new reward function that is automatically set to the rewardFunction for the current MDP environment
				Map<String, double[][]> newlyComputed_Policy_Value_H_Q_matrices = PolicySolver.runPolicyIteration(environment, irlAlgo, null);
				
				Pair<Double, DoubleMatrix> LLHAndGradientForTable_i_fromNewWeightMatrix = llhFunctIRL.computeLogLikelihoodAndGradient_Bayesian(environment, irlAlgo, boundUpdatedWeightMatrix, updatedSAPairCountsForSubsetOfTrajectories, newlyComputed_Policy_Value_H_Q_matrices.get("P"), null, null, true);
		
				Pair<Double, double[][]> PriorProbAndGradientForTable_i_fromNewWeightMatrix= priorIRL.computeLogPriorAndGradient(boundUpdatedWeightMatrix);
				
				logPosteriorProbability_updated = LLHAndGradientForTable_i_fromNewWeightMatrix.getFirst() + PriorProbAndGradientForTable_i_fromNewWeightMatrix.getFirst(); //NEW log posteriorProbability of REWARD Function = NEW log likelihood + NEW priorProbability of REWARD FUNCTION
				
				gradient_updated_forTablei = LLHAndGradientForTable_i_fromNewWeightMatrix.getSecond().add(new DoubleMatrix(PriorProbAndGradientForTable_i_fromNewWeightMatrix.getSecond()));  //NEW gradient = gradientBasedOnQMatrixLikelihood + gradientBasedOnPrior
		
				DoubleMatrix sumGradients = gradient_updated_forTablei.add(gradient_forTablei);
				DoubleMatrix prod1= sumGradients.mul(scalingParameter*0.5);
				DoubleMatrix sum1= new DoubleMatrix(randomShift).add(prod1);
				DoubleMatrix g_Updated1 = MatrixUtilityJBLAS.squaredMatrix(sum1);
				////Reformatting so calculations are correct....DoubleMatrix g_Updated1 = MatrixUtilityJBLAS.squaredMatrix(new DoubleMatrix(randomShift).add(gradient_updated_forTablei.add(gradient_forTablei).mul(scalingParameter*0.5)));
	
				double gNumerator = Math.exp(g_Updated1.sum()*-0.5);
				double fNumerator = Math.exp(logPosteriorProbability_updated);
				double fDenominator = Math.exp(logPosteriorProbability);
				double gDenominator = Math.exp(MatrixUtilityJBLAS.squaredMatrix(new DoubleMatrix(randomShift)).sum()*-0.5);
				double probQuotient =  (fNumerator*gNumerator)/(fDenominator*gDenominator);
				
				double rand2 = RNG.nextDouble();
				if(Double.compare(probQuotient, rand2)>0) {
		    		tableWeightVectors.put(table_i, boundUpdatedWeightMatrix);
		    		tablePolicyVectors.put(table_i, newlyComputed_Policy_Value_H_Q_matrices.get("P"));
		    		tableValueVectors.put(table_i, newlyComputed_Policy_Value_H_Q_matrices.get("V"));
		    		tableQVectors.put(table_i, newlyComputed_Policy_Value_H_Q_matrices.get("Q"));
		    		
		    		
		    		restaurantLikelihoods.put(table_i, LLHAndGradientForTable_i_fromNewWeightMatrix.getFirst());
		    		restaurantPriors.put(table_i, PriorProbAndGradientForTable_i_fromNewWeightMatrix.getFirst());
		    		
		    		restGradientsLLH.put(table_i, LLHAndGradientForTable_i_fromNewWeightMatrix.getSecond());

		    		
		    		restGradientsPrior.put(table_i, new DoubleMatrix(PriorProbAndGradientForTable_i_fromNewWeightMatrix.getSecond()));
	
			    }
				weight_policy_q_vectorMapForTable_i.clear(); //need to reset the weight, policy, vector for each reward update iteration
			}//end for-loop for rewardd update iterations
		}// end of else-condition 
	    rmap5 = new RestaurantMap(tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, tableAssignmentMatrix);
	    rmap5._restLikeLihoods = restaurantLikelihoods;
	    rmap5._restPriors = restaurantPriors;
	    rmap5._restGradientsFromLLH = restGradientsLLH;
	    rmap5._restGradientsFromPrior = restGradientsPrior;
	    //JK 6.23.2019 saving updated station-action-count map as well
	    //rmap5._saPairCountsInfoForSubsetOfTrajectories = updatedSAPairCountsForSubsetOfTrajectories;
	    
	    return rmap5;
	    		
		
	}
	public static double computeMinProbabilityQuotient(double[][] currentTblWeightVector_customer_i, double [][] currentTblValueVector_customer_i, double [][] altTblWeightVector_customer_i, double[][] altTblValueVector_customer_i, MDP env, IRLAlgorithm irlAlgo, Multimap<Integer,double[]> stateActPairCountsInfoForAllTrajectories, Integer customer_i) {
				
				//Compute likelihood (according to softmax distribution) for customer_i originally assigned table (based on Q matrix computed from weight and value matrices) 
				//Recall:The log of a quotient is the difference of the logs;Therefore, the rule for division is to subtract the logarithms
				//Since the likelihood computation involves the quotient of two log() values, we simply subtract them to obtain the qBasedLikelihood
				double [][] qMatrix1 = PolicySolver.policyImprovementStep(env, currentTblValueVector_customer_i, currentTblWeightVector_customer_i).get("Q"); //returns the log-based qMatrix of dimension [ numStates x numActions]
				double [][] qMatrixeta1 = MatrixUtilityJBLAS.scalarMultiplication(qMatrix1, irlAlgo.getLikelihood().getEta());
				DoubleMatrix qMatrixeta1RealMat = new DoubleMatrix(qMatrixeta1);
				
				double [][] expQLLH1= MatrixUtilityJBLAS.exp(qMatrixeta1);
				double [][] sumeQLLH1 = MatrixUtilityJBLAS.sumPerRow(expQLLH1); //returns column matrix, with sum across columns per row; thus for each trajectory/customer we obtain sum of q-values
				double [][] logSumQLLH1 = MatrixUtilityJBLAS.log(sumeQLLH1); //returns log of each element in column matrix
				DoubleMatrix logSumMatrix1 = new DoubleMatrix(logSumQLLH1);
				
				DoubleMatrix qBasedLikelihood1 = MatrixUtilityJBLAS.elementwiseSubtractionByColumnVector(qMatrixeta1RealMat, logSumMatrix1); //=loglikelihood of original restaurant table assigned to trajectory 'customer_i' ; [numStates x numActions] matrix
				
				//Compute likelihood (according to softmax distribution) for customer_i alternative/new table (based on Q matrix computed from weight and value matrices) ; same as above
				double [][] qMatrix2 = PolicySolver.policyImprovementStep(env, altTblValueVector_customer_i, altTblWeightVector_customer_i).get("Q");

				double [][] qMatrixeta2 = MatrixUtilityJBLAS.scalarMultiplication(qMatrix2, irlAlgo.getLikelihood().getEta());
				
				DoubleMatrix qMatrixeta2realMat = new DoubleMatrix(qMatrixeta2);
				
				double [][] expQLLH2= MatrixUtilityJBLAS.exp(qMatrixeta2);
				double [][] sumeQLLH2 = MatrixUtilityJBLAS.sumPerRow(expQLLH2); //returns column matrix, with sum across columns per row
				double [][] logSumQLLH2 = MatrixUtilityJBLAS.log(sumeQLLH2); //returns log of each element in column matrix
				DoubleMatrix logSumMatrix2 = new DoubleMatrix(logSumQLLH2);
				///JK this already a column matrix RealVector logSumMatrix2realVec = logSumMatrix2.getColumnVector(0);
				
				DoubleMatrix qBasedLikelihood2 = MatrixUtilityJBLAS.elementwiseSubtractionByColumnVector(qMatrixeta2realMat, logSumMatrix2); //loglikelihood of alternative/new restaurant table assigned to trajectory 'customer_i'; [numStates x numActions] matrix

				//Compute quotient of likelihoods ( between original table likelihood and alternative/new table likelihood)
				double logLLHQuotient = 0;

				if(customer_i == null) { //calculate likelihood quotients based on ALL customers/trajectories (this is only used for DPM_MH without Langevin diffusion)
					if (!stateActPairCountsInfoForAllTrajectories.isEmpty()) {
						 //for each sub-trajectory listed in map;recall that this Map of <Integer, stateActionPairCounts> begins at integer =1; (since this map can be representative of a subset/whole of original trajectory dataset, the first sub-trajectory in this map is not necessarily customer 1.
						//JK. 6.23.2019 , rewrote to start at traj=0; for each sub-trajectory listed in map;recall that this Map of <Integer, stateActionPairCounts> begins at integer =1; (since this map can be representative of a subset/whole of original trajectory dataset, the first sub-trajectory in this map is not necessarily customer 1.
						for (int traj_j: stateActPairCountsInfoForAllTrajectories.keySet()) {
							//System.out.println("stateActPairCountsInfoForSubsetOfTrajectories keyset size = "+stateActPairCountsInfoForSubsetOfTrajectories.keySet().size());
							List<double []> saPairCountsForTrajJ = (List<double[]>) stateActPairCountsInfoForAllTrajectories.get(traj_j);
							
							for(int observedSA=0; observedSA< saPairCountsForTrajJ.size(); observedSA++) { //for each OBSERVED state-action pair (with count >1) in trajectory j
								Double state = saPairCountsForTrajJ.get(observedSA)[0]; //get state for that observed sa-pair
								Double action = saPairCountsForTrajJ.get(observedSA)[1]; //get action
								Double count = saPairCountsForTrajJ.get(observedSA)[2]; //get count ; number of times s,a was observed in trajectory j
								logLLHQuotient = logLLHQuotient + (qBasedLikelihood1.get(state.intValue(), action.intValue())-qBasedLikelihood2.get(state.intValue(), action.intValue()))*count;
							}
						}
					}
				}
				else {
					if (!stateActPairCountsInfoForAllTrajectories.isEmpty()) {
							List<double []> saPairCountsForTrajJ = (List<double[]>) stateActPairCountsInfoForAllTrajectories.get(customer_i);
							
							for(int observedSA=0; observedSA< saPairCountsForTrajJ.size(); observedSA++) { //for each OBSERVED state-action pair (with count >1) in trajectory j
								Double state = saPairCountsForTrajJ.get(observedSA)[0]; //get state for that observed sa-pair
								Double action = saPairCountsForTrajJ.get(observedSA)[1]; //get action
								Double count = saPairCountsForTrajJ.get(observedSA)[2]; //get count ; number of times s,a was observed in trajectory j
								logLLHQuotient = logLLHQuotient + (qBasedLikelihood1.get(state.intValue(), action.intValue())-qBasedLikelihood2.get(state.intValue(), action.intValue()))*count;
							}
						
					}
					
				}
				
				//we want to return the non-log based probability
				double minProbability = Math.exp(logLLHQuotient);
				return minProbability;
	}
	
	


}
