package CRC_Prediction;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.jblas.DoubleMatrix;

import CRC_Prediction.Utils.MatrixUtilityJBLAS;
import CRC_Prediction.Utils.VectorUtility;

/**
 * Markov Decision Process class where problem environment can be defined
 * @author m186806
 *
 */
public class MDP implements Serializable {
	
	static final long serialVersionUID = 1;
	
		//Set parameters specific to the problem we wish to solve;
		public  String _name;
		/**
		 * Discount factor for RL reward calculations
		 */
		public  double _discount = 0.9;
		/**
		 *# of experiments (problem instances; 10 sets of behavior data) jk
		 */
//		public  int _numberExperiments = 10; 
		// # of experts (this is the GROUND TRUTH # of tables for the generated trajectory; 
		// i.e. our MH sampled restaurant will infer that the number of tables = X, where X will ideally end up being equal to numExperts)
		public  int _numberTrueTables  = 3; 
		/**
		 * Number of trajectories to be generated per expert
		 */
//		public  int _numberDemoTrajectoriesPerTable    = 2;	// Default
		public  int _numberDemoTrajectoriesPerTable    = 4;  
		/**
		 * # of steps in each trajectory
		 */
		public  int _numberStepsPerTrajectory    = 40;   
		public  int _initialRandomSeed  = 1;   // initial random seed
		//Obtain the trajectories consisting of 40 time steps and measure the performance as we
		//increase the number of trajectories per reward function.
		//public static int _basisVectorSize =8; //we defined the number of states explicitly
		//blockSize = 2;
		public  double _noise = 0.3;
		public  double _newExpertProbability = 0.0;
		/**
		 * // # of new trajectories (for transfer learning task) jk
		 */
		public  int _newExperiments  = 20;  
		public  int _newTrajectorySteps  = _numberStepsPerTrajectory;
		
		public  int _numberStates;
		public  int _numberRewardFeatures;
		public  int _numberActions;
		public  boolean _useSparseMatrix = false;
		
		/**
		 * Each element of map corresponds to 1 out of the _numberActions possible; i.e. _transitionMatrix = numStates x numStatex x numActions; 
		 * Each element represents the probability of transitioning to the next state (row) given the current state(col), 
		 * for the action a_i (HashMap index integer)
		 */
		public  HashMap<Integer, double[][]> _transitionMatrix; 
		
		/**
		 * This is the TRANSPOSE of the _transitionmatrix.
		 * Each matrix elements represents the DISCOUNTED probability of transitioning from current state (row) to the next state(col), 
		 * for the action a_i  (HashMap index integer)
		 */
		//public  HashMap<Integer, double [][]> _discountedTransitionMatrix;
		public  DoubleMatrix _discountedTransitionMatrix;

		
		/**
		 *list of (numActions) [nStates x nFeatures] matrices: 
		 *for each action, we have a matrix specifying which one of the 16 feature-indices is associated with each of the 64 possible states, 
		 *given the specific action; each matrix within this list will indicate the presence/absence (1/0) of a given feature f_j at given state 's_i'
		 */
		public  ArrayList<double[][]> _stateFeatureMatrix;  
		
		public  double [] _startDistribution;
		public  double [][] _weightMatrix;
		/**
		 * Placeholder variable to store reward function computed with current MDP environment variables and parameters
		 */
		public  double[][] _rewardFunction;
 
		
		/**
		 * Constructor for Markov Decision Process
		 * @param environmentName
		 * @param initSeed
		 * @param discount
		 */
		public MDP(String environmentName, int initSeed, double discount, int numStates, int numActions, int numRewardFeatures, boolean useSparse) {
			setMDPName(environmentName);
			setInitialSeed(initSeed);
			setDiscount(discount);
			setNumStates(numStates);
			setNumActions(numActions);
			setNumRewardFeatures(numRewardFeatures);
			setIfSparse(useSparse); 
			
			initializeMDP();
			//create initial transiton matr
			System.out.println("Finished initializing MDP");
			
		    
		}
		
		private void initializeMDP() {
			
//			MersenneTwisterFastIRL randomNumGen = new MersenneTwisterFastIRL(getInitialSeed());	// GTD No longer used
			
			// Initialize the 2 transition matrices: state transition: T(s',s,a) = P(s'|s,a)
			//_transitionmatrix models the probability of the next state (row), given current state (col) for action a_i (HashMap Integer index)
			//_discountedTransitonMatrix models the probability of the next state (col), given the current state(row) for action a_I (HashMap Integer index)
			HashMap<Integer, double[][]> transitionMatrix = new HashMap <Integer, double[][]>();
			//HashMap<Integer, double[][]> discountedTMatrix = new HashMap<Integer, double[][]>();
			DoubleMatrix discountedTMatrix ;

			for(int a=0; a< _numberActions; a++) {
				double[][] tMatrixForAction_a = new double[_numberStates][_numberStates];
//				double [][] dtMatrixForAction_a = new double[_numberStates][_numberStates];	// GTD Not used
				
				transitionMatrix.put(a, tMatrixForAction_a);
				//discountedTMatrix.put(a, dtMatrixForAction_a);
			}
			
			
	//List state indices for each coordinate:
//		for(int y=1; y<8+1; y++) { 
// 			
// 			for(int x=1; x<8+1; x++ ) {
// 				int stateIndex = generateIndexFromCoordinates(x, y, 8);
// 				System.out.println("(x,y)|("+x+","+y+") = "+stateIndex);
// 				
// 			}
// 		}
	//Determine the neighbors of each state based on knowledge of the physical grid.
			for(int y =1; y< 8+1; y++) {
				for (int x=1; x<8+1; x++) {
					// set the index value of the 'state' at location (x,y) to integer value between[1:64]
					int stateIndex = generateIndexFromCoordinates(x, y, 8);
					// create 4x1 zero array corresponding to the different possible states that we can transition to 
					// from state 's' via one of 4 possible actions.
					int [] possibleNextStates = new int[4];
					//set the index value of the 'state' that we transition to from 's' after executing action 0 ='north'
					possibleNextStates[0] = generateIndexFromCoordinates(x, y-1, 8);
					//set the index value of the 'state' that we transition to from 's' after executing action 1 ='south'
					possibleNextStates[1] = generateIndexFromCoordinates(x, y+1, 8);
					//set the index value of the 'state' that we transition to from 's' after executing action 2 ='west'
					possibleNextStates[2] = generateIndexFromCoordinates(x-1, y, 8);
					//set the index value of the 'state' that we transition to from 's' after executing action 3 ='east'
					possibleNextStates[3] = generateIndexFromCoordinates(x+1, y, 8);
					
					//use knowledge about the neighbors of state 's' in order to set ALL the transition probabilities from it; 
					for(int a_i=0; a_i<_numberActions; a_i++) {
						double[][] tMatrixForAction_ai = transitionMatrix.get(a_i);
						//System.out.println("actual action: "+a_i);
						for(int a_j=0; a_j <_numberActions; a_j++) {
							System.out.println("possibleNextStates[a_j] ="+possibleNextStates[a_j]);
							System.out.println("+			x,y ="+x+","+y);

							System.out.println("stateIndex ="+stateIndex);
							//account for stochastic transition probability of moving to any neighbor state by random chance
							tMatrixForAction_ai[possibleNextStates[a_j]][stateIndex]= 
									tMatrixForAction_ai[possibleNextStates[a_j]][stateIndex] + (_noise/(double)_numberActions);
//							System.out.println ("			transition prob of stochastic move " + a_j + " :" + 
//												tMatrixForAction_ai[possibleNextStates[a_j]][stateIndex]);
						}
						//'normalize?' the transition probability of moving to the next state that is actually associated with the given action
						//% sum up the total prob=1, by setting the transition probability of the state corresponding to action 'a' = 0.3/4 +.7 
						// (i.e. inc. the probability of the state associated with the given action)
						tMatrixForAction_ai[possibleNextStates[a_i]][stateIndex] = tMatrixForAction_ai[possibleNextStates[a_i]][stateIndex]+1 - _noise;

					}
				}
			}

//
// check that total transition probability..is within bounds (is not negative, and if greater than 1 than at least less than maximum error amount of 1e-6)
			for (int s=0; s<_numberStates; s++) {
				for (int a=0; a<_numberActions; a++) {
// 					System.out.println("Checking for error... (state,action)=("+s+","+a+")");
					double [][] transitionMatrForAction_a = transitionMatrix.get(a);
					double error = Math.abs(MatrixUtilityJBLAS.sumPerColumn(transitionMatrForAction_a)[s]-1);
					// determine number of non zero matrix elements and if total probability of transitions for s,a pair is >1 (more than > 1e-6) OR 
					// if the probability of any of the possible NEXT states is NEGATIVE or GREATER THAN 1
					if (error >1e-6 || MatrixUtilityJBLAS.sumPerColumn(MatrixUtilityJBLAS.lessEqualityComparison(transitionMatrForAction_a, 0.0))[s]>0 || 
						MatrixUtilityJBLAS.sumPerColumn(MatrixUtilityJBLAS.greaterEqualityComparison(transitionMatrForAction_a, 1.0))[s]>0)
					{
						System.out.println("sum of column "+s+" = "+MatrixUtilityJBLAS.sumPerColumn(transitionMatrForAction_a)[s]);
						System.out.println("Error ="+error);
						System.out.println ("# of NEXT STATES with probabilities being negative = " + 
											MatrixUtilityJBLAS.sumPerColumn(MatrixUtilityJBLAS.lessEqualityComparison(transitionMatrForAction_a, 0.0))[s]);
						System.out.println ("# of NEXT states with probabilities >1 = " + 
										MatrixUtilityJBLAS.sumPerColumn(MatrixUtilityJBLAS.greaterEqualityComparison(transitionMatrForAction_a, 1.0))[s]);
						throw new java.lang.RuntimeException("bad transition probability found!!!");
					}
					
				}
			}
			
			
			//Create DISCOUNTED TRANSITION MATRIX by transposing _transitionMatrix and multiplying by discount factor
//			for (int a=0; a< _numberActions; a++) {
//				double [][] transposedTMatrix = MatrixUtility.transpose(transitionMatrix.get(a));
//				discountedTMatrix.put(a, MatrixUtility.scalarMultiplication(transposedTMatrix, _discount));
//			}
			discountedTMatrix = MatrixUtilityJBLAS.convertHashMapToRealMatrixCols(transitionMatrix);
//			double [][] dMatrixFooAs2Darray = discountedTMatrix.toArray2();	// GTD Not used

			DoubleMatrix dMatrix = discountedTMatrix.transpose().mul( _discount);

//			double[][] dmatriDBlarray = dMatrix.toArray2();	// GTD Not used
//
//			create state x feature matrix: specifying which one of the 16 feature-indices is associated with each of the 64 possible states
			double [][] stateFeatureMatrix = new double [_numberStates][_numberRewardFeatures];
			for (int y=1 ; y<8+1; y++) {
				for (int x=1; x<8+1 ; x++) {
					int stateIndex = generateIndexFromCoordinates(x, y, 8); //index value of state corresponding to location (x,y)
					Double i = Math.ceil(x/2.0);
					Double j = Math.ceil(y/2.0);
					//get the index value associated with the given block region [1:16]
					int f = generateIndexFromCoordinates(i.intValue(), j.intValue(), 8/2);
					stateFeatureMatrix [stateIndex][f]= 1;//set/indicate the index value of the feature associated with the given state 's'
				}
			}
//
//			initial state distribution

			double [] initialDistribution = new double [_numberStates];
			for (int r =0; r< initialDistribution.length; r++) {
				initialDistribution[r]= (double)1/_numberStates; // set the intial probability of each state = 1/64 =b_0(s_i)
			}
			
//			weight vector  (sum of weights can be >1?)jk
			double [][] weightMatr = new double [_numberRewardFeatures][1]; //creates a 16x1 array of zeros
			//returns a row vector containing a random permutation of the integers from 1 to nF-1=15 inclusive
			int [] randomPermutation1 = VectorUtility.createPermutatedVector(_numberRewardFeatures,0);
			Double k = Math.ceil(0.3*_numberRewardFeatures); // 0.3 *16 = 4.8 = 5 (when rounded up)
			//get the sub-vector of 0:4 integer values from row vector randomPermutation1
			int [] subPermutation1  = VectorUtility.rangeOfVector(randomPermutation1, 0, k.intValue()-1, 1);
			double [][] rdmWeights = MersenneTwisterFastIRL.RandomUniformMatrix(_numberRewardFeatures, 1, 1);

			// set the weight vector values for a subset of the elements in the 16x1 weight vector to a  k x 1 array of (uniformly distributed)
			// random values minus 1; minus so that they are all negative?!
			for(int r: subPermutation1 ) {
				weightMatr[r][0]= rdmWeights[r][0]-1;
			}
			weightMatr[weightMatr.length-1][0] =1; //set last index of array =1
			
			
			//set MDP class variables
			_startDistribution = initialDistribution;
			_transitionMatrix = transitionMatrix;
			//_discountedTransitionMatrix = discountedTMatrix;
			_discountedTransitionMatrix = dMatrix;

			
			// essentially using duplicate of same state-feature matrix for each action (since in mazeworld, 
			// reward features are state-based and NOT action based, as a result, 
			// the reward features pertinent for a given action will be the same for all actions)
			ArrayList<double[][]> sfMatrixList = new ArrayList<double[][]>();
			for (int a=0; a<_numberActions; a++) {
				sfMatrixList.add(MatrixUtilityJBLAS.deepCopy(stateFeatureMatrix));
			}
			_stateFeatureMatrix = sfMatrixList;
			
			//double [] alternativeWeights = 
//					new double [] {0,0, -0.693872546465222,0,0,0,0,-0.220763770897981,0,0,-0.0211367640875018,-0.977827468701620,0,0,-0.0893416390569715,1};
			_weightMatrix = weightMatr;
			//_weightMatrix = MatrixUtils.createColumnRealMatrix(alternativeWeights).getData();
			
			// Reshape the weighted state-feature matrix into a 64x4 matrix
			RealMatrix rewardMatrix = MatrixUtils.createRealMatrix(_numberStates, _numberActions);
			for (int a =0; a< _numberActions; a++) {
				rewardMatrix.setColumnMatrix(a, MatrixUtils.createRealMatrix(stateFeatureMatrix).multiply(MatrixUtils.createRealMatrix(weightMatr)));
			}
			
			_rewardFunction = rewardMatrix.getData();
		
		}//end initializeMDP()
		
		
		//set methods
		public void setMDPName(String envName) {
			_name = envName;
		}
		public void setInitialSeed(int seedVal) {
			_initialRandomSeed = seedVal;
		}
		public void setDiscount(double d) {
			_discount =d;
		}
		public void setNumStates(int nStates) {
			_numberStates = nStates;
		}
		public void setIfSparse(boolean isSparse) {
			_useSparseMatrix = isSparse;
		}
		public void setNumActions(int nActions) {
			_numberActions = nActions;
		}
		public void setNumRewardFeatures(int numRFeatures) {
			_numberRewardFeatures = numRFeatures;
		}
		
		public void setTransitionMatrix( HashMap<Integer, double[][]> tMatrix) {
			_transitionMatrix = tMatrix;
		}
//		public void setDiscountedTransitionMatrix(HashMap<Integer, double[][]> dtMatrix) {
//			_discountedTransitionMatrix = dtMatrix;
//		}
		public void setDiscountedTransitionMatrix(DoubleMatrix dtMatrix) {
			_discountedTransitionMatrix = dtMatrix;
		}
		public void setStateFeatureMatrix( ArrayList<double[][]> stateFeatureMatrix) {
			_stateFeatureMatrix = stateFeatureMatrix;
		}
		
		public void setStartDistribution( double [] startDistribution) {
			_startDistribution = startDistribution;
		}
		public void setWeight( double [][] wt) {
			_weightMatrix =wt;
		}
		public void setRewardFunction( double[][] rwrdFunction) {
			_rewardFunction = rwrdFunction;
		}
		
		//get methods
		public String getMDPName() {
			return _name;
		}
		public int getInitialSeed() {
			return _initialRandomSeed;
		}
		public double getDiscount() {
			return _discount;
		}
		public int getNumStates() {
			return _numberStates;
		}
		public int getNumActions() {
			return _numberActions;
		}
		public boolean getIfSparse() {
			return _useSparseMatrix;
		}
		public int getNumRewardFeatures() {
			return _numberRewardFeatures;
		}
		public HashMap<Integer ,double[][]> getTransitionMatrix() {
			return _transitionMatrix;
		}
//		public HashMap<Integer, double[][]> getDiscountedTransitionMatrix(){
//			return _discountedTransitionMatrix;
//		}
		public DoubleMatrix getDiscountedTransitionMatrix(){
			return _discountedTransitionMatrix;
		}
		public ArrayList<double[][]> getStateFeatureMatrix() {
			return _stateFeatureMatrix;
		}
		
		public double [] getStartDistribution() {
			return _startDistribution;
		}
		public double [][] getWeight() {
			return _weightMatrix;
		}
		public double[][] getRewardFunction() {
			return _rewardFunction;
		}
		public int getNumTrueTables() {
			return _numberTrueTables;
		}
		public int getNumDemoTrajectoriesPerTable() {
			return _numberDemoTrajectoriesPerTable;
		}
		public int getNumStepsPerTrajectory() {
			return _numberStepsPerTrajectory;
		}
		
		public int generateIndexFromCoordinates(int xCoord, int yCoord, int dimension1) {
			//computes the numerical index of a 'state' based on its x,y coordinates
			int x = Math.max(1, Math.min(dimension1, xCoord));
			int y = Math.max(1, Math.min(dimension1, yCoord));
			int index = (y - 1)*dimension1 + x-1;

			return index;
		}
		
		
		





}
