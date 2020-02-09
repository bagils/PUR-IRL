
package CRC_Prediction;


import CRC_Prediction.Utils.*;
import com.datastax.driver.core.Row;
import com.datastax.driver.core.Session;
import java.io.*;
import java.util.*;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.jblas.DoubleMatrix;


/**
 * Markov Decision Process class where problem environment for 'CANCER' can be defined
 * This builds on top of original MDP class but is designed to build state feature-matrices and
 * transition-matrices from data as opposed to expert knowledge of a grid
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
public class MDPCancer implements Serializable
{
	
	static final long serialVersionUID = 1;
	// Set parameters specific to the problem we wish to solve;
	public String						_name;
	/**
	 * Discount factor for RL reward calculations
	 */
	public double						_discount						= 0.9;
	/**
	 * # of experiments (problem instances; 10 sets of behavior data) jk
	 */
	public int							_numberExperiments				= 10;
	/**
	 * # of experts (this is the GROUND TRUTH # of tables for the generated trajectory; 
	 * i.e. our MH sampled restaurant will infer that the number of tables = X, where X will ideally end up being equal to numExperts)
	 */
	public int							_numberTrueTables				= 3;
	/**
	 * Number of trajectories to be generated per expert
	 */
//	public int							_numberDemoTrajectoriesPerTable	= 2; //default
	public int							_numberDemoTrajectoriesPerTable	= 4;

	/**
	 * # of steps in each trajectory
	 */
	public int							_numberStepsPerTrajectory		= 40;
	public int							_initialRandomSeed				= 1;						// initial random seed
	// Obtain the trajectories consisting of 40 time steps and measure the performance as we
	// increase the number of trajectories per reward function.
	// public static int _basisVectorSize =8; //we defined the number of states explicitly
	// blockSize = 2;
	public double						_noise							= 0.3;
	public double						_newExpertProbability			= 0.0;
	/**
	 * // # of new trajectories (for transfer learning task) jk
	 */
	public int							_newExperiments					= 20;
	public int							_newTrajectorySteps				= _numberStepsPerTrajectory;
	
	public int							_numberStates;
	public int							_numberRewardFeatures;
	public int							_numberActions;
	
	public int							_numberStateBasisVectors;									// JK: added number of basis vectors 2.27.2019
	public int							_numberActionBasisVectors;
	
	public boolean						_useSparseMatrix				= false;
	
	
	/**
	 * Each element of map corresponds to 1 out of the _numberActions possible; i.e.
	 * _transitionMatrix = numStates x numStatex x numActions;
	 * Each element represents the probability of transitioning to the next state (row) given the
	 * current state(col), for the action a_i (Map index integer)
	 */
	public Map<Integer, double[][]>	_transitionMatrix;
	
	/**
	 * This is the TRANSPOSE of the _transitionmatrix.
	 * Each matrix elements represents the DISCOUNTED probability of transitioning from current
	 * state (row) to the next state(col), for the action a_i (Map index integer)
	 */
	public DoubleMatrix					_discountedTransitionMatrix;
	
	/**
	 * list of (numActions) [nStates x nFeatures] matrices:
	 * for each action, we have a matrix specifying which one of the 16 feature-indices is
	 * associated with each of the 64 possible states,
	 * given the specific action; each matrix within this list will indicate the presence/absence
	 * (1/0) of a given feature f_j at given state 's_i'
	 */
	public Map<Integer, double[][]>		_stateFeatureMatrixMAP;
	
	public double[]						_startDistribution;
	public double[][]					_weightMatrix;
	/**
	 * Placeholder variable to store reward function computed with current MDP environment variables
	 * and parameters
	 */
	public double[][]					_rewardFunction;
	
	// necessary to access Cassandra dB tables
	// private static String serverIPZ = "127.0.0.1";
	// private static String keyspacejkZ = "crckeyspace";
	// private static Cluster clusterZ = Cluster.builder().addContactPoint(serverIP).build();
	// private static Session sessionZ = cluster.connect(keyspacejk);
	// JK 3.19.2019: we need to use the same cassandra session constructed in
	// MainProgramCancer.class
	private static Session				_cassSession;
	
	//JK 7.19.2019 Added to differentiate between cancer trajectories (in which action 0 is used for padding) and GridWorld (where action 0 is a cardinal direction)
	public boolean						_isGridWorldEnv					= false;

	
	
	/**
	 * Constructor for Markov Decision Process with pre-existing Cassandra table info
	 * 
	 * @param environmentName
	 * @param initSeed
	 * @param discount
	 */
	public MDPCancer (String environmentName, int initSeed, double discount, int numStates, int numActions, int numRewardFeatures, 
					  int numStateBasisVectors, int numActionBasisVectors, boolean useSparse, Session cassandraSession)
			throws IOException
	{
		setMDPName (environmentName);
		setInitialSeed (initSeed);
		setDiscount (discount);
		setNumStates (numStates);
		setNumActions (numActions);
		setNumRewardFeatures (numRewardFeatures);
		setNumStateBasisVectors (numStateBasisVectors);
		setNumActionBasisVectors (numActionBasisVectors);
		setIfSparse (useSparse);
		
		_cassSession = cassandraSession;
		
		// create initial transition matrix
		initializeMDP ();
	}
	
	
	/**
	 * Constructor for Markov Decision Process from existing one
	 * 
	 * @param base	{@link MDPCancer} to copy from
	 */
	public MDPCancer (MDPCancer base)
	{
		setMDPName (base._name);
		setInitialSeed (base._initialRandomSeed);
		setDiscount (base._discount);
		setNumStates (base._numberStates);
		setNumActions (base._numberActions);
		setNumRewardFeatures (base._numberRewardFeatures);
		setNumStateBasisVectors (base._numberStateBasisVectors);
		setNumActionBasisVectors (base._numberActionBasisVectors);
		setIfSparse (base._useSparseMatrix);
		setIfGridWorld(base._isGridWorldEnv);
		
//		_cassSession = base._cassSession;
		
		_stateFeatureMatrixMAP = new HashMap<Integer, double[][]> (base._stateFeatureMatrixMAP);
		_transitionMatrix = new HashMap<Integer, double[][]> (base._transitionMatrix);
		_discountedTransitionMatrix = base._discountedTransitionMatrix.dup ();
		_startDistribution = copy (base._startDistribution);
		_weightMatrix = copy (base._weightMatrix);
		_rewardFunction = copy (base._rewardFunction);
		
//		initializeMDP ();
		// create initial transition matrix
	}
	
	
	/**
	 * Constructor for Markov Decision Process from scratch using VCF files and TreeTraversal
	 * instance
	 * 
	 * @param environmentName
	 * @param initSeed
	 * @param discount
	 */
	public MDPCancer (String environmentName, int initSeed, double discount, int numStates, int numActions, int numRewardFeatures, 
					  int numStateBasisVectors, int numActionBasisVectors, boolean useSparse, TreeTraversal traversedTreeInstance, 
					  Session cassandraSession) throws IOException
	{
		setMDPName (environmentName);
		setInitialSeed (initSeed);
		setDiscount (discount);
		setNumStates (numStates);
		setNumActions (numActions);
		setNumRewardFeatures (numRewardFeatures);
		setNumStateBasisVectors (numStateBasisVectors);
		setNumActionBasisVectors (numActionBasisVectors);
		setIfSparse (useSparse);
		
		_cassSession = cassandraSession;
		
		// create initial transition matrix
		initializeMDP (traversedTreeInstance);
		
	}
	
	
	/**
	 * Constructor for TOY-MODEL Markov Decision Process
	 * 
	 * @param environmentName
	 * @param initSeed
	 * @param discount
	 */
	public MDPCancer (String environmentName, int initSeed, double discount, int numStates, int numActions, int numRewardFeatures, 
					  boolean useSparse, boolean useGridWEnvironment, int numSimulExperts, int numDemoTrajsPerSimulExpert)
	{
		setMDPName (environmentName);
		setInitialSeed (initSeed);
		setDiscount (discount);
		setNumStates (numStates);
		setNumActions (numActions);
		setNumRewardFeatures (numRewardFeatures);
		setIfSparse (useSparse);
		setIfGridWorld (useGridWEnvironment);
		setNumTrueTables (numSimulExperts);
		setNumDemoTrajectoriesPerTable (numDemoTrajsPerSimulExpert);
		
//		initializeTOYMDP (); // starts at actions from '0' (which is considered the default END action used for padding)
		// create initial transition matrix
		initializeTOYMDPV2 (); // starts at actions from '0' (which is considered the default END action used for padding)
	}
	
	
	private void initializeTOYMDPV2() {
		
		int gridLength = (int) Math.sqrt(_numberStates);
		
		Map<Integer, double[][]> safMatrixMAP = new HashMap<Integer, double[][]> ();

		
		
		// Initialize the 2 transition matrices: state transition: T(s',s,a) = P(s'|s,a)
		//_transitionmatrix models the probability of the next state (row), given current state (col) for action a_i (HashMap Integer index)
		//_discountedTransitonMatrix models the probability of the next state (col), given the current state(row) for action a_I (HashMap Integer index)
		HashMap<Integer, double[][]> transitionMatrix = new HashMap <Integer, double[][]>();
		DoubleMatrix discountedTMatrix ;

		for(int a=0; a< _numberActions; a++) {
			double[][] tMatrixForAction_a = new double[_numberStates][_numberStates];
			
			transitionMatrix.put(a, tMatrixForAction_a);
		}
		

//Determine the neighbors of each state based on knowledge of the physical grid.
		for(int y =1; y< gridLength+1; y++) {
			for (int x=1; x<gridLength+1; x++) {
				// set the index value of the 'state' at location (x,y) to integer value between[1:64]
				int stateIndex = generateIndexFromCoordinates(x, y, gridLength);
				// create 4x1 zero array corresponding to the different possible states that we can transition to 
				// from state 's' via one of 4 possible actions.
				int [] possibleNextStates = new int[4];
				//set the index value of the 'state' that we transition to from 's' after executing action 0 ='north'
				possibleNextStates[0] = generateIndexFromCoordinates(x, y-1, gridLength);
				//set the index value of the 'state' that we transition to from 's' after executing action 1 ='south'
				possibleNextStates[1] = generateIndexFromCoordinates(x, y+1, gridLength);
				//set the index value of the 'state' that we transition to from 's' after executing action 2 ='west'
				possibleNextStates[2] = generateIndexFromCoordinates(x-1, y, gridLength);
				//set the index value of the 'state' that we transition to from 's' after executing action 3 ='east'
				possibleNextStates[3] = generateIndexFromCoordinates(x+1, y, gridLength);
				
				//use knowledge about the neighbors of state 's' in order to set ALL the transition probabilities from it; 
				for(int a_i=0; a_i<_numberActions; a_i++) {
					double[][] tMatrixForAction_ai = transitionMatrix.get(a_i);
					//System.out.println("actual action: "+a_i);
					for (int a_j = 0; a_j < _numberActions; a_j++)
					{
						System.out.println ("possibleNextStates[a_j] =" + possibleNextStates[a_j]);
						System.out.println ("+			x,y =" + x + "," + y);
						
						System.out.println ("stateIndex =" + stateIndex);
						// account for stochastic transition probability of moving to any neighbor
						// state by random chance
						tMatrixForAction_ai[possibleNextStates[a_j]][stateIndex] = 
								tMatrixForAction_ai[possibleNextStates[a_j]][stateIndex] + (_noise / (double) _numberActions);
//						System.out.println (" transition prob of stochastic move " + a_j + " :" + tMatrixForAction_ai[possibleNextStates[a_j]][stateIndex]);
					}
					//'normalize?' the transition probability of moving to the next state that is actually associated with the given action
					//% sum up the total prob=1, by setting the transition probability of the state corresponding to action 'a' = 0.3/4 +.7 
					// (i.e. inc. the probability of the state associated with the given action)
					tMatrixForAction_ai[possibleNextStates[a_i]][stateIndex] = tMatrixForAction_ai[possibleNextStates[a_i]][stateIndex]+1 - _noise;

				}
			}
		}

//
//check that total transition probability..is within bounds (is not negative, and if greater than 1 than at least less than maximum error amount of 1e-6)
		for (int s=0; s<_numberStates; s++) {
			for (int a=0; a<_numberActions; a++) {
//					System.out.println("Checking for error... (state,action)=("+s+","+a+")");
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
//		for (int a=0; a< _numberActions; a++) {
//			double [][] transposedTMatrix = MatrixUtility.transpose(transitionMatrix.get(a));
//			discountedTMatrix.put(a, MatrixUtility.scalarMultiplication(transposedTMatrix, _discount));
//		}
		discountedTMatrix = MatrixUtilityJBLAS.convertHashMapToRealMatrixCols(transitionMatrix);

		DoubleMatrix dMatrix = discountedTMatrix.transpose().mul( _discount);

//
//		create state x feature matrix: specifying which one of the 16 feature-indices is associated with each of the 64 possible states
		double [][] stateFeatureMatrix = new double [_numberStates][_numberRewardFeatures];
		for (int y=1 ; y<gridLength+1; y++) {
			for (int x=1; x<gridLength+1 ; x++) {
				int stateIndex = generateIndexFromCoordinates(x, y, gridLength); //index value of state corresponding to location (x,y)
				Double i = Math.ceil(x/2.0);
				Double j = Math.ceil(y/2.0);
				//get the index value associated with the given block region [1:16]
				int f = generateIndexFromCoordinates(i.intValue(), j.intValue(), gridLength/2);
				stateFeatureMatrix [stateIndex][f]= 1;//set/indicate the index value of the feature associated with the given state 's'
			}
		}
//
//		initial state distribution

		double [] initialDistribution = new double [_numberStates];
		for (int r =0; r< initialDistribution.length; r++) {
			initialDistribution[r]= (double)1/_numberStates; // set the intial probability of each state = 1/64 =b_0(s_i)
		}
		
//		weight vector  (sum of weights can be >1?)jk
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
		_discountedTransitionMatrix = dMatrix;

		
		// essentially using duplicate of same state-feature matrix for each action (since in mazeworld, 
		// reward features are state-based and NOT action based, as a result, 
		// the reward features pertinent for a given action will be the same for all actions)
		for (int a=0; a<_numberActions; a++) {
			safMatrixMAP.put (a, MatrixUtilityJBLAS.deepCopy(stateFeatureMatrix));
		}
		

		_stateFeatureMatrixMAP = safMatrixMAP;
		
		_weightMatrix = weightMatr;
		
		// Reshape the weighted state-feature matrix into a 64x4 matrix
		RealMatrix rewardMatrix = MatrixUtils.createRealMatrix(_numberStates, _numberActions);
		for (int a =0; a< _numberActions; a++) {
			rewardMatrix.setColumnMatrix(a, MatrixUtils.createRealMatrix(stateFeatureMatrix).multiply(MatrixUtils.createRealMatrix(weightMatr)));
		}
		
		_rewardFunction = rewardMatrix.getData();
	
	}//end initializeTOYMDP()
	
	
	
	protected void initializeTOYMDP ()	// Not called by anything GTD 12/24/19
	{
		
		Map<Integer, double[][]> safMatrixMAP = new HashMap<Integer, double[][]> ();

		
		
		// Initialize the 2 transition matrices: state transition: T(s',s,a) = P(s'|s,a)
		//_transitionmatrix models the probability of the next state (row), given current state (col) for action a_i (HashMap Integer index)
		//_discountedTransitonMatrix models the probability of the next state (col), given the current state(row) for action a_I (HashMap Integer index)
		HashMap<Integer, double[][]> transitionMatrix = new HashMap <Integer, double[][]>();
		//HashMap<Integer, double[][]> discountedTMatrix = new HashMap<Integer, double[][]>();
		DoubleMatrix discountedTMatrix ;

		for(int a=0; a< _numberActions; a++) {
			double[][] tMatrixForAction_a = new double[_numberStates][_numberStates];
			
			transitionMatrix.put(a, tMatrixForAction_a);
		}
		
		
//List state indices for each coordinate:
//	for(int y=1; y<8+1; y++) { 
//			
//			for(int x=1; x<8+1; x++ ) {
//				int stateIndex = generateIndexFromCoordinates(x, y, 8);
//				System.out.println("(x,y)|("+x+","+y+") = "+stateIndex);
//				
//			}
//		}
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
					for (int a_j = 0; a_j < _numberActions; a_j++)
					{
						System.out.println ("possibleNextStates[a_j] =" + possibleNextStates[a_j]);
						System.out.println ("+			x,y =" + x + "," + y);
						
						System.out.println ("stateIndex =" + stateIndex);
						// account for stochastic transition probability of moving to any neighbor
						// state by random chance
						tMatrixForAction_ai[possibleNextStates[a_j]][stateIndex] = 
								tMatrixForAction_ai[possibleNextStates[a_j]][stateIndex] + (_noise / (double) _numberActions);
//						System.out.println (" transition prob of stochastic move " + a_j + " :" + tMatrixForAction_ai[possibleNextStates[a_j]][stateIndex]);
					}
					//'normalize?' the transition probability of moving to the next state that is actually associated with the given action
					//% sum up the total prob=1, by setting the transition probability of the state corresponding to action 'a' = 0.3/4 +.7 
					// (i.e. inc. the probability of the state associated with the given action)
					tMatrixForAction_ai[possibleNextStates[a_i]][stateIndex] = tMatrixForAction_ai[possibleNextStates[a_i]][stateIndex]+1 - _noise;

				}
			}
		}

//
//check that total transition probability..is within bounds (is not negative, and if greater than 1 than at least less than maximum error amount of 1e-6)
		for (int s = 0; s < _numberStates; s++)
		{
			for (int a = 0; a < _numberActions; a++)
			{
				// System.out.println("Checking for error... (state,action)=("+s+","+a+")");
				double[][] transitionMatrForAction_a = transitionMatrix.get (a);
				double error = Math.abs (MatrixUtilityJBLAS.sumPerColumn (transitionMatrForAction_a)[s] - 1);
				// determine number of non zero matrix elements and if total probability of transitions for s,a pair is >1 (more than > 1e-6) OR
				// if the probability of any of the possible NEXT states is NEGATIVE or GREATER THAN 1
				if ((error > 1e-6) ||
					(MatrixUtilityJBLAS.sumPerColumn (MatrixUtilityJBLAS.lessEqualityComparison (transitionMatrForAction_a, 0.0))[s] > 0) || 
					(MatrixUtilityJBLAS.sumPerColumn (MatrixUtilityJBLAS.greaterEqualityComparison (transitionMatrForAction_a, 1.0))[s] > 0))
				{
					System.out.println ("sum of column " + s + " = " + MatrixUtilityJBLAS.sumPerColumn (transitionMatrForAction_a)[s]);
					System.out.println ("Error =" + error);
					System.out.println ("# of NEXT STATES with probabilities being negative = " + 
										MatrixUtilityJBLAS.sumPerColumn (MatrixUtilityJBLAS.lessEqualityComparison (transitionMatrForAction_a, 0.0))[s]);
					System.out.println ("# of NEXT states with probabilities >1 = " + 
										MatrixUtilityJBLAS.sumPerColumn (MatrixUtilityJBLAS.greaterEqualityComparison (transitionMatrForAction_a, 1.0))[s]);
					throw new java.lang.RuntimeException ("bad transition probability found!!!");
				}
				
			}
		}		
		
		//Create DISCOUNTED TRANSITION MATRIX by transposing _transitionMatrix and multiplying by discount factor
//		for (int a=0; a< _numberActions; a++) {
//			double [][] transposedTMatrix = MatrixUtility.transpose(transitionMatrix.get(a));
//			discountedTMatrix.put(a, MatrixUtility.scalarMultiplication(transposedTMatrix, _discount));
//		}
		discountedTMatrix = MatrixUtilityJBLAS.convertHashMapToRealMatrixCols(transitionMatrix);

		DoubleMatrix dMatrix = discountedTMatrix.transpose().mul( _discount);

//
//		create state x feature matrix: specifying which one of the 16 feature-indices is associated with each of the 64 possible states
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
//		initial state distribution

		double [] initialDistribution = new double [_numberStates];
		for (int r =0; r< initialDistribution.length; r++) {
			initialDistribution[r]= (double)1/_numberStates; // set the intial probability of each state = 1/64 =b_0(s_i)
		}
		
//		weight vector  (sum of weights can be >1?)jk
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
		_discountedTransitionMatrix = dMatrix;

		
		// essentially using duplicate of same state-feature matrix for each action (since in mazeworld, 
		// reward features are state-based and NOT action based, as a result, 
		// the reward features pertinent for a given action will be the same for all actions)
		for (int a=0; a<_numberActions; a++) {
			safMatrixMAP.put (a, MatrixUtilityJBLAS.deepCopy(stateFeatureMatrix));
		}
		

		_stateFeatureMatrixMAP = safMatrixMAP;
		
		_weightMatrix = weightMatr;
		
		// Reshape the weighted state-feature matrix into a 64x4 matrix
		RealMatrix rewardMatrix = MatrixUtils.createRealMatrix(_numberStates, _numberActions);
		for (int a =0; a< _numberActions; a++) {
			rewardMatrix.setColumnMatrix(a, MatrixUtils.createRealMatrix(stateFeatureMatrix).multiply(MatrixUtils.createRealMatrix(weightMatr)));
		}
		
		_rewardFunction = rewardMatrix.getData();
	
	}//end initializeTOYMDP()


	
	
	
	
	/**
	 * Initialize MDP using state-transition info from Cassandra table
	 * 
	 * @throws IOException
	 */
	private void initializeMDP () throws IOException
	{
		Map<Integer, Map<Integer, Integer>> mapOfPossibleNextStates = new HashMap<> ();
		
		String cqlQueryStateKeys = "SELECT * FROM nextpossiblestatesmapping_table";
		for (Row a_row : _cassSession.execute (cqlQueryStateKeys))
		{
			Integer currentStateINTj = a_row.getInt ("currentstateint");
			Map<Integer, Integer> nextStateTransitionsForCurrStatej = a_row.getMap ("nextTransitionsMAP", Integer.class, Integer.class);
			
			mapOfPossibleNextStates.put (currentStateINTj, nextStateTransitionsForCurrStatej);
		}
		
		
		Map<Integer, double[][]> safMatrixMAP = new HashMap<Integer, double[][]> ();
		
		Map<UUID, Integer> basisVectorToIntMAP = new HashMap<UUID, Integer> ();
		
		String cqlGetBasisVectorsStatement = "select * FROM statebasisvectors_table";
		for (Row b_row : _cassSession.execute (cqlGetBasisVectorsStatement))
		{
			UUID bvectoruuid = b_row.getUUID ("basisvectoruuid");
			Long bvector_longval = b_row.getLong ("basisvectorint");
			
			Integer bvectorINTEGER = bvector_longval.intValue ();
			basisVectorToIntMAP.put (bvectoruuid, bvectorINTEGER);
		}
		Set<UUID> basisVectorMapKeys = basisVectorToIntMAP.keySet ();
		
		String cqlStatementSelectActionINTS = "select actionint from actionspace_table ALLOW FILTERING";
		for (Row a_row : _cassSession.execute (cqlStatementSelectActionINTS))
		{
			Long action_ai_longval = a_row.getLong ("actionint");
			int action_ai_int = action_ai_longval.intValue ();
			double[][] action_ai_featureMatrix = new double[_numberStates][_numberRewardFeatures];
			if (action_ai_int < _numberRewardFeatures)
			{
				// recall that first K columns of reward feature matrix corresponding to the state-basis-vector-based reward features, 
				// the remaining correspond to driver-gene-name-action-based reward features
				int actionRewardFeatureint = action_ai_int + _numberStateBasisVectors;
				for (int s = 0; s < _numberStates; s++)
				{
					action_ai_featureMatrix[s][actionRewardFeatureint] = 1;
				}
			}
			// TODO: JK 3.6.2019: Do we need to account for action0 which is NOT created in the
			// actionSpace because it is a default padding value
			
			String cqlStatementSelectStates = "select * FROM statespace_table";
			for (Row s_row : _cassSession.execute (cqlStatementSelectStates))
			{
				
				Long state_sj_longval = s_row.getLong ("stateint");
				int state_sj_int = state_sj_longval.intValue ();
				for (UUID basisVectoruuid_b : basisVectorMapKeys)
				{
					String bVectoruuid_bAsString = basisVectoruuid_b.toString ();
					boolean isbasisVectorbPertinent = s_row.getBool (bVectoruuid_bAsString);
					if (isbasisVectorbPertinent == true)
					{
						int basisVectorint_b = basisVectorToIntMAP.get (basisVectoruuid_b).intValue ();
						action_ai_featureMatrix[state_sj_int][basisVectorint_b] = 1;
					}
				}
			}
			
			// add the sfmatrix associated with this action into safMatrixMAP
			safMatrixMAP.put (action_ai_int, action_ai_featureMatrix);
			
		}
		_stateFeatureMatrixMAP = safMatrixMAP;
		
		
		// Initialize the 2 transition matrices: state transition: T(s',s,a) = P(s'|s,a)
		// _transitionmatrix models the probability of the next state (row), given current state
		// (col) for action a_i (Map Integer index)
		// _discountedTransitonMatrix models the probability of the next state (col), given the
		// current state(row) for action a_I (Map Integer index)
		Map<Integer, double[][]> transitionMatrix = new HashMap<Integer, double[][]> ();
		// Map<Integer, double[][]> discountedTMatrix = new HashMap<Integer, double[][]>(); // we never ran this in toy-model testing
		DoubleMatrix discountedTMatrix;
		
		for (int a = 0; a < _numberActions; a++)
		{
			double[][] tMatrixForAction_a = new double[_numberStates][_numberStates];
			
			transitionMatrix.put (a, tMatrixForAction_a);
		}
		
		Map<Integer, Integer> possibleNextStatesForCurrState_s;
		for (int s = 0; s < _numberStates; s++)
		{
			possibleNextStatesForCurrState_s = mapOfPossibleNextStates.get (s);
			
			for (int a_i = 0; a_i < _numberActions; a_i++)
			{ // use knowledge about the neighbors of state 's' in order to set ALL the transition
				// probabilities from it;
				double[][] tMatrixForAction_ai = transitionMatrix.get (a_i);
				// System.out.println("actual action: "+a_i);
				for (int a_j = 0; a_j < _numberActions; a_j++)
				{
					// System.out.println("possibleNextStates[a_j]
					// ="+possibleNextStatesForCurrState_s.get(a_j));
					
					// System.out.println("currentstateINT ="+s);
					int	state = possibleNextStatesForCurrState_s.get (a_j);
					// account for stochastic transition probability of moving to any neighbor state by random chance
					tMatrixForAction_ai[state][s] = tMatrixForAction_ai[state][s] + (_noise / (double) _numberActions); 
					// System.out.println(" transition prob of stochastic move "+a_j+" :"
					// +tMatrixForAction_ai[possibleNextStates[a_j]][stateIndex]);
				}
				// 'normalize?' the transition probability of moving to the next state that is
				// actually associated with the given action
				if (s == 0)
				{
					// you get a precision error for state0 which persists into state0 regardless of the action, 
					// resulting in the sum of (.3/2167) * 2167 = 1.000000000000004 due to floating point error, instead of 1.0.
					tMatrixForAction_ai[possibleNextStatesForCurrState_s.get (a_i)][s] = 1;
				}
				else
				{
					int	state = possibleNextStatesForCurrState_s.get (a_i);
					// % sum up the total prob = 1, by setting the transition probability of the state corresponding to action 
					// 'a' = 0.3/4 +.7 (i.e. inc. the probability of the state associated with the given action)
					tMatrixForAction_ai[state][s] = tMatrixForAction_ai[state][s] + 1 - _noise;
				}
			}
			
		}
		
		// check that total transition probability..is within bounds (is not negative, and if
		// greater than 1 than at least less than maximum error amount of 1e-6)
		for (int s = 0; s < _numberStates; s++)
		{
			for (int a = 0; a < _numberActions; a++)
			{
// 				System.out.println ("Checking for error... (state,action)=(" + s + "," + a + ")");
				double[][] transitionMatrForAction_a = transitionMatrix.get (a);
				double error = Math
						.abs (MatrixUtilityJBLAS.sumPerColumn (transitionMatrForAction_a)[s] - 1);
				// determine number of non zero matrix elements and if total probability of
				// transitions for s,a pair is >1 (more than > 1e-6) OR if the probability of any of
				// the possible NEXT states is NEGATIVE or GREATER THAN 1
				if ((error > 1e-6) || 
					(MatrixUtilityJBLAS.sumPerColumn (MatrixUtilityJBLAS.lessEqualityComparison (transitionMatrForAction_a, 0.0))[s] > 0) || 
					(MatrixUtilityJBLAS.sumPerColumn (MatrixUtilityJBLAS.greaterEqualityComparison (transitionMatrForAction_a, 1.0))[s] > 0))
				{
					System.out.println ("sum of column " + s + " = "+ MatrixUtilityJBLAS.sumPerColumn (transitionMatrForAction_a)[s]);
					System.out.println ("Error =" + error);
					System.out.println ("# of NEXT STATES with probabilities being negative = " + 
										MatrixUtilityJBLAS.sumPerColumn (MatrixUtilityJBLAS .lessEqualityComparison (transitionMatrForAction_a, 0.0))[s]);
					System.out.println ("# of NEXT states with probabilities >1 = " + 
										MatrixUtilityJBLAS.sumPerColumn (MatrixUtilityJBLAS.greaterEqualityComparison (transitionMatrForAction_a, 1.0))[s]);
					throw new java.lang.RuntimeException ("bad transition probability found!!!");
				}
				
			}
		}
		
		// Create DISCOUNTED TRANSITION MATRIX by transposing _transitionMatrix and multiplying by
		// discount factor
		// for (int a=0; a< _numberActions; a++) {
		// double [][] transposedTMatrix = MatrixUtility.transpose(transitionMatrix.get(a));
		// discountedTMatrix.put(a, MatrixUtility.scalarMultiplication(transposedTMatrix,
		// _discount));
		// }
		discountedTMatrix = MatrixUtilityJBLAS.convertHashMapToRealMatrixCols (transitionMatrix);
		
		DoubleMatrix dMatrix = discountedTMatrix.transpose ().mul (_discount);
		
		//
		// initial state distribution
		
		double[] initialDistribution = new double[_numberStates];
		for (int r = 0; r < initialDistribution.length; r++)
		{
			initialDistribution[r] = (double) 1 / _numberStates; // set the initial probability of each state = 1/64 =b_0(s_i)
		}
		
		// weight vector (sum of weights can be >1?)jk
		double[][]	weightMatr = new double[_numberRewardFeatures][1]; // creates a 16x1 array of zeros
		// returns a row vector containing a random permutation of the integers from 1 to nF-1=15 inclusive
		int[]		randomPermutation1 = VectorUtility.createPermutatedVector (_numberRewardFeatures, 0);
		Double		k = Math.ceil (0.3 * _numberRewardFeatures); // 0.3 *16 = 4.8 = 5 (when rounded up)
		// get the sub-vector of 0:4 integer values from row vector randomPermutation1
		int[]		subPermutation1 = VectorUtility.rangeOfVector (randomPermutation1, 0, k.intValue () - 1, 1);
		double[][]	rdmWeights = MersenneTwisterFastIRL.RandomUniformMatrix (_numberRewardFeatures, 1, 1);
		
		for (int r : subPermutation1)
		{
			weightMatr[r][0] = rdmWeights[r][0] - 1; // set the weight vector values for a subset of the elements in the 16x1 weight vector to
														// a k x 1 array of (uniformly distributed)random values minus 1; minus
														// so that they are all negative?!
		}
		weightMatr[weightMatr.length - 1][0] = 1; // set last index of array = 1
		
		// set MDP class variables
		_startDistribution = initialDistribution;
		_transitionMatrix = transitionMatrix;
		// _discountedTransitionMatrix = discountedTMatrix;
		_discountedTransitionMatrix = dMatrix;
		
		_weightMatrix = weightMatr;
		
		// Reshape the weighted state-feature matrix into a 64x4 matrix
		RealMatrix rewardMatrix = MatrixUtils.createRealMatrix (_numberStates, _numberActions);
		double[][] sfmatrixForaction_a;
		for (int a = 0; a < _numberActions; a++)
		{
			sfmatrixForaction_a = _stateFeatureMatrixMAP.get (a);
			rewardMatrix.setColumnMatrix (a, MatrixUtils.createRealMatrix (sfmatrixForaction_a).multiply (MatrixUtils.createRealMatrix (weightMatr)));
		}
		
		_rewardFunction = rewardMatrix.getData ();
		
	}// end initializeMDP()
	
	
	/**
	 * Initialize MDP using TreeTraversal class object
	 * 
	 * @param traversedTreeInst
	 * @throws IOException
	 */
	private void initializeMDP (TreeTraversal traversedTreeInst) throws IOException
	{
		Map<Integer, Map<Integer, Integer>> mapOfPossibleNextStates = traversedTreeInst.getNextPossibleStatesMAPForAllStates ();
		Map<Integer, double[][]> safMatrixMAP = new HashMap<Integer, double[][]> ();
		Map<UUID, Integer> basisVectorToIntMAP = new HashMap<UUID, Integer> ();
		
		String cqlGetBasisVectorsStatement = "select * FROM statebasisvectors_table";
		for (Row b_row : _cassSession.execute (cqlGetBasisVectorsStatement))
		{
			UUID bvectoruuid = b_row.getUUID ("basisvectoruuid");
			Long bvector_longval = b_row.getLong ("basisvectorint");
			
			Integer bvectorINTEGER = bvector_longval.intValue ();
			basisVectorToIntMAP.put (bvectoruuid, bvectorINTEGER);
		}
		Set<UUID> basisVectorMapKeys = basisVectorToIntMAP.keySet ();
		
		String cqlStatementSelectActionINTS = "select actionint from actionspace_table ALLOW FILTERING";
		for (Row a_row : _cassSession.execute (cqlStatementSelectActionINTS))
		{
			Long action_ai_longval = a_row.getLong ("actionint");
			int action_ai_int = action_ai_longval.intValue ();
			double[][] action_ai_featureMatrix = new double[_numberStates][_numberRewardFeatures];
			if (action_ai_int < _numberRewardFeatures)
			{
				// recall that first K columns of reward feature matrix corresponding to the state-basis-vector-based reward features, 
				// the remaining correspond to driver-gene-name-action-based reward features
				int actionRewardFeatureint = action_ai_int + _numberStateBasisVectors;
				for (int s = 0; s < _numberStates; s++)
				{
					action_ai_featureMatrix[s][actionRewardFeatureint] = 1;
				}
			}
			// TODO: JK 3.6.2019: Do we need to account for action0 which is NOT created in the
			// actionSpace because it is a default padding value
			
			String cqlStatementSelectStates = "select * FROM statespace_table";
			for (Row s_row : _cassSession.execute (cqlStatementSelectStates))
			{
				Long	state_sj_longval = s_row.getLong ("stateint");
				int		state_sj_int = state_sj_longval.intValue ();
				for (UUID basisVectoruuid_b : basisVectorMapKeys)
				{
					String bVectoruuid_bAsString = basisVectoruuid_b.toString ();
					boolean isbasisVectorbPertinent = s_row.getBool (bVectoruuid_bAsString);
					if (isbasisVectorbPertinent == true)
					{
						int basisVectorint_b = basisVectorToIntMAP.get (basisVectoruuid_b).intValue ();
						action_ai_featureMatrix[state_sj_int][basisVectorint_b] = 1;
					}
				}
			}
			
			// add the sfmatrix associated with this action into safMatrixMAP
			safMatrixMAP.put (action_ai_int, action_ai_featureMatrix);
			
		}
		_stateFeatureMatrixMAP = safMatrixMAP;
		
		
		// Initialize the 2 transition matrices: state transition: T(s',s,a) = P(s'|s,a)
		// _transitionmatrix models the probability of the next state (row), given current state
		// (col) for action a_i (Map Integer index)
		// _discountedTransitonMatrix models the probability of the next state (col), given the
		// current state(row) for action a_I (Map Integer index)
		Map<Integer, double[][]> transitionMatrix = new HashMap<Integer, double[][]> ();
		// Map<Integer, double[][]> discountedTMatrix = new HashMap<Integer, double[][]>(); //we
		// never ran this in toy-model testing
		DoubleMatrix discountedTMatrix;
		
		for (int a = 0; a < _numberActions; a++)
		{
			double[][] tMatrixForAction_a = new double[_numberStates][_numberStates];
			
			transitionMatrix.put (a, tMatrixForAction_a);
		}
		
		Map<Integer, Integer> possibleNextStatesForCurrState_s;
		for (int s = 0; s < _numberStates; s++)
		{
			possibleNextStatesForCurrState_s = mapOfPossibleNextStates.get (s);
			
			for (int a_i = 0; a_i < _numberActions; a_i++)
			{ // use knowledge about the neighbors of state 's' in order to set ALL the transition
				// probabilities from it;
				double[][] tMatrixForAction_ai = transitionMatrix.get (a_i);
				// System.out.println("actual action: "+a_i);
				for (int a_j = 0; a_j < _numberActions; a_j++)
				{
					// System.out.println("possibleNextStates[a_j]
					// ="+possibleNextStatesForCurrState_s.get(a_j));
					
					// System.out.println("currentstateINT ="+s);
					int	state = possibleNextStatesForCurrState_s.get (a_j);
					// account for stochastic transition probability of moving to any neighbor state by random chance
					tMatrixForAction_ai[state][s] = tMatrixForAction_ai[state][s] + (_noise / (double) _numberActions);
					// System.out.println(" transition prob of stochastic move "+a_j+" :"
					// +tMatrixForAction_ai[possibleNextStates[a_j]][stateIndex]);
				}
				// 'normalize?' the transition probability of moving to the next state that is
				// actually associated with the given action
				if (s == 0)
				{
					// you get a precision error for state0 which persists into state0 regardless of the action, 
					// resulting in the sum of (.3 / 2167) * 2167 = 1.000000000000004 due to floating point error, instead of 1.0.
					tMatrixForAction_ai[possibleNextStatesForCurrState_s.get (a_i)][s] = 1.0;
				}
				else
				{
					int	state = possibleNextStatesForCurrState_s.get (a_i);
					// % sum up the total prob = 1, by setting the transition probability of the state corresponding to 
					// action 'a' = 0.3/4 +.7 (i.e. inc. the probability of the state associated with the given action)
					tMatrixForAction_ai[state][s] = tMatrixForAction_ai[state][s] + 1 - _noise;
				}
			}
			
		}
		
		// check that total transition probability..is within bounds (is not negative, and if
		// greater than 1 than at least less than maximum error amount of 1e-6)
		for (int s = 0; s < _numberStates; s++)
		{
			for (int a = 0; a < _numberActions; a++)
			{
// 				System.out.println ("Checking for error... (state,action)=(" + s + "," + a + ")");
				double[][] transitionMatrForAction_a = transitionMatrix.get (a);
				double error = Math.abs (MatrixUtilityJBLAS.sumPerColumn (transitionMatrForAction_a)[s] - 1);
				// determine number of non zero matrix elements and if total probability of
				// transitions for s,a pair is >1 (more than > 1e-6) OR if the probability of any of
				// the possible NEXT states is NEGATIVE or GREATER THAN 1
				if ((error > 1e-6) || 
					(MatrixUtilityJBLAS.sumPerColumn (MatrixUtilityJBLAS.lessEqualityComparison (transitionMatrForAction_a, 0.0))[s] > 0) || 
					(MatrixUtilityJBLAS.sumPerColumn (MatrixUtilityJBLAS.greaterEqualityComparison (transitionMatrForAction_a, 1.0))[s] > 0))
				{
					System.out.println ("sum of column " + s + " = " + MatrixUtilityJBLAS.sumPerColumn (transitionMatrForAction_a)[s]);
					System.out.println ("Error =" + error);
					System.out.println ("# of NEXT STATES with probabilities being negative = " + 
										MatrixUtilityJBLAS.sumPerColumn (MatrixUtilityJBLAS.lessEqualityComparison (transitionMatrForAction_a, 0.0))[s]);
					System.out.println ("# of NEXT states with probabilities >1 = " + 
										MatrixUtilityJBLAS.sumPerColumn (MatrixUtilityJBLAS.greaterEqualityComparison (transitionMatrForAction_a, 1.0))[s]);
					throw new java.lang.RuntimeException ("bad transition probability found!!!");
				}
				
			}
		}
		
		// Create DISCOUNTED TRANSITION MATRIX by transposing _transitionMatrix and multiplying by
		// discount factor
		// for (int a=0; a< _numberActions; a++) {
		// double [][] transposedTMatrix = MatrixUtility.transpose(transitionMatrix.get(a));
		// discountedTMatrix.put(a, MatrixUtility.scalarMultiplication(transposedTMatrix,
		// _discount));
		// }
		discountedTMatrix = MatrixUtilityJBLAS.convertHashMapToRealMatrixCols (transitionMatrix);
		
		DoubleMatrix dMatrix = discountedTMatrix.transpose ().mul (_discount);
		
		//
		
		// initial state distribution
		
		double[] initialDistribution = new double[_numberStates];
		for (int r = 0; r < initialDistribution.length; r++)
		{
			initialDistribution[r] = (double) 1 / _numberStates; // set the initial probability of each state = 1/64 =b_0(s_i)
		}
		
		// weight vector (sum of weights can be >1?)jk
		double[][]	weightMatr = new double[_numberRewardFeatures][1]; // creates a 16x1 array of zeros
		// returns a row vector containing a random permutation of the integers from 1 to nF-1=15 inclusive
		int[]		randomPermutation1 = VectorUtility.createPermutatedVector (_numberRewardFeatures, 0);
		Double		k = Math.ceil (0.3 * _numberRewardFeatures); // 0.3 *16 = 4.8 = 5 (when rounded up)
		// get the sub-vector of 0:4 integer values from row vector randomPermutation1
		int[]		subPermutation1 = VectorUtility.rangeOfVector (randomPermutation1, 0, k.intValue () - 1, 1);
		double[][]	rdmWeights = MersenneTwisterFastIRL.RandomUniformMatrix (_numberRewardFeatures, 1, 1);
		
		for (int r : subPermutation1)
		{
			weightMatr[r][0] = rdmWeights[r][0] - 1; // set the weight vector values for a subset of the elements in the 16x1 weight vector to
														// a k x 1 array of (uniformly distributed)random values minus 1; minus
														// so that they are all negative?!
		}
		weightMatr[weightMatr.length - 1][0] = 1; // set last index of array =1
		
		// set MDP class variables
		_startDistribution = initialDistribution;
		_transitionMatrix = transitionMatrix;
		// _discountedTransitionMatrix = discountedTMatrix;
		_discountedTransitionMatrix = dMatrix;
		
		_weightMatrix = weightMatr;
		
		// Reshape the weighted state-feature matrix into a 64x4 matrix
		RealMatrix rewardMatrix = MatrixUtils.createRealMatrix (_numberStates, _numberActions);
		double[][] sfmatrixForaction_a;
		for (int a = 0; a < _numberActions; a++)
		{
			sfmatrixForaction_a = _stateFeatureMatrixMAP.get (a);
			rewardMatrix.setColumnMatrix (a, MatrixUtils.createRealMatrix (sfmatrixForaction_a).multiply (MatrixUtils.createRealMatrix (weightMatr)));
		}
		
		_rewardFunction = rewardMatrix.getData ();
		
	}// end initializeMDP()
	
	
	// set methods
	public void setMDPName (String envName)
	{
		_name = envName;
	}
	
	
	public void setInitialSeed (int seedVal)
	{
		_initialRandomSeed = seedVal;
	}
	
	
	public void setDiscount (double d)
	{
		_discount = d;
	}
	
	
	public void setNumStates (int nStates)
	{
		_numberStates = nStates;
	}
	
	
	public void setIfSparse (boolean isSparse)
	{
		_useSparseMatrix = isSparse;
	}
	
	public void setIfGridWorld(boolean isGridWorldEnv) {
		_isGridWorldEnv = isGridWorldEnv;
	}
	
	
	public void setNumActions (int nActions)
	{
		_numberActions = nActions;
	}
	
	
	public void setNumRewardFeatures (int numRFeatures)
	{
		_numberRewardFeatures = numRFeatures;
	}
	
	
	public void setNumActionBasisVectors (int nActionBVectors)
	{
		_numberActionBasisVectors = nActionBVectors;
	}
	
	
	public void setNumStateBasisVectors (int nStateBVectors)
	{
		_numberStateBasisVectors = nStateBVectors;
	}
	
	
	public void setTransitionMatrix (Map<Integer, double[][]> tMatrix)
	{
		_transitionMatrix = tMatrix;
	}
	
	
	// public void setDiscountedTransitionMatrix(Map<Integer, double[][]> dtMatrix) {
	// _discountedTransitionMatrix = dtMatrix;
	// }
	public void setDiscountedTransitionMatrix (DoubleMatrix dtMatrix)
	{
		_discountedTransitionMatrix = dtMatrix;
	}
	
	

	public void setStateFeatureMatrixMAP (Map<Integer, double[][]> stateFeatureMatrix_map)
	{
		_stateFeatureMatrixMAP = stateFeatureMatrix_map;
	}
	
	
	public void setStartDistribution (double[] startDistribution)
	{
		_startDistribution = startDistribution;
	}
	
	
	public void setWeight (double[][] wt)
	{
		_weightMatrix = wt;
	}
	
	
	public void setRewardFunction (double[][] rwrdFunction)
	{
		_rewardFunction = rwrdFunction;
	}
	
	//JK added 9.1.2019
	public void setNumTrueTables(int numSimulExperts) {
		_numberTrueTables = numSimulExperts;
	}
	
	public void setNumDemoTrajectoriesPerTable (int numDemoTrajsPerExpert)
	{
		_numberDemoTrajectoriesPerTable = numDemoTrajsPerExpert ;
	}
	
	
	// get methods
	public String getMDPName ()
	{
		return _name;
	}
	
	
	public int getInitialSeed ()
	{
		return _initialRandomSeed;
	}
	
	
	public double getDiscount ()
	{
		return _discount;
	}
	
	
	public int getNumStates ()
	{
		return _numberStates;
	}
	
	
	public int getNumActions ()
	{
		return _numberActions;
	}
	
	
	public int getNumActionBVectors ()
	{
		return _numberActionBasisVectors;
	}
	
	
	public int getNumStateBVectors ()
	{
		return _numberStateBasisVectors;
	}
	
	
	public boolean getIfSparse ()
	{
		return _useSparseMatrix;
	}
	
	
	public int getNumRewardFeatures ()
	{
		return _numberRewardFeatures;
	}
	
	
	public Map<Integer, double[][]> getTransitionMatrix ()
	{
		return _transitionMatrix;
	}
	
	
//	public Map<Integer, double[][]> getDiscountedTransitionMatrix ()
//	{
//		return _discountedTransitionMatrix;
//	}
	
	
	public DoubleMatrix getDiscountedTransitionMatrix ()
	{
		return _discountedTransitionMatrix;
	}
	

	
	
	public Map<Integer, double[][]> getStateFeatureMatrixMAP ()
	{
		return _stateFeatureMatrixMAP;
	}
	
	
	public double[] getStartDistribution ()
	{
		return _startDistribution;
	}
	
	
	public double[][] getWeight ()
	{
		return _weightMatrix;
	}
	
	
	public double[][] getRewardFunction ()
	{
		return _rewardFunction;
	}
	
	
	public int getNumTrueTables ()
	{
		return _numberTrueTables;
	}
	
	
	public int getNumDemoTrajectoriesPerTable ()
	{
		return _numberDemoTrajectoriesPerTable;
	}
	
	
	public int getNumStepsPerTrajectory ()
	{
		return _numberStepsPerTrajectory;
	}
	
	
	public int generateIndexFromCoordinates (int xCoord, int yCoord, int dimension1)
	{
		// computes the numerical index of a 'state' based on its x,y coordinates
		int x = Math.max (1, Math.min (dimension1, xCoord));
		int y = Math.max (1, Math.min (dimension1, yCoord));
		int index = (y - 1) * dimension1 + x - 1;
		
		return index;
	}
	
	
	/**
	 * Class method for DEserializing an MDPCancer object from an OBJECT InputStream to generate a
	 * fully instantiated MDPCancer model
	 * 
	 * @param in ObjectInputStream to read from
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	@SuppressWarnings ("unchecked")
	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException
	{
		_name = (String) in.readObject ();
		_discount = in.readDouble ();
		_numberExperiments = in.readInt ();
		_numberTrueTables = in.readInt ();
		_numberDemoTrajectoriesPerTable = in.readInt ();
		_numberStepsPerTrajectory = in.readInt ();
		_initialRandomSeed = in.readInt ();
		_noise = in.readDouble ();
		_newExpertProbability = in.readDouble ();
		_newExperiments = in.readInt ();
		_newTrajectorySteps = in.readInt ();
		_numberStates = in.readInt ();
		_numberRewardFeatures = in.readInt ();
		_numberActions = in.readInt ();
		_numberStateBasisVectors = in.readInt ();
		_numberActionBasisVectors = in.readInt ();
		_useSparseMatrix = in.readBoolean ();
		_transitionMatrix = (Map<Integer, double[][]>) in.readObject ();
		_discountedTransitionMatrix = (DoubleMatrix) in.readObject ();
		_stateFeatureMatrixMAP = (Map<Integer, double[][]>) in.readObject ();
		_startDistribution = (double[]) in.readObject ();
		_weightMatrix = (double[][]) in.readObject ();
		_rewardFunction = (double[][]) in.readObject ();
	}
	
	
	/**
	 * Class method for SERIALIZING the current MDPCancer class object, (and its fully instantiated model), 
	 * into an input-specified OBJECT OutputStream
	 * 
	 * @param out ObjectOutputStream to write into
	 * @throws IOException
	 */
	private void writeObject (ObjectOutputStream out) throws IOException
	{
		out.writeObject (_name);
		out.writeDouble (_discount);
		out.writeInt (_numberExperiments);
		out.writeInt (_numberTrueTables);
		out.writeInt (_numberDemoTrajectoriesPerTable);
		out.writeInt (_numberStepsPerTrajectory);
		out.writeInt (_initialRandomSeed);
		out.writeDouble (_noise);
		out.writeDouble (_newExpertProbability);
		out.writeInt (_newExperiments);
		out.writeInt (_newTrajectorySteps);
		out.writeInt (_numberStates);
		out.writeInt (_numberRewardFeatures);
		out.writeInt (_numberActions);
		out.writeInt (_numberStateBasisVectors);
		out.writeInt (_numberActionBasisVectors);
		out.writeBoolean (_useSparseMatrix);
		out.writeObject (_transitionMatrix);
		out.writeObject (_discountedTransitionMatrix);
		out.writeObject (_stateFeatureMatrixMAP);
		out.writeObject (_startDistribution);
		out.writeObject (_weightMatrix);
		out.writeObject (_rewardFunction);
	}
	
	
	/**
	 * Make a copy of a double[][]
	 * 
	 * @param theArray	double[][] to copy, must be valid
	 * @return	new double[][], every row a new double[]
	 * @throws NullPointerException	If {@code theArray} is null, or any row is null
	 */
	public static final double[][] copy (double[][] theArray) throws NullPointerException
	{
		int			size = theArray.length;
		double[][]	results = new double[size][];
		
		for (int i = 0; i < size; ++i)
		{
			double[]	row = theArray[i];
			int			len = row.length;
			double[]	newRow = results[i] = new double[len];
			
			System.arraycopy (row, 0, newRow, 0, len);
		}
		
		return results;
	}
	
	
	/**
	 * Make a copy of a double[]
	 * 
	 * @param theArray	double[] to copy, must be valid
	 * @return	double[]
	 * @throws NullPointerException	If {@code theArray} is null
	 */
	public static final double[] copy (double[] theArray) throws NullPointerException
	{
		int			size = theArray.length;
		double[]	results = new double[size];
		
		System.arraycopy (theArray, 0, results, 0, size);
		
		return results;
	}
	
}
