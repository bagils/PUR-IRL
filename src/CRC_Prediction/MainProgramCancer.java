
package CRC_Prediction;


import java.io.*;
import java.nio.file.*;
import java.util.*;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.jblas.DoubleMatrix;

import com.datastax.driver.core.BoundStatement;
import com.datastax.driver.core.Cluster;
import com.datastax.driver.core.PreparedStatement;
import com.datastax.driver.core.ResultSet;
import com.datastax.driver.core.Row;
import com.datastax.driver.core.Session;
import com.jprofiler.api.controller.Controller;
import CRC_Prediction.Utils.TreeTraversal;
import CRC_Prediction.Utils.VectorUtility;
import gnu.trove.set.hash.THashSet;


/// To run from command line using fat jar file of snapshot 0.0.2 found in ~/sandboxForJava
/// directory
// java -jar IRLJK_MAVEN-0.0.2-SNAPSHOT-jar-with-dependencies.jar -ancestryRelationsListFile
/// /Users/m186806/tIDFilteredDriverOnlyVCFs/allTIDFilteredSUBFILES/listOfAncestryFiles.txt
/// -driversSubcloneVCFsDirectoriesListFile
/// /Users/m186806/tIDFilteredDriverOnlyVCFs/allTIDFilteredSUBFILES/listOfSampleSpecificDRIVERSSubcloneVCFsDirectories.txt
/**
 * 
 * @author John Kalantari
 * Copyright 2019, Mayo Foundation for Medical Education and Research
 *
 */
public class MainProgramCancer
{
	
	// necessary to access Cassandra dB tables
	// private static String serverIP = "127.0.0.1";
	// private static String keyspacejk = "crckeyspace";
	// private static Cluster _cluster = Cluster.builder().addContactPoint(serverIP).build();
	// private static Session _session = _cluster.connect(keyspacejk);
	// private static String _serverIP = null; // GTD Not used
	// private static String _keyspacejk = null; // GTD Not used
	private static Cluster			_cluster	= null;
	private static Session			_session	= null;
	
	protected IRLAlgorithm			_irlAlgo;
	protected MDPCancer				_mdp;
	protected IRLRestaurant			_bestRestaurant;
	protected List<IRLRestaurant>	_restaurantIterations;
	
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
		
//		System.loadLibrary ("libjniconverge");
    }

	/**
	 * Test that matrix multiplication and addition work, then exit the program
	 */
	protected static void testMatrixJNI ()
	{
//		InferenceAlgoCancer.matrixTest (7032, 299, 1e-12);
		
		double			epsilon = 5.0;
		int				maximumIterations = 1;
		double[][]		fxnData = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
		DoubleMatrix	fxnMatrix = new DoubleMatrix (fxnData);
		double[][]		mulData = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
		DoubleMatrix	mulMatrix = new DoubleMatrix (mulData);
		double[][]		addData = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
		DoubleMatrix	addMatrix = new DoubleMatrix (addData);
		
		double[][]	results = InferenceAlgoCancer.convergeMatrix (fxnMatrix, mulMatrix, addMatrix, maximumIterations, epsilon);
		
		printMatrix ("mulMatrix", mulMatrix.toArray2 ());
		printMatrix ("fxnMatrix", fxnMatrix.toArray2 ());
		printMatrix ("addMatrix", addMatrix.toArray2 ());
		printMatrix ("results", results);
		
		System.exit (0);
	}
	
	
	/**
	 * Print out the contents of a 1d matrix
	 * 
	 * @param title	Title line to print out, if not null and not empty
	 * @param row	The double[] to print out
	 */
	protected static final void printMatrix (String title, double[] row)
	{
		if ((title != null) && !title.isEmpty ())
			System.out.println (title);
		
		int	numCols = row.length;
		
		for (int j = 0; j < numCols; ++j)
		{
			if (j != 0)
				System.out.print (", ");
			System.out.print (row[j]);
		}
		System.out.println ();
	}
	
	
	/**
	 * 
	 * 
	 * @param title	Title line to print out, if not null and not empty
	 * @param theMatrix	2d matrix to print out, one row per line
	 */
	private static final void printMatrix (String title, double[][] theMatrix)
	{
		if ((title != null) && !title.isEmpty ())
			System.out.println (title);
		
		int			numRows = theMatrix.length;
		int			numCols = theMatrix[0].length;
		
		for (int i = 0; i < numRows; ++i)
		{
			double[]	row = theMatrix[i];
			
			for (int j = 0; j < numCols; ++j)
			{
				if (j != 0)
					System.out.print (", ");
				System.out.print (row[j]);
			}
			System.out.println ();
		}
	}
	
	
	/**
	 * This function extracts paths from subclonal phylogenetic trees by running new TreeTraversal
	 * instance
	 * 
	 * @param env
	 * @param irlAlgo
	 * @return
	 */
	public static List<double[][]> getCancerPathsDataFromTreeTraversal (MDPCancer env, IRLAlgorithmCancer irlAlgo, TreeTraversal traversedtreeobj)
	{
		List<double[][]> dataTrajectories = new ArrayList<double[][]> ();
		Map<String, List<double[][]>> 
			doublesStateAndActiontrajsForAllSamplesMap = traversedtreeobj.getDoublesStateAndActiontrajectoriesForAllSamplesMap ();
		for (List<double[][]> trajList : doublesStateAndActiontrajsForAllSamplesMap.values ())
		{
			dataTrajectories.addAll (trajList);
		}
		System.out.println (dataTrajectories.size () + " trajectories generated");
		
		return dataTrajectories;
	}
	
	
	/**
	 * This function extracts paths from subclonal phylogenetic trees as stored in Cassandra db
	 * 
	 * @param env
	 * @param irlAlgo
	 * @return
	 */
	public static List<double[][]> getCancerPathsDataFromDatabase (MDPCancer env, IRLAlgorithmCancer irlAlgo, int maxTrajsPerSample)
	{
		
		List<double[][]> dataTrajectories = new ArrayList<double[][]> ();
		double[][] trajCustomer = new double[2][5];
//		String samplename = null;	// GTD Not used
//		UUID trajuuid = null;	// GTD Not used
		Long action0 = null;
		Long action1 = null;
		Long action2 = null;
		Long action3 = null;
		Long action4 = null;
		Long state0 = null;
		Long state1 = null;
		Long state2 = null;
		Long state3 = null;
		Long state4 = null;
		
//		String cqlSelectSubsetOfSampleNamesfrom_trajcounttable = "select samplename from sampletrajcounts_table where trajcount <150000 ALLOW FILTERING";
		String cqlSelectSubsetOfSampleNamesfrom_trajcounttable = "select samplename from crckeyspace.sampletrajcounts_table where trajcount <"
				+ maxTrajsPerSample + " ALLOW FILTERING";
		
		for (Row name_row : _session.execute (cqlSelectSubsetOfSampleNamesfrom_trajcounttable))
		{
			String samplenameX = name_row.getString ("samplename");
			
			String cqlSelecStateActionTrajectories_table = "select * from stateactiontrajectories_table where samplename=\'" + samplenameX + "\'";
			
			for (Row row : _session.execute (cqlSelecStateActionTrajectories_table))
			{
				
//				samplename = row.getString ("samplename");	// GTD Not used
//				trajuuid = row.getUUID ("trajuuid");	// GTD Not used
				action0 = row.getLong ("action0");
				action1 = row.getLong ("action1");
				action2 = row.getLong ("action2");
				action3 = row.getLong ("action3");
				action4 = row.getLong ("action4");
				state0 = row.getLong ("state0");
				state1 = row.getLong ("state1");
				state2 = row.getLong ("state2");
				state3 = row.getLong ("state3");
				state4 = row.getLong ("state4");
				
				trajCustomer[0][0] = state0.intValue ();
				trajCustomer[0][1] = state1.intValue ();
				trajCustomer[0][2] = state2.intValue ();
				trajCustomer[0][3] = state3.intValue ();
				trajCustomer[0][4] = state4.intValue ();
				trajCustomer[1][0] = action0.intValue ();
				trajCustomer[1][1] = action1.intValue ();
				trajCustomer[1][2] = action2.intValue ();
				trajCustomer[1][3] = action3.intValue ();
				trajCustomer[1][4] = action4.intValue ();
				dataTrajectories.add (trajCustomer.clone ());
				
			}
			// String cqlSelecStateActionTrajectories_table = "select * from stateactiontrajectories_table";
			// double [][] trajCustomer= new double[2][5];
			//
			// for (Row row: _session.execute(cqlSelecStateActionTrajectories_table)) {
			//
			// String samplename =row.getString("samplename");
			// UUID trajuuid = row.getUUID("trajuuid");
			// Long action0 = row.getLong("action0");
			// Long action1 = row.getLong("action1");
			// Long action2 = row.getLong("action2");
			// Long action3 = row.getLong("action3");
			// Long action4 = row.getLong("action4");
			// Long state0 = row.getLong("state0");
			// Long state1 = row.getLong("state1");
			// Long state2 = row.getLong("state2");
			// Long state3 = row.getLong("state3");
			// Long state4 = row.getLong("state4");
			//
			// trajCustomer[0][0] = state0.intValue();
			// trajCustomer[0][1] = state1.intValue();
			// trajCustomer[0][2] = state2.intValue();
			// trajCustomer[0][3] = state3.intValue();
			// trajCustomer[0][4] = state4.intValue();
			// trajCustomer[1][0] = action0.intValue();
			// trajCustomer[1][1] = action1.intValue();
			// trajCustomer[1][2] = action2.intValue();
			// trajCustomer[1][3] = action3.intValue();
			// trajCustomer[1][4] = action4.intValue();
			// dataTrajectories.add(trajCustomer.clone());
			
		}
		
		System.out.println (dataTrajectories.size () + " trajectories extracted from database");
		
		return dataTrajectories;
	}
	
	
	public static List<double[][]> generateDemoData (MDPCancer env, IRLAlgorithmCancer irlAlgo)
	{
		List<double[][]> dataTrajectories = new ArrayList<double[][]> ();
		MersenneTwisterFastIRL RNG = new MersenneTwisterFastIRL ();
		
		int numDemoTrajPerTable = env.getNumDemoTrajectoriesPerTable ();
		int numFeatures = env.getNumRewardFeatures ();
		int numStepsPerTraj = env.getNumStepsPerTrajectory ();
		double discountFactor = env.getDiscount ();
		
		// XXX:The size of of each of these lists corresponds to the number of ground-truth tables
		// specified by env.getTrueTables in the restaurant. Each element in the list is a column
		// matrix for that table.
//		List<double[][]> tableWeightVectors = new ArrayList<double[][]> (); // stores the	// GTD Not used
																			// weight-vector associated with each table index/value;
//		List<double[][]> tablePolicyVectors = new ArrayList<double[][]> (); // each policy is a	// GTD Not used
																			// column matrix of dimension numStates x 1
//		List<double[][]> tableValueVectors = new ArrayList<double[][]> (); // each value is a	// GTD Not used
																			// column matrix of dimension numStates x 1 (i.e. it is NOT a row vector)
		
		for (int table = 1; table < env.getNumTrueTables () + 1; table++)
		{
			
			// sample a weightMatrix for each unique table/reward-function
			double[][] weightMatr = new double[numFeatures][1];
			int[] randomPermutation1 = VectorUtility.createPermutatedVector (numFeatures, 0);
			Double k = Math.ceil (0.3 * numFeatures);
			int[] subVectorOfRndPermutation = VectorUtility.rangeOfVector (randomPermutation1, 0,
					k.intValue () - 1, 1); // for each ground-truth table/expert randomly select a
											// subset of features that are indicative/relevant for
											// that table
			double[][] randomWeights = MersenneTwisterFastIRL.RandomUniformMatrix (numFeatures, 1, 1);
			
			for (int r : subVectorOfRndPermutation)
			{ // randomly generate weights for each feature of relevance for the given ground-truth table/expert
				weightMatr[r][0] = (randomWeights[r][0] * 2) - 1;
			}
			
			RewardFunctionGenerationCancer.generateWeightedRewardFunction (env, weightMatr);
			
			// begin generating demo trajectories for current ground truth table/expert
			Map<String, double[][]> policy_Value_H_Q_matricesForDemoData = PolicySolverCancer.runPolicyIteration (env, irlAlgo, null);
			double[][] policyUsedtoCreateDemoTrajs = policy_Value_H_Q_matricesForDemoData.get ("P");
			System.out.println ("Ground truth policy of table " + table + ":");
			for (int s = 0; s < policyUsedtoCreateDemoTrajs.length; s++)
			{
				System.out.println ("state " + s + " : action " + policyUsedtoCreateDemoTrajs[s][0]);
			}
			
			// Sample trajectories by executing policy piL
			double[] valueFunctionsArray = new double[numDemoTrajPerTable];
			for (int traj_i = 0; traj_i < numDemoTrajPerTable; traj_i++)
			{ // for each demo trajectory to be created (of a given reward-function/table type)
				// NOTE: each trajectory consists of a 1. state-sequence and 2. action-sequence, each with same number of steps
				double[][] trajectory_i = new double[2][numStepsPerTraj];
				
				int sampleState = SampleMultinomialIRL.sampleSingleStateFromMultinomial (100, env.getStartDistribution (), RNG);
				double value_i = 0.0; // initialize the value function of policy for current trajectory
				for (int step = 0; step < numStepsPerTraj; step++)
				{ // for each step of the trajectory
					// retrieve the action to be executed at the current state according to the policy used to generate the demo trajectories
					Double action = policyUsedtoCreateDemoTrajs[sampleState][0];
					// retrieve the reward value for the state-action pair according to the reward function stored in the MDP env.rewardFunction
					Double rewardValue = env.getRewardFunction ()[sampleState][action.intValue ()];
					// update the value function with the discounted reward; XXX: Does the exponent need to be ^(step#-1)? or just ^step
					value_i = value_i + (rewardValue * Math.pow (discountFactor, (step)));
					// set the state for the current step in this trajectory
					trajectory_i[0][step] = (double) sampleState;
					// set the action for the current step in this trajectory
					trajectory_i[1][step] = action;
					
					RealMatrix transitionMatrixForAction_i = MatrixUtils.createRealMatrix (env.getTransitionMatrix ().get (action.intValue ()));
					// sample the NEXT STATE according to the transition matrix for current state 'sampleState' and action 'a'
					sampleState = SampleMultinomialIRL.sampleSingleStateFromMultinomial (10, transitionMatrixForAction_i.getColumn (sampleState), RNG);
				}
				valueFunctionsArray[traj_i] = value_i;
				dataTrajectories.add (traj_i, trajectory_i); // add trajectory to trajectory dataset
			}
			// compute the mean of value functions for all trajectories
//			double valueFunctionMean = VectorUtility.getMean (valueFunctionsArray);	// GTD Not used
			// compute used the variance of value functions for all trajectories
//			double valueFunctionVariance = VectorUtility.getVariance (valueFunctionsArray);	// GTD Not used
		}
		System.out.println (dataTrajectories.size () + " trajectories generated");
		return dataTrajectories;
	}
	
	
	public static void main (String[] args) throws Exception, FileNotFoundException, IOException, IllegalArgumentException
	{
//		testMatrixJNI ();
		// Parse command line arguments
		CRC_Prediction.CommandLineIRLOptions.ParseReturn parseReturn = CommandLineIRLOptions.parse (args);
		
		int numberOfStatesInDBINT = 0;
		int numberOfActionsInDBINT = 0;
		int numberOfStateBasisVectors = 0;
		int numberOfActionBasisVectors = 0;
		int numberOfRewardFeatures = 0;
		int numStatesInMDP = 0;
		int numActionsInMDP = 0;
		Long numberOfActionsInDBLONG = Long.valueOf (0);
		Long numberOfStatesInDBLONG = Long.valueOf (0);
		Long numberOfStateBVectorsInDB = Long.valueOf (0);
		long	seed = parseReturn._seed;
		int		numThreads = parseReturn._numThreads;
		int		numGPUs = parseReturn._numCuda;
		boolean maxTablesBool = parseReturn._maxTablesDoExist;
		boolean profile = parseReturn._profile;
		int		numTables = maxTablesBool ? parseReturn._maxTablesInRestaurant : 0;
		
		double inverseTemperatureEta = parseReturn._inverseTemperatureEta; // 10.0; //eta parameter
																			// for bayesian IRL
		double rewardFuction_mean = parseReturn._rewardFunctionMean; // 0.0; //gaussian prior mu
																		// parameter
		double rewardFunction_stDev = parseReturn._rewardFunctionStDev; // 0.1; //gaussian prior
																		// sigma parameter
		double alpha = parseReturn._alphaConcentration; // 1.0; //concentration parameter
		double discountHyperParam = parseReturn._discount; // 0.0; //discount hyperparameter
		
		int maxMHIterations = parseReturn._maxMHIterations; // 1000;
		int iterationsForTableAssignmentUpdate = parseReturn._iterationsForTableAssignmentUpdate; // 2;
																									// //default
		int iterationsForRewardFuctionUpdate = parseReturn._iterationsForRewardFunctionUpdate;// 10;
																								// //default
		int iterationsForTransferLearning = parseReturn._iterationsForTransferLearning;// 100;
		double discountVal = parseReturn._discountValForMDP;
		
		int numExperiments = parseReturn._numExperiments;

		
		// String inputPathForLISTOFAncestryRelationshipsFILE_str =
		// "/Users/m186806/tIDFilteredDriverOnlyVCFs/allTIDFilteredSUBFILES/listOfAncestryFiles.txt";
		// //text file specifies the path to the AncestryRelationship .txt file specific the
		// relationship between subclone idxs
		// String inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILE_str =
		// "/Users/m186806/tIDFilteredDriverOnlyVCFs/allTIDFilteredSUBFILES/listOfSampleSpecificDRIVERSSubcloneVCFsDirectories.txt";
		// // //text file specifying the path to the directory containing all DRIVER-EDGE subvcfs
		// corresponding to the actions leading to the subclones for a given tumor-sample
		String inputPathForLISTOFAncestryRelationshipsFILE_str = parseReturn._inputPathForAncestryRelationshipsListFile.getAbsolutePath ();
		String inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILE_str = 
				parseReturn._inputPathForSubcloneDriversDirPathsListFile.getAbsolutePath ();
		
		String outputDirPathStr = parseReturn._outputDir;
		
		String serverIP = parseReturn._dbserverIP;
		String keyspacejk = parseReturn._dbkeyspace;
		_cluster = Cluster.builder ().addContactPoint (serverIP).build ();
		_session = _cluster.connect (keyspacejk);
		
		TreeTraversal traversedTreeInstance = null;
		
		boolean insertPathsForDBDynamically = parseReturn._insertPathsDynamically;
		
		// TODO: create command line flag for this boolean
		// perform tree-traversal while storing countmatricesMap in cassandra table
		if (parseReturn._treeTraversalWithDB)
		{
			// JK added 5.1.2019: we should reset treeTraversalCountMatricesMap_table in Cassandra
			// every time we re-run tree traversal
			String traversalIDstr = "fooTreeTraversalIntance";
			traversedTreeInstance = new TreeTraversal (_cluster, _session, inputPathForLISTOFAncestryRelationshipsFILE_str, 
														inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILE_str, 
														outputDirPathStr, traversalIDstr, insertPathsForDBDynamically);
			System.out.println ("****************Finished creating/running TreeTraversal instance with db!");
			System.out.println ("*****NOTE: Use cassandraCSVImport.sh to upload stateActionTrajectories from .csv files into Cassandra DB");
			// numStatesInMDP = traversedTreeInstance.getNumStatesInMDP();
			// numActionsInMDP = traversedTreeInstance.getNumActionsInMDP();
		}
		// if you are NOT uploading paths from the dB, you still need to perform tree-traversal
		// this calls the normal tree-traversal constructor which does NOT store countMatricesMap in-memory
		else if (!parseReturn._pathsFromDB)
		{
			traversedTreeInstance = new TreeTraversal (_cluster, _session, inputPathForLISTOFAncestryRelationshipsFILE_str, 
														inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILE_str, 
														outputDirPathStr, insertPathsForDBDynamically);
			System.out.println ("****************Finished creating/running TreeTraversal instance!");
			// numStatesInMDP = traversedTreeInstance.getNumStatesInMDP();
			// numActionsInMDP = traversedTreeInstance.getNumActionsInMDP();
		}
		
		String cqlCountNumACTIONSInActionSpaceStatement = "select count(*) from actionspace_table";
		String cqlCountNumSTATESInStateSpaceStatement = "select count(*) FROM statespace_table";
		String cqlCountNumBASISVECTORSStatement = "select count(*) FROM statebasisvectors_table";
		ResultSet rsActions = _session.execute (cqlCountNumACTIONSInActionSpaceStatement);
		ResultSet rsStates = _session.execute (cqlCountNumSTATESInStateSpaceStatement);
		ResultSet rsBVectors = _session.execute (cqlCountNumBASISVECTORSStatement);
		
		for (Row r_a : rsActions)
		{
			numberOfActionsInDBLONG = r_a.getLong (0);
		}
		for (Row r_s : rsStates)
		{
			numberOfStatesInDBLONG = r_s.getLong (0);
		}
		for (Row r_b : rsBVectors)
		{
			numberOfStateBVectorsInDB = r_b.getLong (0);
		}
		
		numberOfStatesInDBINT = numberOfStatesInDBLONG.intValue ();
		numberOfActionsInDBINT = numberOfActionsInDBLONG.intValue ();
		numberOfStateBasisVectors = numberOfStateBVectorsInDB.intValue ();
		
		numStatesInMDP = numberOfStatesInDBINT;
		numActionsInMDP = numberOfActionsInDBINT;
		
		numberOfRewardFeatures = (numActionsInMDP) + numberOfStateBasisVectors;
		
		THashSet<IRLRestaurant> MHRestarurantSamplesSet = new THashSet<IRLRestaurant> ();
//		IRLRestaurant bestMHSampledRestaurant = new IRLRestaurant ();
		IRLRestaurant bestMHSampledRestaurant = null;
		
		LikelihoodFunctionCancer llhfunction = new LikelihoodFunctionCancer (inverseTemperatureEta);
		Prior prior = new Prior ("gaussian", rewardFuction_mean, rewardFunction_stDev, 3);
		// Prior prior = new Prior("uniform", rewardFuction_mean, rewardFunction_stDev, 4 );
		
		IRLAlgorithmCancer irlalgo = new IRLAlgorithmCancer ("CRCIRL", llhfunction, prior, 1, alpha, discountHyperParam, maxMHIterations, 
															 iterationsForTableAssignmentUpdate, iterationsForRewardFuctionUpdate, 
															 iterationsForTransferLearning);
		MDPCancer reinforcementLearningEnvironment = null;
		List<double[][]> demoTrajectories = null;
		
		IRLRestaurantFactory irlfactory = new IRLRestaurantFactory (parseReturn._outputDir);
		MDPCancerFactory mdpfactory = new MDPCancerFactory (parseReturn._outputDir);
		
		// JK. 6.13.2019 adding ability to pre-load existing IRLRestaurant and MDPCAncer object
		if (!parseReturn._startFromScratch)
		{
			// retrieve the last modified .serialized files in specified directory
			Path dir = Paths.get (parseReturn._serializationDirectory); // specify your directory
			
			// Find most recent serialized IRLRestaurant in specified directory
			Optional<Path> latestRestaurantFilePath = Files.list (dir) // here we get the stream with full directory listing
					.filter (f -> !Files.isDirectory (f)) // exclude subdirectories from listing
					.filter (f -> f.toString ().endsWith (".restaurant.serialized"))
					.max (Comparator.comparingLong (f -> f.toFile ().lastModified ())); // finally get the last file using simple comparator by
																						// lastModified field
			
			if (latestRestaurantFilePath.isPresent ()) // your folder may be empty
			{
				// do your code here, lastFilePath contains all you need
				System.out.println ("Found our serialized resaurant:" + latestRestaurantFilePath);
				Path serializedRestaurantFilePath = latestRestaurantFilePath.get ();
				File preexistingIRLRestaurantFile = serializedRestaurantFilePath.toFile ();
				bestMHSampledRestaurant = irlfactory.get (preexistingIRLRestaurantFile);
			}
			else
			{
				System.err.println (" Serialized IRLRestaurant  was NOT found in specified directory. Will start from scratch!");
				parseReturn._startFromScratch = true;
			}
			
			// Find most recent serialized MDPCancer object in specified directory
			Optional<Path> latestMDPFilePath = Files.list (dir) // here we get the stream with full directory listing
					.filter (f -> !Files.isDirectory (f)) // exclude subdirectories from listing
					.filter (f -> f.toString ().endsWith (".mdpcancer.serialized"))
					.max (Comparator.comparingLong (f -> f.toFile ().lastModified ())); // finally get the last file using simple comparator by
																						// lastModified field
			
			if (latestMDPFilePath.isPresent ()) // your folder may be empty
			{
				System.out.println ("Found our serialized MDPCancer obj:" + latestMDPFilePath);
				Path serializedMDPFilePath = latestMDPFilePath.get ();
				File preexistingMDPCancerFile = serializedMDPFilePath.toFile ();
				reinforcementLearningEnvironment = mdpfactory.get (preexistingMDPCancerFile);
			}
			else
			{
				System.err.println (" Serialized MDPCancer  was NOT found in specified directory. Will start from scratch. ");
				parseReturn._startFromScratch = true;
			}
			
		}
		
		if (parseReturn._startFromScratch)
		{
			System.out.println ("We are building IRLRestaurant from stratch...");
			bestMHSampledRestaurant = new IRLRestaurant ();
			
			if (!parseReturn._pathsFromDB)
			{
				System.out.println ("Getting cancer paths from tree traversal");
				reinforcementLearningEnvironment = new MDPCancer ("microbial-influence", 1, discountVal, numStatesInMDP, numActionsInMDP, 
																  numberOfRewardFeatures, numberOfStateBasisVectors, numberOfActionBasisVectors, false, 
																  traversedTreeInstance, _session);
				demoTrajectories = getCancerPathsDataFromTreeTraversal (reinforcementLearningEnvironment, irlalgo, traversedTreeInstance);
			}
			else
			{
				System.out.println ("Getting cancer paths from database");
				reinforcementLearningEnvironment = new MDPCancer ("microbial-influence", 1, discountVal, numStatesInMDP, numActionsInMDP, 
																  numberOfRewardFeatures, numberOfStateBasisVectors, numberOfActionBasisVectors, false, 
																  _session);
				int maxTrajsToExtractPerSampleFromDB = parseReturn._maxTrajsPerSampleFromDB;
				demoTrajectories = getCancerPathsDataFromDatabase (reinforcementLearningEnvironment, irlalgo, maxTrajsToExtractPerSampleFromDB);
			}
		}
		else
		{ // Start CRP with VALID pre-existing IRLRestaurant and MDPCancer instances
			System.out.println ("Starting CRP-IRL with valid deserialized IRLRestaurant and MDPCancer instances!!!");
			if (!parseReturn._pathsFromDB)
			{
				System.out.println ("Getting cancer paths from tree traversal");
				demoTrajectories = getCancerPathsDataFromTreeTraversal (reinforcementLearningEnvironment, irlalgo, traversedTreeInstance);
			}
			else
			{
				System.out.println ("Getting cancer paths from database");
				int maxTrajsToExtractPerSampleFromDB = parseReturn._maxTrajsPerSampleFromDB;
				demoTrajectories = getCancerPathsDataFromDatabase (reinforcementLearningEnvironment, irlalgo, maxTrajsToExtractPerSampleFromDB);
				
			}
			
		}
		
		if (profile)
		{
			Controller.startCPURecording (true);
			Controller.startCallTracer (100000, false, false);
			Controller.startThreadProfiling ();
		}
		
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//JK 10.25.2019 Start storing transitionMatrix into cassandra for use by PUR-MCTS
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		@SuppressWarnings ("null")
		Map<Integer,double[][]> transitionMatricesMAP = reinforcementLearningEnvironment.getTransitionMatrix();
		String truncateTransitionMatrixTableCQLStatement = "TRUNCATE transitionmatrix_table";
		try
		{
			_session.execute (truncateTransitionMatrixTableCQLStatement);
		}
		catch (RuntimeException oops)
		{
			oops.printStackTrace ();
		}
		String cqlMatrixInsertQuery = "INSERT INTO transitionmatrix_table (actionint , nextStateRow , currentStateCol , probvalue ) VALUES (?, ?, ?,?)";
		PreparedStatement preparedStatement = _session.prepare (cqlMatrixInsertQuery);
		Set<Integer> actionIntKeys = transitionMatricesMAP.keySet ();
		double[][] transitionMatrixForActionINTi = null;
		int numTMatrixRows = 0;
		int numTMatrixCols = 0;
		double probDBLValue = 0.0;

		for (Integer actionINTi : actionIntKeys)
		{
			transitionMatrixForActionINTi = transitionMatricesMAP.get (actionINTi);
			numTMatrixRows = transitionMatrixForActionINTi.length;
			numTMatrixCols =  transitionMatrixForActionINTi[0].length;
			for (int rowINTj=0; rowINTj< numTMatrixRows; rowINTj++ ) {
				for (int colINTk=0; colINTk < numTMatrixCols; colINTk++) {
					probDBLValue = transitionMatrixForActionINTi[rowINTj][colINTk];
					BoundStatement boundStatement = preparedStatement.bind (actionINTi, rowINTj, colINTk, probDBLValue);
					_session.execute (boundStatement);
				}
			}
		}
		///////////end of updating Cassandra transitionMatrix_table////////////
		
		
		
		System.out.println ("********Calling ChineseRestaurantProcessInference()");
		if (numGPUs > 0)
		{
			Thread	curThread = Thread.currentThread ();
			String	name = curThread.getName ();
			
			name = name + InferenceAlgoCancer.kGPU + (numGPUs / 2);
			curThread.setName (name);
			InferenceAlgoCancer.initGPUs (numGPUs);
		}

		//JK 8.28.2019 added for v6 0.3.0
		for (int e = 0; e < numExperiments; e++)
		{
			if (!parseReturn._computeOccupancyWithDB)
			{
				_cluster.close ();
				System.out.println ("compute occupancy-matrix in-memory");
				
				InferenceAlgoCancer.ChineseRestaurantProcessInference (reinforcementLearningEnvironment, demoTrajectories, irlalgo, 
																		MHRestarurantSamplesSet, bestMHSampledRestaurant, seed, numThreads, numGPUs, 
																		numTables, parseReturn._startFromScratch, irlfactory, mdpfactory, profile);
			}
			else
			{
				System.out.println ("compute occupancy-matrix and store in db");
				InferenceAlgoCancer.ChineseRestaurantProcessInferenceWithDatabase (reinforcementLearningEnvironment, demoTrajectories, irlalgo, 
																					MHRestarurantSamplesSet, bestMHSampledRestaurant, _session, seed, 
																					numThreads, numGPUs, numTables, parseReturn._startFromScratch, 
																					irlfactory, mdpfactory, profile);
				_cluster.close (); //JK added 6.17.2019 to close communication to cassandra cluster (otherwise will remain on indefinitely)
			}
		}
		if (profile)
		{
	        Controller.saveSnapshot (new File ("Profile." + InferenceAlgoCancer.timeStampStr () + ".jps"));
			Controller.stopCPURecording ();
			Controller.stopThreadProfiling ();
		}
		
	}
	
}
