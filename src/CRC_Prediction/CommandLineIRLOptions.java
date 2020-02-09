
package CRC_Prediction;


import java.io.File;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

/**
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
@SuppressWarnings ("deprecation")
public class CommandLineIRLOptions
{
	
	public static Options getOptions ()
	{
		
		// on command line, specify your input using -datafile ; -depth; -nodes ; -bytes ; -url ;
			// -saveto
		Options options = new Options ();
		
		options.addOption ("ancestryRelationsListFile", true, "Fully specified path to inputPathForLISTOFAncestryRelationshipsFILE_str");
		options.addOption ("driversSubcloneVCFsDirectoriesListFile", true, 
							"Fully specified path to inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILE_str");
		options.addOption ("inverseTemperatureEta", true, "Specify IRL (double) parameter: eta");
		options.addOption ("alphaConcentration", true, "Specify CRP (double) parameter: alpha");
		options.addOption ("discount", true, "Specify CRP (double) parameter: discount");
		options.addOption ("rewardFunctionMean", true, "Specify IRL (double) parameter: mean");
		options.addOption ("rewardFunctionStdDev", true, "Specify IRL (double) parameter: std dev");
		options.addOption ("maxMHIterations", true, "Specify IRL (int) parameter: number of MH iterations");
		options.addOption ("maxTableAssignmentUpdateIterations", true, "Specify IRL (int) parameter: iterationsForTableAssignmentUpdate");
		options.addOption ("maxRewardFunctionUpdateIterations", true, "Specify IRL (int) parameter: iterationsForRewardFunctionUpdate");
		options.addOption ("maxTransferLearningIterations", true, "Specify IRL (int) parameter: iterationsForTransferLearning");
		options.addOption ("discvountValueForMDP", true, "Specify IRL (double) parameter: discount val for MDP");
		options.addOption ("dbserver", true, "Specify IP address of Cassandra server (local is 127.0.0.1)");
		options.addOption ("dbkeyspace", true, 
							"Specify keyspace of Cassandra database containing statespace, actionspace, statebasisvectors, etc " + 
							"(default for CRC is crckeyspace)");
		options.addOption ("pathsFromDB", true, 
							"Indicate whether paths will be extracted from state-action-trajectories stored in stateactiontrajectories_table in cassandra");
		options.addOption ("outputDir", true, "Specify absolute path to the output directory for outputting .csv files");
		options.addOption ("computeOccupancyWithDB", true, 
							"Indicate whether computeOccupancy(), updateRewardFunctions(), computeLogPosteriorProb(), " + 
							"ChineseRestaurantProcessInference() will rely on cassandra table for LLH and gradient calculations");
		options.addOption ("maxTrajsPerSampleFromDB", true, 
							"Specify the maximum possible number of trajectories (per sample) to extract from " + 
							"stateActionTrajectories_table to use as customers");
		options.addOption ("treeTraversalWithDB", true, 
							"Indicate whether you want to store countMatricesMAP into cassandra table when performing tree-traversal");
		options.addOption ("insertPathsToDBDynamically", true, 
							"Indicate whether paths will be DYNAMICALLY created and inserted as state-action-trajectories into " + 
							"stateactiontrajectories_table in cassandra");
		options.addOption ("seed", true, "Seed for the RNG to use");
		options.addOption ("numThreads", true, "Number of threads to use for parallelizable tasks, minimum of 1");
		options.addOption ("numGPUs", true, "Number of GPUs to use for parallelizable tasks, minimum of 0");
		options.addOption ("startIRLfromScratch", true, 
							"Indicate whether IRL algorithm will begin with randomly initialized restaurant and table arrangement " + 
							"or resume from existing restaurant");
		options.addOption ("serializationDir", true, 
							"Specify absolute path to the directory for retrieving serialized IRLRestaurant and MDPCancer object files");
		options.addOption ("profile", true, "Indicate whether should profile the app usign JProfile");
        options.addOption ("maxTables", true, "Specify IRL (int) parameter: maximum number of tables in restaurant");
        options.addOption ("maxTablesExist", true, 
        					"Indicate whether you are pre-determining the maximum number of tables in restaurant. Default maxTables=numObservations");
		
		options.addOption ("serializedRestaurant", true, 
							"Specify path to the file of a specific serialized IRLRestaurant object file, whose tables you wish to evaluate");
		options.addOption ("serializedMDP", true, 
				"Specify path to the file of a specific serialized MDP object file you wish to evaluate");
		options.addOption ("numExperiments", true, 
				"Specify IRL (int) parameter: number of experiments to run");
		options.addOption ("numMDPstates", true, "Specify IRL (int) parameter: number of states in (Gridworld) MDP");
		//options.addOption ("numMDPactions", true, "Specify IRL (int) parameter: number of actions in (Gridworld) MDP");
		options.addOption ("numSimulatedExperts", true, "Specify Gridworld initialization (int) parameter: number of experts in (Gridworld) to simulate");

		options.addOption ("numDemonstrationsPerExpert", true, "Specify Gridworld initialization (int) parameter: number of demonstrations per expert in (Gridworld) simulation");

		

		return options;
	}
	
	
	public static ParseReturn parse (String[] args)
	{
		
		Options options = getOptions ();
		CommandLineParser clp = new BasicParser ();
		CommandLine cl = null;
		
		try
		{
			cl = clp.parse (options, args);
		}
		catch (ParseException pe)
		{
			System.err.println ("Parsing failed.  Reason: " + pe.getMessage ());
			System.out.println ();
			
			HelpFormatter formatter = new HelpFormatter ();
			formatter.printHelp ("IRLCommandLineParse", options);
			
			System.exit (1);
		}
		
		ParseReturn parseReturn = new ParseReturn ();
		
		// -ancestryRelationsListFile flag
		@SuppressWarnings ("null") // Compiler doesn't understand System.exit
		String fileNameofAncestryRelList = cl.getOptionValue ("ancestryRelationsListFile");
		if (fileNameofAncestryRelList != null)
		{
			System.out.println ("Congrats!" + fileNameofAncestryRelList + " is a valid data file for the ancestryRelationsListFile!");
			File preexistingAncestryRelationsListFile = new File (fileNameofAncestryRelList);
			parseReturn._inputPathForAncestryRelationshipsListFile = preexistingAncestryRelationsListFile;
		}
		else
		{
			System.err.println ("INVALID ancestryRelationsListFile specified.");
		}
		
		// -driversSubcloneVCFsDirectoriesListFile flag
		String fileNameofDriversSubcloneVCFsDirectoriesList = cl.getOptionValue ("driversSubcloneVCFsDirectoriesListFile");
		if (fileNameofDriversSubcloneVCFsDirectoriesList != null)
		{
			System.out.println ("Congrats!" + fileNameofDriversSubcloneVCFsDirectoriesList + 
								" is a valid data file for the driversSubcloneVCFsDirectoriesListFile!");
			File preexistingdriversSubcloneVCFsDirectoriesListFile = new File (fileNameofDriversSubcloneVCFsDirectoriesList);
			parseReturn._inputPathForSubcloneDriversDirPathsListFile = preexistingdriversSubcloneVCFsDirectoriesListFile;
		}
		else
		{
			System.err.println ("INVALID driversSubcloneVCFsDirectoryListFile specified.");
		}
		
		// -inverseTemperatureEta flag
		if (cl.getOptionValue ("inverseTemperatureEta") != null)
		{
			parseReturn._inverseTemperatureEta = Double.parseDouble (cl.getOptionValue ("inverseTemperatureEta"));
		}
		
		// -alphaConcentration flag
		if (cl.getOptionValue ("alphaConcentration") != null)
		{
			parseReturn._alphaConcentration = Double.parseDouble (cl.getOptionValue ("alphaConcentration"));
		}
		
		// -discount flag
		if (cl.getOptionValue ("discount") != null)
		{
			parseReturn._discount = Double.parseDouble (cl.getOptionValue ("discount"));
		}
		
		// -rewardFunctionMean flag
		if (cl.getOptionValue ("rewardFunctionMean") != null)
		{
			parseReturn._rewardFunctionMean = Double.parseDouble (cl.getOptionValue ("rewardFunctionMean"));
		}
		
		// -rewardFunctionStdDev flag
		if (cl.getOptionValue ("rewardFunctionStdDev") != null)
		{
			parseReturn._rewardFunctionStDev = Double.parseDouble (cl.getOptionValue ("rewardFunctionStdDev"));
		}
		
		// -maxMHIterations flag
		if (cl.getOptionValue ("maxMHIterations") != null)
		{
			parseReturn._maxMHIterations = Integer.parseInt (cl.getOptionValue ("maxMHIterations"));
		}
		
		// -maxTableAssignmentUpdateIterations flag
		if (cl.getOptionValue ("maxTableAssignmentUpdateIterations") != null)
		{
			parseReturn._iterationsForTableAssignmentUpdate = Integer.parseInt (cl.getOptionValue ("maxTableAssignmentUpdateIterations"));
		}
		
		// -maxRewardFunctionUpdateIterations flag
		if (cl.getOptionValue ("maxRewardFunctionUpdateIterations") != null)
		{
			parseReturn._iterationsForRewardFunctionUpdate = Integer.parseInt (cl.getOptionValue ("maxRewardFunctionUpdateIterations"));
		}
		
		// -maxTransferLearningIterations flag
		if (cl.getOptionValue ("maxTransferLearningIterations") != null)
		{
			parseReturn._iterationsForTransferLearning = Integer.parseInt (cl.getOptionValue ("maxTransferLearningIterations"));
		}
		
		// -discvountValueForMDP flag
		if (cl.getOptionValue ("discvountValueForMDP") != null)
		{
			parseReturn._discountValForMDP = Double.parseDouble (cl.getOptionValue ("discvountValueForMDP"));
		}
		
		// -dbserver
		String dbserverIPAddress = cl.getOptionValue ("dbserver");
		// System.out.println("dbserverIPAddress from dbserver input argument ="+dbserverIPAddress);
		if (dbserverIPAddress != null)
		{
			parseReturn._dbserverIP = dbserverIPAddress;
		}
		else
		{
			System.out.println ("No IP address provided. Using default local IP address 127.0.0.1 instead.");
//			parseReturn._dbserverIP = "127.0.0.1";
		}
		
		// -dbkeyspace
		String dbkeyspaceName = cl.getOptionValue ("dbkeyspace");
		if (dbkeyspaceName != null)
		{
			parseReturn._dbkeyspace = dbkeyspaceName;
			System.out.println ("Using cassandra keyspace :" + parseReturn._dbkeyspace);
		}
		else
		{
			System.out.println ("Cassandra keyspace not specified. Using default crckeyspace keyspace instead.");
//			parseReturn._dbkeyspace = "crckeyspace";
		}
		
		// -pathsFromDB flag
		if (cl.getOptionValue ("pathsFromDB") != null)
		{
			parseReturn._pathsFromDB = Boolean.parseBoolean (cl.getOptionValue ("pathsFromDB"));
		}
		
		// -outputDir flag
		String outputDirStr = cl.getOptionValue ("outputDir");
		if (outputDirStr != null)
		{
			parseReturn._outputDir = outputDirStr;
		}
		else
		{
			System.out.println ("Output directory was not specified. Using default working directory.");
		}
		
		// -computeOccupancyWithDB flag
		if (cl.getOptionValue ("computeOccupancyWithDB") != null)
		{
			parseReturn._computeOccupancyWithDB = Boolean.parseBoolean (cl.getOptionValue ("computeOccupancyWithDB"));
		}
		
		// -maxTrajsPerSampleFromDB flag
		if (cl.getOptionValue ("maxTrajsPerSampleFromDB") != null)
		{
			parseReturn._maxTrajsPerSampleFromDB = Integer.parseInt (cl.getOptionValue ("maxTrajsPerSampleFromDB"));
		}
		
		// -treeTraversalWithDB flag
		if (cl.getOptionValue ("treeTraversalWithDB") != null)
		{
			parseReturn._treeTraversalWithDB = Boolean.parseBoolean (cl.getOptionValue ("treeTraversalWithDB"));
		}
		
		// -insertPathsToDBDynamically
		if (cl.getOptionValue ("insertPathsToDBDynamically") != null)
		{
			parseReturn._insertPathsDynamically = Boolean.parseBoolean (cl.getOptionValue ("insertPathsToDBDynamically"));
		}
		
		// -startIRLfromScratch
		if (cl.getOptionValue ("startIRLfromScratch") != null)
		{
			parseReturn._startFromScratch = Boolean.parseBoolean (cl.getOptionValue ("startIRLfromScratch"));
		}
		
		// -profile
		if (cl.getOptionValue ("profile") != null)
		{
			parseReturn._profile = Boolean.parseBoolean (cl.getOptionValue ("profile"));
		}
		
		// -serializationDir
		String serialDirStr = cl.getOptionValue ("serializationDir");
		if (serialDirStr != null)
		{
			System.out.println ("You have specified serialization directory: " + serialDirStr);
			File serialDir = new File (serialDirStr);
			if (serialDir.isDirectory ())
			{
				parseReturn._serializationDirectory = serialDirStr;
				parseReturn._startFromScratch = false; // i.e if you are specifying a serialization directory, you are implying that you wish
														// to use serialized objects found in this directory
			}
			else
			{
				System.out.println ("Serialization directory does NOT exist. Will instead build IRL from scratch");
				parseReturn._startFromScratch = true;
			}
			
		}
		else
		{
			System.out.println ("Serialization directory was not specified. Will build IRL from scratch");
			parseReturn._startFromScratch = true;
		}
		
		// -seed
		String	test = cl.getOptionValue ("seed");
		if (test != null)
		{
			parseReturn._seed = Long.parseLong (test);
		}
		
		// -numThreads
		test = cl.getOptionValue ("numThreads");
		if (test != null)
		{
			parseReturn._numThreads = Integer.parseInt (test);
		}
		
		// -numGPUs
		test = cl.getOptionValue ("numGPUs");
		if (test != null)
		{
			parseReturn._numCuda = Integer.parseInt (test);
		}
		
		// -maxTables
		if (cl.getOptionValue ("maxTables") != null)
		{
			parseReturn._maxTablesInRestaurant = Integer.parseInt (cl.getOptionValue ("maxTables"));
			parseReturn._maxTablesDoExist = true;
		}
		
		// -maxTablesExist
		if (cl.getOptionValue ("maxTablesExist") != null)
		{
			parseReturn._maxTablesDoExist = Boolean.parseBoolean (cl.getOptionValue ("maxTablesExist"));
		}
		
		
		// -serializedRestaurant
		String serialRestaurantStr = cl.getOptionValue ("serializedRestaurant");
		if (serialRestaurantStr != null)
		{
			System.out.println ("You have specified serialized restaurant for evaluation: " + serialRestaurantStr);
			File serialFile = new File (serialRestaurantStr);
			if (serialFile.isFile())
			{
				parseReturn._serializedRestaurant = serialRestaurantStr;
				parseReturn._evalSerializedRestaurant = true;
			}
			else
			{
				System.out.println ("Serialized restaurant does NOT exist. ");
				parseReturn._evalSerializedRestaurant = false;

			}
			
		}
		
		// -serializedMDP
		String serialMDPStr = cl.getOptionValue ("serializedMDP");
		if (serialMDPStr != null)
		{
			System.out.println ("You have specified serialized MDP for evaluation: " + serialMDPStr);
			File serialFile = new File (serialMDPStr);
			if (serialFile.isFile())
			{
				parseReturn._serializedMDP = serialMDPStr;
				parseReturn._evalSerializedMDP = true;
			}
			else
			{
				System.out.println ("Serialized MDP does NOT exist. ");
				parseReturn._evalSerializedMDP = false;

			}
			
		}
		
		//-numExperiments
		if (cl.getOptionValue ("numExperiments") != null)
		{
			parseReturn._numExperiments = Integer.parseInt (cl.getOptionValue ("numExperiments"));
		}
		
		//-numMDPstates
		if (cl.getOptionValue ("numMDPstates") != null)
		{
			parseReturn._numMDPstates = Integer.parseInt (cl.getOptionValue ("numMDPstates"));
		}
//		//-numMDPactions  (NOTE: in gridworld there is ALWAYS only 4 possible cardinal actions; only the state space can increase,not the action space)
//		if (cl.getOptionValue ("numMDPactions") != null)
//		{
//			parseReturn._numMDPactions = Integer.parseInt (cl.getOptionValue ("numMDPactions"));
//		}
		//-numSimulatedExperts
		if (cl.getOptionValue ("numSimulatedExperts") != null)
		{
			parseReturn._numSimulatedExperts = Integer.parseInt (cl.getOptionValue ("numSimulatedExperts"));
		}
		
		//-numDemonstrationsPerExpert
		if (cl.getOptionValue ("numDemonstrationsPerExpert") != null)
		{
			parseReturn._numDemonstrationsPerExpert = Integer.parseInt (cl.getOptionValue ("numDemonstrationsPerExpert"));
		}
		
				
		
		return parseReturn;
	}
	
	
	public static class ParseReturn
	{
		public File		_inputPathForAncestryRelationshipsListFile		= null;
		public File		_inputPathForSubcloneDriversDirPathsListFile	= null;
		public double	_inverseTemperatureEta							= 10.0;
		public double	_alphaConcentration								= 1.0;
		public double	_discount										= 0.0;
		public double	_rewardFunctionMean								= 0.0;
		public double	_rewardFunctionStDev							= 0.1;
		// public int _maxMHIterations = 10;
		// public int _iterationsForTableAssignmentUpdate = 100;
		// public int _iterationsForRewardFunctionUpdate = 2;
		// public int _maxMHIterations = 1000;
		// public int _iterationsForTableAssignmentUpdate = 250;
		// public int _iterationsForRewardFunctionUpdate = 100;
//		public int		_maxMHIterations								= 100;
//		public int		_iterationsForTableAssignmentUpdate				= 10;
//		public int		_iterationsForRewardFunctionUpdate				= 50;
		// public int _iterationsForRewardFunctionUpdate = 1; //JK changed 5.7.2019
		
		public int		_maxMHIterations								= 100;
		public int		_iterationsForTableAssignmentUpdate				= 2;
		public int		_iterationsForRewardFunctionUpdate				= 10;
		
		public int		_iterationsForTransferLearning					= 100;
		public double	_discountValForMDP								= 0.9;
		public String	_dbserverIP										= "127.0.0.1";
		public String	_dbkeyspace										= "crckeyspace";
		public boolean	_pathsFromDB									= false;
		public String	_outputDir										= "";
		public boolean	_computeOccupancyWithDB							= false;
		public int		_maxTrajsPerSampleFromDB						= 150000;
		public boolean	_treeTraversalWithDB							= false;
		public boolean	_insertPathsDynamically							= false;		// added after version 0.31
		public boolean	_startFromScratch								= true;
		public boolean	_profile										= false;
		public String	_serializationDirectory							= "";
		
		public long		_seed											= 0;
		public int		_numThreads										= 1;
		public int		_numCuda										= 0;
    	public int		_maxTablesInRestaurant							= 0;
    	public boolean	_maxTablesDoExist								= false;
    	//added v6 0.2.7
    	public String 	_serializedRestaurant							="";
    	public boolean 	_evalSerializedRestaurant						= false;
    	public String 	_serializedMDP									="";
    	public boolean 	_evalSerializedMDP								= false;
    	public int 		_numExperiments									= 1;
    	public int 		_numMDPstates									=64;
    	public int 		_numMDPactions									=4;
    	public int 		_numSimulatedExperts							=3;
    	public int 		_numDemonstrationsPerExpert 					=2;


    	
		
	}
}
