
package CRC_Prediction.Utils;


import com.datastax.driver.core.*;
import com.google.common.collect.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.math3.util.Pair;
//import org.rosuda.REngine.REngineException;


public class TreeTraversal
{
	
	public Map<String, double[][]>					_transitionCountMatrixList;
	public Map<String, double[][]>					_cooccurenceCountMatrixList;
	public double									_noise					= 0.3;
	private static final int						_stateActionTrajLength	= 5;
	
	private static final Pattern					tabDelimiter			= Pattern.compile ("\t");
	private static final Pattern					commaDelimiter			= Pattern.compile (",");
	// private static final Pattern colonDelimiter = Pattern.compile(":"); // GTD Not used
	private static final Pattern					semiColonDelimiter		= Pattern.compile (";");
	
	private static Cluster							cluster;
	private static Session							session;
	
	public List<TrajectoryTimeStep>					leafTimeStepsAcrossAllTrajectories;
	
	public Map<Integer, Map<Integer, Integer>>	_nextPossibleStatesMAPForAllStates;
	
	public int										_numStatesInMDP			= 0;
	public int										_numActionsInMDP		= 0;
	
	public Map<String, List<double[][]>>			_doublesStateAndActiontrajectoriesForAllSamplesMap;
	// textfilespecifiesthepathtotheAncestryRelationship.txtfilespecifictherelationshipbetweensubcloneidxs
	public String									_inputPathForLISTOFAncestryRelationshipsFILEstr;
	// textfilespecifyingthepathtothedirectorycontainingallDRIVER-EDGEsubvcfscorrespondingtotheactionsleadingtothesubclonesforagiventumor-sample
	public String									_inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILEstr;
	public static String							_outputDirPathStr;
	
	
	public TreeTraversal (Cluster clusterObj, Session sessionObj,
			String inputPathForLISTOFAncestryRelationshipsFILEstring,
			String inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILEstring,
			String outPathDirString, boolean insertPathsIntoDBDynamic) 
	{
		
		cluster = clusterObj;
		session = sessionObj;
		_inputPathForLISTOFAncestryRelationshipsFILEstr = inputPathForLISTOFAncestryRelationshipsFILEstring;
		_inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILEstr = inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILEstring;
		_outputDirPathStr = outPathDirString;
		
		String cqlSelectEnsemblGeneIDTranscriptIDSymbol_table = "select * from ensembl_gene_transcript_symbol_fordrivergenes";
		
		Map<Integer, String> intToGeneSymbol = new HashMap<Integer, String> ();
		Map<String, Integer> geneSymbolToInt = new HashMap<String, Integer> ();
		HashMultimap<String, String> geneSymbolToTIDs = HashMultimap.create ();
		Map<String, String> tidToGeneSymbol = new HashMap<String, String> ();
		
		Map<String, Integer> tidToMUTactionINTmap = new HashMap<String, Integer> ();
		Map<String, Integer> tidToMETHYLactionINTmap = new HashMap<String, Integer> ();
		
		Map<String, UUID> tidToMUTactionUUIDmap = new HashMap<String, UUID> ();
		Map<String, UUID> tidToMETHYLactionUUIDmap = new HashMap<String, UUID> ();
		
		Map<UUID, Integer> uuidToIntegerMAPforActions = new HashMap<UUID, Integer> ();
		
		for (Row row : session.execute (cqlSelectEnsemblGeneIDTranscriptIDSymbol_table))
		{
			// String rowAsString = row.toString();
//			String geneID = row.getString (0);	// GTD Not used
			String transcriptID = row.getString (1);
			String geneSymbol = row.getString (2);
			geneSymbolToTIDs.put (geneSymbol, transcriptID);
			tidToGeneSymbol.put (transcriptID, geneSymbol);
		}
		
		Set<String> driverGeneSymbolSet = geneSymbolToTIDs.keySet ();
		
		int gsymbolCount = 1; // We start at actionINT=1 as opposed to zero, actionINT=0 will be
								// reserved for padding purposes.
		for (String gSymbol : driverGeneSymbolSet)
		{
			intToGeneSymbol.put (gsymbolCount, gSymbol);
			geneSymbolToInt.put (gSymbol, gsymbolCount);
			
			gsymbolCount++;
			
		}
		int totalNumDriverGeneSymbolsWithValidtranscripts = driverGeneSymbolSet.size ();
		
		// int actionINTcount =1; //start the actionINT space with 1 instead of zero. This will
		// allows us to fill to pad paths with length < _stateActionTrajLength with zero's
		String truncateActionSpaceTableWithNewActionsCQLStatement = "TRUNCATE actionspace_table";
		session.execute (truncateActionSpaceTableWithNewActionsCQLStatement);
		for (int geneInt : intToGeneSymbol.keySet ())
		{
			String geneSymbol = intToGeneSymbol.get (geneInt);
			
			UUID mutactionuid = UUID.randomUUID ();
			UUID methylationuid = UUID.randomUUID ();
			// int actionINTForMUT = actionINTcount; //the genetINT is actually better
			// int actionINTForMETHYL =
			// actionINTcount+totalNumDriverGeneSymbolsWithValidtranscripts;
			int actionINTForMUT = geneInt;
			int actionINTForMETHYL = geneInt + totalNumDriverGeneSymbolsWithValidtranscripts;
			uuidToIntegerMAPforActions.put (mutactionuid, actionINTForMUT);
			uuidToIntegerMAPforActions.put (methylationuid, actionINTForMETHYL);
			
			Set<String> transcriptIDsSet = geneSymbolToTIDs.get (geneSymbol);
			
			for (String tid : transcriptIDsSet)
			{
				tidToMUTactionUUIDmap.put (tid, mutactionuid);
				tidToMETHYLactionUUIDmap.put (tid, methylationuid);
				tidToMUTactionINTmap.put (tid, Integer.valueOf (actionINTForMUT));
				tidToMETHYLactionINTmap.put (tid, Integer.valueOf (actionINTForMETHYL));
			}
			
			// NOTE: because some gene symbols contain a hyphen in their names, which causes an
			// error in cassandra, they must be enclosed in double quotes.
			// cql statement (<nameOfColumnA>, <nameOfColumnC>,...) values(<value to add to
			// columnA>, <value to add to columnC>,...)
			String insertMUTAction = "INSERT INTO actionspace_table (actionuuid,\""
					+ geneSymbol + "\",actionint, isMutation, isMethylation) values(" + mutactionuid
					+ ",true," + actionINTForMUT + ", true, false)";
			String insertMETHYLAction = "INSERT INTO actionspace_table (actionuuid,\""
					+ geneSymbol + "\",actionint, isMutation, isMethylation) values("
					+ methylationuid + ",true," + actionINTForMETHYL + ", false, true)";
			
			session.execute (insertMUTAction);
			session.execute (insertMETHYLAction);
			
			// actionINTcount++;
		}
		
		/////////////////////////////// Add padding action actionINT=0
		/////////////////////////////// (action0)//////////////////////
		UUID paddingAction0uid = UUID.randomUUID ();
		int paddingaction0INT = 0;
		String insertPaddingAction0 = "INSERT INTO actionspace_table (actionuuid,actionint, isMutation, isMethylation) values("
				+ paddingAction0uid + "," + paddingaction0INT + ", false, false)";
		session.execute (insertPaddingAction0);
		/////////////////////////////////////////////////////////////////////////////////////////////
		
		// Create mapping based on Cassandra db state_space table (based on GLFM created
		// states/patterns)
		Map<UUID, Integer> uuidToIntegerMapForStatesInDB = new HashMap<UUID, Integer> ();
		UUID stateUUIDinDB = null;
		Integer stateINTinDB = 0;
		Long stateINT_longval_inDB = (long) 0;
		
		String cqlGetStateRowsStatement = "select * from statespace_table";
		ResultSet rsStates = session.execute (cqlGetStateRowsStatement);
		for (Row r_state : rsStates)
		{
			stateUUIDinDB = r_state.getUUID ("stateuuid");
			stateINT_longval_inDB = r_state.getLong ("stateint");
			
			stateINTinDB = stateINT_longval_inDB.intValue ();
			uuidToIntegerMapForStatesInDB.put (stateUUIDinDB, stateINTinDB);
		}
		
		String truncateStateActionTrajTableCQLStatement = "TRUNCATE stateactiontrajectories_table";
		session.execute (truncateStateActionTrajTableCQLStatement);
		
		String truncateSampleNameTableCQLStatement = "TRUNCATE wgssamplenames_table";
		session.execute (truncateSampleNameTableCQLStatement);
		
		String truncateTrajCountsTableCQLStatement = "TRUNCATE sampleTrajCounts_table";
		session.execute (truncateTrajCountsTableCQLStatement);
		
		// Create stateactiontrajectories_table with appropriate number of columns = number of
		// state-action pairs in each trajectory
		// NOTE: this will cause an error if these columns already exist
		// String cqlCreateStateActionTrajectoriesTableStatement = "create TABLE
		// stateactiontrajectories_table(trajuuid UUID PRIMARY KEY, sampleName text)";
		// for(int l=0; l< _stateActionTrajLength; l++) {
		// String cqlADDStateColToTrajectoriesTableStatement = "alter TABLE
		// stateactiontrajectories_table ADD state"+l+" bigint";
		// String cqlADDActionColToTrajectoriesTableStatement = "alter TABLE
		// stateactiontrajectories_table ADD action"+l+" bigint";
		// session.execute(cqlADDStateColToTrajectoriesTableStatement);
		// session.execute(cqlADDActionColToTrajectoriesTableStatement);
		// }
		
		// These should be inputted as arguments to the constructor!
		// String inputPathForLISTOFAncestryRelationshipsFILEstr =
		// "/Users/m186806/tIDFilteredDriverOnlyVCFs/allTIDFilteredSUBFILES/listOfAncestryFiles.txt";
		// //text file specifies the path to the AncestryRelationship .txt file specific the
		// relationship between subclone idxs
		// String inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILEstr =
		// "/Users/m186806/tIDFilteredDriverOnlyVCFs/allTIDFilteredSUBFILES/listOfSampleSpecificDRIVERSSubcloneVCFsDirectories.txt";
		// // //text file specifiying the path to the directory containing all DRIVER-EDGE subvcfs
		// corresponding to the actions leading to the subclones for a given tumor-sample
		
		leafTimeStepsAcrossAllTrajectories = new ArrayList<TrajectoryTimeStep> ();
		
		// Initialize the 2 transition matrices: state transition: T(s',s,a) = P(s'|s,a)
		// _transitionmatrix models the probability of the next state (row), given current state
		// (col) for action a_i (Map Integer index)
		// _discountedTransitonMatrix models the probability of the next state (col), given the
		// current state(row) for action a_I (Map Integer index)
		
		/// COUNT number of ACTIONS in actionspace_table
		String cqlCountNumACTIONSInActionSpaceStatement = "select count(*) from actionspace_table";
		ResultSet rsActionsCOUNT = session.execute (cqlCountNumACTIONSInActionSpaceStatement);
		Long numberOfActionsInDB = (long) 0;
		int numberOfActionsInMDP = 0;
		for (Row r_a : rsActionsCOUNT)
		{
			numberOfActionsInDB = r_a.getLong (0);
		}
		// numberOfActionsInMDP = numberOfActionsInDB.intValue()+1; //although 0 is not included in
		// action-space(because it is not explicity assigned to any gene-symbol) it is a DEFAULT
		// padding action
		numberOfActionsInMDP = numberOfActionsInDB.intValue (); // this is only true if we DID
																// create an actionINT=0 in
																// actionspace_table
																// that corresponds to the padding
																// action
		setNumActionsInMDP (numberOfActionsInMDP);
		////////////////////////////////////////////////
		
		/// Count number of STATES in statespace_table
		String cqlCountNumSTATESInStateSpaceStatement = "select count(*) FROM statespace_table";
		ResultSet rsStatesCOUNT = session.execute (cqlCountNumSTATESInStateSpaceStatement);
		Long numberOfStatesInDB = (long) 0;
		int numberOfStatesInMDP = 0;
		for (Row r_s : rsStatesCOUNT)
		{
			numberOfStatesInDB = r_s.getLong (0);
		}
		// numberOfStatesInMDP = numberOfStatesInDB.intValue()+1; //although 0 is not included in
		// state-space(because it is explicity associated with the NORMAL STATE) it is a DEFAULT
		// padding state
		numberOfStatesInMDP = numberOfStatesInDB.intValue ();
		setNumStatesInMDP (numberOfStatesInMDP);
		///////////////////////////////////////////////////
		
		// Initialize map of count-matrices; one for each possible action
		Map<Integer, double[][]> countMatricesMAP = new HashMap<Integer, double[][]> ();
		double[][] countMatrixfor_ai;
		for (int a = 0; a < numberOfActionsInMDP; a++)
		{
			countMatrixfor_ai = new double[numberOfStatesInMDP][numberOfStatesInMDP]; // recall,
																						// number of
																						// states in
																						// statespace
																						// does not
																						// include
																						// stateINT
																						// 0, which
																						// is
																						// reserved
																						// as a
																						// default
																						// value for
																						// padding
																						// stateActionTrajectories
																						// based on
																						// subclonal
																						// paths
																						// which
																						// have
																						// lengths <
																						// _stateActionTrajLength
			countMatricesMAP.put (Integer.valueOf (a), countMatrixfor_ai);
		}
		
		////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////
		
		////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////
		
//		BufferedOutputStream bos = null;	// GTD Not used
		
		System.out.println ("******Running iterateStateNodesInTreesFromFULLSUBCLONEAncestryFiles()\n");
		Map<String, MutablePair<Queue<int[]>, List<TreeTraversal.Node>>> specificPathsTraversedInEachSampleTreeMap = iterateStateNodesInTreesFromFULLSUBCLONEAncestryFiles (
				_inputPathForLISTOFAncestryRelationshipsFILEstr);
		
		System.out.println ("******Running iterateActionEdgesInTreesFromDRIVERSSUBCLONEAncestryFiles()\n");
		iterateActionEdgesInTreesFromDRIVERSSUBCLONEAncestryFiles (
				_inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILEstr,
				tidToMUTactionUUIDmap, tidToMETHYLactionUUIDmap,
				specificPathsTraversedInEachSampleTreeMap, true, false);
		
		Map<String, List<double[][]>> doublesStateAndActiontrajectoriesForAllSamplesMap = new HashMap<String, List<double[][]>> ();
		/// JK 3.1.2019 this hashmap can eventually be turned into the List<double[][]> used by the CRP-IRL as input trajectories
		
		List<Double> singleDBLTrajectoryOfStates = new ArrayList<Double> ();
		List<Double> singleDBLTrajectoryOfActions = new ArrayList<Double> ();
		String cqlInsertSampleNameStatement = "";
		
		/// JK: creating state-action trajectories
		System.out.println ("******Create initial state-action trajectories\n");
		for (String sampleX : specificPathsTraversedInEachSampleTreeMap.keySet ())
		{
			
			MutablePair<Queue<int[]>, List<TreeTraversal.Node>> traversedPathsPair = specificPathsTraversedInEachSampleTreeMap.get (sampleX);
			List<TreeTraversal.Node> treeXNodes = traversedPathsPair.right;
			// the left item of the MutablePair is the queue of int []'s corresponding to the various paths in treeX
			
			for (Node currNode : treeXNodes)
			{
				
				if (currNode.Id == 0)
				{ // beginning at the node corresponding to subclone with idx '0'
					traverseStateActionTrajBFS (currNode, null, leafTimeStepsAcrossAllTrajectories);
				}
				
			}
			
			// we are creating an arraylist of double[][] because we do NOT know the number of state-action trajectories (double [][] arrays) 
			// across all paths in sampleX's tree (note: however, we must specify a prefixed length length for each sa trajectory for CRP-IRL to work.)
			List<double[][]> listOfStateActionDoublesTrajectoriesForSampleX = new ArrayList<double[][]> ();
			
			// print out trajectory based on reverse-constructing each possible path from each
			// leaf-node
			for (TrajectoryTimeStep leafTS : leafTimeStepsAcrossAllTrajectories)
			{
				
				// This is a recursive function that will generate the double[] of states and
					// double[] of actions for a single path.
				printSTATEACTIONDOUBLETrajectory (leafTS, singleDBLTrajectoryOfStates, singleDBLTrajectoryOfActions, uuidToIntegerMAPforActions, 
												  uuidToIntegerMapForStatesInDB);
				if (!singleDBLTrajectoryOfStates.isEmpty ())
				{
					double[][] stateActionTraj = new double[2][_stateActionTrajLength];
					Double s_dbl = (double) 0;
					Double a_dbl = (double) 0;
					int lastIndexElementOfCurrentTraj = singleDBLTrajectoryOfStates.size () - 1;
					for (int i = 0; i < singleDBLTrajectoryOfStates.size (); i++)
					{
						s_dbl = singleDBLTrajectoryOfStates.get (i);
						a_dbl = singleDBLTrajectoryOfActions.get (i);
						// recall, the List<Double> singleDBLTrajectoryOfStates is appended to,
						// starting from the leaf node of the path(thus element 0 in the arraylist
						// is actually the last element in the path)
						// problem is that not all paths will have the maximum length =
						// _stateActionTrajLength
						stateActionTraj[0][lastIndexElementOfCurrentTraj - i] = s_dbl;
						stateActionTraj[1][lastIndexElementOfCurrentTraj - i] = a_dbl;
					}
					// add double[][] corresponding to a single s-a trajectory for the current sampleX
					listOfStateActionDoublesTrajectoriesForSampleX.add (stateActionTraj);
					singleDBLTrajectoryOfStates.clear ();
					singleDBLTrajectoryOfActions.clear ();
				}
			}
			System.out.println ("Inspecting sample " + sampleX);
			if (listOfStateActionDoublesTrajectoriesForSampleX.size () > 0)
			{
				List<double[][]> listOfStateActionSequenceDoublesFromNormalStartStateForSampleX;
				
				// Store the sample name that is associated with all of the subclonal trajectories.
				// We need these so we can iterate through stateactiontrajectories_table more
				// efficiently
				cqlInsertSampleNameStatement = "INSERT INTO wgssamplenames_table (samplename) values(\'" + sampleX + "\') ";
				// the action associated with a subclone is executed by the subclone before it (thus we need to associate the
				// action with previous subclone's stateINT)
				
				session.execute (cqlInsertSampleNameStatement);
				System.out.println ("ADDED sample " + sampleX + " to wgssamplenames_table");
				// ***NOTE: the List<double[][]> listOfStateActionDoublesTrajectoriesForSampleX is
				// the stateINT and actionINT extracted from subcloneX's FULLsubclone.X.vcf and
				// DRIVERS.subclone.X.vcf files, respectively. It does NOT actually correspond to
				// the state-action sequence that we claim is representative of a subclone's
				// evolution.
				listOfStateActionSequenceDoublesFromNormalStartStateForSampleX = saveSTATEACTIONDOUBLETrajectoriesForSingleSampleXToDB (
						listOfStateActionDoublesTrajectoriesForSampleX, sampleX, countMatricesMAP,
						insertPathsIntoDBDynamic);
				// System.out.println("Finished inserting each path into
				// stateactiontrajectories_table for sample "+sampleX);
				doublesStateAndActiontrajectoriesForAllSamplesMap.put (sampleX, 
																	   new ArrayList<> (listOfStateActionSequenceDoublesFromNormalStartStateForSampleX));
				
				listOfStateActionSequenceDoublesFromNormalStartStateForSampleX.clear ();
			}
			leafTimeStepsAcrossAllTrajectories.clear ();
			listOfStateActionDoublesTrajectoriesForSampleX.clear ();
		}
		
		// this Map of List<double[][]>objects could eventually be merged into a SINGLEList<double[][]> and used as input to CRP-IRL
		setDoublesStateAndActiontrajectoriesForAllSamplesMap (doublesStateAndActiontrajectoriesForAllSamplesMap);
		
		Map<Integer, Map<Integer, Integer>> nextPossibleStatesMAPPING = createMappingOfPossibleStatesForEachState (
				countMatricesMAP, numberOfStatesInMDP, numberOfActionsInMDP);
		setNextPossibleStatesMAPForAllStates (nextPossibleStatesMAPPING);
		// saveCountMatricesMAPtoCSV(countMatricesMAP, numberOfStatesInMDP);
		
		String truncateNextStateTransitionsTableCQLStatement = "TRUNCATE nextPossibleStatesMapping_table";
		session.execute (truncateNextStateTransitionsTableCQLStatement);
		
		String cqlMapQuery = "UPDATE nextPossibleStatesMapping_table SET nextTransitionsMAP = ? ";
		cqlMapQuery += "WHERE currentStateINT = ?";
		
		PreparedStatement preparedStatement = session.prepare (cqlMapQuery);
		Set<Integer> stateIntKeys = nextPossibleStatesMAPPING.keySet ();
		for (Integer stateINTj : stateIntKeys)
		{
			Map<Integer, Integer> nextPossibleTransitionMapForCurrentStateINTj = nextPossibleStatesMAPPING
					.get (stateINTj);
			BoundStatement boundStatement = preparedStatement
					.bind (nextPossibleTransitionMapForCurrentStateINTj, stateINTj);
			session.execute (boundStatement);
		}
		
	}
	
	
	/***
	 * 2nd TreeTraversal constructor, that builds countMatricesMAP in cassandra database and NOT
	 * in-memory Map (which could be causing garbage collection (GC) pauses)
	 * 
	 * @param clusterObj
	 * @param sessionObj
	 * @param inputPathForLISTOFAncestryRelationshipsFILEstring
	 * @param inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILEstring
	 * @param outPathDirString
	 *            traversalID: used to differentiate between instances of TreeTraversals that may
	 *            have been run/stored in DB; also added this field to differentiate this
	 *            constructor from original constructor which does not store countMatricesMAP in
	 *            database
	 * @throws IOException
	 */
	public TreeTraversal (Cluster clusterObj, Session sessionObj,
			String inputPathForLISTOFAncestryRelationshipsFILEstring,
			String inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILEstring,
			String outPathDirString, String traversalID, boolean insertPathsIntoDBDynamic)
			throws IOException
	{
		
		cluster = clusterObj;
		session = sessionObj;
		_inputPathForLISTOFAncestryRelationshipsFILEstr = inputPathForLISTOFAncestryRelationshipsFILEstring;
		_inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILEstr = inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILEstring;
		_outputDirPathStr = outPathDirString;
		
		String cqlSelectEnsemblGeneIDTranscriptIDSymbol_table = "select * from ensembl_gene_transcript_symbol_fordrivergenes";
		
		Map<Integer, String> intToGeneSymbol = new HashMap<Integer, String> ();
		Map<String, Integer> geneSymbolToInt = new HashMap<String, Integer> ();
		HashMultimap<String, String> geneSymbolToTIDs = HashMultimap.create ();
		Map<String, String> tidToGeneSymbol = new HashMap<String, String> ();
		
		Map<String, Integer> tidToMUTactionINTmap = new HashMap<String, Integer> ();
		Map<String, Integer> tidToMETHYLactionINTmap = new HashMap<String, Integer> ();
		
		Map<String, UUID> tidToMUTactionUUIDmap = new HashMap<String, UUID> ();
		Map<String, UUID> tidToMETHYLactionUUIDmap = new HashMap<String, UUID> ();
		
		Map<UUID, Integer> uuidToIntegerMAPforActions = new HashMap<UUID, Integer> ();
		
		for (Row row : session.execute (cqlSelectEnsemblGeneIDTranscriptIDSymbol_table))
		{
			// String rowAsString = row.toString();
//			String geneID = row.getString (0);	// GTD Not used
			String transcriptID = row.getString (1);
			String geneSymbol = row.getString (2);
			geneSymbolToTIDs.put (geneSymbol, transcriptID);
			tidToGeneSymbol.put (transcriptID, geneSymbol);
		}
		
		Set<String> driverGeneSymbolSet = geneSymbolToTIDs.keySet ();
		
		int gsymbolCount = 1; // We start at actionINT=1 as opposed to zero, actionINT=0 will be
								// reserved for padding purposes.
		for (String gSymbol : driverGeneSymbolSet)
		{
			intToGeneSymbol.put (gsymbolCount, gSymbol);
			geneSymbolToInt.put (gSymbol, gsymbolCount);
			
			gsymbolCount++;
			
		}
		int totalNumDriverGeneSymbolsWithValidtranscripts = driverGeneSymbolSet.size ();
		
		// int actionINTcount =1; //start the actionINT space with 1 instead of zero. This will
		// allows us to fill to pad paths with length < _stateActionTrajLength with zero's
		String truncateActionSpaceTableWithNewActionsCQLStatement = "TRUNCATE actionspace_table";
		session.execute (truncateActionSpaceTableWithNewActionsCQLStatement);
		for (int geneInt : intToGeneSymbol.keySet ())
		{
			String geneSymbol = intToGeneSymbol.get (geneInt);
			
			UUID mutactionuid = UUID.randomUUID ();
			UUID methylationuid = UUID.randomUUID ();
			// int actionINTForMUT = actionINTcount; //the genetINT is actually better
			// int actionINTForMETHYL =
			// actionINTcount+totalNumDriverGeneSymbolsWithValidtranscripts;
			int actionINTForMUT = geneInt;
			int actionINTForMETHYL = geneInt + totalNumDriverGeneSymbolsWithValidtranscripts;
			uuidToIntegerMAPforActions.put (mutactionuid, actionINTForMUT);
			uuidToIntegerMAPforActions.put (methylationuid, actionINTForMETHYL);
			
			Set<String> transcriptIDsSet = geneSymbolToTIDs.get (geneSymbol);
			
			for (String tid : transcriptIDsSet)
			{
				tidToMUTactionUUIDmap.put (tid, mutactionuid);
				tidToMETHYLactionUUIDmap.put (tid, methylationuid);
				tidToMUTactionINTmap.put (tid, Integer.valueOf (actionINTForMUT));
				tidToMETHYLactionINTmap.put (tid, Integer.valueOf (actionINTForMETHYL));
			}
			
			// NOTE: because some gene symbols contain a hyphen in their names, which causes an
			// error in cassandra, they must be enclosed in double quotes.
			// cql statement (<nameOfColumnA>, <nameOfColumnC>,...) values(<value to add to
			// columnA>, <value to add to columnC>,...)
			String insertMUTAction = "INSERT INTO actionspace_table (actionuuid,\""
					+ geneSymbol + "\",actionint, isMutation, isMethylation) values(" + mutactionuid
					+ ",true," + actionINTForMUT + ", true, false)";
			String insertMETHYLAction = "INSERT INTO actionspace_table (actionuuid,\""
					+ geneSymbol + "\",actionint, isMutation, isMethylation) values("
					+ methylationuid + ",true," + actionINTForMETHYL + ", false, true)";
			
			session.execute (insertMUTAction);
			session.execute (insertMETHYLAction);
			
			// actionINTcount++;
		}
		
		/////////////////////////////// Add padding action actionINT=0
		/////////////////////////////// (action0)//////////////////////
		UUID paddingAction0uid = UUID.randomUUID ();
		int paddingaction0INT = 0;
		String insertPaddingAction0 = "INSERT INTO actionspace_table (actionuuid,actionint, isMutation, isMethylation) values("
				+ paddingAction0uid + "," + paddingaction0INT + ", false, false)";
		session.execute (insertPaddingAction0);
		/////////////////////////////////////////////////////////////////////////////////////////////
		
		// Create mapping based on Cassandra db state_space table (based on GLFM created
		// states/patterns)
		Map<UUID, Integer> uuidToIntegerMapForStatesInDB = new HashMap<UUID, Integer> ();
		UUID stateUUIDinDB = null;
		Integer stateINTinDB = 0;
		Long stateINT_longval_inDB = (long) 0;
		
		String cqlGetStateRowsStatement = "select * from statespace_table";
		ResultSet rsStates = session.execute (cqlGetStateRowsStatement);
		for (Row r_state : rsStates)
		{
			stateUUIDinDB = r_state.getUUID ("stateuuid");
			stateINT_longval_inDB = r_state.getLong ("stateint");
			
			stateINTinDB = stateINT_longval_inDB.intValue ();
			uuidToIntegerMapForStatesInDB.put (stateUUIDinDB, stateINTinDB);
		}
		
		String truncateStateActionTrajTableCQLStatement = "TRUNCATE stateactiontrajectories_table";
		session.execute (truncateStateActionTrajTableCQLStatement);
		
		String truncateSampleNameTableCQLStatement = "TRUNCATE wgssamplenames_table";
		session.execute (truncateSampleNameTableCQLStatement);
		
		String truncateTrajCountsTableCQLStatement = "TRUNCATE sampleTrajCounts_table";
		session.execute (truncateTrajCountsTableCQLStatement);
		
		// Create stateactiontrajectories_table with appropriate number of columns = number of
		// state-action pairs in each trajectory
		// NOTE: this will cause an error if these columns already exist
		// String cqlCreateStateActionTrajectoriesTableStatement = "create TABLE
		// stateactiontrajectories_table(trajuuid UUID PRIMARY KEY, sampleName text)";
		// for(int l=0; l< _stateActionTrajLength; l++) {
		// String cqlADDStateColToTrajectoriesTableStatement = "alter TABLE
		// stateactiontrajectories_table ADD state"+l+" bigint";
		// String cqlADDActionColToTrajectoriesTableStatement = "alter TABLE
		// stateactiontrajectories_table ADD action"+l+" bigint";
		// session.execute(cqlADDStateColToTrajectoriesTableStatement);
		// session.execute(cqlADDActionColToTrajectoriesTableStatement);
		// }
		
		// These should be inputted as arguments to the constructor!
		// String inputPathForLISTOFAncestryRelationshipsFILEstr =
		// "/Users/m186806/tIDFilteredDriverOnlyVCFs/allTIDFilteredSUBFILES/listOfAncestryFiles.txt";
		// //text file specifies the path to the AncestryRelationship .txt file specific the
		// relationship between subclone idxs
		// String inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILEstr =
		// "/Users/m186806/tIDFilteredDriverOnlyVCFs/allTIDFilteredSUBFILES/listOfSampleSpecificDRIVERSSubcloneVCFsDirectories.txt";
		// // //text file specifiying the path to the directory containing all DRIVER-EDGE subvcfs
		// corresponding to the actions leading to the subclones for a given tumor-sample
		
		leafTimeStepsAcrossAllTrajectories = new ArrayList<TrajectoryTimeStep> ();
		
		// Initialize the 2 transition matrices: state transition: T(s',s,a) = P(s'|s,a)
		// _transitionmatrix models the probability of the next state (row), given current state
		// (col) for action a_i (Map Integer index)
		// _discountedTransitonMatrix models the probability of the next state (col), given the
		// current state(row) for action a_I (Map Integer index)
		
		/// COUNT number of ACTIONS in actionspace_table
		String cqlCountNumACTIONSInActionSpaceStatement = "select count(*) from actionspace_table";
		ResultSet rsActionsCOUNT = session.execute (cqlCountNumACTIONSInActionSpaceStatement);
		Long numberOfActionsInDB = (long) 0;
		int numberOfActionsInMDP = 0;
		for (Row r_a : rsActionsCOUNT)
		{
			numberOfActionsInDB = r_a.getLong (0);
		}
		// numberOfActionsInMDP = numberOfActionsInDB.intValue()+1; //although 0 is not included in
		// action-space(because it is not explicity assigned to any gene-symbol) it is a DEFAULT
		// padding action
		numberOfActionsInMDP = numberOfActionsInDB.intValue (); // this is only true if we DID create an actionINT=0 in actionspace_table
																// that corresponds to the padding action
		setNumActionsInMDP (numberOfActionsInMDP);
		////////////////////////////////////////////////
		
		/// Count number of STATES in statespace_table
		String cqlCountNumSTATESInStateSpaceStatement = "select count(*) FROM statespace_table";
		ResultSet rsStatesCOUNT = session.execute (cqlCountNumSTATESInStateSpaceStatement);
		Long numberOfStatesInDB = (long) 0;
		int numberOfStatesInMDP = 0;
		for (Row r_s : rsStatesCOUNT)
		{
			numberOfStatesInDB = r_s.getLong (0);
		}
		// numberOfStatesInMDP = numberOfStatesInDB.intValue()+1; //although 0 is not included in
		// state-space(because it is explicity associated with the NORMAL STATE) it is a DEFAULT
		// padding state
		numberOfStatesInMDP = numberOfStatesInDB.intValue ();
		setNumStatesInMDP (numberOfStatesInMDP);
		///////////////////////////////////////////////////
		
		// JK added 5.1.2019: we should reset treetraversalcountmatricesmap_table in Cassandra
		// everytime we are going to run tree traversal
		String truncateCountMatricesMAPTableCQLStatement = "TRUNCATE treetraversalcountmatricesmap_table";
		session.execute (truncateCountMatricesMAPTableCQLStatement);
		
		// XXX: countMatricesMAP will be stored as a table in Cassandra database instead; this
		// should ease working memory GC requirements during tree-traversal
		// //Initialize map of count-matrices; one for each possible action
		// Map<Integer, double[][]> countMatricesMAP = new HashMap<Integer, double[][]>();
		// double[][] countMatrixfor_ai;
		// for(int a=0; a< numberOfActionsInMDP; a++) {
		// countMatrixfor_ai =new double[numberOfStatesInMDP][numberOfStatesInMDP]; //recall, number
		// of states in statespace does not include stateINT 0, which is reserved as a default value
		// for padding stateActionTrajectories based on subclonal paths which have lengths <
		// _stateActionTrajLength
		// countMatricesMAP.put(Integer.valueOf(a), countMatrixfor_ai);
		// }
		
		////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////
		
		////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////
		
//		BufferedOutputStream bos = null;	// GTD Not used
		
		System.out.println ("******Running iterateStateNodesInTreesFromFULLSUBCLONEAncestryFiles()\n");
		Map<String, MutablePair<Queue<int[]>, List<TreeTraversal.Node>>> specificPathsTraversedInEachSampleTreeMap = 
				iterateStateNodesInTreesFromFULLSUBCLONEAncestryFiles (_inputPathForLISTOFAncestryRelationshipsFILEstr);
		
		System.out.println ("******Running iterateActionEdgesInTreesFromDRIVERSSUBCLONEAncestryFiles()\n");
		iterateActionEdgesInTreesFromDRIVERSSUBCLONEAncestryFiles (
				_inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILEstr,
				tidToMUTactionUUIDmap, tidToMETHYLactionUUIDmap,
				specificPathsTraversedInEachSampleTreeMap, true, false);
		
		Map<String, List<double[][]>> doublesStateAndActiontrajectoriesForAllSamplesMap = new HashMap<String, List<double[][]>> ();
		/// JK 3.1.2019
		/// this hashmap can eventually be turned into the List<double[][]> used by the CRP-IRL as input trajectories
		
		List<Double> singleDBLTrajectoryOfStates = new ArrayList<Double> ();
		List<Double> singleDBLTrajectoryOfActions = new ArrayList<Double> ();
		String cqlInsertSampleNameStatement = "";
		
		/// JK: creating state-action trajectories
		System.out.println ("******Create initial state-action trajectories\n");
		for (String sampleX : specificPathsTraversedInEachSampleTreeMap.keySet ())
		{
			MutablePair<Queue<int[]>, List<TreeTraversal.Node>> traversedPathsPair = specificPathsTraversedInEachSampleTreeMap.get (sampleX);
			List<TreeTraversal.Node> treeXNodes = traversedPathsPair.right; // the left item of the
																			// MutablePair is the
																			// queue of int []'s
																			// corresponding to the
																			// various paths in
																			// treeX
			
			for (Node currNode : treeXNodes)
			{
				
				if (currNode.Id == 0)
				{ // beginning at the node corresponding to subclone with idx '0'
					traverseStateActionTrajBFS (currNode, null, leafTimeStepsAcrossAllTrajectories);
				}
				
			}
			
			List<double[][]> listOfStateActionDoublesTrajectoriesForSampleX = new ArrayList<double[][]> (); // we
																											// are
																											// creating
																											// an
																											// arraylist
																											// of
																											// double[][]
																											// because
																											// we
																											// do
																											// NOT
																											// know
																											// the
																											// number
																											// of
																											// state-action
																											// trajectories
																											// (double
																											// [][]
																											// arrays)
																											// across
																											// all
																											// paths
																											// in
																											// sampleX's
																											// tree
																											// (note:
																											// however,
																											// we
																											// must
																											// specify
																											// a
																											// prefixed
																											// length
																											// length
																											// for
																											// each
																											// sa
																											// trajectory
																											// for
																											// CRP-IRL
																											// to
																											// work.)
			
			// print out trajectory based on reverse-constructing each possible path from each
			// leaf-node
			for (TrajectoryTimeStep leafTS : leafTimeStepsAcrossAllTrajectories)
			{
				
				// This is a recursive function that will generate the double[] of states and
					// double[] of actions for a single path.
				printSTATEACTIONDOUBLETrajectory (leafTS, singleDBLTrajectoryOfStates,
						singleDBLTrajectoryOfActions, uuidToIntegerMAPforActions,
						uuidToIntegerMapForStatesInDB);
				if (singleDBLTrajectoryOfStates.size () > 0)
				{
					double[][] stateActionTraj = new double[2][_stateActionTrajLength];
					Double s_dbl = (double) 0;
					Double a_dbl = (double) 0;
					int lastIndexElementOfCurrentTraj = singleDBLTrajectoryOfStates.size () - 1;
					for (int i = 0; i < singleDBLTrajectoryOfStates.size (); i++)
					{
						s_dbl = singleDBLTrajectoryOfStates.get (i);
						a_dbl = singleDBLTrajectoryOfActions.get (i);
						// recall, the List<Double> singleDBLTrajectoryOfStates is appended to,
						// starting from the leaf node of the path(thus element 0 in the arraylist
						// is actually the last element in the path)
						// problem is that not all paths will have the maximum length =
						// _stateActionTrajLength
						stateActionTraj[0][lastIndexElementOfCurrentTraj - i] = s_dbl;
						stateActionTraj[1][lastIndexElementOfCurrentTraj - i] = a_dbl;
					}
					listOfStateActionDoublesTrajectoriesForSampleX.add (stateActionTraj); // add
																							// double[][]
																							// corresponding
																							// to a
																							// single
																							// s-a
																							// trajectory
																							// for
																							// the
																							// current
																							// sampleX
					singleDBLTrajectoryOfStates.clear ();
					singleDBLTrajectoryOfActions.clear ();
				}
			}
			System.out.println ("Inspecting sample " + sampleX);
			if (listOfStateActionDoublesTrajectoriesForSampleX.size () > 0)
			{
				
				List<double[][]> listOfStateActionSequenceDoublesFromNormalStartStateForSampleX;
				
				// Store the samplename that is associated with all of the subclonal trajectories.
				// We need these so we can iterate through stateactiontrajectories_table more
				// efficiently
				cqlInsertSampleNameStatement = "INSERT INTO wgssamplenames_table (samplename) values(\'"
						+ sampleX + "\') "; // the action associated with a subclone is executed by
											// the subclone before it (thus we need to associate the
											// action with previous subclone's stateINT)
				
				session.execute (cqlInsertSampleNameStatement);
				System.out.println ("ADDED sample " + sampleX + " to wgssamplenames_table");
				// ***NOTE: the List<double[][]> listOfStateActionDoublesTrajectoriesForSampleX is
				// the stateINT and actionINT extracted from subcloneX's FULLsubclone.X.vcf and
				// DRIVERS.subclone.X.vcf files, respectively. It does NOT actually correspond to
				// the state-action sequence that we claim is representative of a subclone's
				// evolution.
				listOfStateActionSequenceDoublesFromNormalStartStateForSampleX = saveSTATEACTIONDOUBLETrajectoriesAndCountMatricesMapForSingleSampleXToDB (
						listOfStateActionDoublesTrajectoriesForSampleX, sampleX,
						insertPathsIntoDBDynamic);
				// System.out.println("Finished inserting each path into stateactiontrajectories_table for sample "+sampleX);
				doublesStateAndActiontrajectoriesForAllSamplesMap.put (sampleX, 
						new ArrayList<> (listOfStateActionSequenceDoublesFromNormalStartStateForSampleX));
				
				listOfStateActionSequenceDoublesFromNormalStartStateForSampleX.clear ();
			}
			leafTimeStepsAcrossAllTrajectories.clear ();
			listOfStateActionDoublesTrajectoriesForSampleX.clear ();
		}
		
		setDoublesStateAndActiontrajectoriesForAllSamplesMap (doublesStateAndActiontrajectoriesForAllSamplesMap);
		// this Map of List<double[][]> objects could eventually be merged into a SINGLE List<double[][]> and used as input to CRP-IRL
		
		Map<Integer, Map<Integer, Integer>> nextPossibleStatesMAPPING = 
				createMappingOfPossibleStatesForEachStateWithDatabase (numberOfStatesInMDP, numberOfActionsInMDP);
		setNextPossibleStatesMAPForAllStates (nextPossibleStatesMAPPING);
		// saveCountMatricesMAPtoCSV(countMatricesMAP, numberOfStatesInMDP);
		
		String truncateNextStateTransitionsTableCQLStatement = "TRUNCATE nextPossibleStatesMapping_table";
		session.execute (truncateNextStateTransitionsTableCQLStatement);
		
		// store the Map of HashMaps 'nextPossibleStatesMAPPING' as the cassandra table where
		// the key = currentStateINT and the value= a SET of pairs actionINT:nextStateINT
		String cqlMapQuery = "UPDATE nextPossibleStatesMapping_table SET nextTransitionsMAP = ? ";
		cqlMapQuery += "WHERE currentStateINT = ?";
		
		PreparedStatement preparedStatement = session.prepare (cqlMapQuery);
		Set<Integer> stateIntKeys = nextPossibleStatesMAPPING.keySet ();
		for (Integer stateINTj : stateIntKeys)
		{
			Map<Integer, Integer> nextPossibleTransitionMapForCurrentStateINTj = nextPossibleStatesMAPPING
					.get (stateINTj);
			BoundStatement boundStatement = preparedStatement
					.bind (nextPossibleTransitionMapForCurrentStateINTj, stateINTj);
			session.execute (boundStatement);
		}
		
	}
	/////////////// end of 2nd TreeTraversal() constructor
	
	
	public static void printTrajectory (TrajectoryTimeStep ts, List<String> traj)
	{
		traj.add ("Action:" + ts.currentActionuid.toString ());
		if (ts.previous != null)
		{
			printTrajectory (ts.previous, traj);
		}
	}
	
	
	public static void printACTIONINTEGERTrajectory (TrajectoryTimeStep ts, List<Integer> traj,
			Map<UUID, Integer> uuidToIntMapForActions)
	{
		
		UUID currentActionAsUUID = ts.currentActionuid;
//		UUID currentStateAsUUID = ts.currentStateuid;	// GTD Not used
		Integer currActionAsINT = uuidToIntMapForActions.get (currentActionAsUUID);
		traj.add (currActionAsINT);
		if (ts.previous != null)
		{
			printACTIONINTEGERTrajectory (ts.previous, traj, uuidToIntMapForActions);
		}
	}
	
	
	public static void printSTATEACTIONDOUBLETrajectory (TrajectoryTimeStep ts,
			List<Double> statetraj, List<Double> actiontraj,
			Map<UUID, Integer> uuidToIntMapForActions,
			Map<UUID, Integer> uuidToIntMapForStates)
	{
		
		UUID currentActionAsUUID = ts.currentActionuid;
		UUID currentStateAsUUID = ts.currentStateuid;
		Integer currActionAsINT = uuidToIntMapForActions.get (currentActionAsUUID);
		Integer currStateAsINT = uuidToIntMapForStates.get (currentStateAsUUID);
		if (ts.currentStateuid != null)
		{ // if the current subclone has not been assigned a stateuuid{by Node() constructor called
			// in generatePathsInTree()}from treepaths_table, this means that this
			// subclone vcf file was not yet processed by GLFM and assigned a state (remember this
			// table only contains subclones for which the GLFM function has processed and found
			// basis vectors and assigned to a state/pattern)
			actiontraj.add ((double) currActionAsINT);
			statetraj.add ((double) currStateAsINT);
		}
		if (ts.previous != null)
		{
			printSTATEACTIONDOUBLETrajectory (ts.previous, statetraj, actiontraj,
					uuidToIntMapForActions, uuidToIntMapForStates);
		}
	}
	
	
	public static void printoutSTRINGTrajectoriesForSampleToTextFile (
			BufferedOutputStream buff_output_stream,
			Map<String, List<List<String>>> strtrajectoriesForAllSamplesMAP)
			throws FileNotFoundException, IOException
	{
		/// Print out each trajectory to text file
		for (String sampleX : strtrajectoriesForAllSamplesMAP.keySet ())
		{
			try
			{
				System.out.println ("try statement iterating through strtrajectoriesForAllSamplesMap currently at sample : " + sampleX);
				List<List<String>> listOfTrajsForSampleX = strtrajectoriesForAllSamplesMAP.get (sampleX);
				
				File outFile = new File (sampleX + ".actionTrajectories");
				buff_output_stream = new BufferedOutputStream (new FileOutputStream (outFile));
				
//				int l;	// GTD Not used
//				byte[] buffer = new byte[1024 * 8];	// GTD Not used
				for (List<String> traj : listOfTrajsForSampleX)
				{
					for (String actionStr : traj)
					{
						UUID origUUIDAction = UUID.fromString (actionStr);
						byte[] actionAsByteArray = UuidUtils.asBytes (origUUIDAction);
						buff_output_stream.write (actionAsByteArray);
					}
				}
				
				System.out.println ("wrote out all trajectories for sample : " + sampleX + " into " + outFile);
				
			}
			finally
			{
				if (buff_output_stream != null)
				{
					buff_output_stream.close ();
				}
				
			}
		}
	}
	
	
	public static void printoutINTEGERTrajectoriesForALLSamplesINMapToTextFile (
			BufferedOutputStream buff_output_stream,
			Map<String, List<List<Integer>>> integertrajectoriesForAllSamplesMAP)
			throws FileNotFoundException, IOException
	{
		/// Print out each trajectory to text file
		for (String sampleX : integertrajectoriesForAllSamplesMAP.keySet ())
		{
			try
			{
				System.out.println ("try statement iterating through strtrajectoriesForAllSamplesMap currently at sample : " + sampleX);
				List<List<Integer>> listOfINTTrajsForSampleX = integertrajectoriesForAllSamplesMAP.get (sampleX);
				
				File outFile = new File (sampleX + ".actionTrajectories");
				buff_output_stream = new BufferedOutputStream (new FileOutputStream (outFile));
				
//				int l;	// GTD Not used
//				byte[] buffer = new byte[1024 * 8];	// GTD Not used
				for (List<Integer> traj : listOfINTTrajsForSampleX)
				{
					for (Integer actionINTEGER : traj)
					{
						int action_int = actionINTEGER.intValue ();
						buff_output_stream.write (action_int);
					}
				}
				
				System.out.println ("wrote out all trajectories for sample : " + sampleX + " into " + outFile);
				
			}
			finally
			{
				if (buff_output_stream != null)
				{
					buff_output_stream.close ();
				}
				
			}
		}
	}
	
	
	public static void printoutACTIONINTEGERTrajectoriesForSingleSampleXToTextFile (
			BufferedOutputStream buff_output_stream, List<List<Integer>> sampleXIntTrajectories,
			String sampleXName) throws FileNotFoundException, IOException
	{
		/// Print out each trajectory for sampleX to text file
		
		System.out.println ("try statement iterating through strtrajectoriesForAllSamplesMap currently at sample : " + sampleXName);
		
		// File outFile = new File(sampleXName+".actionTrajectories");
		// buff_output_stream = new BufferedOutputStream(new FileOutputStream(outFile));
		
//		int l;	// GTD Not used
//		byte[] buffer = new byte[1024 * 8];	// GTD Not used
		int trajCounter = 0;
		for (List<Integer> traj : sampleXIntTrajectories)
		{
			try
			{
				File outFile = new File (sampleXName + ".actionTrajectories" + trajCounter);
				buff_output_stream = new BufferedOutputStream (new FileOutputStream (outFile));
				for (Integer actionINTEGER : traj)
				{
					int action_int = actionINTEGER.intValue ();
					buff_output_stream.write (action_int);
				}
				trajCounter++;
			}
			finally
			{
				if (buff_output_stream != null)
				{
					buff_output_stream.close ();
				}
			}
			if (trajCounter > 10)
			{ // stop prematurely for debugging purposes
				break;
			}
		}
		
		// System.out.println("wrote out all trajectories for sample : "+sampleXName+" into
		// "+outFile);
		
	}
	
	
	public static List<double[][]> saveSTATEACTIONDOUBLETrajectoriesForSingleSampleXToDB (
			List<double[][]> sampleXDBLTrajectories, String sampleXName,
			Map<Integer, double[][]> countMatricesMap, boolean insertPathsIntoDBDynamically)
	{
		/// insert each trajectory for sampleX to cassandra db table
		double state_dbl = (double) 0;
		double action_dbl = (double) 0;
		int state_asint = 0;
		int action_asint = 0;
		
		int trajCounter = 0;
		String cqlInsertSAPairStatement = "";
//		String cqlUploadCSVPathsToDB_Statement = "";	// GTD Not used
		
		double[][] countMatrixfor_actioni;
		List<double[][]> sampleXDoubleTrajectoriesListFromNormalStart = new ArrayList<double[][]> ();
		// cqlInsertSAPairStatement=cqlInsertSAPairStatement+"BEGIN UNLOGGED BATCH";
		
		// Batch batch = QueryBuilder.unloggedBatch();
		
		for (double[][] traj : sampleXDBLTrajectories)
		{
			
			double[][] stateActionSequenceFromNormalStartState = new double[2][_stateActionTrajLength];
			
			UUID sa_trajUUID = UUID.randomUUID ();
			
			int trajectoryLength = traj[0].length;
			int curr_state_asint = 1; // Assume all trajectories begin at NORMAL STATE which has
										// stateINT=1
			
			if (trajectoryLength == _stateActionTrajLength)
			{
				for (int t = 0; t < _stateActionTrajLength; t++)
				{
					state_dbl = traj[0][t];
					action_dbl = traj[1][t];
					
					state_asint = (int) state_dbl;
					action_asint = (int) action_dbl;
					
					countMatrixfor_actioni = countMatricesMap.get (action_asint);
					countMatrixfor_actioni[curr_state_asint][state_asint] = countMatrixfor_actioni[curr_state_asint][state_asint] + 1;
					countMatricesMap.put (action_asint, countMatrixfor_actioni);
					
					if (insertPathsIntoDBDynamically)
					{
						// original insert statement
						// the action associated with a subclone is executed by the subclone before it (thus we need to associate the 
						// action with previous subclone's stateINT)
						cqlInsertSAPairStatement = "INSERT INTO stateactiontrajectories_table (trajuuid,state" + t + ",action" + t + 
													",samplename) values(" + sa_trajUUID + "," + curr_state_asint + "," + action_asint + ",\'" + 
													sampleXName + "\')";
						// System.out.println(cqlInsertSAPairStatement);
//						session.execute (cqlInsertSAPairStatement);
						
						session.execute (new SimpleStatement (cqlInsertSAPairStatement).setReadTimeoutMillis (65000));
						
						// batch insert via appending to string
						// cqlInsertSAPairStatement = cqlInsertSAPairStatement+" INSERT INTO
						// stateactiontrajectories_table
						// (trajuuid,state"+t+",action"+t+",samplename)
						// values("+sa_trajUUID+","+curr_state_asint+","+
						// action_asint+",\'"+sampleXName+"\');"; //the action associated with a
						// subclone is executed by the subclone before it (thus we need to associate
						// the action with previous subclone's stateINT)
					}
					
					stateActionSequenceFromNormalStartState[0][t] = curr_state_asint; // state at
																						// time t
					stateActionSequenceFromNormalStartState[1][t] = action_asint; // action at time
																					// t
					
					// JK: NEXT-state becomes 'current' state for next time-step in trajectory
					curr_state_asint = state_asint;
				}
				
				sampleXDoubleTrajectoriesListFromNormalStart.add (stateActionSequenceFromNormalStartState);
				trajCounter++;
			} // closes if statement if(trajectoryLength == _stateActionTrajLength)
			else
			{
				System.out.println ("irregular length trajectory has length =" + trajectoryLength);
			}
			// JK 3.19.2019: Commented out in order to see full memory load for storing ALL expanded
			// paths
			// if (trajCounter >10) { //stop prematurely for debugging purposes
			// break;
			// }
		}
		saveTrajectoriestoCSV (sampleXName, sampleXDoubleTrajectoriesListFromNormalStart);
		System.out.println ("Adding " + trajCounter + " trajectories via batch insert.....");
		
		// This never worked because COPY is a cqlsh shell command and not a standard CQL that the
		// datastax java driver recognizes.
		// if(!insertPathsIntoDBDynamically) {
		// cqlUploadCSVPathsToDB_Statement = "copy stateactiontrajectories_table
		// (samplename,
		// trajuuid,state0,action0,state1,action1,state2,action2,state3,action3,state4,action4) FROM
		// \'"+_outputDirPathStr+"/"+sampleXName+".TRAJECTORIES.csv\' with HEADER=false";
		// session.execute(cqlUploadCSVPathsToDB_Statement);
		// }
		
		String cqlInsertTrajCountStatement = "INSERT INTO sampleTrajCounts_table (trajcount, samplename) values(" + 
											  trajCounter + ",\'" + sampleXName + "\')";
		// System.out.println(cqlInsertTrajCountStatement);
		session.execute (cqlInsertTrajCountStatement);
		
		// cqlInsertSAPairStatement = cqlInsertSAPairStatement+"APPLY BATCH;";
		// session.execute(cqlInsertSAPairStatement);
		return sampleXDoubleTrajectoriesListFromNormalStart;
		
	}
	
	
	/**
	 * Alternative to saveSTATEACTIONDOUBLETrajectoriesForSingleSampleXToDB() to store
	 * countMatricesMAP in cassandra table instead of in-memory
	 * 
	 * @param sampleXDBLTrajectories
	 * @param sampleXName
	 * @return
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static List<double[][]> saveSTATEACTIONDOUBLETrajectoriesAndCountMatricesMapForSingleSampleXToDB (
			List<double[][]> sampleXDBLTrajectories, String sampleXName,
			boolean insertPathsIntoDBDynamically) throws FileNotFoundException, IOException
	{
		/// insert each trajectory for sampleX to cassandra db table
		double state_dbl = (double) 0;
		double action_dbl = (double) 0;
		int state_asint = 0;
		int action_asint = 0;
		
		int trajCounter = 0;
		String cqlInsertSAPairStatement = "";
		
//		String cqlUploadCSVPathsToDB_Statement = "";	// GTD Not used
		
		// double [][] countMatrixfor_actioni;
		List<double[][]> sampleXDoubleTrajectoriesListFromNormalStart = new ArrayList<double[][]> ();
		// cqlInsertSAPairStatement=cqlInsertSAPairStatement+"BEGIN UNLOGGED BATCH";
		
		// Batch batch = QueryBuilder.unloggedBatch();
		
		// Long countVal =null;
		
		for (double[][] traj : sampleXDBLTrajectories)
		{
			
			double[][] stateActionSequenceFromNormalStartState = new double[2][_stateActionTrajLength];
			
			UUID sa_trajUUID = UUID.randomUUID ();
			
			int trajectoryLength = traj[0].length;
			int curr_state_asint = 1; // Assume all trajectories begin at NORMAL STATE which has
										// stateINT=1
			
			if (trajectoryLength == _stateActionTrajLength)
			{
				for (int t = 0; t < _stateActionTrajLength; t++)
				{
					state_dbl = traj[0][t];
					action_dbl = traj[1][t];
					
					state_asint = (int) state_dbl;
					action_asint = (int) action_dbl;
					
					// JK 5.2.2019 :replaced Map countMatricesMAP with database table
					// update treetraversalcountmatricesmap_table set count=count+1
					// where actionint=8 AND currentstateint=1 AND nextstateint=1;
					String cqlUpdateCountMatricesMapTableStatement = "update treetraversalcountmatricesmap_table " + 
																	 "set count=count+1 where actionINT=" + action_asint + 
																	 " AND currentStateINT=" + curr_state_asint + " AND nextStateINT=" + state_asint;
					session.execute (cqlUpdateCountMatricesMapTableStatement);
					
					// select count from treetraversalcountmatricesmap_table where
					// actionint=8;
//					String cqlGetCountStatement = "select count FROM treetraversalcountmatricesmap_table where actionINT="
//							+ action_asint + " AND currentStateINT=" + curr_state_asint
//							+ " AND nextStateINT=" + state_asint;	// GTD Not used
					
					// JK: we never need to actually see what the count value is in real-time
					// for (Row row: session.execute(cqlGetCountStatement)) {
					// //String rowAsString = row.toString();
					// countVal = row.getLong("count");
					// System.out.println(countVal);
					// //System.out.println(rowAsString);
					// }
					
					if (insertPathsIntoDBDynamically)
					{
						// original insert statement
						cqlInsertSAPairStatement = "INSERT INTO stateactiontrajectories_table (trajuuid,state"
								+ t + ",action" + t + ",samplename) values(" + sa_trajUUID + ","
								+ curr_state_asint + "," + action_asint + ",\'" + sampleXName
								+ "\')"; // the action associated with a subclone is executed by the
											// subclone before it (thus we need to associate the
											// action with previous subclone's stateINT)
						// System.out.println(cqlInsertSAPairStatement);
						session.execute (cqlInsertSAPairStatement);
						
						// batch insert via appending to string
						// cqlInsertSAPairStatement = cqlInsertSAPairStatement+" INSERT INTO
						// stateactiontrajectories_table
						// (trajuuid,state"+t+",action"+t+",samplename)
						// values("+sa_trajUUID+","+curr_state_asint+","+
						// action_asint+",\'"+sampleXName+"\');"; //the action associated with a
						// subclone is executed by the subclone before it (thus we need to associate
						// the action with previous subclone's stateINT)
					}
					
					stateActionSequenceFromNormalStartState[0][t] = curr_state_asint; // state at
																						// time t
					stateActionSequenceFromNormalStartState[1][t] = action_asint; // action at time
																					// t
					
					// JK: NEXT-state becomes 'current' state for next time-step in trajectory
					curr_state_asint = state_asint;
				}
				
				sampleXDoubleTrajectoriesListFromNormalStart
						.add (stateActionSequenceFromNormalStartState);
				trajCounter++;
			} // closes if statement if(trajectoryLength == _stateActionTrajLength)
			else
			{
				System.out.println ("irregular length trajectory has length =" + trajectoryLength);
			}
			// JK 3.19.2019: Commented out in order to see full memory load for storing ALL expanded
			// paths
			// if (trajCounter >10) { //stop prematurely for debugging purposes
			// break;
			// }
		}
		saveTrajectoriestoCSV (sampleXName, sampleXDoubleTrajectoriesListFromNormalStart);
		System.out.println ("Adding " + trajCounter + " trajectories via batch insert.....");
		
		// This never worked because COPY is a cqlsh shell command and not a standard CQL that the
		// datastax java driver recognizes.
		// if(!insertPathsIntoDBDynamically) {
		// cqlUploadCSVPathsToDB_Statement = "COPY stateactiontrajectories_table
		// (samplename,
		// trajuuid,state0,action0,state1,action1,state2,action2,state3,action3,state4,action4) FROM
		// \'"+_outputDirPathStr+"/"+sampleXName+".TRAJECTORIES.csv\' with HEADER=false;";
		// session.execute(cqlUploadCSVPathsToDB_Statement);
		// }
		
		String cqlInsertTrajCountStatement = "INSERT INTO sampleTrajCounts_table (trajcount, samplename) values("
				+ trajCounter + ",\'" + sampleXName + "\')";
		// System.out.println(cqlInsertTrajCountStatement);
		session.execute (cqlInsertTrajCountStatement);
		
		// cqlInsertSAPairStatement = cqlInsertSAPairStatement+"APPLY BATCH;";
		// session.execute(cqlInsertSAPairStatement);
		return sampleXDoubleTrajectoriesListFromNormalStart;
		
	}
	
	
	public static void readInByteTrajectoryFile (File fName) throws FileNotFoundException, IOException
	{
		String					sampleXName = fName.getName ();
		BufferedOutputStream	buff_output_stream = null;
		BufferedInputStream		bis = null;
		
		try
		{
			System.out.println ("try statement iterating through strtrajectoriesForAllSamplesMap currently at sample : " + sampleXName);
			
			File outFile = new File (sampleXName + ".INTactionTrajectories");
			buff_output_stream = new BufferedOutputStream (new FileOutputStream (outFile));
			bis = new BufferedInputStream (new FileInputStream (fName));
			
			int l;
			byte[] buffer = new byte[1024 * 8];
			try (Writer w = new OutputStreamWriter (buff_output_stream, "UTF-8"))
			{
				while ((l = bis.read (buffer)) > -1)
				{
					Integer lasint = Integer.valueOf (l);
					String lasString = lasint.toString ();
					System.out.println (lasString);
					
					w.write (lasString);
				}
				w.close ();
				
			}
			
			System.out.println ("wrote out byte-trajectory in terms of ints for sample : " + sampleXName + " into " + outFile);
			
		}
		finally
		{
			if (buff_output_stream != null)
			{
				buff_output_stream.close ();
			}
			if (bis != null)
			{
				bis.close ();
			}
			
		}
		
	}
	
	
	public static Map<String, MutablePair<Queue<int[]>, List<TreeTraversal.Node>>> iterateStateNodesInTreesFromFULLSUBCLONEAncestryFiles (
			String inputPathForLISTOFAncestryRelationshipsFILEStr)
	{
		
		// <key> name of sample, <value> queue of integer lists for given sample-tree , each list
			// corresponds to an ordered path in the tree
		Map<String, MutablePair<Queue<int[]>, List<TreeTraversal.Node>>> 
			mapOfAllTreesSpecificPaths = new HashMap<String, MutablePair<Queue<int[]>, List<TreeTraversal.Node>>> ();
		
		try
		{
			//////// For each ChildToParentRelationShip...txt file for a given tree, generate the
			//////// set of possible paths(in terms of the sequence subclone indices traversed) and
			//////// save to Map<String, Queue<int []>> treeSpecificPathsList
			Path pathForLISTOFAncestryRelationshipsFILE = Paths
					.get (inputPathForLISTOFAncestryRelationshipsFILEStr); // text file will contain
																			// the full path to the
																			// ChildToParentAncestry
																			// text file describing
																			// the parent subclone
																			// for each child
																			// subclone via a list
																			// of the form: <child
																			// subclone idx>:<parent
																			// subclone idx>
			List<String> listOfsubcloneAncestryRelationshipsTextFiles = Files
					.readAllLines (pathForLISTOFAncestryRelationshipsFILE);
			// set up a loop over the input data
			for (String treeXAncestryTextFileStr : listOfsubcloneAncestryRelationshipsTextFiles)
			{
				// System.out.println(treeXAncestryTextFileStr);
				MutablePair<Queue<int[]>, List<TreeTraversal.Node>> pathsAndNodesInTreeX = generatePathsInTree (treeXAncestryTextFileStr);
				
				String ancestryFileName = Paths.get (treeXAncestryTextFileStr).getFileName ().toString ();
				String delimsA = "[.]";
				String[] tokensA = ancestryFileName.split (delimsA);
				String partA = tokensA[1];
				String delimsB = "[_]";
				String[] tokensB = partA.split (delimsB);
				String sampleName = tokensB[0] + "_" + tokensB[1];
				
				// parse the name of ChildToParent text file name to only keep the actual sample
				// name as the key for our hashmap
				// ChildToParentRelationShipForMostInformativeTree16569-DMMR-TUMOR-s065F14xC1C02_16569-DMMR-NORMAL-s065F18xC1G02_10000_2500.txt
				// we want only sample name :
				// 16569-DMMR-TUMOR-s065F14xC1C02_16569-DMMR-NORMAL-s065F18xC1G02
//				Queue<int[]> pathsInTreeX = pathsAndNodesInTreeX.getLeft ();	// GTD Not used
				mapOfAllTreesSpecificPaths.put (sampleName, pathsAndNodesInTreeX);
				
			}
		}
		catch (Exception ex)
		{
			if (ex instanceof IOException)
			{
				
				ex.printStackTrace ();
				
			}
//			else if (ex instanceof REngineException)
//			{
//				ex.printStackTrace ();
//				
//			}
			else
			{
				throw new RuntimeException (ex);
			}
		}
		
		return mapOfAllTreesSpecificPaths;
	}
	
	
	// public static Queue<int []> generatePathsInTree(String
	// inputPathForAncestryRelationshipsTextFile) {
	public static MutablePair<Queue<int[]>, List<TreeTraversal.Node>> generatePathsInTree (
			String inputPathForAncestryRelationshipsTextFile)
	{
		
		List<TreeTraversal.Node> nodesInSampleTree = new ArrayList<TreeTraversal.Node> ();
		
		Multimap<Integer, Integer> parentToChildRelMultimap = ArrayListMultimap.create ();
		Map<Integer, Integer> childToParentMap = new HashMap<Integer, Integer> ();
		HashSet<Integer> uniqueSubclIndices = new HashSet<Integer> ();
		Queue<int[]> pathsFromRoot = null;
		
		try
		{
			Path inputpath = Paths.get (inputPathForAncestryRelationshipsTextFile);
			List<String> inputlines = Files.readAllLines (inputpath);
			
			// set up a loop over the input data
			for (String line : inputlines)
			{
				String[] arrayOfStrings = line.split (":");
				// assuming structure is <child>:<parent>
				// arrayOfStrings[0] = child
				// arrayOfStrings[1] = parent
				for (String a : arrayOfStrings)
				{
					System.out.println (a);
					uniqueSubclIndices.add (Integer.parseInt (a));
				}
				// map <parent, children>
				parentToChildRelMultimap.put (Integer.parseInt (arrayOfStrings[1]), Integer.parseInt (arrayOfStrings[0]));
				
				// map <child, parent>
				childToParentMap.put (Integer.parseInt (arrayOfStrings[0]), Integer.parseInt (arrayOfStrings[1]));
			}
			
			for (Integer uniqueIdx : uniqueSubclIndices)
			{
				// Node subclone_node = new Node(uniqueIdx, inputpath.getFileName().toString());
				Node subclone_node = new Node (uniqueIdx, inputpath.getFileName ().toString (), session);
				
				nodesInSampleTree.add (subclone_node);
			}
			// after all unique nodes have been created. we can set their ancestral relationships
			for (Node subclone : nodesInSampleTree)
			{
				List<Node> childrenNodes = new ArrayList<Node> ();
				
				for (Integer childId : parentToChildRelMultimap.get (subclone.Id))
				{
					Node childNode = nodesInSampleTree.get (childId);
					if (childNode.node_stateUUID != null)
					{
						childrenNodes.add (nodesInSampleTree.get (childId));
					}
					else
					{
						System.out.println ("NULL stateuuid for :" + childNode.VCFFileName);
					}
				}
				subclone.setChildren (childrenNodes);
				Integer parentId = childToParentMap.get (subclone.Id);
				if (parentId != null)
				{
					Node parent = nodesInSampleTree.get (parentId);
					subclone.setParentNode (parent);
				}
				
			}
			
			pathsFromRoot = traverseBFS (nodesInSampleTree.get (0));
			
			// will use the set of int [] paths after we have obtained the set of state-signatures
		}
		catch (IOException ie)
		{
			ie.printStackTrace ();
		}
		
		MutablePair<Queue<int[]>, List<TreeTraversal.Node>> treeXPaths = new MutablePair<Queue<int[]>, List<TreeTraversal.Node>> (
				pathsFromRoot, nodesInSampleTree);
		return treeXPaths;
	}
	
	
	public static void iterateActionEdgesInTreesFromDRIVERSSUBCLONEAncestryFiles (
			String inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPaths,
			Map<String, UUID> tidToMUTactionUUIDmap,
			Map<String, UUID> tidToMETHYLactionUUIDmap,
			Map<String, MutablePair<Queue<int[]>, List<TreeTraversal.Node>>> specificPathsTraversedInEachSampleTreeMAP,
			boolean isMutationVCF, boolean isMethylationVCF)
	{
		try
		{
			Path listOfTreeSpecificSubcloneDRIVERSOnlyDirPaths = Paths
					.get (inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPaths); // text file
																						// will
																						// contain
																						// the full
																						// path to
																						// the
																						// subdirectory
																						// containing
																						// the set
																						// of
																						// subclone
																						// vcf's
																						// (containing
																						// only the
																						// DRIVER
																						// variants)
																						// for each
																						// CRC
																						// sample
																						// best-tree,
			List<String> allTreeDRIVERDirPaths = Files
					.readAllLines (listOfTreeSpecificSubcloneDRIVERSOnlyDirPaths); // list of paths
																					// to
																					// subdirectories
																					// for all
																					// tumor-samples;
																					// each path
																					// points to the
																					// directory in
																					// which the
																					// subclones
																					// (with only
																					// DRIVER
																					// variants) of
																					// a tumor
																					// sample X are
																					// stored
			
			// We also need to generate the array of actions (int []) corresponding to the various
			// drivers that may have lead to the given subclone (state-signature) in treeX
			// i.e. we want to generate int [] for each subclone of the current sample and add it to
			// childToDriversListMap Map<String, HashMultimap<Integer, int []>>
			// . This is needed for subsequent computations when initializing the stochastic model
			// with counts for each state_A --> state_B transition for a given action_1
			// now we need to iterate through all subclone vcf files containing only the DRIVER
			// variants
			for (String treeXDRIVERSubclonesdirPath : allTreeDRIVERDirPaths)
			{ // each line is the path to a single tumor's sample subdirectory of subclone (DRIVER
				// variants only) vcf files
				// Assuming the subclone DRIVERS vcf filename is
				// subclone.0.unique.ANN.SNPEFF.reduced.FILTERED.16264-DMMR-TUMOR-s024F17xC1A04_16264-DMMR-NORMAL-s024F16xC1H03.vcf
				List<Path> subCloneDRIVERSVCFfilesInDir = Files
						.walk (Paths.get (treeXDRIVERSubclonesdirPath))
						.filter (Files::isRegularFile).collect (Collectors.toList ()); // get list
																						// of paths
																						// of the
																						// DRIVER
																						// subvcf
																						// files for
																						// a SINGLE
																						// tumor-sample
//				String firstDRIVERSsubclFileName = subCloneDRIVERSVCFfilesInDir.get (0).getFileName ().toString ();	// GTD Not used
//				String delimsB = "[.]";	// GTD Not used
//				String[] tokensB = firstDRIVERSsubclFileName.split (delimsB);	// GTD Not used
//				String sampleNameB = tokensB[7]; // XXX:******assuming that 7th field in filename	// GTD Not used
													// corresponds to the full sample name
				
				// generate HashMultiMap<Integer, int[]> for current sample X, which associates an
				// int[] of actions (drivers) with each subclone in treeX.
				updateDRIVERActionEdgesForEachSubcloneInTreeX (subCloneDRIVERSVCFfilesInDir,
						tidToMUTactionUUIDmap, tidToMETHYLactionUUIDmap, isMutationVCF,
						isMethylationVCF, specificPathsTraversedInEachSampleTreeMAP);
				
			}
			
		}
		catch (Exception ex)
		{
			if (ex instanceof IOException)
			{
				
				ex.printStackTrace ();
				
			}
//			else if (ex instanceof REngineException)
//			{
//				ex.printStackTrace ();
//				
//			}
			else
			{
				throw new RuntimeException (ex);
			}
		}
		
	}
	
	
	public static void traverseStateActionTrajBFS (Node stateNode,
			TrajectoryTimeStep parenttimestep, List<TrajectoryTimeStep> leafTimeSteps)
	{
		HashSet<UUID> parentEdgeActions = stateNode.getParentEdgeActions ();
		for (UUID eAction : parentEdgeActions)
		{
			TrajectoryTimeStep currTimestep = new TrajectoryTimeStep (parenttimestep,
					stateNode.node_stateUUID, eAction);
			if (stateNode.childrenNodeList.size () > 0)
			{
				for (Node childnode : stateNode.childrenNodeList)
				{
					traverseStateActionTrajBFS (childnode, currTimestep, leafTimeSteps);
				}
			}
			leafTimeSteps.add (currTimestep);
			
		}
	}
	
	
	public void traverseDFS (Node root)
	{
		Stack<Node> s = new Stack<Node> ();
		Stack<String> sPath = new Stack<> ();
		Stack<Integer> sSum = new Stack<> ();
		s.push (root);
		sPath.push (root.Id + "");
		sSum.push (root.Id);
		
		while (!s.isEmpty ())
		{
			// Pop out
			Node head = s.pop ();
			String headPath = sPath.pop ();
			Integer headSum = sSum.pop ();
			if (head.childrenNodeList == null || head.childrenNodeList.isEmpty ())
			{ // Leaf
				System.out.println (headPath + "(" + headSum + ")");
				continue;
			}
			for (Node child : head.childrenNodeList)
			{
				String path = headPath + "->" + child.Id;
				Integer sum = headSum + child.Id;
				// Push on stack
				s.push (child);
				sPath.push (path);
				sSum.push (sum);
			}
		}
	}
	
	
	public static Queue<int[]> traverseBFS (Node root)
	{
		Queue<Node> qOfNodesToProcess = new LinkedList<> ();
		Queue<int[]> qOfPaths = new LinkedList<> ();
		Queue<int[]> qOfPathsFromRoot = new LinkedList<> ();
		Queue<Integer> qSum = new LinkedList<> ();
		qOfNodesToProcess.add (root);
		int[] rootPath = new int[1];
		rootPath[0] = root.Id;
		qOfPaths.add (rootPath);
		qSum.add (root.Id);
		
		while (!qOfNodesToProcess.isEmpty ())
		{
			// Poll the q
			Node head = qOfNodesToProcess.poll ();
			int[] headPath = qOfPaths.poll ();
			
			Integer headSum = qSum.poll ();
			if (head.childrenNodeList == null || head.childrenNodeList.isEmpty ())
			{ // Leaf
				System.out.println (headPath + "(" + headSum + ")");
				continue;
			}
			for (Node child : head.childrenNodeList)
			{
				int hpLength = headPath.length;
				int[] path = new int[hpLength + 1];
				for (int i = 0; i < hpLength; i++)
				{
					path[i] = headPath[i];
				}
				path[hpLength] = child.Id;
				
				Integer sum = headSum + child.Id;
				// Add to the q
				qOfNodesToProcess.add (child);
				qOfPaths.add (path);
				qSum.add (sum);
				if (child.childrenNodeList == null || child.childrenNodeList.isEmpty ())
				{
					qOfPathsFromRoot.add (path);
				}
			}
			
		}
		return qOfPathsFromRoot;
	}
	
	
	public static Map<Integer, int[]> determineSetOfPossibleActionsResponsibleForEachSubcloneInTreeX (
			List<Path> pathsForEachSubcloneVCFFile,
			HashMultimap<Integer, MutablePair<String[], String[]>> actionIDXtoVariantDRIVERSMAP,
			Map<String, String> tIDToGeneSymbolMap)
	{
		Map<Integer, int[]> mapOfActions = new HashMap<Integer, int[]> ();
		
		Map<String, Pattern> transcriptToPatternMap = new HashMap<String, Pattern> ();
		for (String t : tIDToGeneSymbolMap.keySet ())
		{
			String tidpattstring = ".*" + t + ".*";
			
			Pattern tpattern = Pattern.compile (tidpattstring);
			transcriptToPatternMap.put (t, tpattern);
			
		}
		
		try
		{
			
			for (int sfileCount = 0; sfileCount < pathsForEachSubcloneVCFFile.size (); sfileCount++)
			{ // simply want to iterate through all paths for the subclone vcfs in the List<Path<
				Path subcloneVCFFilepath = pathsForEachSubcloneVCFFile.get (sfileCount);
				
				// Assume the subclone DRIVERS vcf filename is
				// subclone.0.unique.ANN.SNPEFF.reduced.FILTERED.16264-DMMR-TUMOR-s024F17xC1A04_16264-DMMR-NORMAL-s024F16xC1H03.vcf
				// we want to extract the subclone index number, which in this example is in
				// position 1: "0"
				String subcloneVCFFileName = subcloneVCFFilepath.getFileName ().toString ();
				System.out.println ("Running determineSetOfPossibleActionsResponsibleForEachSubcloneInTreeX() to parse variants in vcf file :"
								+ subcloneVCFFileName);
				String delims = "[.]";
				String[] tokens = subcloneVCFFileName.split (delims);
				String currSubCloneIndex = tokens[1]; // XXX******assuming that position 1 in
														// filename corresponds to the subclone
														// index
				
				List<String> vcfLineObjects = Files.readAllLines (subcloneVCFFilepath);
				
				int[] currSubCloneActionList = new int[0];
				
				Pattern vpattern = Pattern.compile (".*#.*");
//				Pattern apcpattern = Pattern.compile (".*APC.*");	// GTD Not used
				
				List<String> actualVariants = new ArrayList<String> ();
//				String tabDelims = "[\\t]";	// GTD Not used
//				String commaDelims = "[,]";	// GTD Not used
//				String semicolonDelims = "[;]";	// GTD Not used
				
				for (String variantLine : vcfLineObjects)
				{
					
					if (vpattern.matcher (variantLine).matches () == false)
					{ // if the given variant line of the vcf does NOT contain the ## characters
						actualVariants.add (variantLine);
//						String[] variantLineAsTokens = variantLine.split (tabDelims);	// GTD Not used
						
						/// alternative vcf parser method
						List<String> data = toList (tabDelimiter, variantLine);
						
						// extract INFO field of VCF
						ListMultimap<String, String> infoField = null;
						if (!data.get (7).equals ("") && !data.get (7).equals ("."))
						{
							infoField = ArrayListMultimap.create ();
							List<String> props = toList (semiColonDelimiter, data.get (7));
							for (String prop : props)
							{
								int idx = prop.indexOf ('=');
								if (idx == -1)
								{
									infoField.put (prop, "");
								}
								else
								{
									String key = prop.substring (0, idx);
									String value = prop.substring (idx + 1);
									infoField.putAll (key, toList (commaDelimiter, value));
								}
							}
						}
						
						List<String> ANNField = (infoField != null) ? infoField.get ("ANN") : new ArrayList<> ();
						
						// for (Pattern tpattern:transcriptToPatternMap.values()) {
						Set<String> tidkeys = transcriptToPatternMap.keySet ();
						for (String tidStr : tidkeys)
						{
							Pattern tpattern = transcriptToPatternMap.get (tidStr);
							for (int a = 0; a < ANNField.size (); a++)
							{
								if (tpattern.matcher (ANNField.get (a)).matches () == true)
								{
									/// XXX: Uncomment this print statement when needed for
									/// debugging
									// System.out.println(tidStr+" found in: "+ANNField.get(a));
								}
							}
						}
						
//						String refObj = variantLineAsTokens[3]; // retrieve the REF allele(s) for	// GTD Not used
																// this variant
//						String altObj = variantLineAsTokens[4]; // retrieve the ALT allele(s) for	// GTD Not used
																// this variant
						/// XXX: Uncomment this print statement when needed for debugging
						// System.out.println("REF:"+refObj+" | ALT:"+altObj);
						
						// convert the variant object to its respective ACTION by looking at
						// actionIDtoVariantDRIVERSObjectsMAP
//						String[] altVariantAsTokens = altObj.split (commaDelims);	// GTD Not used
//						String[] refVariantAsTokens = refObj.split (commaDelims);	// GTD Not used
						// if( altVariantAsTokens.length >1 || refVariantAsTokens.length>1 ) { //if
						// there are >1 REF or ALT allele
						// //There are multiple alternative alleles to which the reference allele
						// can change into
						// for(int i=0; i< altVariantAsTokens.length; i++) {
						// String altToken = altVariantAsTokens[i];
						// String [] foo2 = {altToken};
						// for (int j=0; j<refVariantAsTokens.length; j++) {
						// String refToken = refVariantAsTokens[j];
						// String [] foo1 = {refToken};
						// MutablePair<String[], String[]> aTokePair = new MutablePair<>(foo1,
						// foo2);
						// int matchingActionIdx =
						// checkContentsOfActionToVariantMultiMap(actionIDXtoVariantDRIVERSMAP,aTokePair);
						// if(matchingActionIdx > -1) {
						// System.out.println(refObj+">"+altObj+"variant matches the pre-defined
						// action: "+matchingActionIdx);
						//
						// int aListLength = currSubCloneActionList.length;
						// int[] newAList = new int[aListLength+1];
						// for (int k=0; k< currSubCloneActionList.length; k++) {
						// newAList[k]= currSubCloneActionList[k];
						// }
						// newAList[aListLength] = matchingActionIdx;
						// currSubCloneActionList = newAList;
						// }
						// }
						// }
						// }// close if-condition for REF or ALT variants that may consist of
						// multiple alleles
						// else {
						// String [] foo1 = {refObj};
						// String [] foo2 = {altObj};
						// MutablePair<String[], String[]> aTokePair = new MutablePair<>(foo1,
						// foo2);
						// int matchingActionIdx =
						// checkContentsOfActionToVariantMultiMap(actionIDXtoVariantDRIVERSMAP,aTokePair);
						// if(matchingActionIdx > -1) {
						// System.out.println(refObj+">"+altObj+"variant matches the pre-defined
						// action: "+matchingActionIdx);
						// int aListLength = currSubCloneActionList.length;
						// int[] newAList = new int[aListLength+1];
						// for (int k=0; k< currSubCloneActionList.length; k++) {
						// newAList[k]= currSubCloneActionList[k];
						// }
						// newAList[aListLength] = matchingActionIdx;
						// currSubCloneActionList = newAList;
						// }
						// }
					}
				}
				mapOfActions.put (Integer.parseInt (currSubCloneIndex), currSubCloneActionList);
			} // close for-loop iterating throuth each subclone filepath
			
		}
		catch (Exception ex)
		{
			if (ex instanceof IOException)
			{
				
				ex.printStackTrace ();
				
			}
//			else if (ex instanceof REngineException)
//			{
//				ex.printStackTrace ();
//				
//			}
			else
			{
				throw new RuntimeException (ex);
			}
		}
		return mapOfActions;
		
	}
	
	
	public static int checkContentsOfActionToVariantMultiMap (
			HashMultimap<Integer, MutablePair<String[], String[]>> actionIDToVariantMAP,
			MutablePair<String[], String[]> variantToCheck)
	{
		
		int actionContainingVariantOfInterest = Integer.MIN_VALUE;
		// lets assume that the actionIDToVariant map only contains key and values in which the
		// String [] is of length =1, i.e. they only contain one string, but each action(key) can
		// contain MULTIPLE MutablePair<String[], String[]> (i.e. we don't care if some vcf variants
		// contain multiple possible REF and/or ALT alleles, C>CT,CTTC (each possiblity will be
		// examined against the set of MutablePairs for a given actionidx)
		// thus the variant of interest's MutablePair can contain String[] with multiple alleles in
		// the REF(key) and ALT(value) of MutablePair<String[], String[]> variantToCheck
		int keyLength = variantToCheck.getKey ().length;
		int valLength = variantToCheck.getValue ().length;
		String[] varkey = variantToCheck.getKey ();
		String[] varval = variantToCheck.getValue ();
		for (int i = 0; i < keyLength; i++)
		{
			String strkey_i = varkey[i];
			
			for (int j = 0; j < valLength; j++)
			{
				String strval_j = varval[j];
				for (Integer actionIDx : actionIDToVariantMAP.keys ())
				{
					Set<MutablePair<String[], String[]>> setOfValsForActionX = actionIDToVariantMAP
							.get (actionIDx);
					for (MutablePair<String[], String[]> variantZ : setOfValsForActionX)
					{
						String zkey = variantZ.getKey ()[0];
						String zval = variantZ.getValue ()[0];
						if (strkey_i.matches (zkey))
						{
							if (strval_j.matches (zval))
							{
								System.out.println ("This variant matches an existing ACTION!");
								actionContainingVariantOfInterest = actionIDx.intValue ();
								return actionContainingVariantOfInterest;
							}
						}
					}
					
				}
				
			}
		}
		
		return actionContainingVariantOfInterest;
	}
	
	
	public void setTransitionCountMatrixList (Map<String, double[][]> tranCountMatrixList)
	{
		_transitionCountMatrixList = tranCountMatrixList;
	}
	
	
	public void setCooccurenceCountMatrixList (Map<String, double[][]> coCountMatrixList)
	{
		_cooccurenceCountMatrixList = coCountMatrixList;
	}
	
	
	private static List<String> toList (Pattern delimiterPattern, String inputString)
	{
		String[] array = delimiterPattern.split (inputString);
		List<String> list = new ArrayList<> (array.length);
		Collections.addAll (list, array);
		return list;
	}
	
	
	public static MutablePair<Map<String, String>, HashMultimap<String, String>> createTranscriptToGeneSymbolMapAndReverseMap (
			Path pathToBioMartInputDataFile)
	{
		// biomart_customGenIDTRANSCRIPTandSymbol.txt
		
		Map<String, String> transcriptIDtoGeneSymbolMap = new HashMap<> ();
		HashMultimap<String, String> geneSymbolToTranscriptIDMap = HashMultimap.create ();
		String tabDelims = "[\\t]";
		
		try
		{
			List<String> lineObjects = Files.readAllLines (pathToBioMartInputDataFile);
			for (String tline : lineObjects)
			{
				
				String[] lineAsTokens = tline.split (tabDelims);
				String tIDtoken = lineAsTokens[0];
				String geneSymboltoken = lineAsTokens[2];
				
				transcriptIDtoGeneSymbolMap.put (tIDtoken, geneSymboltoken);
				geneSymbolToTranscriptIDMap.put (geneSymboltoken, tIDtoken);
			}
			
		}
		catch (Exception ex)
		{
			if (ex instanceof IOException)
			{
				
				ex.printStackTrace ();
				
			}
//			else if (ex instanceof REngineException)
//			{
//				ex.printStackTrace ();
//				
//			}
			else
			{
				throw new RuntimeException (ex);
			}
		}
		
		MutablePair<Map<String, String>, HashMultimap<String, String>> tidToGeneMapAndReverseMap = new MutablePair<> (
				transcriptIDtoGeneSymbolMap, geneSymbolToTranscriptIDMap);
		return tidToGeneMapAndReverseMap;
	}
	
	
	public static void updateDRIVERActionEdgesForEachSubcloneInTreeX (
			List<Path> pathsForEachSubcloneVCFFile, Map<String, UUID> tidToMUTactionUUIDMAP,
			Map<String, UUID> tidToMETHYLactionUUIDMAP, boolean isMutationVCF,
			boolean isMethylationVCF,
			Map<String, MutablePair<Queue<int[]>, List<TreeTraversal.Node>>> specificPathsTraversedInEachSampleTree_MAP)
	{
		
		Map<String, Pattern> transcriptToPatternMap = new HashMap<String, Pattern> ();
		
		if (isMutationVCF)
		{
			for (String t : tidToMUTactionUUIDMAP.keySet ())
			{
				String tidpattstring = ".*" + t + ".*";
				
				Pattern tpattern = Pattern.compile (tidpattstring);
				transcriptToPatternMap.put (t, tpattern);
				
			}
			
			try
			{
				
				for (int sfileCount = 0; sfileCount < pathsForEachSubcloneVCFFile
						.size (); sfileCount++)
				{ // simply want to iterate through all paths for the subclone vcfs in the
					// List<Path<
					Path subcloneVCFFilepath = pathsForEachSubcloneVCFFile.get (sfileCount);
					
					// Assume the subclone DRIVERS vcf filename is
					// subclone.0.unique.ANN.SNPEFF.reduced.FILTERED.16264-DMMR-TUMOR-s024F17xC1A04_16264-DMMR-NORMAL-s024F16xC1H03.vcf
					// we want to extract the subclone index number, which in this example is in
					// position 1: "0"
					String subcloneVCFFileName = subcloneVCFFilepath.getFileName ().toString ();
					String delims = "[.]";
					String[] tokens = subcloneVCFFileName.split (delims);
					String currSubCloneIndex = tokens[1]; // XXX******assuming that position 1 in
															// filename corresponds to the subclone
															// index
					
					String tumorSampleName = tokens[7];
					
					MutablePair<Queue<int[]>, List<TreeTraversal.Node>> pathsAndNodesTraversedInTreeX = specificPathsTraversedInEachSampleTree_MAP
							.get (tumorSampleName); // retrieve the sample-tree associated with
													// tumorSampleName and its corresponding set of
													// paths
					
//					Queue<int[]> orderOfSubIdxsInTreeX = pathsAndNodesTraversedInTreeX.getLeft ();	// GTD Not used
					List<TreeTraversal.Node> nodesInTreeX = pathsAndNodesTraversedInTreeX
							.getRight ();
					Node nodeOfInterest = null;
					for (Node nodeClone : nodesInTreeX)
					{
						if (nodeClone.Id == Integer.parseInt (currSubCloneIndex))
						{
							nodeOfInterest = nodeClone;
							break;
						}
					}
					
					List<String> vcfLineObjects = Files.readAllLines (subcloneVCFFilepath);
					
//					int[] currSubCloneActionList = new int[0];	// GTD Not used
					
					Pattern vpattern = Pattern.compile (".*#.*");
					// Pattern apcpattern = Pattern.compile(".*APC.*");
					
					List<String> actualVariants = new ArrayList<String> ();
//					String tabDelims = "[\\t]";	// GTD Not used
//					String commaDelims = "[,]";	// GTD Not used
//					String semicolonDelims = "[;]";	// GTD Not used
					
					for (String variantLine : vcfLineObjects)
					{
						
						if (vpattern.matcher (variantLine).matches () == false)
						{ // if the given variant line of the vcf does NOT contain the ## characters
							actualVariants.add (variantLine);
//							String[] variantLineAsTokens = variantLine.split (tabDelims);	// GTD Not used
							
							/// alternative vcf parser method
							List<String> data = toList (tabDelimiter, variantLine);
							
							// extract INFO field of VCF
							ListMultimap<String, String> infoField = null;
							if (!data.get (7).equals ("") && !data.get (7).equals ("."))
							{
								infoField = ArrayListMultimap.create ();
								List<String> props = toList (semiColonDelimiter, data.get (7));
								for (String prop : props)
								{
									int idx = prop.indexOf ('=');
									if (idx == -1)
									{
										infoField.put (prop, "");
									}
									else
									{
										String key = prop.substring (0, idx);
										String value = prop.substring (idx + 1);
										infoField.putAll (key, toList (commaDelimiter, value));
									}
								}
							}
							
							List<String> ANNField = (infoField != null) ? infoField.get ("ANN") : new ArrayList<> ();
							
							Set<String> tidkeys = transcriptToPatternMap.keySet ();
							for (String tidStr : tidkeys)
							{
								Pattern tpattern = transcriptToPatternMap.get (tidStr);
								for (int a = 0; a < ANNField.size (); a++)
								{
									if (tpattern.matcher (ANNField.get (a)).matches () == true)
									{
										// XXX: Uncomment this print statement only for debugging
										// purposes
										// System.out.println(tidStr+" found in: "+ANNField.get(a));
										UUID actionUUID = tidToMUTactionUUIDMAP.get (tidStr);
										if (nodeOfInterest != null)
											nodeOfInterest.appendParentEdgeAction (actionUUID);
										
									}
								}
							}
							
//							String refObj = variantLineAsTokens[3]; // retrieve the REF allele(s)	// GTD Not used
																	// for this variant
//							String altObj = variantLineAsTokens[4]; // retrieve the ALT allele(s)	// GTD Not used
																	// for this variant
							/// XXX: Uncomment this print statement when needed for debugging
							// System.out.println("REF:"+refObj+" | ALT:"+altObj);
							
							// convert the variant object to its respective ACTION by looking at
							// actionIDtoVariantDRIVERSObjectsMAP
//							String[] altVariantAsTokens = altObj.split (commaDelims);	// GTD Not used
//							String[] refVariantAsTokens = refObj.split (commaDelims);	// GTD Not used
							
						}
					}
				} // close for-loop iterating throuth each subclone filepath
				
			}
			catch (Exception ex)
			{
				if (ex instanceof IOException)
				{
					
					ex.printStackTrace ();
					
				}
//				else if (ex instanceof REngineException)
//				{
//					ex.printStackTrace ();
//					
//				}
				else
				{
					throw new RuntimeException (ex);
				}
			}
		}
	}
	
	
	/*
	 * Convert set of trajectories generated from sampleX's phylogenetic tree traversal into a .csv
	 * file
	 */
	public static void saveTrajectoriestoCSV (String sampleXName,
			List<double[][]> sampleXDoubleTrajectoriesListFromNormalStart)
	{
		
		// List<double[][]> sampleXDoubleTrajectoriesListFromNormalStart = new
			// ArrayList<double[][]>();
		
		try
		{
			String countMatrixCSVNAME = _outputDirPathStr + "/" + sampleXName + ".TRAJECTORIES.csv"; // JK
																										// 5.9.2019
																										// CHANGED
																										// NAMING
																										// FORMAT
																										// for
																										// csv
																										// so
																										// we
																										// can
																										// automatically
																										// import
																										// .csv
																										// files
																										// to
																										// stateActionTrajectories_tabe
																										// during
																										// saveSTATEACTIONDOUBLETrajectoriesForSingleSampleXToDB()
																										// methods
			// DEPRECATEDString countMatrixCSVNAME =
			// _outputDirPathStr+"/trajectoriesForSamplex_"+sampleXName+".csv";
			// String countMatrixCSVNAME =
			// "/Users/m186806/trajectoriesForSamplex_"+sampleXName+".csv";
			// String countMatrixCSVNAME =
			// "/research/labs/microbiome/chia/m186806/superSandbox/trajectoriesForSamplex_"+sampleXName+".csv";
			
			int numTrajectoriesInSampleX = sampleXDoubleTrajectoriesListFromNormalStart.size ();
			int trajLength = sampleXDoubleTrajectoriesListFromNormalStart.get (1)[0].length;
			FileWriter writer = new FileWriter (countMatrixCSVNAME);
			for (int t = 0; t < numTrajectoriesInSampleX; t++)
			{
				writer.append (sampleXName);
				writer.append (',');
				UUID trajUUID = UUID.randomUUID ();
				writer.append (trajUUID.toString ()); // uuid of trajectory
				writer.append (',');
				for (int j = 0; j < trajLength; j++)
				{
					writer.append (String.valueOf (
							(int) sampleXDoubleTrajectoriesListFromNormalStart.get (t)[0][j])); // state
																								// at
																								// timestep
																								// j
					writer.append (',');
					writer.append (String.valueOf (
							(int) sampleXDoubleTrajectoriesListFromNormalStart.get (t)[1][j])); // action
																								// at
																								// timestep
																								// j
					if (j < (trajLength - 1))
					{
						writer.append (',');
					}
				}
				writer.append ('\n');
				writer.flush ();
			}
			writer.close ();
			
		}
		catch (Exception e)
		{
			e.printStackTrace ();
		}
	}
	
	
	public static void saveCountMatricesMAPtoCSV (Map<Integer, double[][]> countMatricesMap,
			int numStates)
	{
		try
		{
			double[][] countMatrixForaction_i;
			Set<Integer> actionKeys = countMatricesMap.keySet ();
			for (int action_i : actionKeys)
			{
				// String cqlDROPCountMatrixTableStatement = "DROP TABLE
				// countmatrixforaction_"+action_i; //only needed to include in this
				// loop because it was faster than manually deleting ~600ish countmatrix tables
				// stored in cassandra
				// session.execute(cqlDROPCountMatrixTableStatement);
				
				countMatrixForaction_i = countMatricesMap.get (action_i);
				String countMatrixCSVNAME = "/Users/m186806/countMatrixFor_action_" + action_i
						+ ".csv";
				FileWriter writer = new FileWriter (countMatrixCSVNAME);
				for (int i = 0; i < numStates; i++)
				{
					int j;
					for (j = 0; j < (numStates - 1); j++)
					{
						writer.append (String.valueOf ((int) countMatrixForaction_i[i][j]));
						writer.append (',');
					}
					writer.append (String.valueOf ((int) countMatrixForaction_i[i][j]));
					writer.append ('\n');
					writer.flush ();
				}
				writer.close ();
				
			}
			
		}
		catch (Exception e)
		{
			e.printStackTrace ();
		}
	}
	
	
	public static void saveCountMatricesMAPToDB (Map<Integer, double[][]> countMatricesMap,
			int numStates)
	{
		
		double[][] countMatrixForaction_i;
		Set<Integer> actionKeys = countMatricesMap.keySet ();
		for (int action_i : actionKeys)
		{
			String cqlCreateCountMatrixTableStatement = "create TABLE countmatrixforaction_"
					+ action_i + "(currentstateint bigint PRIMARY KEY)";
			session.execute (cqlCreateCountMatrixTableStatement); // only need to run this once;
																	// comment out after created
			for (int s = 0; s < numStates; s++)
			{
				String cqlADDStateColToTrajectoriesTableStatement = "alter TABLE countmatrixforaction_"
						+ action_i + " ADD nextstateint" + s + " bigint";
				session.execute (cqlADDStateColToTrajectoriesTableStatement); // only need to run
																				// this once;
																				// comment out after
																				// created
			}
			countMatrixForaction_i = countMatricesMap.get (action_i);
			for (int r = 0; r < numStates; r++)
			{
				for (int c = 0; c < numStates; c++)
				{
					double count = countMatrixForaction_i[r][c];
					int countINT = (int) count;
					String cqlInsertCellToCountTablestatement = "INSERT INTO countmatrixforaction_"
							+ action_i + " (currentstateint,nextstateint" + c + ") values(" + r
							+ "," + countINT + ")";
					session.execute (cqlInsertCellToCountTablestatement);
				}
			}
			// do we want to store [numStates x numStates] count matrix for each action_i? Or should
			// we compute the transition matrix immediately? and store that it instead?
		}
		
	}
	
	
	public static Map<Integer, Map<Integer, Integer>> createMappingOfPossibleStatesForEachState (
			Map<Integer, double[][]> countMatricesMap, int numberOfStates, int numberOfActions)
	{
		// Map<currentStateINT, Map<actionINT, nextStateINT>>
		Map<Integer, Map<Integer, Integer>> mapOfPossibleNextStatesForAllStates = new HashMap<Integer, Map<Integer, Integer>> ();
		for (int state_j = 0; state_j < numberOfStates; state_j++)
		{
			Map<Integer, Integer> possibleNextStatesForState_j = new HashMap<Integer, Integer> (); // for
																										// state_j:
																										// the
																										// mapping
																										// between
																										// <key>actionINT
																										// and
																										// <value>
																										// nextStateINT
			mapOfPossibleNextStatesForAllStates.put (state_j, possibleNextStatesForState_j);
		}
		double[][] countMatrixForaction_i;
		double[] countsForCurrentState_r;
		double[] subRowCountVector_r;
		int mostProbableNextState = 0;
		Pair<Integer, Double> positionAndValueOfMaxCol;
		Set<Integer> actionKeys = countMatricesMap.keySet ();
		for (int action_i : actionKeys)
		{
			if (action_i != 0)
			{ // we don't want to consider those actions that were solely used for padding purposes
				countMatrixForaction_i = countMatricesMap.get (action_i);
				for (int currStater = 1; currStater < numberOfStates; currStater++)
				{ // do not consider stateINT=0 (row 0) corresponding to the state used for padding
					countsForCurrentState_r = countMatrixForaction_i[currStater];
					subRowCountVector_r = VectorUtility.subVector (countsForCurrentState_r, 1,
							numberOfStates - 1);
					positionAndValueOfMaxCol = VectorUtility
							.maxPositionAndVal (subRowCountVector_r);
					if (positionAndValueOfMaxCol.getSecond () > 0)
					{ // if there exists a nextStateINT with a count >0
						mostProbableNextState = positionAndValueOfMaxCol.getFirst () + 1; // remember
																							// its
																							// returning
																							// the
																							// index
																							// value
																							// of
																							// the
																							// subvector
																							// that
																							// begins
																							// at
																							// state1
																							// and
																							// not
																							// at
																							// state0
					}
					else
					{// otherwise if the maximum value is 0; then we never observed a nextStateINT
						// for currentstateINT
						mostProbableNextState = currStater; // if under action_i, no NEXTSTATEINT
															// has ever been observed to occur, we
															// simply assume that currentStateINT
															// will persist in its given state when
															// action_i is executed
					}
					mapOfPossibleNextStatesForAllStates.get (currStater).put (action_i,
							mostProbableNextState); // iteratively updating for each currentStateINT
													// what the next possible state is if action_i
													// was the action executed
					
				}
				mapOfPossibleNextStatesForAllStates.get (0).put (action_i, 0); // for each action_i,
																				// if the
																				// currentStateInt=0,
																				// we will always
																				// persist in
																				// stateInt=0
				
			}
			else if (action_i == 0)
			{
				for (int currStater = 0; currStater < numberOfStates; currStater++)
				{
					mapOfPossibleNextStatesForAllStates.get (currStater).put (action_i, 0); // regardless
																							// of
																							// whatetever
																							// currentStateINT,
																							// if
																							// action_i=0
																							// is
																							// executed,
																							// you
																							// will
																							// proceed
																							// to
																							// stateINT=0.
				}
			}
			
		}
		
		return mapOfPossibleNextStatesForAllStates;
	}
	
	
	public static Map<Integer, Map<Integer, Integer>> createMappingOfPossibleStatesForEachStateWithDatabase (int numberOfStates, int numberOfActions)
	{
		// Map<currentStateINT, Map<actionINT, nextStateINT>>
		Map<Integer, Map<Integer, Integer>> mapOfPossibleNextStatesForAllStates = new HashMap<Integer, Map<Integer, Integer>> ();
		for (int state_j = 0; state_j < numberOfStates; state_j++)
		{
			Map<Integer, Integer> possibleNextStatesForState_j = new HashMap<Integer, Integer> (); // for
																										// state_j:
																										// the
																										// mapping
																										// between
																										// <key>actionINT
																										// and
																										// <value>
																										// nextStateINT
			mapOfPossibleNextStatesForAllStates.put (state_j, possibleNextStatesForState_j);
		}
//		double[][] countMatrixForaction_i;	// GTD Not used
//		double[] countsForCurrentState_r;	// GTD Not used
//		double[] subRowCountVector_r;	// GTD Not used
		int mostProbableNextState = 0;
//		Pair<Integer, Double> positionAndValueOfMaxCol;	// GTD Not used
		Long maxCountVal = null;
		// int maxCountNextState =0;
		int action_iKey = 0;
		
		for (int a = 0; a < numberOfActions; a++)
		{
			
			// We cannot use the actionINTs in treeTraversalCountMatricesmap_table because this only
			// contains those actionINT values that were observed in our experimental paths, which
			// may only include only a subset of all possible actions.
			// However, the mappOfPossibleNextStatesForAllStates needs to account for every possible
			// action
			///// String cqlGetDistinctActionKeysStatement = "select distinct actionint FROM
			// treetraversalcountmatricesmap_table";
			///// for (Row akeyrow: session.execute(cqlGetDistinctActionKeysStatement)) {
			
			//// action_iKey = akeyrow.getInt("actionINT");
			action_iKey = a;
			
			if (action_iKey != 0)
			{ // we don't want to consider those actions that were solely used for padding purposes
				for (int currStater = 1; currStater < numberOfStates; currStater++)
				{ // do not consider stateINT=0 (row 0) corresponding to the state used for padding
					
					mostProbableNextState = currStater;
					
					String cqlGetMaxCountValForCurrentStateStatement = "select max(count) FROM treetraversalcountmatricesmap_table where actionint="
							+ action_iKey + " and currentstateint=" + currStater;
					for (Row crow : session.execute (cqlGetMaxCountValForCurrentStateStatement))
					{
						maxCountVal = crow.getLong ("system.max(count)");
					}
					
					String cqlGetNextStateIntForMaxValStatement = "select * from treetraversalcountmatricesmap_table where actionint="
							+ action_iKey + " and currentstateint =" + currStater + " and count="
							+ maxCountVal + " ALLOW FILTERING";
					for (Row srow : session.execute (cqlGetNextStateIntForMaxValStatement))
					{
						mostProbableNextState = srow.getInt ("nextstateint"); // if such a state
																				// does NOT exist
																				// ideally
																				// maxCountNextStateInt
																				// will remain the
																				// same as
																				// currentStateINT
					}
					
					// if(positionAndValueOfMaxCol.getSecond() >0) {
					// mostProbableNextState = positionAndValueOfMaxCol.getFirst()+1; //remember its
					// returning the index value of the subvector that begins at state1 and not at
					// state0
					// }
					// else {
					// mostProbableNextState = currStater; //if under action_i, no NEXTSTATEINT has
					// ever been observed to occur, we simply assume that currentStateINT will
					// persist in its given state when action_i is executed
					// }
					mapOfPossibleNextStatesForAllStates.get (currStater).put (action_iKey,
							mostProbableNextState); // iteratively updating for each currentStateINT
													// what the next possible state is if action_i
													// was the action executed
					
				}
				mapOfPossibleNextStatesForAllStates.get (0).put (action_iKey, 0); // for each
																					// action_i, if
																					// the
																					// currentStateInt=0,
																					// we will
																					// always
																					// persist in
																					// stateInt=0
				
			}
			else if (action_iKey == 0)
			{
				for (int currStater = 0; currStater < numberOfStates; currStater++)
				{
					mapOfPossibleNextStatesForAllStates.get (currStater).put (action_iKey, 0); // regardless
																								// of
																								// whatetever
																								// currentStateINT,
																								// if
																								// action_i=0
																								// is
																								// executed,
																								// you
																								// will
																								// proceed
																								// to
																								// stateINT=0.
				}
			}
			
		}
		
		return mapOfPossibleNextStatesForAllStates;
	}
	
	
	public void setNextPossibleStatesMAPForAllStates (
			Map<Integer, Map<Integer, Integer>> nextPossibleStatesMappingForAllStates)
	{
		_nextPossibleStatesMAPForAllStates = nextPossibleStatesMappingForAllStates;
	}
	
	
	public Map<Integer, Map<Integer, Integer>> getNextPossibleStatesMAPForAllStates ()
	{
		return _nextPossibleStatesMAPForAllStates;
	}
	
	
	public void setNumStatesInMDP (int totalNumStates)
	{
		_numStatesInMDP = totalNumStates;
	}
	
	
	public int getNumStatesInMDP ()
	{
		return _numStatesInMDP;
	}
	
	
	public void setNumActionsInMDP (int totalNumActions)
	{
		_numActionsInMDP = totalNumActions;
	}
	
	
	public int getNumActionsInMDP ()
	{
		return _numActionsInMDP;
	}
	
	
	public void setDoublesStateAndActiontrajectoriesForAllSamplesMap (Map<String, List<double[][]>> dblSATrajsForAllSamplesMAP)
	{
		_doublesStateAndActiontrajectoriesForAllSamplesMap = dblSATrajsForAllSamplesMAP;
	}
	
	
	public Map<String, List<double[][]>> getDoublesStateAndActiontrajectoriesForAllSamplesMap ()
	{
		return _doublesStateAndActiontrajectoriesForAllSamplesMap;
	}
	
	
	public static class TrajectoryTimeStep
	{
		
		protected TrajectoryTimeStep	previous;
//		private TrajectoryTimeStep		next;	// GTD Not used
		protected UUID					currentStateuid;
		protected UUID					currentActionuid;
		
		
		public TrajectoryTimeStep (TrajectoryTimeStep parent, UUID currStateUID, UUID currActionUID)
		{
			this.previous = parent;
			this.currentStateuid = currStateUID;
			this.currentActionuid = currActionUID;
		}
	}
	
	
	public static class Node
	{
		int				Id;
		String			VCFFileName;
		Node			parentNode;
		List<Node>		childrenNodeList;
		
		// JK 2.12.2019: adding states and edges
		UUID			node_stateUUID;
		
		HashSet<UUID>	setOfChildrenEdge_actions;
		HashSet<UUID>	setOfParentEdge_actions;
		
		
		public Node (int subcloneIdx, String fileName)
		{
			Id = subcloneIdx;
			VCFFileName = fileName;
			setOfChildrenEdge_actions = new HashSet<UUID> ();
			setOfParentEdge_actions = new HashSet<UUID> ();
		}
		
		
		public Node (int subcloneIdx, String fileName, Session cassSession)
		{
			Id = subcloneIdx;
			// VCFFileName = fileName;
			setOfChildrenEdge_actions = new HashSet<UUID> ();
			setOfParentEdge_actions = new HashSet<UUID> ();
			String ancestryFileName = fileName;
			String period_delimitter = "[.]";
			String[] tokensA = ancestryFileName.split (period_delimitter);
			String partA = tokensA[1];
			String underscore_delimitter = "[_]";
			String[] tokensB = partA.split (underscore_delimitter);
			String sampleName = tokensB[0] + "_" + tokensB[1];
			// note this HAS to be the same for all VCF files!!
			String subCloneWGSName = "subclone." + subcloneIdx+ ".unique.ANN.SNPEFF.reduced.FILTERED." + sampleName;
			
			// note this HAS to be the same for all VCF files!!
			VCFFileName = "subclone." + subcloneIdx + ".unique.ANN.SNPEFF.reduced.FILTERED."+ sampleName + ".vcf";
			
			String cqlGetStateStatement = "select * FROM treepaths_table where subclonewgsname='" + subCloneWGSName + "' allow filtering";
			
			for (Row row : cassSession.execute (cqlGetStateStatement))
			{
				// String rowAsString = row.toString();
				UUID stuuid = row.getUUID ("stateuuid");
				node_stateUUID = stuuid;
				
				System.out.println (node_stateUUID);
				
				// System.out.println(rowAsString);
				
			}
			
		}
		
		
		public void setChildren (List<Node> childrenNodes)
		{
			childrenNodeList = childrenNodes;
		}
		
		
		public void setParentNode (Node parent)
		{
			parentNode = parent;
		}
		
		
		public void setNodeStateUUID (UUID stateUUID)
		{
			node_stateUUID = stateUUID;
		}
		
		
		public void setChildEdgeActions (HashSet<UUID> childActionsSet)
		{
			setOfChildrenEdge_actions = childActionsSet;
		}
		
		
		public void setParentEdgeActions (HashSet<UUID> parentActionsSet)
		{
			setOfParentEdge_actions = parentActionsSet;
		}
		
		
		public void appendChildEdgeAction (UUID childActionUUID)
		{
			setOfChildrenEdge_actions.add (childActionUUID);
		}
		
		
		public void appendParentEdgeAction (UUID parentActionUUID)
		{
			setOfParentEdge_actions.add (parentActionUUID);
		}
		
		
		public HashSet<UUID> getParentEdgeActions ()
		{
			return setOfParentEdge_actions;
		}
		
		
		public HashSet<UUID> getChildEdgeActions ()
		{
			return setOfChildrenEdge_actions;
		}
	}
	
	
	public static void main (String args[]) 
	{
		String serverIP = "127.0.0.1";
		String keyspacejk = "crckeyspace";
		Cluster clusterX = Cluster.builder ().addContactPoint (serverIP).build ();
		Session sessionX = cluster.connect (keyspacejk);
		
		// text file specifies the path to the AncestryRelationship .txt file specific the relationship between subclone idxs
		String inputPathForLISTOFAncestryRelationshipsFILE_str = 
				"/Users/m186806/tIDFilteredDriverOnlyVCFs/allTIDFilteredSUBFILES/listOfAncestryFiles.txt";
		// text file specifiying the path to the directory containing all DRIVER-EDGE subvcfs corresponding to the actions 
		// leading to the subclones for a given tumor-sample
		String inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILE_str = 
				"/Users/m186806/tIDFilteredDriverOnlyVCFs/allTIDFilteredSUBFILES/listOfSampleSpecificDRIVERSSubcloneVCFsDirectories.txt";
		
		String outputPathForCSVFilesStr = "/Users/m186806";
		boolean insertPathsIntoDBAsDynamicallyCreated = false;
		
		@SuppressWarnings ("unused")
		TreeTraversal treeTravInstance = new TreeTraversal (clusterX, sessionX,
				inputPathForLISTOFAncestryRelationshipsFILE_str,
				inputPathForLISTOFTreeSpecificSubcloneDRIVERSOnlyDirPathsFILE_str,
				outputPathForCSVFilesStr, insertPathsIntoDBAsDynamicallyCreated);
		
		cluster.close ();
		System.out.println ("Cluster closed");
	}
	
}
