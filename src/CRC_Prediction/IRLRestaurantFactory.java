/**
 * 
 */
package CRC_Prediction;

/**
 * @author John Kalantari
 * Copyright 2019, Mayo Foundation for Medical Education and Research
 *
 */



import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.Map.Entry;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import org.jblas.DoubleMatrix;

import com.datastax.driver.core.Cluster;
import com.datastax.driver.core.ResultSet;
import com.datastax.driver.core.Row;
import com.datastax.driver.core.Session;

import CRC_Prediction.Utils.MatrixUtilityJBLAS;

/**
 * Implements the byte-based IRLRestaurant factory to generate an IRLRestaurant model initialzed with specific parameters.
*/
public class IRLRestaurantFactory  {

	String _outputDirectoryNameStr;
	
	public IRLRestaurantFactory(String outputDirName) {
		_outputDirectoryNameStr = outputDirName;
	}

    /**
     * 
     * Deserialize and instantiate IRLRestaurant from file (containing serialized IRLRestaurant object)
     */
    public IRLRestaurant get(File serializedIRLRestaurant) {
            try {
            	System.out.println("Trying to get() an IRLRestaurant from file:"+ serializedIRLRestaurant);
                ObjectInputStream ois = null;
                IRLRestaurant otox = null;
                
                //from the input URL create a new ObjectInputStram object
                //..url argument should be in the form:-otomurl file:///users/john/documents/workspace/Henri/serializedLearnedOtomaton.txt
                //ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(serializedIRLRestaurant)));
                ois = new ObjectInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(serializedIRLRestaurant))));

                
                //reads an object from ois object and casts it as an IRLRestaurant class object
                //....NOTE: this method SHOULD call IRLRestaurant's own readObject() class method for proper deserialization!
                otox = (IRLRestaurant) ois.readObject();
                
                ois.close();

 
                //return the IRLRestaurant 'otox' to be used as the model with its already existing parameters
                return otox;
                
            } catch (ClassNotFoundException ex) {
                Logger.getLogger(IRLRestaurantFactory.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(IRLRestaurantFactory.class.getName()).log(Level.SEVERE, null, ex);
            }
        
        return null;
    }
    
    public void write(IRLRestaurant bestirlrestaurant) throws Exception, FileNotFoundException, IOException, IllegalArgumentException { 
    	
		String timeStamp = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss.SSS").format(new Date());


		String outputFileNameForSerializedIRLRestaurant = _outputDirectoryNameStr+"/"+timeStamp+".restaurant.serialized";
        //ObjectOutputStream oos1 = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(outputFileNameForSerializedIRLRestaurant)));
        ObjectOutputStream oos1 = new ObjectOutputStream(new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(outputFileNameForSerializedIRLRestaurant))));

        try {
        	//Store the best IRLRestaurant model
        	
            oos1.writeObject(bestirlrestaurant);
            System.out.println("Stored the best IRLRestaurant model : "+outputFileNameForSerializedIRLRestaurant);
            
            oos1.flush();
            
            oos1.close();
            
        } catch (IOException ioe) {
            System.err.println("Error writing IRLRestaurant model file -- probably corrupt -- try again");
            throw ioe;
        }
    	
    }
    
    
    public void uploadToCassandra(IRLRestaurant irlrestaurant, Session cassSession, String fileNameOfSerializedIRLRestaurant) {
    	
    	System.out.println("Trying to upload an IRLRestaurant to cassandra DB keyspace:"+ cassSession.getLoggedKeyspace());
    	
    	Map<Integer, double[][]> weightMatrices = irlrestaurant.getWeightMatrices();
    	Map<Integer, double[][]> policyFunctions = irlrestaurant.getPolicyMatrices();
    	Map<Integer, double[][]> valueFunctions = irlrestaurant.getValueMatrices();
    	Map<Integer, double[][]> qFunctions = irlrestaurant.getQMatrices();
//    	Map<Integer, double[][]> rewardFunctions = new HashMap<Integer, double[][]> ();	// Not used // GTD 12/24/19
    	
    	
		////////////////////////////////////////////////////////////////////
		// Create mapping based on Cassandra db state_space table (based on GLFM created
		// states/patterns)
		Map<UUID, Integer> stateUUIDToIntegerMapForStatesInDB = new HashMap<UUID, Integer> ();
		Map<Integer, UUID> stateINTToUUIDMapForStatesInDB = new HashMap<Integer, UUID> ();

		UUID stateUUIDinDB = null;
		Integer stateINTinDB = 0;
		Long stateINT_longval_inDB = (long) 0;
		String cqlGetStateRowsStatement = "select * from statespace_table";
		ResultSet rsStates = cassSession.execute (cqlGetStateRowsStatement);
		for (Row r_state : rsStates)
		{
			stateUUIDinDB = r_state.getUUID ("stateuuid");
			stateINT_longval_inDB = r_state.getLong ("stateint");
			
			stateINTinDB = stateINT_longval_inDB.intValue ();
			stateUUIDToIntegerMapForStatesInDB.put (stateUUIDinDB, stateINTinDB);
			stateINTToUUIDMapForStatesInDB.put(stateINTinDB, stateUUIDinDB);
		}
		////////////////////////////////////////////////////////////////////

		
		////////////////////////////////////////////////////////////////////
		//Create mapping based on Cassandra db basis-vectors table
		Map<UUID, Integer> basisVectorUUIDToIntMAP = new HashMap<UUID, Integer> ();
		Map<Integer, UUID> basisVectorINTToUUIDMAP = new HashMap<Integer, UUID> ();

		
		
		String cqlGetBasisVectorsStatement = "select * FROM statebasisvectors_table";
		for (Row b_row : cassSession.execute (cqlGetBasisVectorsStatement))
		{
			UUID bvectoruuid = b_row.getUUID ("basisvectoruuid");
			Long bvector_longval = b_row.getLong ("basisvectorint");
			
			Integer bvectorINTEGER = bvector_longval.intValue ();
			basisVectorUUIDToIntMAP.put (bvectoruuid, bvectorINTEGER);
			basisVectorINTToUUIDMAP.put(bvectorINTEGER, bvectoruuid);
		}
		////////////////////////////////////////////////////////////////////
		
		
		////////////////////////////////////////////////////////////////////
		// Create mapping based on Cassandra db action_space table 
		Map<UUID, Integer> actionUUIDToIntegerMapForActionsInDB = new HashMap<UUID, Integer> ();
		Map<Integer, UUID> actionINTToUUIDMapForActionsInDB = new HashMap<Integer, UUID> ();

		
		UUID actionUUIDinDB = null;
		Integer actionINTinDB = 0;
		Long actionINT_longval_inDB = (long) 0;
		String cqlGetActionRowsStatement = "select * from actionspace_table";
		ResultSet rsActions = cassSession.execute (cqlGetActionRowsStatement);
		for (Row r_action : rsActions)
		{
			actionUUIDinDB = r_action.getUUID ("actionuuid");
			actionINT_longval_inDB = r_action.getLong ("actionint");
			
			actionINTinDB = actionINT_longval_inDB.intValue ();
			actionUUIDToIntegerMapForActionsInDB.put (actionUUIDinDB, actionINTinDB);
			actionINTToUUIDMapForActionsInDB.put(actionINTinDB, actionUUIDinDB);
			
		}
		////////////////////////////////////////////////////////////////////
		
		String cqlInsertWeightFunctionFeatureStatement = "";
		String cqlInsertPolicyFeatureStatement = "";
		String cqlInsertValueFunctionFeatureStatement = "";
		String cqlInsertQFunctionFeatureStatement = "";
		
		int numBVectors = basisVectorINTToUUIDMAP.size ();
//		int numStates = stateINTToUUIDMapForStatesInDB.size ();	// Not used // GTD 12/24/19
//		int numActions = actionINTToUUIDMapForActionsInDB.size ();	// Not used // GTD 12/24/19
		
		// Add each table's weightmatrix to Cassandra table
		for (Entry<Integer, double[][]> entry : weightMatrices.entrySet ())
		{
			Integer tableIdx = entry.getKey ();
			double[][] weightMatrix = entry.getValue ();
			// tblRewardFunction = recapitulateRewardFunction(environment, weightMatrix);
			// rewardFunctions.put(tableIdx, tblRewardFunction);
			int numRows = weightMatrix.length;
			int k = 0;
			
			// Add one weight-matrix feature at a time
			for (int j = 0; j < numRows; j++)
			{
				
				if (j < numBVectors)
				{ // recall that the first |numBVectors| reward features are basis-vector specific (
					
					UUID basisVUUID = basisVectorINTToUUIDMAP.get (j);
					cqlInsertWeightFunctionFeatureStatement = "INSERT INTO crckeyspace.weightFunctions_table (serializationFileName, tableINT, " +
															  "rewardFeatureINT, basisVectorINT, basisvectoruuid, weight) values ('" + 
															  fileNameOfSerializedIRLRestaurant + "'," + tableIdx + "," + j + "," + j + "," + 
															  basisVUUID + "," + weightMatrix[j][0] + ")";
				}
				else
				{ // rest of reward features are action specific
					k = j - numBVectors;
					UUID actionUUID = actionINTToUUIDMapForActionsInDB.get (k);
					cqlInsertWeightFunctionFeatureStatement = "INSERT INTO crckeyspace.weightFunctions_table (serializationFileName, tableINT, " +
															  "rewardFeatureINT , actionINT , actionuuid , weight) values ('" + 
															  fileNameOfSerializedIRLRestaurant + "'," + tableIdx + "," + j + "," + k + "," + 
															  actionUUID + "," + weightMatrix[j][0] + ")";
				}
				cassSession.execute (cqlInsertWeightFunctionFeatureStatement);
			}
		}
		
		//Add each table's policy function to Cassandra table
		for (Entry<Integer, double[][]> entry : policyFunctions.entrySet ())
		{
			
			Integer tableIdx = entry.getKey ();
			double[][] tblPolicyFunction = entry.getValue ();
			
			int numRows = tblPolicyFunction.length;
//			int k = 0;	// Not used // GTD 12/24/19
			
			// Add policy-matrix rows to Cassandra table
			for (int j = 0; j < numRows; j++)
			{
				
				UUID stateUUID = stateINTToUUIDMapForStatesInDB.get (j);
				int actionDBLtoInteger = (int) tblPolicyFunction[j][0];
				cqlInsertPolicyFeatureStatement = 
						"INSERT INTO crckeyspace.policies_table (serializationFileName, tableINT, stateINT, stateuuid, expectedPolicyAction) " + 
						"values ('" + fileNameOfSerializedIRLRestaurant + "'," + tableIdx + "," + j + "," + stateUUID + "," + actionDBLtoInteger +")";
				
				cassSession.execute (cqlInsertPolicyFeatureStatement);
			}
		}
		
		// Add each table's value function to Cassandra table
		for (Entry<Integer, double[][]> entry : valueFunctions.entrySet ())
		{
			
			Integer tableIdx = entry.getKey ();
			double[][] tblValueFunction = entry.getValue ();
			
			int numRows = tblValueFunction.length;
//			int k = 0;	// Not used // GTD 12/24/19
			
			// Add value-matrix rows to Cassandra table
			for (int j = 0; j < numRows; j++)
			{
				
				UUID stateUUID = stateINTToUUIDMapForStatesInDB.get (j);
				cqlInsertValueFunctionFeatureStatement = 
					"INSERT INTO crckeyspace.valueFunctions_table (serializationFileName, tableINT, stateINT, stateuuid, valueFunctionFeatureWeight) " +
					"values ('" + fileNameOfSerializedIRLRestaurant + "'," + tableIdx + "," + j + "," + stateUUID + "," + tblValueFunction[j][0] + ")";
				
				cassSession.execute (cqlInsertValueFunctionFeatureStatement);
			}
		}
		
		// Add each table's Q function to Cassandra table
		for (Entry<Integer, double[][]> entry : qFunctions.entrySet ())
		{
			
			Integer tableIdx = entry.getKey ();
			double[][] tblQFunction = entry.getValue ();
			
			int numRows = tblQFunction.length; // this is a full matrix of dimension = [numStates x numActions]
			int numCols = tblQFunction[0].length;
//			int Y = 0;	// Not used // GTD 12/24/19
//			int Z = 0;	// Not used // GTD 12/24/19
			
			// Add q-matrix rows to Cassandra table
			for (int j = 0; j < numRows; j++)
			{
				
				// Y = j%numStates;//state
				// Z= Math.floorDiv(j, numStates); //action
				UUID stateUUID = stateINTToUUIDMapForStatesInDB.get (j);
				for (int k = 0; k < numCols; k++)
				{
					UUID actionUUID = actionINTToUUIDMapForActionsInDB.get (k);
					cqlInsertQFunctionFeatureStatement = 
							"INSERT INTO crckeyspace.qFunctions_table (serializationFileName, tableINT, stateINT, stateuuid, actionINT, actionuuid, " +
							" qFunctionFeatureWeight ) values ('" + fileNameOfSerializedIRLRestaurant + "'," + tableIdx + "," + j + "," + 
							stateUUID + "," + k + "," + actionUUID + "," + tblQFunction[j][k] + ")";
					cassSession.execute (cqlInsertQFunctionFeatureStatement);
					
				}
				
			}
		}
		
	}
	
	
	public void uploadToCassandra (IRLRestaurant irlrestaurant, Session cassSession,
			String fileNameOfSerializedIRLRestaurant, MDPCancer envmdp,
			String fileNameOfSerializedMDp)
	{
		
		System.out.println ("Trying to upload an IRLRestaurant to cassandra DB keyspace:"
				+ cassSession.getLoggedKeyspace ());
		
		Map<Integer, double[][]> weightMatrices = irlrestaurant.getWeightMatrices ();
		Map<Integer, double[][]> policyFunctions = irlrestaurant.getPolicyMatrices ();
		Map<Integer, double[][]> valueFunctions = irlrestaurant.getValueMatrices ();
		Map<Integer, double[][]> qFunctions = irlrestaurant.getQMatrices ();
		Map<Integer, double[][]> rewardFunctions = new HashMap<Integer, double[][]> ();
		
		////////////////////////////////////////////////////////////////////
		// Create mapping based on Cassandra db state_space table (based on GLFM created
		// states/patterns)
		Map<UUID, Integer> stateUUIDToIntegerMapForStatesInDB = new HashMap<UUID, Integer> ();
		Map<Integer, UUID> stateINTToUUIDMapForStatesInDB = new HashMap<Integer, UUID> ();
		
		UUID stateUUIDinDB = null;
		Integer stateINTinDB = 0;
		Long stateINT_longval_inDB = (long) 0;
		String cqlGetStateRowsStatement = "select * from statespace_table";
		ResultSet rsStates = cassSession.execute (cqlGetStateRowsStatement);
		for (Row r_state : rsStates)
		{
			stateUUIDinDB = r_state.getUUID ("stateuuid");
			stateINT_longval_inDB = r_state.getLong ("stateint");
			
			stateINTinDB = stateINT_longval_inDB.intValue ();
			stateUUIDToIntegerMapForStatesInDB.put (stateUUIDinDB, stateINTinDB);
			stateINTToUUIDMapForStatesInDB.put(stateINTinDB, stateUUIDinDB);
		}
		////////////////////////////////////////////////////////////////////

		
		////////////////////////////////////////////////////////////////////
		//Create mapping based on Cassandra db basis-vectors table
		Map<UUID, Integer> basisVectorUUIDToIntMAP = new HashMap<UUID, Integer> ();
		Map<Integer, UUID> basisVectorINTToUUIDMAP = new HashMap<Integer, UUID> ();

		
		
		String cqlGetBasisVectorsStatement = "select * FROM statebasisvectors_table";
		for (Row b_row : cassSession.execute (cqlGetBasisVectorsStatement))
		{
			UUID bvectoruuid = b_row.getUUID ("basisvectoruuid");
			Long bvector_longval = b_row.getLong ("basisvectorint");
			
			Integer bvectorINTEGER = bvector_longval.intValue ();
			basisVectorUUIDToIntMAP.put (bvectoruuid, bvectorINTEGER);
			basisVectorINTToUUIDMAP.put(bvectorINTEGER, bvectoruuid);
		}
		////////////////////////////////////////////////////////////////////
		
		
		////////////////////////////////////////////////////////////////////
		// Create mapping based on Cassandra db action_space table 
		Map<UUID, Integer> actionUUIDToIntegerMapForActionsInDB = new HashMap<UUID, Integer> ();
		Map<Integer, UUID> actionINTToUUIDMapForActionsInDB = new HashMap<Integer, UUID> ();

		
		UUID actionUUIDinDB = null;
		Integer actionINTinDB = 0;
		Long actionINT_longval_inDB = (long) 0;
		String cqlGetActionRowsStatement = "select * from actionspace_table";
		ResultSet rsActions = cassSession.execute (cqlGetActionRowsStatement);
		for (Row r_action : rsActions)
		{
			actionUUIDinDB = r_action.getUUID ("actionuuid");
			actionINT_longval_inDB = r_action.getLong ("actionint");
			
			actionINTinDB = actionINT_longval_inDB.intValue ();
			actionUUIDToIntegerMapForActionsInDB.put (actionUUIDinDB, actionINTinDB);
			actionINTToUUIDMapForActionsInDB.put(actionINTinDB, actionUUIDinDB);
			
		}
		////////////////////////////////////////////////////////////////////
		
		String cqlInsertWeightFunctionFeatureStatement = "";
		String cqlInsertPolicyFeatureStatement = "";
		String cqlInsertValueFunctionFeatureStatement = "";
		String cqlInsertQFunctionFeatureStatement = "";
		String cqlInsertRewardFunctionFeatureStatement = "";
		
		int numBVectors = basisVectorINTToUUIDMAP.size ();
//		int numStates = stateINTToUUIDMapForStatesInDB.size ();	// Not used // GTD 12/24/19
//		int numActions = actionINTToUUIDMapForActionsInDB.size ();	// Not used // GTD 12/24/19
		
		double[][] tblRewardFunction = null;
		
		// Add each table's weightmatrix to Cassandra table
		for (Entry<Integer, double[][]> entry : weightMatrices.entrySet ())
		{
			
			Integer tableIdx = entry.getKey();
			double [][] weightMatrix = entry.getValue();
			//tblRewardFunction = recapitulateRewardFunction(environment, weightMatrix);
			//rewardFunctions.put(tableIdx, tblRewardFunction);
			int numRows = weightMatrix.length;
			int k=0;
			
			//Add one weight-matrix feature at a time
			for(int j=0; j< numRows; j++) {
				
				if(j <numBVectors) { //recall that the first |numBVectors| reward features are basis-vector specific (
					
					UUID basisVUUID = basisVectorINTToUUIDMAP.get(j);
					cqlInsertWeightFunctionFeatureStatement = "INSERT INTO crckeyspace.weightFunctions_table(serializationFileName   , "
							+ "tableINT , rewardFeatureINT , basisVectorINT , basisvectoruuid , "
							+ "weight ) values('"+fileNameOfSerializedIRLRestaurant+"',"+tableIdx+","	
							+ j + ","+j+"," + basisVUUID + ","+ weightMatrix[j][0]+")";
				}
				else { //rest of reward features are action specific
					k = j-numBVectors;
					UUID actionUUID = actionINTToUUIDMapForActionsInDB.get(k);
					cqlInsertWeightFunctionFeatureStatement = "INSERT INTO crckeyspace.weightFunctions_table(serializationFileName, "
							+ "tableINT , rewardFeatureINT , actionINT , actionuuid , "
							+ "weight) values('"+fileNameOfSerializedIRLRestaurant+"',"+tableIdx+","	
							+ j + ","+k+"," + actionUUID + ","+ weightMatrix[j][0]+")";
				}
				cassSession.execute (cqlInsertWeightFunctionFeatureStatement);
			}
			
			//recapitulate reward function from this table's weight-matrix and to map of all reward functions for restaurant (will add to Cassandra in subsequent loop)
			tblRewardFunction = recapitulateRewardFunction(envmdp, weightMatrix);
			rewardFunctions.put(tableIdx, tblRewardFunction);
		}
		
		//Add each table's policy function to Cassandra table
		for (Entry<Integer, double[][]> entry : policyFunctions.entrySet ())
		{
			
			Integer tableIdx = entry.getKey ();
			double[][] tblPolicyFunction = entry.getValue ();
			
			int numRows = tblPolicyFunction.length;
//			int k = 0;	// Not used // GTD 12/24/19
			
			// Add policy-matrix rows to Cassandra table
			for (int j = 0; j < numRows; j++)
			{
				UUID stateUUID = stateINTToUUIDMapForStatesInDB.get (j);
				int actionDBLtoInteger = (int) tblPolicyFunction[j][0];
				cqlInsertPolicyFeatureStatement = 
					"INSERT INTO crckeyspace.policies_table (serializationFileName, tableINT, stateINT, stateuuid, expectedPolicyAction) values ('" + 
					fileNameOfSerializedIRLRestaurant + "'," + tableIdx + "," + j + "," + stateUUID + "," + actionDBLtoInteger + ")";
				
				cassSession.execute (cqlInsertPolicyFeatureStatement);
			}
		}		
		
		
		//Add each table's value function to Cassandra table
		for (Entry<Integer, double[][]> entry : valueFunctions.entrySet ())
		{
			Integer tableIdx = entry.getKey ();
			double[][] tblValueFunction = entry.getValue ();
			
			int numRows = tblValueFunction.length;
//			int k = 0;	// Not used // GTD 12/24/19
			
			// Add value-matrix rows to Cassandra table
			for (int j = 0; j < numRows; j++)
			{
				
				UUID stateUUID = stateINTToUUIDMapForStatesInDB.get (j);
				cqlInsertValueFunctionFeatureStatement = 
					"INSERT INTO crckeyspace.valueFunctions_table (serializationFileName, tableINT, stateINT, stateuuid, valueFunctionFeatureWeight) " +
					"values ('" + fileNameOfSerializedIRLRestaurant + "'," + tableIdx + "," + j + "," + stateUUID + "," + tblValueFunction[j][0] + ")";
				
				cassSession.execute (cqlInsertValueFunctionFeatureStatement);
			}
		}		
		
		// Add each table's Q function to Cassandra table
		for (Entry<Integer, double[][]> entry : qFunctions.entrySet ())
		{
			Integer tableIdx = entry.getKey ();
			double[][] tblQFunction = entry.getValue ();
			
			int numRows = tblQFunction.length; // this is a full matrix of dimension = [numStates x numActions]
			int numCols = tblQFunction[0].length;
//			int Y = 0;	// Not used // GTD 12/24/19
//			int Z = 0;	// Not used // GTD 12/24/19
			
			// Add q-matrix rows to Cassandra table
			for (int j = 0; j < numRows; j++)
			{
				// Y = j%numStates;//state
				// Z= Math.floorDiv(j, numStates); //action
				UUID stateUUID = stateINTToUUIDMapForStatesInDB.get (j);
				for (int k = 0; k < numCols; k++)
				{
					UUID actionUUID = actionINTToUUIDMapForActionsInDB.get (k);
					cqlInsertQFunctionFeatureStatement = 
							"INSERT INTO crckeyspace.qFunctions_table (serializationFileName, tableINT, stateINT, stateuuid, actionINT, actionuuid, " +
							"qfunctionfeatureweight) values ('" + fileNameOfSerializedIRLRestaurant + "'," + tableIdx + "," + j + "," + stateUUID + 
							"," + k + "," + actionUUID + "," + tblQFunction[j][k] + ")";
					cassSession.execute (cqlInsertQFunctionFeatureStatement);
				}
			}
		}		
		
		// Add each table's REWARD function to Cassandra table
		for (Entry<Integer, double[][]> entry : rewardFunctions.entrySet ())
		{
			Integer tableIdx = entry.getKey ();
			tblRewardFunction = entry.getValue ();
			
			int numRows = tblRewardFunction.length; // this is a full matrix of dimension = [numStates x numActions]
			int numCols = tblRewardFunction[0].length;
//			int Y = 0;	// Not used // GTD 12/24/19
//			int Z = 0;	// Not used // GTD 12/24/19
			
			// Add reward function matrix rows to Cassandra table
			for (int j = 0; j < numRows; j++)
			{
//				Y = j % numStates;// state
//				Z = Math.floorDiv (j, numStates); // action
				UUID stateUUID = stateINTToUUIDMapForStatesInDB.get (j);
				for (int k = 0; k < numCols; k++)
				{
					UUID actionUUID = actionINTToUUIDMapForActionsInDB.get (k);
					cqlInsertRewardFunctionFeatureStatement = 
						"INSERT INTO crckeyspace.rewardfunctions_table (serializationFileName, tableINT, stateINT, stateuuid, actionINT, actionuuid, " +
						"rewardweight) values ('" + fileNameOfSerializedIRLRestaurant + "'," + tableIdx + "," + j + "," + stateUUID + "," + k + "," + 
						actionUUID + "," + tblRewardFunction[j][k] + ")";
					cassSession.execute (cqlInsertRewardFunctionFeatureStatement);
				}
				
			}
		}
		
	} // end uploadToCassandra() method
    
    
    public double[][] recapitulateRewardFunction(MDPCancer env, double[][] weightMatrix) {

//		RealMatrix rewardMatrix = MatrixUtils.createRealMatrix (env.getNumStates (), env.getNumActions ());	// Not used // GTD 12/24/19
//		RealMatrix weightedFMatrixBlock = null;	// Not used // GTD 12/24/19
		Map<Integer, double[][]> safmatrixMAP = env.getStateFeatureMatrixMAP ();
		
		///////////////////////////////////////////////////////////
		//JK alternative method to compute reward function 
		DoubleMatrix Fmatrix = MatrixUtilityJBLAS.convertMultiDimMatrixMap(safmatrixMAP);
		DoubleMatrix fwMatrix = Fmatrix.mmul(new DoubleMatrix(weightMatrix));
		DoubleMatrix fwREWARDmatrix = fwMatrix.reshape(env.getNumStates(), env.getNumActions());
		double [][] fwRewardDBLArray = fwREWARDmatrix.toArray2();
		return fwRewardDBLArray;
		///////////////////////////////////////////////////////////
				
//		for (Entry<Integer, double[][]> entry : safmatrixMAP.entrySet ())
//		{
//			Integer action_iINTEGER = entry.getKey ();
//			int action_i = action_iINTEGER.intValue ();
//			double[][] sfmatrixBLOCK = entry.getValue (); // retrieves the 64x16 matrix indicating which features are
//														 // pertinent for each state when executed with action_i;
//			if (sfmatrixBLOCK == null)
//			{
//				System.out.println ("sfmatrixBLOCK is null for action_i: " + action_i);
//				System.err.println ("sfmatrixBLOCK is null for action_i: " + action_i);
//				continue;
//			}
//			weightedFMatrixBlock = MatrixUtility.multiplyMatricesWithMatrixUtils (sfmatrixBLOCK, weightMatrix); 
//			// a 64x1 vector/matrix ( 64x16 matrix * 16x1 matrix) should become a 64x1 vector/matrix after multiplying 64x16 matrix (indicating
//			// pertinence/non-pertinence (0/1) of each feature) * 16x1 matrix indicating the weight of each feature for the
//			rewardMatrix.setColumnMatrix (action_i, weightedFMatrixBlock); // set the reward function for all state-action_i
//		}
//		double[][] rewardMatrix2DArray = rewardMatrix.getData ();
//		return rewardMatrix2DArray;
    }
    
    public static void main(String[] args) throws Exception, IllegalArgumentException
	{
		CRC_Prediction.CommandLineIRLOptions.ParseReturn parseReturn = CommandLineIRLOptions.parse (args);
		IRLRestaurant deserializedRestaurant = null;
		MDPCancer deserializedMDP = null;
		IRLRestaurantFactory restFactory = new IRLRestaurantFactory (parseReturn._outputDir);
		MDPCancerFactory mdpFactory = new MDPCancerFactory (parseReturn._outputDir);
		
		String serverIP = parseReturn._dbserverIP;
		String keyspacejk = parseReturn._dbkeyspace;
		Cluster cluster = Cluster.builder ().addContactPoint (serverIP).build ();
		Session csession = cluster.connect (keyspacejk);
		
		if (parseReturn._evalSerializedRestaurant)
		{
			File serialRestrFile = new File (parseReturn._serializedRestaurant);
			deserializedRestaurant = restFactory.get (serialRestrFile);
			String restaurantfileName = serialRestrFile.getName ();
			
			if (parseReturn._evalSerializedMDP)
			{
				File serialMDPFile = new File (parseReturn._serializedMDP);
				deserializedMDP = mdpFactory.get (serialMDPFile);
				String mdpFileName = serialMDPFile.getName ();
				restFactory.uploadToCassandra (deserializedRestaurant, csession, restaurantfileName, deserializedMDP, mdpFileName);
			}
			else
			{
				restFactory.uploadToCassandra (deserializedRestaurant, csession,
						restaurantfileName);
				
			}
		}
		
		cluster.close (); // close communication to cassandra cluster (otherwise will remain on indefinitely)
		
	}
    
}
