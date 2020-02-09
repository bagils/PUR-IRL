/**
 * IRLJK
 *
 * <p>@author Gregory Dougherty</p>
 * Copyright Mayo Clinic, 2020
 *
 */
package CRC_Prediction;

import CRC_Prediction.Utils.MatrixUtilityJBLAS;
import com.google.common.collect.Multimap;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import org.jblas.DoubleMatrix;

/**
 * Class that provides a threaded updater
 *
 * <p>@author Gregory Dougherty</p>
 * Copyright 2019, Mayo Foundation for Medical Education and Research
 */
public class TrajectoryNoDBUpdater implements Runnable
{
	private BlockingQueue<Integer>			customerIndexes;
	private MDPCancer						environment;
	private IRLAlgorithmCancer				irlAlgo;
	private double[][]						tableAssignmentMatrix;
	private Map<Integer, double[][]>		tableWeightVectors;
	private Map<Integer, double[][]>		tablePolicyVectors;
	private Map<Integer, double[][]>		tableValueVectors;
	private Map<Integer, double[][]>		tableQVectors;
	private Map<Integer, Double>			restaurantLikelihoods;
	private Map<Integer, Double>			restaurantPriors;
	private Map<Integer, DoubleMatrix>		restGradientsLLH;
	private Map<Integer, DoubleMatrix>		restGradientsPrior;
	private Multimap<Integer, double[]>		stateActionPairCountsInfoForAllTrajectories;
	private RestaurantMap					rmap;
	
	
	/**
	 * 
	 * @param customerIndexes	{@link BlockingQueue} that gives access to tables that still need to be processed
	 * @param environment
	 * @param irlAlgo
	 * @param tableAssignmentMatrix
	 * @param tableWeightVectors
	 * @param tablePolicyVectors
	 * @param tableValueVectors
	 * @param tableQVectors
	 * @param restaurantLikelihoods
	 * @param restaurantPriors
	 * @param restGradientsLLH
	 * @param restGradientsPrior
	 */
	public TrajectoryNoDBUpdater (BlockingQueue<Integer> customerIndexes, MDPCancer environment, IRLAlgorithmCancer irlAlgo,
			double[][] tableAssignmentMatrix, Map<Integer, double[][]> tableWeightVectors,
			Map<Integer, double[][]> tablePolicyVectors, Map<Integer, double[][]> tableValueVectors,
			Map<Integer, double[][]> tableQVectors, Map<Integer, Double> restaurantLikelihoods,
			Map<Integer, Double> restaurantPriors, Map<Integer, DoubleMatrix> restGradientsLLH,
			Map<Integer, DoubleMatrix> restGradientsPrior, Multimap<Integer, double[]> stateActionPairCountsInfoForAllTrajectories)
	{
		this.customerIndexes = customerIndexes;
		this.environment = environment;
		this.irlAlgo = irlAlgo;
		this.tableAssignmentMatrix = tableAssignmentMatrix;
		this.tableWeightVectors = tableWeightVectors;
		this.tablePolicyVectors = tablePolicyVectors;
		this.tableValueVectors = tableValueVectors;
		this.tableQVectors = tableQVectors;
		this.restaurantLikelihoods = restaurantLikelihoods;
		this.restaurantPriors = restaurantPriors;
		this.restGradientsLLH = restGradientsLLH;
		this.restGradientsPrior = restGradientsPrior;
		this.stateActionPairCountsInfoForAllTrajectories = stateActionPairCountsInfoForAllTrajectories;
		rmap = null;
	}


	/* (non-Javadoc)
	 * @see java.lang.Runnable#run()
	 */
	public void run ()
	{
		Integer			tblIndex;
		
		while ((tblIndex = customerIndexes.poll ()) != null)
		{
			int	customerID = tblIndex.intValue ();
			int	maxTableIndex = (int) MatrixUtilityJBLAS.matrixMaximum (tableAssignmentMatrix);
			
//			System.out.print ("Calling InferenceAlgoCancer.updateTableAssignmentWithDatabase with customerID: ");
//			System.out.println (customerID);
			rmap = InferenceAlgoCancer.updateTableAssignment (0, 0, environment, customerID, maxTableIndex, irlAlgo, tableAssignmentMatrix, 
															  tableWeightVectors, tablePolicyVectors, tableValueVectors, tableQVectors, 
															  restaurantLikelihoods, restaurantPriors, restGradientsLLH, restGradientsPrior, 
															  stateActionPairCountsInfoForAllTrajectories);
			tableAssignmentMatrix = rmap._restaurantAssignmentMatrix;
			tableWeightVectors = rmap._restaurantTableWeightMatrices;
			tablePolicyVectors = rmap._restaurantTablePolicyMatrices;
			tableValueVectors = rmap._restaurantTableValueMatrices;
			tableQVectors = rmap._restaurantTableQMatrices;

			restaurantLikelihoods = rmap._restLikeLihoods;
			restaurantPriors = rmap._restPriors;
			restGradientsLLH = rmap._restGradientsFromLLH;
			restGradientsPrior = rmap._restGradientsFromPrior;
		}
	}
	
	
	/**
	 * @return the rmap
	 */
	public final RestaurantMap getRmap ()
	{
		return rmap;
	}
	
}
