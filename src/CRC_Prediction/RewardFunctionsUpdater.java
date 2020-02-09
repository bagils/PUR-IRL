/**
 * IRLJK
 *
 * <p>@author Gregory Dougherty</p>
 * Copyright Mayo Clinic, 2019
 *
 */
package CRC_Prediction;

import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import org.jblas.DoubleMatrix;

/**
 * Class that provides a threaded updater
 *
 * <p>@author Gregory Dougherty</p>
 * Copyright 2019, Mayo Foundation for Medical Education and Research
 */
public class RewardFunctionsUpdater implements Runnable
{
	private BlockingQueue<Integer>			tableIndexes;
	private List<double[][]>				trajectorySet;
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
	private Map<Integer, RestaurantTable>	restResults;
	
	
	/**
	 * @param tableIndexes	{@link BlockingQueue} that gives access to tables that still need to be processed
	 * @param trajectorySet
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
	 * @param restResults
	 */
	public RewardFunctionsUpdater (BlockingQueue<Integer> tableIndexes, List<double[][]> trajectorySet, MDPCancer environment, 
									IRLAlgorithmCancer irlAlgo, double[][] tableAssignmentMatrix, Map<Integer, double[][]> tableWeightVectors, 
									Map<Integer, double[][]> tablePolicyVectors, Map<Integer, double[][]> tableValueVectors, 
									Map<Integer, double[][]> tableQVectors, Map<Integer, Double> restaurantLikelihoods, 
									Map<Integer, Double> restaurantPriors, Map<Integer, DoubleMatrix> restGradientsLLH, 
									Map<Integer, DoubleMatrix> restGradientsPrior, Map<Integer, RestaurantTable> restResults)
	{
		this.tableIndexes = tableIndexes;
		this.trajectorySet = trajectorySet;
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
		this.restResults = restResults;
	}


	/* (non-Javadoc)
	 * @see java.lang.Runnable#run()
	 */
	public void run ()
	{
		RestaurantTable	rmap6;
		Integer			tblIndex;
		
		while ((tblIndex = tableIndexes.poll ()) != null)
		{
			int	tableID = tblIndex.intValue ();
			
			rmap6 = InferenceAlgoCancer.updateRewardFunctions (trajectorySet, environment, tableID, irlAlgo, tableAssignmentMatrix, 
																tableWeightVectors.get (tableID), tablePolicyVectors.get (tableID), 
																tableValueVectors.get (tableID), tableQVectors.get (tableID), 
																restaurantLikelihoods.get (tableID), restaurantPriors.get (tableID), 
																restGradientsLLH.get (tableID), restGradientsPrior.get (tableID));
			synchronized (restResults)
			{
//				System.out.print ("Values of rmap6._restaurantTableWeightMatrices table: ");
//				System.out.print (tableID);
//				System.out.print (": ");
//				StringUtils.dumpArray (rmap6._restaurantTableWeightMatrices, System.out);
//				System.out.print ("Dump of restResults before Put: ");
//				StringUtils.dumpMap (restResults, System.out);
				restResults.put (Integer.valueOf (tableID), rmap6);
//				System.out.print ("Dump of restResults after Put: ");
//				StringUtils.dumpMap (restResults, System.out);
			}
		}
	}
	
}
