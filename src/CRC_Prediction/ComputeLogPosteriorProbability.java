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

/**
 * 
 *
 * <p>@author Gregory Dougherty</p>
 */
public class ComputeLogPosteriorProbability implements Runnable
{
	private BlockingQueue<Integer>		tableIndexes;
	private List<double[][]>			trajSet;
	private double[][]					tableAssignmentMatrix;
	private Map<Integer, double[][]>	tblWeightVectors;
	private Map<Integer, double[][]>	tblPolicyVectors;
	private MDPCancer					environment;
	private IRLAlgorithmCancer			irlAlgo;
	private boolean						partOfInitialDPMPosterior;
	private double[]					logLikelihoods;
	private double[]					logPriorProbs;
	
	/**
	 * 
	 * @param tableIndexes	{@link BlockingQueue} that gives access to tables that still need to be processed
	 * @param trajSet
	 * @param tableAssignmentMatrix
	 * @param tblWeightVectors
	 * @param tblPolicyVectors
	 * @param environment
	 * @param irlAlgo
	 * @param partOfInitialDPMPosterior
	 * @param logLikelihoods
	 * @param logPriorProbs
	 */
	public ComputeLogPosteriorProbability (BlockingQueue<Integer> tableIndexes, List<double[][]> trajSet, double[][] tableAssignmentMatrix, 
											Map<Integer, double[][]> tblWeightVectors, Map<Integer, double[][]> tblPolicyVectors, 
											MDPCancer environment, IRLAlgorithmCancer irlAlgo, boolean partOfInitialDPMPosterior, 
											double[] logLikelihoods, double[] logPriorProbs)
	{
		this.tableIndexes = tableIndexes;
		this.trajSet = trajSet;
		this.tableAssignmentMatrix = tableAssignmentMatrix;
		this.tblWeightVectors = tblWeightVectors;
		this.tblPolicyVectors = tblPolicyVectors;
		this.environment = environment;
		this.irlAlgo = irlAlgo;
		this.partOfInitialDPMPosterior = partOfInitialDPMPosterior;
		this.logLikelihoods = logLikelihoods;
		this.logPriorProbs = logPriorProbs;
	}
	
	
	public void run ()
	{
		Integer	tblIndex;
		
		while ((tblIndex = tableIndexes.poll ()) != null)
		{
			int	table = tblIndex.intValue ();
			InferenceAlgoCancer.computeLogPosteriorProbabilityForDirichletProcessMixture (
					table, trajSet, tableAssignmentMatrix, tblWeightVectors, tblPolicyVectors, 
					environment, irlAlgo, partOfInitialDPMPosterior, logLikelihoods, logPriorProbs);
		}
	}
	
}
