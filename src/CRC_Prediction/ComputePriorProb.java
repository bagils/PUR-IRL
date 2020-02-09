/**
 * IRLJK
 *
 * <p>@author Gregory Dougherty</p>
 * Copyright Mayo Clinic, 2019
 *
 */
package CRC_Prediction;

import java.util.concurrent.BlockingQueue;

/**
 * 
 *
 * <p>@author Gregory Dougherty</p>
 * Copyright 2019, Mayo Foundation for Medical Education and Research
 */
public class ComputePriorProb implements Runnable
{
	private BlockingQueue<Integer>	tableIndexes;
	private int						tblIndex;
	private double[][]				tableAssignmentMatrix;
	private IRLAlgorithmCancer		irlalgo;
	private double[]				priorProbDistribution;
	
	
	/**
	 * @param tableIndexes	{@link BlockingQueue} that gives access to tables that still need to be processed
	 * @param tblIndex
	 * @param tableAssignmentMatrix
	 * @param irlalgo
	 * @param priorProbDistribution
	 */
	public ComputePriorProb (BlockingQueue<Integer> tableIndexes, int tblIndex, double[][] tableAssignmentMatrix, 
							 IRLAlgorithmCancer irlalgo, double[] priorProbDistribution)
	{
		this.tableIndexes = tableIndexes;
		this.tblIndex = tblIndex;
		this.tableAssignmentMatrix = tableAssignmentMatrix;
		this.irlalgo = irlalgo;
		this.priorProbDistribution = priorProbDistribution;
	}
	
	
	/* (non-Javadoc)
	 * @see java.lang.Runnable#run()
	 */
	public void run ()
	{
		Integer	curIndex;
		
		 // prior probability of table index/label 'table_i' = # of trajectories that have been assigned label/index 'table_i'
		while ((curIndex = tableIndexes.poll ()) != null)
		{
  		// should return a numTraj x 1 column matrix with 1's at entries corresponding to customer assigned to table 'table_i'
			int	tableIndex = curIndex.intValue ();
			int	logMatrSumVal = InferenceAlgoCancer.countMatches (tableAssignmentMatrix, (double) tableIndex);
			
			if (tableIndex == tblIndex)
			{
				// remove the contribution of trajectory 'customer_i' to the prior probability of table index/label being 'tblIndex'
				logMatrSumVal -= 1;
			}
			if (logMatrSumVal > 0)
			{
				// element 0 of this double[] corresponds to table 1!!!   be careful
				priorProbDistribution[tableIndex - 1] = logMatrSumVal - irlalgo.getDiscountHyperparameter ();
			}
			else
			{
				// need this else condition if the count for a tableIndex =0; in which case the
				// priorProb for table would become negative because we are subtracting the val of discount
				priorProbDistribution[tableIndex - 1] = logMatrSumVal; 
				// element 0 of this double[] corresponds to table 1!!!   be careful
			}
			
		}
	}
	
}
