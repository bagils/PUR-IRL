/**
 * IRLJK
 *
 * <p>@author Gregory Dougherty</p>
 * Copyright Mayo Clinic, 2019
 *
 */
package CRC_Prediction;

import java.util.Map;
import java.util.concurrent.BlockingQueue;

/**
 * Class so can multi-thread {@link InferenceAlgoCancer#generateNewWeights(int, MDPCancer, IRLAlgorithmCancer, Map, Map, Map, boolean)}
 *
 * <p>@author Gregory Dougherty</p>
 */
@SuppressWarnings ("javadoc")
public class TableWeightCreator implements Runnable
{
	private BlockingQueue<Integer>		tableIndexes;
	private MDPCancer					env;
	private IRLAlgorithmCancer			irloptions;
	private Map<Integer, double[][]>	tblWeightVectors;
	private Map<Integer, double[][]>	tblPlVectors;
	private Map<Integer, double[][]>	tblVlVectors;
	private Map<Integer, double[][]>	tblQVectors;
	private boolean						changeWeights;
	
	
	/**
	 * @param tableIndexes	{@link BlockingQueue} that gives access to tables that still need to be processed
	 * @param env
	 * @param irloptions
	 * @param tblWeightVectors
	 * @param tblPlVectors
	 * @param tblVlVectors
	 * @param changeWeights
	 */
	public TableWeightCreator (BlockingQueue<Integer> tableIndexes, MDPCancer env, IRLAlgorithmCancer irloptions, 
								Map<Integer, double[][]> tblWeightVectors, Map<Integer, double[][]> tblPlVectors, 
								Map<Integer, double[][]> tblVlVectors, Map<Integer, double[][]> tblQVectors, boolean changeWeights)
	{
		this.tableIndexes = tableIndexes;
		this.env = env;
		this.irloptions = irloptions;
		this.tblWeightVectors = tblWeightVectors;
		this.tblPlVectors = tblPlVectors;
		this.tblVlVectors = tblVlVectors;
		this.tblQVectors = tblQVectors;
		this.changeWeights = changeWeights;
	}
	
	
	/* (non-Javadoc)
	 * @see java.lang.Runnable#run()
	 */
	public void run ()
	{
		Integer	tblIndex;
		
		while ((tblIndex = tableIndexes.poll ()) != null)
		{
			InferenceAlgoCancer.generateNewWeights (tblIndex.intValue (), env, irloptions, tblWeightVectors, tblPlVectors, 
													tblVlVectors, tblQVectors, changeWeights);
		}
	}
	
}
