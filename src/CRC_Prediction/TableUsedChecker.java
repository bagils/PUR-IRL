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
public class TableUsedChecker implements Runnable
{
	private double[][]			tblAssignmentMatrix;
	private List<double[][]>	wMatrix;
	private List<double[][]>	pMatrix;
	private List<double[][]>	vMatrix;
	private List<double[][]>	qMatrix;
	private List<double[]>		tableLabelMatrix;
	private BlockingQueue<Integer>		tableIndexes;
	private Map<Integer, double[][]>	tblWeightVectors;
	private Map<Integer, double[][]>	tblPlVectors;
	private Map<Integer, double[][]>	tblVlVectors;
	private Map<Integer, double[][]>	tblQVectors;
	
	
	/**
	 * @param tableIndexes	{@link BlockingQueue} that gives access to tables that still need to be processed
	 * @param tblAssignmentMatrix
	 * @param wMatrix
	 * @param pMatrix
	 * @param vMatrix
	 * @param qMatrix
	 * @param tableLabelMatrix
	 * @param tblWeightVectors
	 * @param tblPlVectors
	 * @param tblVlVectors
	 * @param tblQVectors
	 */
	public TableUsedChecker (BlockingQueue<Integer> tableIndexes, double[][] tblAssignmentMatrix, List<double[][]> wMatrix, 
							 List<double[][]> pMatrix, List<double[][]> vMatrix,  List<double[][]> qMatrix, List<double[]> tableLabelMatrix, 
							 Map<Integer, double[][]> tblWeightVectors, Map<Integer, double[][]> tblPlVectors, 
							 Map<Integer, double[][]> tblVlVectors, Map<Integer, double[][]> tblQVectors)
	{
		this.tableIndexes = tableIndexes;
		this.tblAssignmentMatrix = tblAssignmentMatrix;
		this.wMatrix = wMatrix;
		this.pMatrix = pMatrix;
		this.vMatrix = vMatrix;
		this.qMatrix = qMatrix;
		this.tableLabelMatrix = tableLabelMatrix;
		this.tblWeightVectors = tblWeightVectors;
		this.tblPlVectors = tblPlVectors;
		this.tblVlVectors = tblVlVectors;
		this.tblQVectors = tblQVectors;
	}
	
	
	/* (non-Javadoc)
	 * @see java.lang.Runnable#run()
	 */
	public final void run ()
	{
		Integer	tblIndex;
		
		while ((tblIndex = tableIndexes.poll ()) != null)
			doCheck (tblIndex.intValue ());
	}
	
	
	/**
	 * 
	 * 
	 * @param tblIndex
	 */
	private final void doCheck (int tblIndex)
	{
		if (InferenceAlgoCancer.hasTrue (tblAssignmentMatrix, tblIndex))
		{
//			DoubleMatrix kthWColumnMatrix = new DoubleMatrix (tblWeightVectors.get (tblIndex));
			double[][]	kthWColumnMatrix = tblWeightVectors.get (tblIndex);
			InferenceAlgoCancer.safeSet (wMatrix, tblIndex, kthWColumnMatrix);
			
			double[][]	kthPColumnMatrix = tblPlVectors.get (tblIndex);
			InferenceAlgoCancer.safeSet (pMatrix, tblIndex, kthPColumnMatrix);
			
			double[][]	kthVColumnMatrix = tblVlVectors.get (tblIndex);
			InferenceAlgoCancer.safeSet (vMatrix, tblIndex, kthVColumnMatrix);
			
			double[][]	kthQColumnMatrix = tblQVectors.get (tblIndex);
			InferenceAlgoCancer.safeSet (qMatrix, tblIndex, kthQColumnMatrix);
			
			double[] tblLabel = new double[] {tblIndex, 0};
//			double[][] tblLabel = new double[][] {{tblIndex, 0}};
			
//			DoubleMatrix kthLabelRowMatrix = new DoubleMatrix (1, 2, tblLabel);
			InferenceAlgoCancer.safeSet (tableLabelMatrix, tblIndex, tblLabel);
		}
	}
	
}
