/**
 * IRLJK
 *
 * <p>@author Gregory Dougherty</p>
 * Copyright Mayo Clinic, 2019
 *
 */
package CRC_Prediction;

import java.util.List;
import java.util.concurrent.BlockingQueue;

/**
 * 
 *
 * <p>@author Gregory Dougherty</p>
 * Copyright 2019, Mayo Foundation for Medical Education and Research
 */
public class TableLabelUpdater implements Runnable
{
	private BlockingQueue<Integer>	tableIndexes;
	private List<double[]>			tableLabelMatrix;
	private double[][]				tblAssignmentMatrix;
	private double[][]				newPartitiontblAssignmentMatrix;
	
	
	/**
	 * @param tableIndexes	{@link BlockingQueue} that gives access to table labels that still need to be processed
	 * @param tableLabelMatrix
	 * @param tblAssignmentMatrix
	 * @param newPartitiontblAssignmentMatrix
	 */
	public TableLabelUpdater (BlockingQueue<Integer> tableIndexes, List<double[]> tableLabelMatrix, 
							  double[][] tblAssignmentMatrix, double[][] newPartitiontblAssignmentMatrix)
	{
		this.tableIndexes = tableIndexes;
		this.tableLabelMatrix = tableLabelMatrix;
		this.tblAssignmentMatrix = tblAssignmentMatrix;
		this.newPartitiontblAssignmentMatrix = newPartitiontblAssignmentMatrix;
	}
	
	
	/* (non-Javadoc)
	 * @see java.lang.Runnable#run()
	 */
	public final void run ()
	{
		Integer	labelIndex;
		
		while ((labelIndex = tableIndexes.poll ()) != null)
			doUpdate (labelIndex.intValue ());
	}
	
	
	/**
	 * DO for a single label
	 * 
	 * @param labelIndex
	 */
	private final void doUpdate (int labelIndex)
	{
		double[]	labelInfo = tableLabelMatrix.get (labelIndex);
		double		oldIndex = labelInfo[0];	// GTD get first item
		int			numRows = tblAssignmentMatrix.length;
		
//		System.out.println ("old table index # " + oldIndex);
		for (int r = 0; r < numRows; ++r)
		{
			double[]	newRow = newPartitiontblAssignmentMatrix[r];
			double[]	theRow = tblAssignmentMatrix[r];
			int			numCols = theRow.length;
			
			for (int c = 0; c < numCols; ++c)
			{
				// replace the table-index associated with row/trajectory 'r' with its NEW table-index
				// replace label-index for the table assigned at position (r,c) in the new assignment matrix with new label
				// 'tableLabelMatrix.get(i).getData()[0][1]'
				if (theRow[c] == oldIndex)
				{
//					double[]	theMatrix = tableLabelMatrix.get (labelIndex);
					double		value = labelInfo[1];	// Get second item
					
//					newPartitiontblAssignmentMatrix.put (r, c, value);	// GTD data is by column, not by row
					newRow[c] = value;	// GTD theMatrix.data is by column, not by row
//					System.out.println ("doUpdate: r: " + r + ", c: " + c + ", value: " + value);
				}
			}
		}
	}
	
}
