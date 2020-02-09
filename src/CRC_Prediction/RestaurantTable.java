
package CRC_Prediction;


import java.io.Serializable;
import org.jblas.DoubleMatrix;


/**
 * Holding class for information that is updated each MH iteration, for one table
 * 
 * @author John Kalantari
 * Copyright 2019, Mayo Foundation for Medical Education and Research
 *
 */
public class RestaurantTable implements Serializable
{
	
	static final long	serialVersionUID	= 1;
	
	public double[][]	_restaurantTableWeightMatrices;
	public double[][]	_restaurantTablePolicyMatrices;
	public double[][]	_restaurantTableValueMatrices;
	public double[][]	_restaurantTableQMatrices;
	
	public double[][]	_restaurantAssignmentMatrix;
	
	public Double		_restLikeLihoods;
	public Double		_restPriors;
	
	public DoubleMatrix	_restGradientsFromLLH;
	public DoubleMatrix	_restGradientsFromPrior;
	
	public double[]		_saPairCountsInfoForSubsetOfTrajectories;
	
	
	public RestaurantTable (double[][] weightMatrices, double[][] policyMatrices, double[][] valueMatrices, double[][] qMatrices, 
							double[][] assignmentMatrix)
	{
		_restaurantTableWeightMatrices = weightMatrices;
		_restaurantTablePolicyMatrices = policyMatrices;
		_restaurantTableValueMatrices = valueMatrices;
		_restaurantTableQMatrices = qMatrices;
		_restaurantAssignmentMatrix = assignmentMatrix;
	}
	
}
