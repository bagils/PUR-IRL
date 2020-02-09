
package CRC_Prediction;


import java.io.Serializable;
import java.util.Map;
import org.jblas.DoubleMatrix;
import com.google.common.collect.Multimap;


/**
 * Holding class for information that is updated each MH iteration
 * 
 * @author m186806
 *
 */
public class RestaurantMap implements Serializable
{
	
	static final long					serialVersionUID	= 1;
	
	public Map<Integer, double[][]>		_restaurantTableWeightMatrices;
	public Map<Integer, double[][]>		_restaurantTablePolicyMatrices;
	public Map<Integer, double[][]>		_restaurantTableValueMatrices;
	public Map<Integer, double[][]>		_restaurantTableQMatrices; //jK added 6.18.2019 qMatrices Map
	
	public double[][]					_restaurantAssignmentMatrix;
	
	public Map<Integer, Double>			_restLikeLihoods;
	public Map<Integer, Double>			_restPriors;
	
	public Map<Integer, DoubleMatrix>	_restGradientsFromLLH;
	public Map<Integer, DoubleMatrix>	_restGradientsFromPrior;
	
	public Multimap<Integer, double[]>	_saPairCountsInfoForSubsetOfTrajectories;
	
	
	public RestaurantMap (Map<Integer, double[][]> weightMatrices, Map<Integer, double[][]> policyMatrices, Map<Integer, double[][]> valueMatrices, 
						  Map<Integer, double[][]> qMatrices, double[][] assignmentMatrix)
	{
		_restaurantTableWeightMatrices = weightMatrices;
		_restaurantTablePolicyMatrices = policyMatrices;
		_restaurantTableValueMatrices = valueMatrices;
		_restaurantTableQMatrices = qMatrices;
		_restaurantAssignmentMatrix = assignmentMatrix;
	}
	
}
