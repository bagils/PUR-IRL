
package CRC_Prediction;


import java.io.*;
import java.util.Map;
import CRC_Prediction.Utils.MatrixUtility;


public class IRLRestaurant implements Serializable
{
	private static final long			serialVersionUID	= 1;
	
	private double[][]					_seatingArrangement;
	private Map<Integer, double[][]>	_weightMatrices;
	private Map<Integer, double[][]>	_policyMatrices;
	private Map<Integer, double[][]>	_valueMatrices;
	private Map<Integer, double[][]>	_qMatrices;
	private double						_logPosteriorProb;
	
	
	public IRLRestaurant ()
	{
		setLogPosteriorProb (Double.NEGATIVE_INFINITY);
	}
	
	
	/**
	 * 6.18.2019: JK modified constructor to include map of qMatrices
	 * @param tblAssignmentMatrix
	 * @param tblWeightMatrices
	 * @param tblPolicyMatrices
	 * @param tblValueMatrices
	 * @param tblQMatrices
	 * @param logPosProb
	 */
	public IRLRestaurant (double[][] tblAssignmentMatrix, Map<Integer, double[][]> tblWeightMatrices, Map<Integer, double[][]> tblPolicyMatrices, 
						  Map<Integer, double[][]> tblValueMatrices, Map<Integer, double[][]> tblQMatrices, double logPosProb)
	{
		setSeatingArrangement (MatrixUtility.deepCopy (tblAssignmentMatrix));
		
		setWeightMatrices (tblWeightMatrices);
		setPolicyMatrices (tblPolicyMatrices);
		setValueMatrices (tblValueMatrices);
		setQMatrices (tblQMatrices);
		setLogPosteriorProb (logPosProb);
	}
	
	
	public static IRLRestaurant clone (IRLRestaurant r)
	{
		IRLRestaurant rest = new IRLRestaurant (r.getSeatingArrangement (), r.getWeightMatrices (), r.getPolicyMatrices (), 
												r.getValueMatrices (), r.getQMatrices (), r.getLogPosteriorProb ());
		return rest;
	}
	
	
	public void setSeatingArrangement (double[][] assignmentMatrix)
	{
		_seatingArrangement = MatrixUtility.deepCopy (assignmentMatrix);
	}
	
	
	public void setWeightMatrices (Map<Integer, double[][]> tableWeightVectors)
	{
		_weightMatrices = tableWeightVectors;
	}
	
	
	public void setPolicyMatrices (Map<Integer, double[][]> tablePolicyVectors)
	{
		_policyMatrices = tablePolicyVectors;
	}
	
	
	public void setValueMatrices (Map<Integer, double[][]> tableValueVectors)
	{
		_valueMatrices = tableValueVectors;
	}
	
	
	public void setQMatrices (Map<Integer, double[][]> tableQVectors)
	{
		_qMatrices = tableQVectors;
	}
	
	
	public void setLogPosteriorProb (double logPP)
	{
		_logPosteriorProb = logPP;
	}
	
	
	public double[][] getSeatingArrangement ()
	{
		return _seatingArrangement;
	}
	
	
	public Map<Integer, double[][]> getWeightMatrices ()
	{
		return _weightMatrices;
	}
	
	
	public Map<Integer, double[][]> getPolicyMatrices ()
	{
		return _policyMatrices;
	}
	
	
	public Map<Integer, double[][]> getValueMatrices ()
	{
		return _valueMatrices;
	}
	
	
	public Map<Integer, double[][]> getQMatrices ()
	{
		return _qMatrices;
	}
	
	
	public Double getLogPosteriorProb ()
	{
		return _logPosteriorProb;
	}
	
	
	/**
	 * Class method for DEserializing an IRLRestaurant object from an OBJECT InputStream to generate
	 * a fully instantiated IRLRestaurant model
	 * 
	 * @param in ObjectInputStream to read from
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	@SuppressWarnings ("unchecked")
	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException
	{
		_seatingArrangement = (double[][]) in.readObject ();
		_weightMatrices = (Map<Integer, double[][]>) in.readObject ();
		_policyMatrices = (Map<Integer, double[][]>) in.readObject ();
		_valueMatrices = (Map<Integer, double[][]>) in.readObject ();
		_qMatrices = (Map<Integer, double[][]>) in.readObject ();
		
		_logPosteriorProb = in.readDouble ();
	}
	
	
	/**
	 * Class method for SERIALIZING the current IRLRestaurant class object,(and its fully
	 * instantiated model), into an input-specified OBJECT OutputStream
	 * 
	 * @param out ObjectOutputStream to write into
	 * @throws IOException
	 */
	private void writeObject (ObjectOutputStream out) throws IOException
	{
		out.writeObject (_seatingArrangement);
		out.writeObject (_weightMatrices);
		out.writeObject (_policyMatrices);
		out.writeObject (_valueMatrices);
		out.writeObject (_qMatrices);
		
		out.writeDouble (_logPosteriorProb);
		
	}
	
}
