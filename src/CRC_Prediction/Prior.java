
package CRC_Prediction;


import org.apache.commons.math3.util.Pair;
import org.jblas.DoubleMatrix;
import CRC_Prediction.Utils.MatrixUtilityJBLAS;


/**
 * Create Prior class object and set its parameters
 * 
 * @author John Kalantari
 * Copyright 2019, Mayo Foundation for Medical Education and Research
 *
 */
public class Prior
{
	
	public String	_distributionName;
	
	public double	_beta;							// hyperparameter for mean of reward (for
													// normal-gamma, beta-gamma distributions)
	public double	_gamma1;						// hyperparmaeter for variance of reward (for
													// normal-gamma, beta-gamma distributions)
	public double	_gamma2;						// hyperparameter for variance of reward (for
													// normal-gamma, beta-gamma distributions)
	public double	_delta;							// discretization level for reward value (for
													// normal-gamma, beta-gamma distributions)
	public double	_numberSamplesMCIntegration;	// number of samples for Monte-Carlo Integration
													// (for beta-gamma distributions)
	
	public double	_mu;							// mean of reward function (for gaussian
													// distribution)
	public double	_sigma;							// std.dev of reward function (for gaussian
													// distribution)
	
	public int		_priorIdentifier;				// 1:normal-gamma, 2:beta-gamma, 3:gaussian,
													// 4:uniform
	
	
	/**
	 * Construct normal-gamma distribution prior
	 * 
	 * @param name
	 * @param prior_identifier
	 * @param beta
	 * @param gamma1
	 * @param gamma2
	 * @param delta
	 * @param priorIdentifier
	 */
	public Prior (String name, int prior_identifier, double beta, double gamma1, double gamma2,
			double delta, int priorIdentifier)
	{
		set_distributionName (name);
		set_beta (beta);
		set_gamma1 (gamma1);
		set_gamma2 (gamma2);
		set_delta (delta);
		set_identifier (priorIdentifier);
	}
	
	
	/**
	 * Construct gaussian distribution prior
	 * 
	 * @param name
	 * @param mean
	 * @param stdDev
	 * @param priorIdentifier
	 */
	public Prior (String name, double mean, double stdDev, int priorIdentifier)
	{
		set_mu (mean);
		set_sigma (stdDev);
		set_identifier (priorIdentifier);
	}
	
	
	public String get_distributionName ()
	{
		return _distributionName;
	}
	
	
	public int get_identifier ()
	{
		return _priorIdentifier;
	}
	
	
	public double get_beta ()
	{
		return _beta;
	}
	
	
	public double get_gamma1 ()
	{
		return _gamma1;
	}
	
	
	public double get_gamma2 ()
	{
		return _gamma2;
	}
	
	
	public double get_delta ()
	{
		return _delta;
	}
	
	
	public double get_mu ()
	{
		return _mu;
	}
	
	
	public double get_sigma ()
	{
		return _sigma;
	}
	
	
	public void set_distributionName (String name)
	{
		_distributionName = name;
	}
	
	
	public void set_identifier (int pIdentifier)
	{
		_priorIdentifier = pIdentifier;
	}
	
	
	public void set_beta (double b)
	{
		_beta = b;
	}
	
	
	public void set_gamma1 (double g1)
	{
		_gamma1 = g1;
	}
	
	
	public void set_gamma2 (double g2)
	{
		_gamma2 = g2;
	}
	
	
	public void set_delta (double d)
	{
		_delta = d;
	}
	
	
	public void set_mu (double m)
	{
		_mu = m;
	}
	
	
	public void set_sigma (double s)
	{
		_sigma = s;
	}
	
	/**
	 * JK 7.25.2019 data validated
	 * @param weightMatrix
	 * @return
	 */
	protected Pair<Double, double[][]> computeLogPriorAndGradient (double[][] weightMatrix)
	{
		
		double priorProb = 0.0;
		DoubleMatrix rewardGradient = null;
		
		if (_priorIdentifier == 4)
		{ // if prior is Uniform
			priorProb = Math.log (1.0);
			rewardGradient = new DoubleMatrix (weightMatrix.length, weightMatrix[0].length); // reward gradient is all zero's
		}
		else	// if(_priorIdentifier==3) //if prior is Gaussian
		{
			DoubleMatrix aMatrix = new DoubleMatrix (weightMatrix);
			addTo (aMatrix, -1.0 * _mu);
			
			if (aMatrix.columns == 1)
				return fastCompute (aMatrix);
			else
			{
				DoubleMatrix aMatrixTransposed = aMatrix.transpose ();
				
				double denominator = Math.pow (_sigma, 2);
				double invDenominator = Math.pow (denominator, -1.0);
				// JK 4.9.2019 the matrix multiplication should be mmul() not element-wise
				DoubleMatrix quotient = aMatrixTransposed.mmul (aMatrix).mul (-0.5).mul (invDenominator);
				
				priorProb = MatrixUtilityJBLAS.sumPerColumn (quotient.toArray2 ())[0];
				rewardGradient = aMatrix.mul (-1.0 * invDenominator);
			}
		}
		
		Pair<Double, double[][]> logPriorProbAndGradient = new Pair<Double, double[][]> (priorProb, rewardGradient.toArray2 ());
		return logPriorProbAndGradient;
	}
	
	
	/**
	 * Compute the usual case, where it's a n x 1 matrix
	 * 
	 * @param aMatrix	{@link DoubleMatrix} to modify and return
	 * @return	a {@link Pair} holding a {@link Double} (priorProb) and a double[][] (rewardGradient)
	 */
	private Pair<Double, double[][]> fastCompute (DoubleMatrix aMatrix)
	{
		double[]	data = aMatrix.data;
		int			len = data.length;
		double		square = 0.0;
		
		// Multiply by its transpose
		for (int i = 0; i < len; ++i)
		{
			double	value = data[i];
			square += value * value;
		}
		
		double	denominator = Math.pow (_sigma, 2);
		double denominatorForPrior = denominator*2.0;
		double	invDenominatorForPrior = Math.pow (denominatorForPrior, -1.0);
		double	invDenominatorForGradient = Math.pow (denominator, -1.0);

		double	priorProb = -1.0*square * invDenominatorForPrior;

		
		mulBy (aMatrix, -1.0 * invDenominatorForGradient); 

		
		return new Pair<Double, double[][]> (priorProb, aMatrix.toArray2 ());
	}
	
	
	/**
	 * Add a value to all the elements of a {@link DoubleMatrix}, changing the DoubleMatrix
	 * 
	 * @param aMatrix	{@link DoubleMatrix} to add to
	 * @param value		Value to add
	 */
	public static final void addTo (DoubleMatrix aMatrix, double value)
	{
		double[]	data = aMatrix.data;
		int			len = data.length;
		
		for (int i = 0; i < len; ++i)
			data[i] += value;
	}
	
	
	/**
	 * Multiply all the elements of a {@link DoubleMatrix} by {@code value}, changing the DoubleMatrix
	 * 
	 * @param aMatrix	{@link DoubleMatrix} to multiply
	 * @param value		Value to multiply by
	 */
	public static final void mulBy (DoubleMatrix aMatrix, double value)
	{
		double[]	data = aMatrix.data;
		int			len = data.length;
		
		for (int i = 0; i < len; ++i)
			data[i] *= value;
	}
	
}
