package CRC_Prediction;

import CRC_Prediction.MersenneTwisterFastIRL;
import java.util.Arrays;

/**
 * 
 * @author John Kalantari
 * Copyright 2019, Mayo Foundation for Medical Education and Research
 *
 */
public class SampleMultinomialIRL {

    public static int[] sampleMultinomial(int n, double[] weights, MersenneTwisterFastIRL rng){
        double tw = 0.0;
        for(double w : weights){
            if(w < 0.0){
                throw new IllegalArgumentException("weights must be non negative");
            }
            tw += w;
        }

        for(int i = 0; i < weights.length; i++){
            weights[i] /= tw;
        }

        int[] sample = new int[weights.length];
        double r, cuSum;
        int l = weights.length;
        for(int i = 0; i < n; i++){
            r = rng.nextDouble();
            cuSum = 0.0;
            for(int j = 0; j < l; j++){
                cuSum += weights[j];
                if(cuSum > r){
                    sample[j]++;
                    break;
                }
            }
        }
        
        return sample;
    }
    
    

    /**
     * Sample single State index between 0 and weights.length-1, according to the multinomial distribution
     * @param n
     * @param weights
     * @param rng
     * @return
     */
    public static int sampleSingleStateFromMultinomial(int n, double[] weights, MersenneTwisterFastIRL rng){
        double tw = 0.0;
        double [] probabilityDistro = new double [weights.length];
        for(double w : weights){
            if(w < 0.0){
                throw new IllegalArgumentException("weights must be non negative");
            }
            tw += w;
        }

        for(int i = 0; i < weights.length; i++){
            probabilityDistro[i] = weights[i]/ tw;
        }

        int sampleState=0;

        boolean haveSampledState= false;
        
        double r, cuSum;
        int l = weights.length;
        //for(int i = 0; i < n; i++){
        while(!haveSampledState) { //JK 7.21.2019 alternative while-loop to continue to compute cumSum rather than resorting to default state
	            r = rng.nextDouble();
	            cuSum = 0.0;
	            for(int j = 0; j < l; j++){ 

	                cuSum += probabilityDistro[j];
	                if(cuSum > r){
	                   
	                			sampleState = j;  

	                			haveSampledState = true;
	                			break;

	                }
	            }
	            if(haveSampledState) {
	            		break;
	            }
        		}
        
        return sampleState;
    }
    
    
    static int	gWhichTable = 0;
    static final boolean cheat = false;
    
    
    /**
     * JK 7.26.2019 data validated 
     * Table index is between 1 and weights.length
     * @param n : is the number of times you want to loop through all elements in the probability distribution; 
     * if a random table-index is not sampled within this loop; then by default table 0 is returned 
     * @param weights
     * @param rng
     * @return
     */
	public static int sampleSingleTableFromMultinomial (int n, double[] weights, MersenneTwisterFastIRL rng)
	{
		double tw = 0.0;
		int l = weights.length;
//		if (cheat)
//		{
//			++gWhichTable;
//			return (gWhichTable % l) + 1;
//		}
		double[] probabilityDistro = new double[l];
		for (double w : weights)
		{
			if (w < 0.0)
			{
				throw new IllegalArgumentException ("weights must be non negative");
			}
			tw += w;
		}
		
		double	cumSum = 0.0;
		for (int i = 0; i < l; i++)
		{
			probabilityDistro[i] = (cumSum += weights[i] / tw);
		}
		
		// JK 3.15.2019: changed sampleTAble=0 to 1; because the returned value is assumed to
		// correspond to a table-index and
		// NOT the position in a double array
		
//		double cuSum;
		// for(int i = 0; i < n; i++){
		while (true)
		{ // JK 7.21.2019 alternative while-loop to continue to compute cumSum rather than resorting
			// to default table
			
			double	r = rng.nextDouble ();	// Last element of probabilityDistro should be 1, by definition
			int		pos = Arrays.binarySearch (probabilityDistro, r);
			if (pos < l)
			{
				if (pos < 0)	// Not an exact match, but gives insertion point, which is what we want
					pos = -(pos);
				else
					++pos;	// Get pos + 1, either way
				return pos;
			}
//			cuSum = 0.0;
//			for (int j = 0; j < l; j++)
//			{
//				
////				cuSum += probabilityDistro[j];
////				if (cuSum > r)
//				if (probabilityDistro[j] > r)
//				{
//					sampleTable = j + 1; // index 0 actually corresponds to table 1, thus table index = j+1;
//					haveSampledTable = true;
//					break;
//				}
//			}
		}
		
	}

    public static int[] deleteCustomersAtRandom(int nDelete, int[] cc, int customers, MersenneTwisterFastIRL rng){
        if(nDelete > customers){
            throw new IllegalArgumentException("nDelete must be <= customers");
        }

        int[] c = new int[cc.length];
        System.arraycopy(cc,0,c,0,cc.length);

        double r, cuSum;
        int[] sample = new int[c.length];
        int l = c.length;
        for(int n = 0; n < nDelete; n++){
            r = rng.nextDouble();
            cuSum = 0.0;
            for(int i = 0; i < l; i++){
                cuSum += (double) c[i] / (double) customers;
                if(cuSum > r){
                    sample[i]++;
                    c[i]--;
                    customers--;
                    break;
                }
                assert i != (l-1);
            }
        }

        assert checkSample(sample, cc, nDelete);

        return sample;
    }

    public static boolean checkSample(int[] sample, int[] cc, int nDelete){
        int c = 0;
        assert sample.length == cc.length;
        for(int i = 0; i < cc.length; i++){
            c += sample[i];
            assert sample[i] <= cc[i];
        }
        return c == nDelete;
    }

    /*public static void main(String[] args){
        int[] dcr = deleteCustomersAtRandom(15, new int[]{20,40,30,60},150, new MersenneTwisterFast(1));
        for(int i : dcr){
            System.out.print(", " + i);
        }
        System.out.println();
    }*/
}

