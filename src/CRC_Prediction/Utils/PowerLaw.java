/**
 * 
 */
package CRC_Prediction.Utils;

/**
 *
 * @(#)PowerLaw.java    ver 1.2  6/20/2005
 *
 * Modified by Weishuai Yang (wyang@cs.binghamton.edu). 
 * 
 * this file is based on T J Finney's Manuscripts Simulation Tool, 2001
 */

import java.util.Random;

/**
 * provides power law selection. Modified by Weishuai Yang this file is based on
 * T J Finney's Manuscripts Simulation Tool, 2001
 */
public class PowerLaw {

    private Random rand;

    /**
     * constructs a power law object using an external random generator
     * 
     * @param r
     *            random generator passed in
     */
    public PowerLaw(Random r) {
        rand = r;
    }

    /**
     * constructs a power law object using an internal random generator
     */
    public PowerLaw() {
        rand = new Random();
    }

    /**
     * get uniformly distributed double in [0, 1]
     */
    public double getRand() {
        return rand.nextDouble();
    }

    /**
     * get uniformly distributed integer in [0, N - 1]
     */
    public int getRandInt(int N) {
        return rand.nextInt(N);
    }

    /**
     * selects item using power law probability of selecting array item: p(ni) =
     * k * (ni^p) k is a normalisation constant p(ni) = 0 if ni is zero, even
     * when p < 0
     * 
     * 
     * @param nums
     *            array of numbers ni
     * @param p
     *            exponent p
     * @return index in [0, array size - 1]
     */

    public int select(double[] nums, double p) {
        // make array of probabilities
        double[] probs = new double[nums.length];
        for (int i = 0; i < probs.length; i++) {
            if (nums[i] == 0)
                probs[i] = 0;
            else
                probs[i] = Math.pow(nums[i], p);
        }

        // sum probabilities
        double sum = 0;
        for (int i = 0; i < probs.length; i++) {
            sum += probs[i];
        }

        // obtain random number in range [0, sum]
        double r = sum * getRand();

        // subtract probs until result negative
        // no of iterations gives required index
        int i;
        for (i = 0; i < probs.length; i++) {
            r -= probs[i];
            if (r < 0) {
                break;
            }
        }
        return i;
    }

    /**
     * select item using Zipf's law
     * 
     * @param size
     *            of ranked array
     * @return index in [0, array size - 1]
     */
    public int zipf(int size) {
        // make array of numbers
        double[] nums = new double[size];
        for (int i = 0; i < nums.length; i++) {
            nums[i] = i + 1;
        }
        // get index using special case of power law
        return select(nums, -1.0);
    }
    
    public static int[] generatePowerLawDistributedDatasetSizings(int numExperts, int numTotalPaths, int exptSeed) {
    	
    	int [] tableSizings = new int[numExperts];
    	int assignedExpert =0;
    	
    	PowerLaw p = new PowerLaw(new Random(exptSeed));
		System.out.println("*****Simulation"+exptSeed+" for "+numTotalPaths+" totalPaths******");
        for (int i = 0; i < numTotalPaths; i++) {
        	 assignedExpert = p.zipf(numExperts);
        	 tableSizings[assignedExpert]++;
             System.out.println(assignedExpert);
         }
    	
    	return tableSizings;
    	
    }

    /**
     * test purpose main
     * 
     * @param args
     *            command line inputs
     */
    public static void main(String[] args) {
    	
    	
        //
//        PowerLaw p = new PowerLaw(new Random(555));

        /*
         * double[] numbers = {0, 1, 2, 3}; for (int i = 0; i < 5; i++) {
         * System.out.println("Select: " + p.select(numbers, -1)); }
         */
        
        //original loop
//        for (int i = 0; i < 50; i++) {
//           // System.out.println("Zipf #"+i+": " + p.zipf(20));
//            System.out.println( p.zipf(20));
//        }
    	
    	//test loop
//        for (int i = 0; i < 60; i++) {
//            System.out.println( p.zipf(3));
//        }
//        
    	
    	int numSimulations =10;
    	int numExperts =3;
    	int numTotalPaths=30;
    	PowerLaw p = null;
    	int randomSeed =0;

    	for(int t=0;t<4; t++) {
	    	for (int s=0; s< numSimulations; s++) {
	    		randomSeed = s;
	    		p = new PowerLaw(new Random(randomSeed));
	    		System.out.println("*****Simulation"+s+" for "+numTotalPaths+" totalPaths******");
		        for (int i = 0; i < numTotalPaths; i++) {
		             System.out.println( p.zipf(numExperts));
		         }
	    	}
	        numTotalPaths+=10;

    	}
    	
    	int[] tableSizings = generatePowerLawDistributedDatasetSizings(3, 30, 0);
    	for (int j=0; j< tableSizings.length; j++) {
    		System.out.println("Table #"+j+" has "+tableSizings[j]+" paths!");
    	}
    	
        
        
    }
}
