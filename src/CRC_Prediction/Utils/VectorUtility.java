package CRC_Prediction.Utils;

import CRC_Prediction.MersenneTwisterFastIRL;
import com.google.common.math.BigIntegerMath;
import java.util.*;
import org.apache.commons.math3.util.Pair;

/**
 * 
 * Static vector manipulation routines for Matlab porting and other numeric
 * operations. The routines work for int and double partly; the class is
 * extended as needed.
 * @author John Kalantari
 * Copyright 2019, Mayo Foundation for Medical Education and Research
 *
 */
public class VectorUtility {

    public static int ndigits = 0;

    public static int colwidth = 0;
    
    public static int[] rangeOfVector(int [] origVector, int start, int end, int step) {

        int[] out = new int[(int) Math.floor((end - start) / step) + 1];
        for (int i = 0; i < out.length; i++) {
            out[i] = origVector[start + step * i];
        }
        return out;
    }

    /**
     * @param start
     * @param end
     * @param step
     * @return [start : step : end]
     */
    public static int[] range(int start, int end, int step) {

        int[] out = new int[(int) Math.floor((end - start) / step) + 1];
        for (int i = 0; i < out.length; i++) {
            out[i] = start + step * i;
        }
        return out;
    }

    /**
     * @param start
     * @param end
     * @return [start : end]
     */
    public static int[] range(int start, int end) {
        return range(start, end, end - start > 0 ? 1 : -1);
    }

    /**
     * create sequence [start : step : end] of double values. TODO: check
     * precision.
     * 
     * @param start
     *            double value of start, if integer, use "1.0" notation.
     * @param end
     *            double value of end, if integer, use "1.0" notation.
     * @param step
     *            double value of step size
     * @return
     */
    public static double[] range(double start, double end, double step) {

        double[] out = new double[(int) Math.floor((end - start) / step) + 1];
        for (int i = 0; i < out.length; i++) {
            out[i] = start + step * i;
        }
        return out;
    }

    /**
     * @param start
     * @param end
     * @return [start : end]
     */
    public static double[] range(double start, double end) {
        return range(start, end, end - start > 0 ? 1 : -1);
    }

    /**
     * sum the elements of vec
     * 
     * @param vec
     * @return
     */
    public static double sum(double[] vec) {
        double sum = 0;
        for (int i = 0; i < vec.length; i++) {
            sum += vec[i];
        }
        return sum;

    }

    /**
     * sum the elements of vec
     * 
     * @param vec
     * @return
     */
    public static int sum(int vec[]) {
        int sum = 0;
        for (int i = 0; i < vec.length; i++)
            sum += vec[i];

        return sum;
    }

    /**
     * cumulative sum of the elements, starting at element 0.
     * 
     * @param vec
     * @return vector containing the cumulative sum of the elements of vec
     */
    public static double[] cumsum(double[] vec) {
        double[] x = new double[vec.length];
        x[0] = vec[0];
        for (int i = 1; i < vec.length; i++) {
            x[i] = vec[i] + x[i - 1];
        }
        return x;
    }
    
    /**
     * Return product of elements in double array
     * @param vec
     * @return
     */
    public static double product(double[] vec) {
        //double prod = 0;
    	double prod = vec[0];
        for (int i = 1; i < vec.length; i++) {
            prod *= vec[i];
        }
        return prod;

    }
    /**
     * Return product of elements in integer array
     * @param vec
     * @return
     */
    public static int product(int[] vec) {
        //int prod = 0;
    		int prod = vec[0];
        for (int i = 1; i < vec.length; i++) {
            prod *= vec[i];
        }
        return prod;

    }
    

    /**
     * maximum value in vec
     * 
     * @param vec
     * @return
     */
    public static int max(int[] vec) {
        int max = vec[0];
        for (int i = 1; i < vec.length; i++) {
            if (vec[i] > max) {
                max = vec[i];
            }
        }
        return max;
    }

    /**
     * maximum value in vec
     * 
     * @param vec
     * @return
     */
    public static double max(double[] vec) {
        double max = vec[0];
        for (int i = 1; i < vec.length; i++) {
            if(Double.compare(vec[i], max) > 0.0) {
        		//if (vec[i] > max) {
                max = vec[i];
            }
        }
        return max;
    }
    
    /**
     * maximum value in vec and its element position
     * JK 5.2.2019: renamed method to reflect that we are returning a pair object in which the first element is the position and the second element is the maximum value of the input argument
     * @param vec
     * @return
     */
    public static Pair<Integer, Double> maxPositionAndVal(double[] vec) {
        double max = vec[0];
        Integer position = 0;
        for (int i = 1; i < vec.length; i++) {
        		if (Double.compare(vec[i], max) >0.0) {
            //if (vec[i] > max)
                max = vec[i];
            		position = i;
        		}
        }
        Pair<Integer,Double> maxPositionAndVal = new Pair<Integer,Double>(position,max);
       
        return maxPositionAndVal;
    }

    /**
     * minimum value in vec
     * 
     * @param vec
     * @return
     */
    public static int min(int[] vec) {
        int min = vec[0];
        for (int i = 1; i < vec.length; i++) {
            if (vec[i] < min)
                min = vec[i];
        }
        return min;
    }

    /**
     * minimum value in vec
     * 
     * @param vec
     * @return
     */
    public static double min(double[] vec) {
        double min = vec[0];
        for (int i = 1; i < vec.length; i++) {
        		if (Double.compare(vec[i], min) <0) {
            //if (vec[i] < min)
                min = vec[i];
        		}
        }
        return min;
    }

    /**
     * @param x
     * @param y
     * @return [x y]
     */
    public static double[] concat(double[] x, double[] y) {
        double[] z = new double[x.length + y.length];
        System.arraycopy(x, 0, z, 0, x.length);
        System.arraycopy(y, 0, z, x.length, y.length);
        return z;
    }

    /**
     * w = [x y z]
     * 
     * @param x
     * @param y
     * @return [x y z]
     */
    public static double[] concat(double[] x, double[] y, double[] z) {
        double[] w = new double[x.length + y.length + z.length];
        System.arraycopy(x, 0, w, 0, x.length);
        System.arraycopy(y, 0, w, x.length, y.length);
        System.arraycopy(y, 0, w, x.length + y.length, z.length);
        return w;
    }

    /**
     * Create new vector of larger size and data of the argument.
     * 
     * @param vector
     *            source array
     * @param moreelements
     *            number of elements to add
     * @return larger vector
     */
    public static double[] increaseSize(final double[] vector, int moreelements) {
        double[] longer = new double[vector.length + moreelements];
        System.arraycopy(vector, 0, longer, 0, vector.length);
        return longer;
    }

    /**
     * Create new matrix of larger size and data of the argument.
     * 
     * @param matrix
     * @param morerows
     * @param morecols
     * @return larger matrix
     */
    public static double[][] increaseSize(final double[][] matrix, int morerows, int morecols) {

        double[][] array2 = new double[matrix.length + morerows][];
        for (int i = 0; i < matrix.length; i++) {
            array2[i] = (morecols > 0) ? increaseSize(matrix[i], morecols) : matrix[i];
        }
        for (int i = matrix.length; i < array2.length; i++) {
            array2[i] = new double[matrix[0].length + morecols];
        }

        return array2;
    }

    /**
     * Create new vector with data of the argument and removed element.
     * 
     * @param vector
     * @param element
     * @return shorter vector
     */
    public static double[] removeElement(final double[] vector, int element) {
        double[] shorter = new double[vector.length - 1];
        System.arraycopy(vector, 0, shorter, 0, element);
        System.arraycopy(vector, element + 1, shorter, element, vector.length - element - 1);
        return shorter;
    }

    /**
     * Create new matrix with data of the argument and removed rows and columns.
     * 
     * @param matrix
     * @param rows
     *            ordered vector of rows to remove
     * @param cols
     *            ordered vector of cols to remove
     * @return smaller matrix
     */
    public static double[][] removeElements(final double[][] matrix, int[] rows, int[] cols) {
        return chooseElements(matrix, rangeComplement(rows, matrix.length), rangeComplement(cols, matrix[0].length));
    }

    /**
     * Create new vector with data of the argument and removed elements.
     * 
     * @param vector
     * @param elements
     *            ordered elements to remove
     * @return smaller vector
     */
    public static double[] removeElements(final double[] vector, int[] elements) {
        return chooseElements(vector, rangeComplement(elements, vector.length));
    }

    /**
     * return the complement of the sorted subset of the set 0:length-1 in
     * Matlab notation
     * 
     * @param set
     *            sorted set of elements < length
     * @param length
     *            of superset of set and its returned complement
     * @return
     */
    public static int[] rangeComplement(int[] set, int length) {
        int[] complement = new int[length - set.length];
        int sindex = 0;
        int cindex = 0;
        for (int i = 0; i < length; i++) {
            if (sindex >= set.length || set[sindex] != i) {
                complement[cindex] = i;
                cindex++;
            } else {
                sindex++;
            }
        }
        return complement;
    }

    /**
     * Create a matrix that contains the rows and columns of the argument matrix
     * in the order given by rows and cols
     * 
     * @param matrix
     * @param rows
     * @param cols
     * @return
     */
    public static double[][] chooseElements(double[][] matrix, int[] rows, int[] cols) {

        double[][] matrix2 = new double[rows.length][cols.length];

        for (int i = 0; i < rows.length; i++) {
            matrix2[i] = chooseElements(matrix[rows[i]], cols);
        }

        return matrix2;
    }

    /**
     * Create vector that contains the elements of the argument in the order as
     * given by keep
     * 
     * @param vector
     * @param keep
     * @return
     */
    public static double[] chooseElements(double[] vector, int[] keep) {
        double[] vector2 = new double[keep.length];

        for (int i = 0; i < keep.length; i++) {
            vector2[i] = vector[keep[i]];
        }
        return vector2;
    }

    /**
     * Create new vector of larger size and data of the argument.
     * 
     * @param vector
     *            source array
     * @param moreelements
     *            number of elements to add
     * @return larger vector
     */
    public static int[] increaseSize(final int[] vector, int moreelements) {
        int[] longer = new int[vector.length + moreelements];
        System.arraycopy(vector, 0, longer, 0, vector.length);
        return longer;
    }

    /**
     * Create new matrix of larger size and data of the argument.
     * 
     * @param matrix
     * @param morerows
     * @param morecols
     * @return larger matrix
     */
    public static int[][] increaseSize(final int[][] matrix, int morerows, int morecols) {

        int[][] array2 = new int[matrix.length + morerows][];
        for (int i = 0; i < matrix.length; i++) {
            array2[i] = (morecols > 0) ? increaseSize(matrix[i], morecols) : matrix[i];
        }
        for (int i = matrix.length; i < array2.length; i++) {
            array2[i] = new int[matrix[0].length + morecols];
        }

        return array2;
    }

    /**
     * Create new vector with data of the argument and removed element.
     * 
     * @param vector
     * @param element
     * @return shorter vector
     */
    public static int[] removeElement(final int[] vector, int element) {
        int[] shorter = new int[vector.length - 1];
        System.arraycopy(vector, 0, shorter, 0, element);
        System.arraycopy(vector, element + 1, shorter, element, vector.length - element - 1);
        return shorter;
    }

    /**
     * Create new matrix with data of the argument and removed rows and columns.
     * 
     * @param matrix
     * @param rows
     *            ordered vector of rows to remove
     * @param cols
     *            ordered vector of cols to remove
     * @return smaller matrix
     */
    public static int[][] removeElements(final int[][] matrix, int[] rows, int[] cols) {
        return chooseElements(matrix, rangeComplement(rows, matrix.length), rangeComplement(cols, matrix[0].length));
    }

    /**
     * Create new vector with data of the argument and removed elements.
     * 
     * @param vector
     * @param elements
     *            ordered elements to remove
     * @return smaller vector
     */
    public static int[] removeElements(final int[] vector, int[] elements) {
        return chooseElements(vector, rangeComplement(elements, vector.length));
    }

    /**
     * Create a matrix that contains the rows and columns of the argument matrix
     * in the order given by rows and cols
     * 
     * @param matrix
     * @param rows
     * @param cols
     * @return
     */
    public static int[][] chooseElements(int[][] matrix, int[] rows, int[] cols) {

        int[][] matrix2 = new int[rows.length][cols.length];

        for (int i = 0; i < rows.length; i++) {
            matrix2[i] = chooseElements(matrix[rows[i]], cols);
        }

        return matrix2;
    }

    /**
     * Create vector that contains the elements of the argument in the order as
     * given by keep
     * 
     * @param vector
     * @param keep
     * @return
     */
    public static int[] chooseElements(int[] vector, int[] keep) {
        int[] vector2 = new int[keep.length];

        for (int i = 0; i < keep.length; i++) {
            vector2[i] = vector[keep[i]];
        }
        return vector2;
    }

    /**
     * prints a double representation of the vector.
     * 
     * @param x
     * @return
     */
    public static String print(int[] x) {
        if (x == null)
            return "null";
        StringBuffer b = new StringBuffer();
        for (int i = 0; i < x.length - 1; i++) {
            b.append(x[i]).append(" ");
        }
        b.append(x[x.length - 1]);
        return b.toString();
    }

    /**
     * prints a double representation of an array.
     * 
     * @param x
     * @return
     */
    public static String print(int[][] x) {
        if (x == null)
            return "null";
        StringBuffer b = new StringBuffer();
        for (int i = 0; i < x.length - 1; i++) {
            b.append(print(x[i])).append("\n");
        }
        b.append(print(x[x.length - 1]));
        return b.toString();
    }

    /**
     * @param len
     * @param factor
     * @return factor * ones(1, len);
     */
    public static double[] ones(int len, double factor) {
        double[] x = new double[len];
        for (int i = 0; i < x.length; i++) {
            x[i] = 1;
        }
        return x;
    }

    /**
     * @param len
     * @param factor
     * @return factor * ones(1, len);
     */
    public static int[] ones(int len, int factor) {
        int[] x = new int[len];
        for (int i = 0; i < x.length; i++) {
            x[i] = factor;
        }
        return x;
    }

    /**
     * @param len
     * @return zeros(1, len)
     */
    public static double[] zeros(int len) {
        return new double[len];
    }
    

    /**
     * 
     * @param len
     * @return
     */
    public static double[] nans(int len) {
        return ones(len, Double.NaN);
    }
    
    public static HashMap<Integer, Double> nansMap(int len) {
    		HashMap<Integer, Double> nMap = new HashMap<Integer, Double>();
    		for (int i=1; i< len+1; i++) {
    			nMap.put(i, Double.NaN);
    		}
        return nMap;
    }

    /**
     * @param len
     * @return ones(1, len)
     */
    public static int[] ones(int len) {
        return ones(len, 1);
    }

    /**
     * cast a double[] to an int[]
     * 
     * @param vec
     * @return
     */
    public static int[] cast(double[] vec) {
        int[] ivec = new int[vec.length];
        for (int i = 0; i < ivec.length; i++) {
            ivec[i] = (int) vec[i];
        }
        return ivec;
    }

    /**
     * cast a double[] to an int[]
     * 
     * @param vec
     * @return
     */
    public static double[] cast(int[] vec) {
        double[] dvec = new double[vec.length];
        for (int i = 0; i < dvec.length; i++) {
            dvec[i] = (double) vec[i];
        }
        return dvec;
    }
    
    public static double [] cast_withstreams(int [] ints) {
    		double[] doubles = Arrays.stream(ints).asDoubleStream().toArray();
    		return doubles;

    }
    


    /**
     * find indices with val
     * 
     * @param vec
     * @param val
     * @return vector with 0-based indices.
     */
    public static int[] find(int[] vec, int val) {
    	List<Integer> v = new ArrayList<Integer>();
        for (int i = 0; i < vec.length; i++) {
            if (vec[i] == val) {
                v.add(i);
            }
        }
        int[] vv = new int[v.size()];
        for (int i = 0; i < vv.length; i++) {
            vv[i] = ( v.get(i)).intValue();
        }
        return vv;
    }

    /**
     * returns a copy of the vector elements with the given indices in the
     * original vector.
     * 
     * @param indices
     * @return
     */
    public static double[] subVector(double[] vec, int[] indices) {
        double[] x = new double[indices.length];
        for (int i = 0; i < x.length; i++) {
            x[i] = vec[indices[i]];
        }
        return x;
    }

    /**
     * returns a copy of the vector elements with the given indices in the
     * original vector.
     * 
     * @param vec
     * @return
     */
    public static int[] subVector(int[] vec, int[] indices) {
        int[] x = new int[indices.length];
        for (int i = 0; i < x.length; i++) {
            x[i] = vec[indices[i]];
        }
        return x;
    }

    /**
     * @param vec
     * @param start
     * @param end
     * @return
     */
    public static double[] subVector(double[] vec, int start, int end) {
        double[] x = new double[end - start + 1];
        for (int i = 0; i <= end - start; i++) {
            x[i] = vec[start + i];
        }
        return x;
    }

    /**
     * set the elements of vec at indices with the respective replacements.
     * TODO: implement views as in the colt library
     * 
     * @param vec
     * @param indices
     * @param replacements
     */
    public static void setSubVector(int[] vec, int[] indices, int[] replacements) {
        for (int i = 0; i < indices.length; i++) {
            vec[indices[i]] = replacements[i];
        }
    }

    /**
     * set the elements of vec at indices with the replacement. 
     * 
     * TODO: implement views as in the colt library
     * 
     * @param vec
     * @param indices
     * @param replacement
     */
    public static void setSubVector(int[] vec, int[] indices, int replacement) {
        for (int i = 0; i < indices.length; i++) {
            vec[indices[i]] = replacement;
        }
    }

    /**
     * add a scalar to the vector
     * 
     * @param vec
     * @param scalar
     */
    public static void add(int[] vec, int scalar) {
        for (int i = 0; i < vec.length; i++) {
            vec[i] += scalar;
        }
    }
    
    /**
     * add a scalar to the vector
     * 
     * @param vec
     * @param scalar
     */
    public static void add(double[] vec, double scalar) {
        for (int i = 0; i < vec.length; i++) {
            vec[i] += scalar;
        }
    }
    /**
     * Add scalar from vector and return result
     * @param vec
     * @param scalar
     * @return resulting vector
     */
    public static double [] add2(double[] vec, double scalar) {
        double [] v2 = new double[vec.length];
    		for (int i = 0; i < vec.length; i++) {
            v2[i]= vec[i] + scalar;
        }
    		return v2;
    }
    
    /**
     * subtract a scalar from the vector
     * 
     * @param vec
     * @param scalar
     */
    public static void subtract(int[] vec, int scalar) {
        for (int i = 0; i < vec.length; i++) {
            vec[i] -= scalar;
        }
    }
    /**
     * subtract a scalar from the vector
     * 
     * @param vec
     * @param scalar
     */
    public static void subtract(double[] vec, double scalar) {
        for (int i = 0; i < vec.length; i++) {
            vec[i] -= scalar;
        }
    }
    
    /**
     * Subtract scalar from vector and return result
     * @param vec
     * @param scalar
     * @return resulting vector
     */
    public static double [] subtract2(double[] vec, double scalar) {
        double [] v2 = new double[vec.length];
    		for (int i = 0; i < vec.length; i++) {
            v2[i]= vec[i] - scalar;
        }
    		return v2;
    }

    /**
     * set the elements of a copy of vec at indices with the respective
     * replacements. TODO: implement views as in the colt library
     * 
     * @param vec
     * @param indices
     * @param replacements
     * @return the copied vector with the replacements;
     */
    public static int[] setSubVectorCopy(int[] vec, int[] indices, int[] replacements) {
        int[] x = new int[vec.length];
        for (int i = 0; i < indices.length; i++) {
            x[indices[i]] = replacements[i];
        }
        return x;
    }

    /**
     * copies a the source to the destination
     * 
     * @param source
     * @return
     */
    public static double[] copy(double[] source) {
        if (source == null)
            return null;
        double[] dest = new double[source.length];
        System.arraycopy(source, 0, dest, 0, source.length);
        return dest;
    }

    /**
     * copies a the source to the destination
     * 
     * @param source
     * @return
     */
    public static int[] copy(int[] source) {
        if (source == null)
            return null;
        int[] dest = new int[source.length];
        System.arraycopy(source, 0, dest, 0, source.length);
        return dest;
    }

    /**
     * multiplicates the vector with a scalar. The argument is modified.
     * 
     * @param ds
     * @param d
     * @return
     */
    public static double[] mult(double[] ds, double d) {
        for (int i = 0; i < ds.length; i++) {
            ds[i] *= d;
        }
        return ds;

    }

    /**
     * multiplicates the vector with a vector (inner product) and returns sum. The argument is
     * not modified.
     * 
     * @param ds
     * @param dt
     * @return
     */
    public static double multAndSum(double[] ds, double[] dt) {
        if (ds.length != dt.length)
            throw new IllegalArgumentException("Vector dimensions must agree.");
        double s = 0;
        for (int i = 0; i < ds.length; i++) {
            s += ds[i] * dt[i];
        }
        return s;
    }
    
    /**
     * Element-wise multiplication
     * @param ds
     * @param dt
     * @return
     */
    public static double [] mult(double[] ds, double[] dt) {
        if (ds.length != dt.length)
            throw new IllegalArgumentException("Vector dimensions must agree.");
        double [] prod =new double[ds.length];
        for (int i = 0; i < ds.length; i++) {
            prod [i] = ds[i] * dt[i];
        }
        return prod;
    }
    
    /**
     * Element-wise addition
     * @param ds
     * @param dt
     * @return
     */
    public static double [] elementWiseAddition(double[] ds, double[] dt) {
        if (ds.length != dt.length)
            throw new IllegalArgumentException("Vector dimensions must agree.");
        double [] prod =new double[ds.length];
        for (int i = 0; i < ds.length; i++) {
            prod [i] = ds[i] + dt[i];
        }
        return prod;
    }


    
    public static int[] createUnitSpaceVector(int vLength, int startVal, int increment) {
	    	//creates a regularly-spaced int vector x, of length=vLength and with start value=startVal, using i as the increment between elements.
	    	int [] x = new int [vLength];
	    	x[0]= startVal;
	    	for (int i=1; i<vLength; i++) {
	    		x[i]= x[i-1]+increment;
	    	}
	    	return x;	
    }
    public static double[] createUnitSpaceVector(int vLength, double startVal, double increment) {
	    	//creates a regularly-spaced double vector x, of length=vLength and with start value=startVal, using i as the increment between elements.
	    	double [] x = new double [vLength];
	    	x[0]= startVal;
	    	for (int i=1; i<vLength; i++) {
	    		x[i]= x[i-1]+increment;
	    	}
	    	return x;	
    }
    
    public static double[] prepopulateVector(int vLength, double val) {
    		double [] pVector =new double [vLength]; 
    		for (int i=0; i<vLength; i++) {
    			pVector[i]= val;
    		}
    		return pVector;
    }
    public static int[] prepopulateVector(int vLength, int val) {
		int [] pVector =new int [vLength]; 
		for (int i=0; i<vLength; i++) {
			pVector[i]= val;
		}
		return pVector;
    }
    
    public static int[] createPermutatedVector(int vLength, int startVal) {
    		int [] pVector =createUnitSpaceVector(vLength, startVal, 1);
    		//MersenneTwisterFastIRL rgen = new MersenneTwisterFastIRL(1); // Random number generator
    		MersenneTwisterFastIRL rgen = new MersenneTwisterFastIRL(); // rather than have the same seed for each call. Use time as intiial seed. Random number generator

    		for (int i=0; i<pVector.length; i++) {
    		    int randomPosition = rgen.nextInt(pVector.length);
    		    int temp = pVector[i];
    		    pVector[i] = pVector[randomPosition];
    		    pVector[randomPosition] = temp;
    		}
     
    		return pVector;
    }
    
    public static double [] pow(double[] arr, int powerVal) {
    		double [] raisedArray = new double[arr.length];
    		
    	    //for (double d: arr)
    		for (int i=0; i< arr.length; i++) {
    	           raisedArray[i] = Math.pow(arr[i], powerVal);     
    	    }
    		return raisedArray;

    }
    public static double [] pow(double[] arr, double [] powerValArray) {
		double [] raisedArray = new double[arr.length];
        if (arr.length != powerValArray.length)
            throw new IllegalArgumentException("Vector dimensions must agree before computing exponent.");
		
		for (int i=0; i< arr.length; i++) {
	           raisedArray[i] = Math.pow(arr[i], powerValArray[i]);     
	    }
		return raisedArray;
    }
    
   
    
//    public static double [] factorial(int[] arr) {
//    		double [] factorialArray = new double [arr.length];
//    		for(int i =0; i<arr.length; i++) {
//        		factorialArray[i]=CombinatoricsUtils.factorial(arr[i]);
//    		}
//    		return factorialArray;
//    }
    //JK replaced 3.11.2019 with guava version that can handle n >20(combinatoricutils.factorial throws arithmetic exception if n is greater than 20)
    public static double [] factorial(int[] arr) {
		double [] factorialArray = new double [arr.length];
		for(int i =0; i<arr.length; i++) {
    		factorialArray[i]= BigIntegerMath.factorial(arr[i]).doubleValue();
		}
		return factorialArray;
}

    public static double getMean(double[] arr) {
        double sum = 0.0;
        int size = arr.length;
        for(double a : arr)
            sum += a;
        return sum/size;
    }

    public static double getVariance(double[] arr) {
        int size = arr.length;
        double mean = getMean(arr);
        double temp = 0;
        for(double a :arr)
            temp += (a-mean)*(a-mean);
        return temp/(size-1);
    }

    public static double getStdDev(double[] arr) {
        return Math.sqrt(getVariance(arr));
    }

    public static double median(double[] arr) {
       Arrays.sort(arr);

       if (arr.length % 2 == 0) {
          return (arr[(arr.length / 2) - 1] + arr[arr.length / 2]) / 2.0;
       } 
       return arr[arr.length / 2];
    }
	public static double [] minComparison(double [] dblArray, double minVal) {
		  	double minimalValuedArray [] = new double [dblArray.length];
		  	double actualMin;
		    for (int i = 0; i < dblArray.length; i++) {
		      actualMin = Math.min(minVal, dblArray[i]);
		      minimalValuedArray[i]= actualMin;
		    }
		    return minimalValuedArray;
	  }
	  public static double [] maxComparison(double [] dblArray, double maxVal) {
		  	double []	maximalValuedArray = new double [dblArray.length];
		  	double actualMax;
		    for (int i = 0; i < dblArray.length; i++) {
		    		actualMax = Math.max(maxVal, dblArray[i]);
		    		maximalValuedArray[i]= actualMax;
		    }
		    return maximalValuedArray;
	  }
	
	public static Double [] sortSet(Set<Double>  dblSet) {
		Set<Double> tset = new TreeSet<Double>(dblSet);
//		Double [] sortedArray = new Double[tset.size()];
//		int i=0;
//		for (Double d : tset) {
//			sortedArray[i] = d;
//			i++;
//		}
//		//Double [] sortedArray = (Double []) tset.toArray();
//		return sortedArray;
		return tset.toArray (new Double[tset.size ()]);
	}
    
    public static void main(String args[]) {
    		int [] parray = createPermutatedVector(10,0);
    		for(int i=0; i<parray.length; i++ ) {
    			System.out.println(i+" : "+parray[i]);
    		}
    }
    
    
    
    
}