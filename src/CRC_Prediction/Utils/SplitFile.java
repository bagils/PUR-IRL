package CRC_Prediction.Utils;
/*
 * Split File
 * 
 * Master's Thesis by Greg Dougherty
 * Created: Mar 18, 2008
 * 
 * Copyright 2007 by Greg Dougherty
 * License to be determined.
 */

import java.util.ArrayList;

/**
 * Class that provides a {@link String} split function that does not do regular expression 
 * processing of the split string
 * 
 * @author Greg Dougherty
 */
public class SplitFile
{
	
	/** Constant to say want String.split to find every possible split in a String */
	public static final int		kReturnAll = -1;
	/** Constant to say want SplitFile.mySplit to return a blank string if separator is last thing in target */
	public static final boolean	kIncludeBlank = true;
	private static final int	kNotFound = -1;
	
	
	/**
	 * Return an array of strings split on the string split.  Unlike the String
	 * class version of this function, split is not treated as a regular expression
	 * 
	 * @param target	string to split up
	 * @param split		string with which to do the splitting
	 * @param maxCol	Maximum number of columns to return. -1 to return all.  If have more than 
	 * {@code maxCol} columns, all remaining columns will be returned as the {@code maxCol} column, 
	 * with the results having a length of {@code maxCol} + 1
	 * @return	An empty array if target is length 0 or null, otherwise an array with lesser of 
	 * {@code maxCol} or n + 1 strings, where n is the number of occurrences of split.
	 * Will return only n strings if the final occurrence of split is at the
	 * end of target.
	 */
	public static final String[] mySplit (String target, String split, int maxCol)
	{
		return mySplit (target, split, maxCol, false, false);
	}
	
	
	/**
	 * Return an array of strings split on the string split.  Unlike the String
	 * class version of this function, split is not treated as a regular expression
	 * 
	 * @param target	String to split up
	 * @param split		String with which to do the splitting
	 * @param maxCol	Maximum number of columns to return. -1 to return all.  If have more than 
	 * {@code maxCol} columns, all remaining columns will be returned as the {@code maxCol} column, 
	 * with the results having a length of {@code maxCol} + 1
	 * @param includeBlank	If true, will include an empty string if final occurrence of split 
	 * is at the end of target 
	 * @return	An empty array if target is length 0 or null, otherwise an array with lesser of 
	 * {@code maxCol} or n + 1 strings, where n is the number of occurrences of split.
	 * Will return only n strings if the final occurrence of split is at the
	 * end of target and includeBlank is false
	 */
	public static final String[] mySplit (String target, String split, int maxCol, boolean includeBlank)
	{
		return mySplit (target, split, maxCol, includeBlank, false);
	}
	
	
	/**
	 * Return an array of strings split on the string split.  Unlike the String
	 * class version of this function, split is not treated as a regular expression
	 * 
	 * @param target	String to split up
	 * @param split		String with which to do the splitting
	 * @param maxCol	Maximum number of columns to return. -1 to return all.  If have more than 
	 * {@code maxCol} columns, all remaining columns will be returned as the {@code maxCol} column, 
	 * with the results having a length of {@code maxCol} + 1
	 * @param includeBlank	If true, will include an empty string if final occurrence of split 
	 * is at the end of target 
	 * @param dropExtra		If true & {@code maxCol} > 0, will return at most {@code maxCol} columns
	 * @return	An empty array if target is length 0 or null, otherwise an array with lesser of 
	 * {@code maxCol} or n + 1 strings, where n is the number of occurrences of split.
	 * Will return only n strings if the final occurrence of split is at the
	 * end of target and includeBlank is false
	 */
	public static final String[] mySplit (String target, String split, int maxCol, boolean includeBlank, boolean dropExtra)
	{
		if (StringUtils.isEmpty (target))
			return new String[0];
		
		ArrayList<String>	items = new ArrayList<String> ();	
		
		int	nextPos, curPos = 0;
		int	len = split.length ();
		int	targetLen = target.length ();
		int	numSplits = 0;
		
		if (maxCol <= 0)
		{
			maxCol = targetLen;
			dropExtra = false;
		}
		
		while ((numSplits < maxCol) && ((nextPos = target.indexOf (split, curPos)) > kNotFound))
		{
			items.add (target.substring (curPos, nextPos));
			curPos = nextPos + len;
			++numSplits;
		}
		
		// If have a string at the end, add it
		if (!dropExtra && (curPos < targetLen))	// This can only be true if numSplits == maxCol
			items.add (target.substring (curPos, targetLen));
		else if ((numSplits < maxCol) && includeBlank)
			items.add ("");
		
		String[]	results = new String[items.size ()];
		return items.toArray (results);
	}
	
}
