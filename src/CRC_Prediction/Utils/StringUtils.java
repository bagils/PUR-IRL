/**
 * Utilities
 *
 * <p>@author Gregory Dougherty</p>
 * Copyright Mayo Clinic, 2014
 *
 */
package CRC_Prediction.Utils;

import static CRC_Prediction.Utils.SplitFile.kReturnAll;
import java.io.*;
import java.text.*;
import java.util.*;
import java.util.Base64;
import java.util.Base64.Encoder;
import java.util.Map.Entry;

/**
 * Class of static String related utilities
 *
 * <p>@author Gregory Dougherty</p>
 */
public class StringUtils
{
	private static Map<Character, Character>	gDNADictionary = null;
	private static Map<Character, Character>	gRNADictionary = null;
	
	/** Default value to return when failing to parse an int (-1) */
	public static final int		kInvalidValue = -1;
	private static final String[]	kQuotes = {"\"", "'"};
	private final static String[]	kHexArray = stringToStringArray ("0123456789ABCDEF");
	/** Default delimiter to use when printing arrays */
	public static final String	kDefaultDelimiter = "\t";
	/** Default delimiter to use when printing arrays that are values in a {@link Map} */
	public static final String	kDefaultSubDelimiter = ", ";
	/** If the sub elements are Lists or Arrays, wrap them with [] */
	public static final boolean	kDoWrapListAndArray = true;
	/** Do not wrap sub elements with [] */
	public static final boolean	kDoNotWrapListAndArray = false;
	/** Write the item with println or equivalent */
	private static final boolean	kDoEndLine = true;
	/** Write the item with print or equivalent, do not end the line after printing this out */
	private static final boolean	kDoNotEndLine = false;
	private static final String		kNewLine = "\n";
	private static final String		kSeparator = ", ";
	private static final int		kSeparatorLen = kSeparator.length ();
	private static final char		kMaskChar = '*';
	/** Character to use when requested the complement of an unrecognized 'base' */
	public static final char		kMissingChar = '.';
	private static final char[][]	kDNACompliments = {{'a', 't'}, {'c', 'g'}, {'g', 'c'}, {'t', 'a'}, {'n', 'n'}, 
	                             	                   {'A', 'T'}, {'C', 'G'}, {'G', 'C'}, {'T', 'A'}, {'N', 'N'}};
	private static final char[][]	kRNACompliments = {{'a', 'u'}, {'c', 'g'}, {'g', 'c'}, {'u', 'a'}, {'n', 'n'}, 
	                             	                   {'A', 'U'}, {'C', 'G'}, {'G', 'C'}, {'U', 'A'}, {'N', 'N'}};
	
	/**
	 * Print out the properties, one to a line key : value
	 * 
	 * @param theProperties	Properties to print out.  Must not be null
	 * @param where			Where to print the results, must be valid and not null
	 */
	public static final void dumpProperties (Properties theProperties, PrintStream where)
	{
		dumpProperties (theProperties, null, where);
	}
	
	
	/**
	 * Print out the properties, one to a line key : value
	 * 
	 * @param theProperties	{@link Properties} to print out.  Must not be null
	 * @param maskProp		If not null, name of property to mask out when dumping {@code theProperties}.
	 * Typically would be a password property 
	 * @param where			Where to print the results, must be valid and not null
	 */
	public static final void dumpProperties (Properties theProperties, String maskProp, PrintStream where)
	{
		Iterator<Entry<Object, Object>>	iter = theProperties.entrySet ().iterator ();
		boolean							testProp = !isEmpty (maskProp);
		
		while (iter.hasNext ())
		{
			Entry<Object, Object>	entry = iter.next ();
			String					key = entry.getKey ().toString ();
			String					value = entry.getValue ().toString ();
			
			where.print (key);
			where.print (" : ");
			if (testProp && maskProp.equals (key))
				where.println (maskString (value, kMaskChar));
			else
				where.println (value);
		}
	}
	
	
	/**
	 * Given a {@link String}, replace all the characters in {@code value} 
	 * with {@code *}
	 * 
	 * @param value		{@link String} to use as a base.  If null or empty will return an empty string
	 * @return	A {@link String}, possible empty, never null
	 */
	public static String maskString (String value)
	{
		return maskString (value, kMaskChar);
	}
	
	
	/**
	 * Given a {@link String} and a {@code maskChar}, replace all the characters in {@code value} 
	 * with {@code maskChar}
	 * 
	 * @param value		{@link String} to use as a base.  If null or empty will return an empty string
	 * @param maskChar	Char to use to replace all characters of {@code value}
	 * @return	A {@link String}, possible empty, never null
	 */
	public static String maskString (String value, char maskChar)
	{
		if (value == null)
			return "";
		
		int	len = value.length ();
		
		if (len == 0)
			return "";
		
		StringBuilder	result = new StringBuilder (len);
		
		for (int i = 0; i < len; ++i)
			result.append (maskChar);
		
		return result.toString ();
	}
	
	
	/**
	 * Take an array and append its elements to a {@link StringBuilder} as a single tab delimited line
	 * 
	 * @param theArray	Array of objects to be concatenated together.  If null will add nothing, 
	 * null elements in {@code theArray} will produce empty columns
	 * @param where		Where to append the results, must be valid and not null
	 * @param <T>		Array type
	 */
	public static final <T> void dumpArray (T[] theArray, StringBuilder where)
	{
		dumpArray (theArray, kDefaultDelimiter, where);
	}
	
	
	/**
	 * Take an array and append its elements to a {@link StringBuilder} as a single delimited line
	 * 
	 * @param theArray	Array of objects to be concatenated together.  If null will add nothing, 
	 * blank line, null elements in {@code theArray} will produce empty columns
	 * @param delimiter	delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param where		Where to append the results, must be valid and not null
	 */
	public static final <T> void dumpArray (T[] theArray, String delimiter, StringBuilder where)
	{
		if (theArray == null)
			return;
		
		if (delimiter == null)
			delimiter = kDefaultDelimiter;
		
		boolean	first = true;
		
		for (T item : theArray)
		{
			if (first)
				first = false;
			else
				where.append (delimiter);
			
			if (item != null)
			{
				if (item instanceof Object[])
					dumpArray ((Object[]) item, delimiter, kDoWrapListAndArray, where);
				else if (item.getClass ().isArray ())
					dumpPrimitiveArray (item, delimiter, kDoWrapListAndArray, where);
				else
					where.append (item.toString ());
			}
		}
	}
	
	
	/**
	 * Take an array and print out its elements to a {@link StringBuilder} as a single delimited line
	 * 
	 * @param theArray	Array of objects to be concatenated together.  If null will print a 
	 * blank line, null elements in {@code theArray} will produce empty columns
	 * @param delimiter	delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param wrap		If true, add [] around output
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Array type
	 */
	public static final <T> void dumpArray (T[] theArray, String delimiter, boolean wrap, StringBuilder where)
	{
		if (theArray == null)
		{
			if (wrap)
				where.append ("[]");
			where.append (kNewLine);
			return;
		}
		
		if (delimiter == null)
			delimiter = kDefaultDelimiter;
		
		boolean	first = true;
		
		if (wrap)
			where.append ("[");
		for (T item : theArray)
		{
			if (first)
				first = false;
			else
				where.append (delimiter);
			
			if (item != null)
			{
				if (item instanceof Object[])
					dumpArray ((Object[]) item, delimiter, kDoWrapListAndArray, where);
				else if (item.getClass ().isArray ())
					dumpPrimitiveArray (item, delimiter, kDoWrapListAndArray, where);
				else
					where.append (item.toString ());
			}
		}
		
		if (wrap)
			where.append ("]");
		where.append (kNewLine);
	}
	
	
	/**
	 * Take an array and print out its elements to a {@link PrintStream} as a single tab delimited line
	 * 
	 * @param theArray	Array of objects to be concatenated together.  If null will print a 
	 * blank line, null elements in {@code theArray} will produce empty columns
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Array type
	 */
	public static final <T> void dumpArray (T[] theArray, PrintStream where)
	{
		dumpArray (theArray, kDefaultDelimiter, kDoWrapListAndArray, where);
	}
	
	
	/**
	 * Take an array and print out its elements to a {@link PrintStream} as a single delimited line
	 * 
	 * @param theArray	Array of objects to be concatenated together.  If null will print a 
	 * blank line, null elements in {@code theArray} will produce empty columns
	 * @param delimiter	delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Array type
	 */
	public static final <T> void dumpArray (T[] theArray, String delimiter, PrintStream where)
	{
		dumpArray (theArray, delimiter, kDoWrapListAndArray, where);
	}
	
	
	/**
	 * Take an array and print out its elements to a {@link PrintStream} as a single delimited line
	 * 
	 * @param theArray	Array of objects to be concatenated together.  If null will print a 
	 * blank line, null elements in {@code theArray} will produce empty columns
	 * @param delimiter	delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param wrap		If true, add [] around output
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Array type
	 */
	public static final <T> void dumpArray (T[] theArray, String delimiter, boolean wrap, PrintStream where)
	{
		if (theArray == null)
		{
			if (wrap)
				where.println ("[]");
			else
				where.println ();
			return;
		}
		
		if (delimiter == null)
			delimiter = kDefaultDelimiter;
		
		boolean	first = true;
		
		if (wrap)
			where.print ("[");
		for (T item : theArray)
		{
			if (first)
				first = false;
			else
				where.print (delimiter);
			
			if (item != null)
			{
				if (item instanceof Object[])
					dumpArray ((Object[]) item, delimiter, kDoWrapListAndArray, where);
				else if (item.getClass ().isArray ())
					dumpPrimitiveArray (item, delimiter, wrap, where);
				else
					where.print (item.toString ());
			}
		}
		
		if (wrap)
			where.println ("]");
		else
			where.println ();
	}
	
	private static final String	kDoubleFormat = "###,###.###";
	private static final DecimalFormat	doubleFormat = new DecimalFormat (kDoubleFormat);
	
	/**
	 * Dump out the contents of an array of primitive types
	 * 
	 * @param item		Item
	 * @param delimiter	delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param where		Where to print the results, must be valid and not null
	 * @param wrap		If true, add [] around output
	 * @param <T>	Type of item
	 */
	private static final <T> void dumpPrimitiveArray (T item, String delimiter, boolean wrap, PrintStream where)
	{
		if (wrap)
			where.print ("[");
		
		if (item instanceof int[])
		{
			boolean	first = true;
			
			for (int value : (int[]) item)
			{
				if (first)
					first = false;
				else
					where.print (delimiter);
				
				where.print (value);
			}
		}
		else if (item instanceof double[])
		{
			boolean	first = true;
			
			for (double value : (double[]) item)
			{
				if (first)
					first = false;
				else
					where.print (delimiter);
				
				where.print (doubleFormat.format (value));
			}
		}
		else if (item instanceof long[])
		{
			boolean	first = true;
			
			for (long value : (long[]) item)
			{
				if (first)
					first = false;
				else
					where.print (delimiter);
				
				where.print (value);
			}
		}
		else if (item instanceof boolean[])
		{
			boolean	first = true;
			
			for (boolean value : (boolean[]) item)
			{
				if (first)
					first = false;
				else
					where.print (delimiter);
				
				where.print (value);
			}
		}
		else 
			where.print (item.toString ());
		
		if (wrap)
			where.print ("]");
	}
	
	
	/**
	 * Dump out the contents of an array of primitive types
	 * 
	 * @param item		Item
	 * @param delimiter	delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param where		Where to print the results, must be valid and not null
	 * @param wrap		If true, add [] around output
	 * @param <T>	Type of item
	 */
	private static final <T> void dumpPrimitiveArray (T item, String delimiter, boolean wrap, StringBuilder where)
	{
		if (wrap)
			where.append ("[");
		
		if (item instanceof int[])
		{
			boolean	first = true;
			
			for (int value : (int[]) item)
			{
				if (first)
					first = false;
				else
					where.append (delimiter);
				
				where.append (value);
			}
		}
		else if (item instanceof double[])
		{
			boolean	first = true;
			
			for (double value : (double[]) item)
			{
				if (first)
					first = false;
				else
					where.append (delimiter);
				
				where.append (doubleFormat.format (value));
			}
		}
		else if (item instanceof long[])
		{
			boolean	first = true;
			
			for (long value : (long[]) item)
			{
				if (first)
					first = false;
				else
					where.append (delimiter);
				
				where.append (value);
			}
		}
		else if (item instanceof boolean[])
		{
			boolean	first = true;
			
			for (boolean value : (boolean[]) item)
			{
				if (first)
					first = false;
				else
					where.append (delimiter);
				
				where.append (value);
			}
		}
		else 
			where.append (item.toString ());
		
		if (wrap)
			where.append ("]");
	}
	
	
	/**
	 * Dump out the contents of an array of primitive types
	 * 
	 * @param item		Item
	 * @param delimiter	delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param where		Where to print the results, must be valid and not null
	 * @param wrap		If true, add [] around output
	 * @param <T>	Type of item
	 * @throws IOException	If can't write to {@code where}
	 */
	private static final <T> void dumpPrimitiveArray (T item, String delimiter, boolean wrap, BufferedWriter where) throws IOException
	{
		if (wrap)
			where.write ("[");
		
		if (item instanceof int[])
		{
			boolean	first = true;
			
			for (int value : (int[]) item)
			{
				if (first)
					first = false;
				else
					where.write (delimiter);
				
				where.write (Integer.toString (value));
			}
		}
		else if (item instanceof double[])
		{
			boolean	first = true;
			
			for (double value : (double[]) item)
			{
				if (first)
					first = false;
				else
					where.write (delimiter);
				
				where.write (doubleFormat.format (value));
			}
		}
		else if (item instanceof long[])
		{
			boolean	first = true;
			
			for (long value : (long[]) item)
			{
				if (first)
					first = false;
				else
					where.write (delimiter);
				
				where.write (Long.toString (value));
			}
		}
		else if (item instanceof boolean[])
		{
			boolean	first = true;
			
			for (boolean value : (boolean[]) item)
			{
				if (first)
					first = false;
				else
					where.write (delimiter);
				
				where.write (Boolean.toString (value));
			}
		}
		else 
			where.write (item.toString ());
		
		if (wrap)
			where.write ("]");
	}
	
	
	/**
	 * Take an array and print out its elements to a {@link BufferedWriter} as a single tab delimited line
	 * 
	 * @param theArray	Array of objects to be concatenated together.  If null will print a 
	 * blank line, null elements in {@code theArray} will produce empty columns
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Array type
	 * @throws IOException	If have problems writing to {@code where}
	 */
	public static final <T> void dumpArray (T[] theArray, BufferedWriter where) throws IOException
	{
		dumpArray (theArray, kDefaultDelimiter, kDoWrapListAndArray, where);
	}
	
	
	/**
	 * Take an array and print out its elements to a {@link BufferedWriter} as a single delimited line
	 * 
	 * @param theArray	Array of objects to be concatenated together.  If null will print a 
	 * blank line, null elements in {@code theArray} will produce empty columns
	 * @param delimiter	delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Array type
	 * @throws IOException	If have problems writing to {@code where}
	 */
	public static final <T> void dumpArray (T[] theArray, String delimiter, BufferedWriter where) throws IOException
	{
		dumpArray (theArray, delimiter, kDoWrapListAndArray, where);
	}
	
	
	/**
	 * Take an array and print out its elements to a {@link BufferedWriter} as a single delimited line
	 * 
	 * @param theArray	Array of objects to be concatenated together.  If null will print a 
	 * blank line, null elements in {@code theArray} will produce empty columns
	 * @param delimiter	delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param wrap		If true, add [] around output
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Array type
	 * @throws IOException	If have problems writing to {@code where}
	 */
	public static final <T> void dumpArray (T[] theArray, String delimiter, boolean wrap, BufferedWriter where) throws IOException
	{
		if (theArray == null)
		{
			if (wrap)
				where.write ("[]");
			where.newLine ();
			return;
		}
		
		if (delimiter == null)
			delimiter = kDefaultDelimiter;
		
		boolean	first = true;
		
		if (wrap)
			where.write ("[");
		for (T item : theArray)
		{
			if (first)
				first = false;
			else
				where.write (delimiter);
			
			if (item != null)
			{
				if (item instanceof Object[])
					dumpArray ((Object[]) item, delimiter, kDoWrapListAndArray, where);
				else if (item.getClass ().isArray ())
					dumpPrimitiveArray (item, delimiter, wrap, where);
				else
					where.write (item.toString ());
			}
		}
		
		if (wrap)
			where.write ("]");
		where.newLine ();
	}
	
	
	/**
	 * Print an item to {@code where} as one line.  If {@code value} is an array of {@link List}, 
	 * print out each of its elements, separated by {@code delimiter}.  if  is a {@link Date}, use 
	 * {@link DateUtils#formatWithTime (Date)} to print out the date
	 * 
	 * @param value		Item to write to {@code where}.  If null will write ""
	 * @param delimiter	delimiter to use.  Must not be null
	 * @param needLine	If true, do println, if false, to print
	 * @param wrap		If true, add [] around arrays or lists
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Element type
	 */
	private static final <T> void dumpItem (T value, String delimiter, boolean needLine, boolean wrap, PrintStream where)
	{
		if (value == null)
			where.print ("\"\"");
		else if (value instanceof Object[])
		{
			dumpArray ((Object[]) value, delimiter, wrap, where);
			needLine = false;
		}
		else if (value instanceof List<?>)
		{
			dumpList ((List<?>) value, delimiter, kDefaultSubDelimiter, wrap, where);
			needLine = false;
		}
		else if (value.getClass ().isArray ())
			dumpPrimitiveArray (value, delimiter, wrap, where);
		else if (value instanceof Date)
			where.print (DateUtils.formatWithTime ((Date) value));
		else
			where.print (value.toString ());
		
		if (needLine)
			where.println ();
	}
	
	
	/**
	 * Print an item to {@code where} as one line.  If {@code value} is an array of {@link List}, 
	 * print out each of its elements, separated by {@code delimiter}.  if  is a {@link Date}, use 
	 * {@link DateUtils#formatWithTime (Date)} to print out the date
	 * 
	 * @param value		Item to write to {@code where}.  If null will write ""
	 * @param delimiter	delimiter to use.  Must not be null
	 * @param needLine	If true, add a newline after printing this item
	 * @param wrap		If true, add [] around arrays or lists
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Element type
	 * @throws IOException	If have problems writing to {@code where}
	 */
	private static final <T> void dumpItem (T value, String delimiter, boolean needLine, boolean wrap, BufferedWriter where) throws IOException
	{
		if (value == null)
			where.write ("\"\"");
		else if (value instanceof Object[])
		{
			dumpArray ((Object[]) value, delimiter, wrap, where);
			needLine = false;
		}
		else if (value instanceof List<?>)
		{
			dumpList ((List<?>) value, delimiter, kDefaultSubDelimiter, wrap, where);
			needLine = false;
		}
		else if (value.getClass ().isArray ())
			dumpPrimitiveArray (value, delimiter, wrap, where);
		else if (value instanceof Date)
			where.write (DateUtils.formatWithTime ((Date) value));
		else
			where.write (value.toString ());
		
		if (needLine)
			where.newLine ();
	}
	
	
	/**
	 * Print an item to {@code where} as one line.  If {@code value} is an array of {@link List}, 
	 * print out each of its elements, separated by {@code delimiter}.  if  is a {@link Date}, use 
	 * {@link DateUtils#formatWithTime (Date)} to print out the date
	 * 
	 * @param value		Item to write to {@code where}.  If null will write ""
	 * @param delimiter	delimiter to use.  Must not be null
	 * @param needLine	If true, add a newline after printing this item
	 * @param wrap		If true, add [] around arrays or lists
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Element type
	 * @throws IOException	If have problems writing to {@code where}
	 */
	private static final <T> void dumpItem (T value, String delimiter, boolean needLine, boolean wrap, StringBuilder where) throws IOException
	{
		if (value == null)
			where.append ("\"\"");
		else if (value instanceof Object[])
		{
			dumpArray ((Object[]) value, delimiter, wrap, where);
			needLine = false;
		}
		else if (value instanceof List<?>)
		{
			dumpList ((List<?>) value, delimiter, kDefaultSubDelimiter, wrap, where);
			needLine = false;
		}
		else if (value.getClass ().isArray ())
			dumpPrimitiveArray (value, delimiter, wrap, where);
		else if (value instanceof Date)
			where.append (DateUtils.formatWithTime ((Date) value));
		else
			where.append (value.toString ());
		
		if (needLine)
			where.append (kNewLine);
	}
	
	
	/**
	 * Take a {@link List} and print out its elements to a {@link PrintStream} as a single delimited line, 
	 * using {@link #kDefaultDelimiter} as the delimiter and {@link #kDefaultSubDelimiter} 
	 * as the delimiter between any array or {@link List} values
	 * 
	 * @param theList	{@link List} of objects to be concatenated together.  If null will print a 
	 * blank line, null elements in {@code theList} will produce empty columns
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		List type
	 */
	public static final <T> void dumpList (List<T> theList, PrintStream where)
	{
		dumpList (theList, kDefaultDelimiter, kDefaultSubDelimiter, kDoNotWrapListAndArray, where);
	}
	
	
	/**
	 * Take a {@link List} and print out its elements to a {@link PrintStream} as a single delimited line, 
	 * using {@code delimiter} as the delimiter and {@link #kDefaultSubDelimiter} as the delimiter 
	 * between any array or {@link List} values
	 * 
	 * @param theList	{@link List} of objects to be concatenated together.  If null will print a 
	 * blank line, null elements in {@code theList} will produce empty columns
	 * @param delimiter	delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		List type
	 */
	public static final <T> void dumpList (List<T> theList, String delimiter, PrintStream where)
	{
		dumpList (theList, delimiter, kDefaultSubDelimiter, kDoNotWrapListAndArray, where);
	}
	
	
	/**
	 * Take a {@link List} and print out its elements to a {@link PrintStream} as a single delimited line
	 * 
	 * @param theList		{@link List} of objects to be concatenated together.  If null will print 
	 * a blank line, null elements in {@code theList} will produce empty columns
	 * @param delimiter		delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param subDelimiter	delimiter to use between array or {@link List} elements in {@code T} 
	 * if {@code T} is an array or list.  If null will use {@link #kDefaultSubDelimiter}
	 * @param where			Where to print the results, must be valid and not null
	 * @param <T>			List type
	 */
	public static final <T> void dumpList (List<T> theList, String delimiter, String subDelimiter, PrintStream where)
	{
		dumpList (theList, delimiter, subDelimiter, kDoNotWrapListAndArray, where);
	}
	
	
	/**
	 * Take a {@link List} and print out its elements to a {@link PrintStream} as a single delimited line
	 * 
	 * @param theList		{@link List} of objects to be concatenated together.  If null will print 
	 * a blank line, null elements in {@code theList} will produce empty columns
	 * @param delimiter		delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param subDelimiter	delimiter to use between array or {@link List} elements in {@code T} 
	 * if {@code T} is an array or list.  If null will use {@link #kDefaultSubDelimiter}
	 * @param wrap			If true, add [] around output
	 * @param where			Where to print the results, must be valid and not null
	 * @param <T>			List type
	 */
	public static final <T> void dumpList (List<T> theList, String delimiter, String subDelimiter, boolean wrap, PrintStream where)
	{
		if (theList == null)
		{
			if (wrap)
				where.println ("[]");
			else
				where.println ();
			return;
		}
		
		if (delimiter == null)
			delimiter = kDefaultDelimiter;
		
		boolean	first = true;
		boolean	subWrap = !delimiter.equals (kNewLine);
		
		if (wrap)
			where.print ("[");
		for (T item : theList)
		{
			if (first)
				first = false;
			else
				where.print (delimiter);
			
			if (item != null)
				dumpItem (item, subDelimiter, kDoNotEndLine, subWrap, where);
		}
		
		if (wrap)
			where.println ("]");
		else
			where.println ();
	}
	
	
	/**
	 * Take a {@link List} and print out its elements to a {@link BufferedWriter} as a single delimited line, 
	 * using {@link #kDefaultDelimiter} as the delimiter and {@link #kDefaultSubDelimiter} 
	 * as the delimiter between any array or {@link List} values
	 * 
	 * @param theList	{@link List} of objects to be concatenated together.  If null will print a 
	 * blank line, null elements in {@code theList} will produce empty columns
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		List type
	 * @throws IOException	If have problems writing to {@code where}
	 */
	public static final <T> void dumpList (List<T> theList, BufferedWriter where) throws IOException
	{
		dumpList (theList, kDefaultDelimiter, kDefaultSubDelimiter, kDoNotWrapListAndArray, where);
	}
	
	
	/**
	 * Take a {@link List} and print out its elements to a {@link BufferedWriter} as a single delimited line, 
	 * using {@code delimiter} as the delimiter and {@link #kDefaultSubDelimiter} as the delimiter 
	 * between any array or {@link List} values
	 * 
	 * @param theList	{@link List} of objects to be concatenated together.  If null will print a 
	 * blank line, null elements in {@code theList} will produce empty columns
	 * @param delimiter	delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		List type
	 * @throws IOException	If have problems writing to {@code where}
	 */
	public static final <T> void dumpList (List<T> theList, String delimiter, BufferedWriter where) throws IOException
	{
		dumpList (theList, delimiter, kDefaultSubDelimiter, kDoNotWrapListAndArray, where);
	}
	
	
	/**
	 * Take a {@link List} and print out its elements to a {@link BufferedWriter} as a single delimited line
	 * 
	 * @param theList		{@link List} of objects to be concatenated together.  If null will print 
	 * a blank line, null elements in {@code theList} will produce empty columns
	 * @param delimiter		delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param subDelimiter	delimiter to use between array or {@link List} elements in {@code T} 
	 * if {@code T} is an array or list.  If null will use {@link #kDefaultSubDelimiter}
	 * @param where			Where to print the results, must be valid and not null
	 * @param <T>			List type
	 * @throws IOException	If have problems writing to {@code where}
	 */
	public static final <T> void dumpList (List<T> theList, String delimiter, String subDelimiter, BufferedWriter where) throws IOException
	{
		dumpList (theList, delimiter, subDelimiter, kDoNotWrapListAndArray, where);
	}
	
	
	/**
	 * Take a {@link List} and print out its elements to a {@link BufferedWriter} as a single delimited line
	 * 
	 * @param theList		{@link List} of objects to be concatenated together.  If null will print 
	 * a blank line, null elements in {@code theList} will produce empty columns
	 * @param delimiter		delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param subDelimiter	delimiter to use between array or {@link List} elements in {@code T} 
	 * if {@code T} is an array or list.  If null will use {@link #kDefaultSubDelimiter}
	 * @param wrap			If true, add [] around output
	 * @param where			Where to print the results, must be valid and not null
	 * @param <T>			List type
	 * @throws IOException	If have problems writing to {@code where}
	 */
	public static final <T> void dumpList (List<T> theList, String delimiter, String subDelimiter, boolean wrap, BufferedWriter where) 
			throws IOException
	{
		if (theList == null)
		{
			if (wrap)
				where.write ("[]");
			where.newLine ();
			return;
		}
		
		if (delimiter == null)
			delimiter = kDefaultDelimiter;
		
		boolean	first = true;
		boolean	subWrap = !delimiter.equals (kNewLine);
		
		if (wrap)
			where.write ("[");
		for (T item : theList)
		{
			if (first)
				first = false;
			else
				where.write (delimiter);
			
			if (item != null)
				dumpItem (item, subDelimiter, kDoNotEndLine, subWrap, where);
		}
		
		if (wrap)
			where.write ("]");
		where.newLine ();
	}
	
	
	/**
	 * Take a {@link List} and print out its elements to a {@link StringBuilder} as a single delimited line
	 * 
	 * @param theList		{@link List} of objects to be concatenated together.  If null will print 
	 * a blank line, null elements in {@code theList} will produce empty columns
	 * @param delimiter		delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param subDelimiter	delimiter to use between array or {@link List} elements in {@code T} 
	 * if {@code T} is an array or list.  If null will use {@link #kDefaultSubDelimiter}
	 * @param wrap			If true, add [] around output
	 * @param where			Where to print the results, must be valid and not null
	 * @param <T>			List type
	 * @throws IOException	If have problems writing to {@code where}
	 */
	public static final <T> void dumpList (List<T> theList, String delimiter, String subDelimiter, boolean wrap, StringBuilder where) 
			throws IOException
	{
		if (theList == null)
		{
			if (wrap)
				where.append ("[]");
			where.append (kNewLine);
			return;
		}
		
		if (delimiter == null)
			delimiter = kDefaultDelimiter;
		
		boolean	first = true;
		boolean	subWrap = !delimiter.equals (kNewLine);
		
		if (wrap)
			where.append ("[");
		for (T item : theList)
		{
			if (first)
				first = false;
			else
				where.append (delimiter);
			
			if (item != null)
				dumpItem (item, subDelimiter, kDoNotEndLine, subWrap, where);
		}
		
		if (wrap)
			where.append ("]");
		where.append (kNewLine);
	}
	
	
	/**
	 * Take a {@link Map} and print out its elements to a {@link PrintStream} as one delimited line 
	 * per entry, using {@link #kDefaultDelimiter} as the delimiter and {@link #kDefaultSubDelimiter} 
	 * as the delimiter between any array or {@link List} values
	 * 
	 * @param theMap	{@link Map} of objects to be printed out.  If null will print a 
	 * blank line, null elements in {@code theMap} will produce empty columns
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Key type
	 * @param <U>		Value type.  If array will use {@link #dumpArray(Object[], String, PrintStream)}
	 */
	public static final <T, U> void dumpMap (Map<T, U> theMap, PrintStream where)
	{
		dumpMap (theMap, kDefaultDelimiter, kDefaultSubDelimiter, where);
	}
	
	
	/**
	 * Take a {@link Map} and print out its elements to a {@link PrintStream} as one delimited line 
	 * per entry, using {@code delimiter} as the delimiter and {@link #kDefaultSubDelimiter} 
	 * as the delimiter between any array or {@link List} values
	 * 
	 * @param theMap	{@link Map} of objects to be printed out.  If null will print a 
	 * blank line, null elements in {@code theMap} will produce empty columns
	 * @param delimiter	delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Key type
	 * @param <U>		Value type.  If array will use {@link #dumpArray(Object[], String, PrintStream)}
	 */
	public static final <T, U> void dumpMap (Map<T, U> theMap, String delimiter, PrintStream where)
	{
		dumpMap (theMap, delimiter, kDefaultSubDelimiter, where);
	}
	
	
	/**
	 * Take a {@link Map} and print out its elements to a {@link PrintStream} as one delimited line 
	 * per entry, using {@code delimiter} as the delimiter and {@code subDelimiter} as the delimiter 
	 * between any array or {@link List} values
	 * 
	 * @param theMap		{@link Map} of objects to be printed out.  If null will print a 
	 * blank line, null elements in {@code theArray} will produce empty columns
	 * @param delimiter		delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param subDelimiter	delimiter to use between array or {@link List} elements in {@code U} 
	 * if {@code U} is an array or list.  If null will use {@link #kDefaultSubDelimiter}
	 * @param where			Where to print the results, must be valid and not null
	 * @param <T>			Key type
	 * @param <U>			Value type.  If array will use {@link #dumpArray(Object[], String, PrintStream)}
	 */
	public static final <T, U> void dumpMap (Map<T, U> theMap, String delimiter, String subDelimiter, PrintStream where)
	{
		if (theMap == null)
		{
			where.println ();
			return;
		}
		
		if (delimiter == null)
			delimiter = kDefaultDelimiter;
		if (subDelimiter == null)
			subDelimiter = kDefaultSubDelimiter;
		
		Set<Entry<T, U>>	entrySet = theMap.entrySet ();
		
		for (Entry<T, U> entry : entrySet)
		{
			T	key = entry.getKey ();
			U	value = entry.getValue ();
			
			dumpItem (key, subDelimiter, kDoNotEndLine, kDoWrapListAndArray, where);
			where.print (delimiter);
			dumpItem (value, subDelimiter, kDoEndLine, kDoWrapListAndArray, where);
		}
	}
	
	
	/**
	 * Take a {@link Map} and print out its elements to a {@link BufferedWriter} as one delimited line 
	 * per entry, using {@link #kDefaultDelimiter} as the delimiter and {@link #kDefaultSubDelimiter} 
	 * as the delimiter between any array or {@link List} values
	 * 
	 * @param theMap	{@link Map} of objects to be printed out.  If null will print a 
	 * blank line, null elements in {@code theMap} will produce empty columns
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Key type
	 * @param <U>		Value type.  If array will use {@link #dumpArray(Object[], String, PrintStream)}
	 * @throws IOException	If have problems writing to {@code where}
	 */
	public static final <T, U> void dumpMap (Map<T, U> theMap, BufferedWriter where) throws IOException
	{
		dumpMap (theMap, kDefaultDelimiter, kDefaultSubDelimiter, where);
	}
	
	
	/**
	 * Take a {@link Map} and print out its elements to a {@link BufferedWriter} as one delimited line 
	 * per entry, using {@code delimiter} as the delimiter and {@link #kDefaultSubDelimiter} 
	 * as the delimiter between any array or {@link List} values
	 * 
	 * @param theMap	{@link Map} of objects to be printed out.  If null will print a 
	 * blank line, null elements in {@code theMap} will produce empty columns
	 * @param delimiter	delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Key type
	 * @param <U>		Value type.  If array will use {@link #dumpArray(Object[], String, PrintStream)}
	 * @throws IOException	If have problems writing to {@code where}
	 */
	public static final <T, U> void dumpMap (Map<T, U> theMap, String delimiter, BufferedWriter where) throws IOException
	{
		dumpMap (theMap, delimiter, kDefaultSubDelimiter, where);
	}
	
	
	/**
	 * Take a {@link Map} and print out its elements to a {@link BufferedWriter} as one delimited line 
	 * per entry, using {@code delimiter} as the delimiter and {@code subDelimiter} as the delimiter 
	 * between any array or {@link List} values
	 * 
	 * @param theMap		{@link Map} of objects to be printed out.  If null will print a 
	 * blank line, null elements in {@code theArray} will produce empty columns
	 * @param delimiter		delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param subDelimiter	delimiter to use between array or {@link List} elements in {@code U} 
	 * if {@code U} is an array or list.  If null will use {@link #kDefaultSubDelimiter}
	 * @param where			Where to print the results, must be valid and not null
	 * @param <T>			Key type
	 * @param <U>			Value type.  If array will use {@link #dumpArray(Object[], String, PrintStream)}
	 * @throws IOException	If have problems writing to {@code where}
	 */
	public static final <T, U> void dumpMap (Map<T, U> theMap, String delimiter, String subDelimiter, BufferedWriter where) throws IOException
	{
		if (theMap == null)
		{
			where.newLine ();
			return;
		}
		
		if (delimiter == null)
			delimiter = kDefaultDelimiter;
		if (subDelimiter == null)
			subDelimiter = kDefaultSubDelimiter;
		
		Set<Entry<T, U>>	entrySet = theMap.entrySet ();
		
		for (Entry<T, U> entry : entrySet)
		{
			T	key = entry.getKey ();
			U	value = entry.getValue ();
			
			dumpItem (key, subDelimiter, kDoNotEndLine, kDoWrapListAndArray, where);
			where.write (delimiter);
			dumpItem (value, subDelimiter, kDoEndLine, kDoWrapListAndArray, where);
		}
	}
	
	
	/**
	 * Take a {@link Map} and print out its elements to a {@link StringBuilder} as one delimited line 
	 * per entry, using {@link #kDefaultDelimiter} as the delimiter and {@link #kDefaultSubDelimiter} 
	 * as the delimiter between any array or {@link List} values
	 * 
	 * @param theMap	{@link Map} of objects to be printed out.  If null will print a 
	 * blank line, null elements in {@code theMap} will produce empty columns
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Key type
	 * @param <U>		Value type.  If array will use {@link #dumpArray(Object[], String, PrintStream)}
	 * @throws IOException	If have problems writing to {@code where}
	 */
	public static final <T, U> void dumpMap (Map<T, U> theMap, StringBuilder where) throws IOException
	{
		dumpMap (theMap, kDefaultDelimiter, kDefaultSubDelimiter, where);
	}
	
	
	/**
	 * Take a {@link Map} and print out its elements to a {@link StringBuilder} as one delimited line 
	 * per entry, using {@code delimiter} as the delimiter and {@link #kDefaultSubDelimiter} 
	 * as the delimiter between any array or {@link List} values
	 * 
	 * @param theMap	{@link Map} of objects to be printed out.  If null will print a 
	 * blank line, null elements in {@code theMap} will produce empty columns
	 * @param delimiter	delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param where		Where to print the results, must be valid and not null
	 * @param <T>		Key type
	 * @param <U>		Value type.  If array will use {@link #dumpArray(Object[], String, PrintStream)}
	 * @throws IOException	If have problems writing to {@code where}
	 */
	public static final <T, U> void dumpMap (Map<T, U> theMap, String delimiter, StringBuilder where) throws IOException
	{
		dumpMap (theMap, delimiter, kDefaultSubDelimiter, where);
	}
	
	
	/**
	 * Take a {@link Map} and print out its elements to a {@link StringBuilder} as one delimited line 
	 * per entry, using {@code delimiter} as the delimiter and {@code subDelimiter} as the delimiter 
	 * between any array or {@link List} values
	 * 
	 * @param theMap		{@link Map} of objects to be printed out.  If null will print a 
	 * blank line, null elements in {@code theArray} will produce empty columns
	 * @param delimiter		delimiter to use, if null will use {@link #kDefaultDelimiter}
	 * @param subDelimiter	delimiter to use between array or {@link List} elements in {@code U} 
	 * if {@code U} is an array or list.  If null will use {@link #kDefaultSubDelimiter}
	 * @param where			Where to print the results, must be valid and not null
	 * @param <T>			Key type
	 * @param <U>			Value type.  If array will use {@link #dumpArray(Object[], String, PrintStream)}
	 * @throws IOException	If have problems writing to {@code where}
	 */
	public static final <T, U> void dumpMap (Map<T, U> theMap, String delimiter, String subDelimiter, StringBuilder where) throws IOException
	{
		if (theMap == null)
		{
			where.append (kNewLine);
			return;
		}
		
		if (delimiter == null)
			delimiter = kDefaultDelimiter;
		if (subDelimiter == null)
			subDelimiter = kDefaultSubDelimiter;
		
		Set<Entry<T, U>>	entrySet = theMap.entrySet ();
		
		for (Entry<T, U> entry : entrySet)
		{
			T	key = entry.getKey ();
			U	value = entry.getValue ();
			
			dumpItem (key, subDelimiter, kDoNotEndLine, kDoWrapListAndArray, where);
			where.append (delimiter);
			dumpItem (value, subDelimiter, kDoEndLine, kDoWrapListAndArray, where);
		}
	}
	
	
	/**
	 * Given a string, return a {@link Boolean}, null if {@code boolValue} is null, return 
	 * {@link Boolean#FALSE} if it has a case-insensitive match to "false", and otherwise returns 
	 * {@link Boolean#TRUE}
	 * 
	 * @param boolValue	String to parse, can be null
	 * @return	A {@link Boolean}, null if {@code boolValue} is null
	 */
	public static final Boolean getBoolean (String boolValue)
	{
		if (boolValue == null)
			return null;
		
		// Trim the passed in string before all tests
		if ((boolValue = boolValue.trim ()).isEmpty ())
			return Boolean.TRUE;
		
		if ("false".equals (boolValue.toLowerCase ()))
			return Boolean.FALSE;
		
		return Boolean.TRUE;
	}
	
	
	/**
	 * Given a string, return it if it isn't null, else return the default string
	 * 
	 * @param testString	{@link String} to use if not null
	 * @param theDefault	{@link String} to use if testString is null
	 * @return	A {@link String}, null only if theDefault is null
	 */
	public static final String getString (String testString, String theDefault)
	{
		if (testString != null)
			return testString;
		
		return theDefault;
	}
	
	
	/**
	 * Given a string array, return the requested index if it exists, else return the default string
	 * 
	 * @param stringArray	{@link String}[] from which to get strings, must not be null
	 * @param which			Index of {@code stringArray} to get, if it exists. Must be >= 0
	 * @param theDefault	{@link String} to use if index is out of bounds
	 * @return	A {@link String}, null only if theDefault is null
	 */
	public static final String getString (String[] stringArray, int which, String theDefault)
	{
		if (stringArray.length <= which)
			return theDefault;
		
		return stringArray[which];
	}
	
	
	/**
	 * Given a string array, return the requested index if it exists, else return the default string
	 * 
	 * @param stringArray	{@link String}[] from which to get strings, must not be null
	 * @param which			Index of {@code stringArray} to get, if it exists. Must be >= 0
	 * @param size			Official size of the array (may be > actual size, ignored if <)
	 * @param endCols		Fixed # of columns that are at the end of the array, 
	 * matters if size > stringArray.length
	 * @param theDefault	{@link String} to use if index is out of bounds
	 * @return	A {@link String}, null only if theDefault is null
	 */
	public static final String getString (String[] stringArray, int which, int size, int endCols, String theDefault)
	{
		if (endCols == 0)	// No end cols, nothing special to do
			return getString (stringArray, which, theDefault);
		
		int	len = stringArray.length;
		
		if (size <= len)	// Array is already of correct size, ignore "size"
		{
			if (len <= which)
				return theDefault;
			
			return stringArray[which];
		}
		
		if (which >= size)	// Past end of "notional" array, return the default
			return theDefault;
		
		int	cutoff = len - endCols;
		
		if (which < cutoff)	// Is it from the front block?
			return stringArray[which];
		
		int	endStart = size - endCols;
		
		if (which >= endStart)	// Is it from the back block?
			return stringArray[(which - endStart) + cutoff];
		
		// Not in front or back block, it's the default
		return theDefault;
	}
	
	
	/**
	 * Given a string array, if the requested index is null replace with {@code theDefault}, 
	 * else do nothing
	 * 
	 * @param stringArray	{@link String}[] to possibly update, must not be null
	 * @param which			Index of {@code stringArray} to test. Must be >= 0
	 * @param theDefault	{@link String} to use if {@code stringArray[which]} is null
	 */
	public static final void setString (String[] stringArray, int which, String theDefault)
	{
		if (stringArray[which] == null)
			stringArray[which] = theDefault;
	}
	
	
	/**
	 * Given a string, if it isn't null and parses to a double, return that double, else return 
	 * {@link Double#NaN}
	 * 
	 * @param theNum	String to parse if not null
	 * @return	A double.  Will return {@link Double#NaN} if can't parse {@code theNum}
	 */
	public static final double getDouble (String theNum)
	{
		return getDouble (theNum, Double.NaN);
	}
	
	
	/**
	 * Given a string, if it isn't null and parses to a double, return that double, 
	 * else return {@code theDefault}
	 * 
	 * @param theNum		String to parse if not null
	 * @param theDefault	Value to return if fail to parse {@code theNum}
	 * @return	A double.  Will return {@code theDefault} if can't parse {@code theNum}
	 */
	public static final double getDouble (String theNum, double theDefault)
	{
		if (isEmpty (theNum))
			return theDefault;
		
		try
		{
			return Double.parseDouble (theNum);
		}
		catch (NumberFormatException oops)
		{
			return theDefault;
		}
	}
	
	
	/**
	 * Given a string, if it isn't null and parses to an int, return that int, else return 
	 * {@link #kInvalidValue}
	 * 
	 * @param theNum	String to parse if not null
	 * @return	An integer.  Will return {@link #kInvalidValue} if can't parse {@code theNum}
	 */
	public static final int getInt (String theNum)
	{
		return getInt (theNum, kInvalidValue);
	}
	
	
	/**
	 * Given a string, if it isn't null and parses to an int, return that int, else return {@code theDefault}
	 * 
	 * @param theNum		String to parse if not null
	 * @param theDefault	Value to return if fail to parse {@code theNum}
	 * @return	An integer.  Will return {@code theDefault} if can't parse {@code theNum}
	 */
	public static final int getInt (String theNum, int theDefault)
	{
		if (isEmpty (theNum))
			return theDefault;
		
		try
		{
			return Integer.parseInt (theNum);
		}
		catch (NumberFormatException oops)
		{
			return theDefault;
		}
	}
	
	
	/**
	 * Parse a string, returning the integers in the string
	 * 
	 * @param parseStr	String to parse for integers
	 * @param splitter	Split string between integers
	 * @return	Array of int[], null if there were any problems at any point
	 */
	public static final int[] getInts (String parseStr, String splitter)
	{
		if (isEmpty (parseStr) || isEmpty (splitter))
			return new int[0];
		
		String[]	numStrs = SplitFile.mySplit (parseStr, splitter, kReturnAll);
		int			i = 0;
		
		try
		{
			int			numResults = numStrs.length;
			int[]		results = new int[numResults];
			
			for (; i < numResults; ++i)
				results[i] = Integer.parseInt (numStrs[i].trim ());
			
			return results;
		}
		catch (NumberFormatException oops)
		{
			// Just fail
//			System.err.print (Integer.toString (i));
//			System.err.print (": '");
//			System.err.print (numStrs[i].trim ());
//			System.err.println ("'");
		}
		
		return null;
	}
	
	
	/**
	 * Parse a string of integers separated by an arbitrary number of spaces, returning the integers 
	 * in the string
	 * 
	 * @param parseStr	String to parse for integers
	 * @return	Array of int[], null if there were any problems at any point
	 */
	public static final int[] getColumnInts (String parseStr)
	{
		if (isEmpty (parseStr))
			return new int[0];
		
		try
		{
			String[]	numStrs = parseStr.trim ().split (" +");
			int			numResults = numStrs.length;
			int[]		results = new int[numResults];
			
			for (int i = 0; i < numResults; ++i)
				results[i] = Integer.parseInt (numStrs[i]);
			
			return results;
		}
		catch (NumberFormatException oops)
		{
			// Just fail
		}
		
		return null;
	}
	
	
	/**
	 * Parse a string, returning the integers in the string
	 * 
	 * @param parseStr		String to parse for integers
	 * @param splitter		Split string between integers
	 * @param rangeMarker	String between range of ints
	 * @return	Array of int[], empty if {@code parseStr} or {@code splitter} are null or empty, 
	 * null if there were any problems parsing the contents of {@code parseStr}
	 */
	public static final int[] getInts (String parseStr, String splitter, String rangeMarker)
	{
		if (isEmpty (parseStr) || isEmpty (splitter))
			return new int[0];
		
		if ((rangeMarker == null) || rangeMarker.isEmpty ())
			return getInts (parseStr, splitter);
		
		try
		{
			String[]	numStrs = SplitFile.mySplit (parseStr, splitter, kReturnAll);
			int			markerLen = rangeMarker.length ();
			int			numResults = numStrs.length;
			int[]		results = new int[numResults];
			
			for (int curPos = 0, curStr = 0; curPos < numResults; ++curPos, ++curStr)
			{
				String	theStr = numStrs[curStr];
				int		pos = theStr.indexOf (rangeMarker);
				
				if (pos > 0)
				{
					int	start = Integer.parseInt (theStr.substring (0, pos));
					int	end = Integer.parseInt (theStr.substring (pos + markerLen));
					int	inserted = end - start;	// Already have space for 1
					
					if (inserted > 0)
					{
						numResults += inserted;
						results = Arrays.copyOf (results, numResults);
						
						for (int j = 0; j < inserted; ++j)
						{
							results[curPos] = start;
							++curPos;
							++start;
						}
					}
					
					results[curPos] = start;	// yes, everyone hits this
				}
				else
					results[curPos] = Integer.parseInt (theStr);
			}
			
			return results;
		}
		catch (NumberFormatException oops)
		{
			// Just fail
		}
		
		return null;
	}
	
	
	/**
	 * Parse a string, returning the long integers in the string
	 * 
	 * @param parseStr	String to parse for longs
	 * @param splitter	Split string between longs
	 * @return	Array of long[], null if there were any problems at any point
	 */
	public static final long[] getLongs (String parseStr, String splitter)
	{
		if (isEmpty (parseStr) || isEmpty (splitter))
			return new long[0];
		
		try
		{
			String[]	numStrs = SplitFile.mySplit (parseStr, splitter, kReturnAll);
			int			numResults = numStrs.length;
			long[]		results = new long[numResults];
			
			for (int i = 0; i < numResults; ++i)
				results[i] = Long.parseLong (numStrs[i]);
			
			return results;
		}
		catch (NumberFormatException oops)
		{
			// Just fail
		}
		
		return null;
	}
	
	
	/**
	 * Take a {@link String} that is supposed to be an integer, parse it, add an int to it, 
	 * and return the result as a String.<br/>
	 * If it doesn't parse to an integer, then will return {@code theNum}
	 * 
	 * @param theNum	{@link String} to parse into an int.  Will return null if {@code theNum} is null
	 * @param toAdd		Value to add to the parsed int
	 * @return	A {@link String}, null if {@code theNum} is null
	 */
	public static final String addToString (String theNum, int toAdd)
	{
		return addToString (theNum, toAdd, null);
	}
	
	
	/**
	 * Take a {@link String} that is supposed to be an integer, parse it, add an int to it, 
	 * and return the result as a String.<br/>
	 * If it doesn't parse to an integer, then if {@code failStr} is not null return {@code failStr}, 
	 * else return {@code theNum}
	 * 
	 * @param theNum	{@link String} to parse into an int.  Will return {@code failStr} if {@code theNum} 
	 * is null
	 * @param toAdd		Value to add to the parsed int
	 * @param failStr	{@link String} to return if {@code theNum} doesn't parse successfully.  If null will 
	 * just return theNum
	 * @return	A {@link String}, only null if {@code theNum} and {@code failStr} are null
	 */
	public static final String addToString (String theNum, int toAdd, String failStr)
	{
		if (theNum == null)
			return failStr;
		
		if (toAdd == 0)
			return theNum;
		
		try
		{
			return "" + (Integer.parseInt (theNum) + toAdd);
		}
		catch (NumberFormatException oops)
		{
			if (failStr != null)
				return failStr;
			
			return theNum;
		}
	}
	
	
	/**
	 * Given a {@link String}, make a {@link List} of {@link String#length ()}{@code + 1} Strings 
	 * each with {@code addStr} added at a different position in {@code theStr}, from before every 
	 * character to between each one to after them all
	 * 
	 * @param theStr	{@link String} to have {@code addStr} added to it
	 * @param addStr	{@link String} to add to {@code theStr}
	 * @return	A {@link List} of {@link String}, only empty if {@code theStr} and {@code theStr} 
	 * are null or empty
	 */
	public static final List<String> addToString (String theStr, String addStr)
	{
		List<String>	results = new ArrayList<> ();
		
		if (isEmpty (theStr))
		{
			if (!isEmpty (addStr))
				results.add (addStr);
			
			return results;
		}
		
		if (isEmpty (addStr))
		{
			results.add (theStr);
			return results;
		}
		
		int				len = theStr.length ();
		StringBuilder	builder = new StringBuilder (len + addStr.length ());
		
		for (int i = 0; i <= len; ++i)
		{
			if (i > 0)
			{
				builder.delete (0, builder.length ());
				builder.append (theStr.substring (0, i));
			}
			
			builder.append (addStr);
			if (i < len)
				builder.append (theStr.substring (i));
			
			results.add (builder.toString ());
		}
		
		return results;
	}
	
	
	/**
	 * Take an array and return a {@link String} of its elements concatenated together
	 * 
	 * @param theArray	Array of objects to be concatenated together.  If null will return "", null 
	 * elements in {@code theArray} will be ignored
	 * @return	A String, never null, possibly empty
	 */
	public static final <T> String arrayToString (T[] theArray)
	{
		return arrayToString (theArray, null);
	}
	
	
	/**
	 * Take an array and return a {@link String} of its elements concatenated together
	 * 
	 * @param theArray	Array of objects to be concatenated together.  If null will return "", null 
	 * elements in {@code theArray} will be ignored
	 * @param delimiter	{@link String} to insert between each element, null or empty to insert nothing
	 * @return	A String, never null, possibly empty
	 */
	public static final <T> String arrayToString (T[] theArray, String delimiter)
	{
		if (theArray == null)
			return "";
		
		StringBuilder	builder = new StringBuilder (theArray.length * 20);
		boolean			first = true;
		
		if ((delimiter != null) && delimiter.isEmpty ())
			delimiter = null;
		
		for (T item : theArray)
		{
			if (first)
				first = false;
			else if (delimiter != null)
				builder.append (delimiter);
			
			if (item != null)
				builder.append (item);
		}
		
		return builder.toString ();
	}
	
	
	/**
	 * Take an array and return a {@link String} of its elements concatenated together, picking the 
	 * elements from cols, in that order.<br/>
	 * If item in cols is negative, count from the end.  If item in cols would produce an 
	 * {@link ArrayIndexOutOfBoundsException}, will add an empty string instead
	 * 
	 * @param theArray	Array of objects to be concatenated together.  If null will return "", null 
	 * elements in {@code theArray} will be ignored
	 * @param cols		Columns from {@code theArray} to use, in order to use them
	 * @param delimiter	{@link String} to insert between each element, null or empty to insert nothing
	 * @return	A String, never null, possibly empty
	 */
	public static final <T> String arrayToString (T[] theArray, int[] cols, String delimiter)
	{
		if ((theArray == null) || (cols == null))
			return "";
		
		StringBuilder	builder = new StringBuilder (theArray.length * 20);
		boolean			first = true;
		int				arrayLen = theArray.length;
		
		if ((delimiter != null) && delimiter.isEmpty ())
			delimiter = null;
		
		for (int which : cols)
		{
			if (first)
				first = false;
			else if (delimiter != null)
				builder.append (delimiter);
			
			if (which < 0)
				which += arrayLen;
			
			T	item = ((which < 0) || (which >= arrayLen)) ? null : theArray[which];
			
			if (item != null)
				builder.append (item);
		}
		
		return builder.toString ();
	}
	
	
    /**
     * Returns a string representation of the contents of the specified array.
     * The string representation consists of a list of the array's elements,
     * enclosed in square brackets (<tt>"[]"</tt>).  Adjacent elements
     * are separated by the characters <tt>", "</tt> (a comma followed
     * by a space).  Elements are converted to strings as by {@link String#valueOf (int)}.  
     * Returns <tt>"null"</tt> if <tt>a</tt> is <tt>null</tt>.
     *
     * @param theArray	The array whose string representation to return
     * @return	A string representation of {@code theArray}
     */
	public static final String arrayToString (byte[] theArray)
	{
		return arrayToString (theArray, false);
	}
	
	
    /**
     * Returns a {@link String} representation of the contents of the specified array, as individual bytes.<br/>
     * The string representation consists of a list of the array's elements, enclosed in square 
     * brackets (<tt>"[]"</tt>).  Adjacent elements are separated by the characters <tt>", "</tt> 
     * (a comma followed by a space).  If {@code asHex} is false, elements are converted to strings 
     * as by {@link String#valueOf (int)}, if {@code asHex} is true the strings will be of the form 
     * (0x??).  Returns <tt>"null"</tt> if <tt>theArray</tt> is <tt>null</tt>.
     *
     * @param theArray	The array whose string representation to return
     * @param asHex		If true, concatenate values as hex, if false, as integers
     * @return	A string representation of {@code theArray}
     */
	public static final String arrayToString (byte[] theArray, boolean asHex)
	{
		if (theArray == null)
			return "null";
		
		int max = theArray.length;
		if (max == 0)
			return "[]";
		
		StringBuilder results = new StringBuilder ();
		results.append ('[');
		
		for (int i = 0; i < max; ++i)
		{
			if (i > 0)
				results.append (", ");
			
			if (asHex)
				appendByteHexString (theArray[i], results);
			else
				results.append (theArray[i]);
		}
		
		results.append (']');
		return results.toString ();
	}
	
	
	/**
	 * Append a byte hexadecimal string (0x??) to a {@link StringBuilder}
	 * 
	 * @param value		byte to use
	 * @param target	{@link StringBuilder} to append to, must be valid and not null
	 * @throws NullPointerException	If {@code target} is null
	 */
	public static final void appendByteHexString (byte value, StringBuilder target)
	{
		target.append ("0x");
		
		String	hex = Integer.toHexString (value);
		int		len = hex.length ();
		
		if (len > 2)
			hex = hex.substring (len - 2, len);
		else if (len < 2)
			target.append ('0');	// Guaranteed that hex will always at least be length 1
		
		target.append (hex);
	}
	
	
    /**
     * Returns a {@link String} representation of the contents of the specified array.<br/>
     * The string representation consists two hexadecimal characters for each byte.<br/>
     * Returns {@code "null"} if {@code bytes} is {@code null}
     *
     * @param bytes	The array whose string representation to return
     * @return	A string representation of {@code bytes}
     */
	public static String bytesToHex (byte[] bytes)
	{
		if (bytes == null)
			return "null";
		
		int				len = bytes.length;
		StringBuilder	result = new StringBuilder (len * 2);
		
		for (int j = 0; j < len; ++j)
		{
			int	value = bytes[j] & 0xFF;
			
			result.append (kHexArray[value >>> 4]);
			result.append (kHexArray[value & 0x0F]);
		}
		
		return result.toString ();
	}
	
	
	/**
	 * Utility routine for the common Boolean test
	 * 
	 * @param tested		String to test
	 * @param nullValue	Value to return true if {@code tested} is null
	 * @return	True if string is not empty and is Boolean true
	 */
	public static final boolean parseBoolean (String tested, boolean nullValue)
	{
		if (tested == null)
			return nullValue;
		
		return !tested.isEmpty () && Boolean.parseBoolean (tested);
	}
	
	
	/**
	 * Utility routine for the common integer task
	 * 
	 * @param toParse		String to parse
	 * @param defaultVal	Return this value if {@code tested} is null or empty
	 * @return	Value of parsed string, {@code defaultVal} if {@code tested} is null or empty
	 */
	public static final int parseInt (String toParse, int defaultVal)
	{
		if (isEmpty (toParse))
			return defaultVal;
		
		return Integer.parseInt (toParse);
	}
	
	
	/**
	 * Take a string that might have quotes around it, and return it without the quotes
	 * 
	 * @param testString	String to de-quote
	 * @return	A String, empty if null was passed in
	 */
	public static final String deQuote (String testString)
	{
		if (isEmpty (testString))
			return "";
		
		for (String quote : kQuotes)
		{
			boolean	start = testString.startsWith (quote);
			boolean	end = testString.endsWith (quote);
			
			if (start || end)
			{
				int	qLen = quote.length ();
				
				if (!end)	// Then start must be true
					return testString.substring (qLen);
				
				int	tLen = testString.length ();
				
				if (!start)	// Then end must be true
					return testString.substring (0, tLen - qLen);
				
				return testString.substring (qLen, tLen - qLen);
			}
		}
		
		return testString;
	}
	
	
	/**
	 * Take a String, split it on {@code delimiter}, and add each of the resulting strings to a 
	 * Set of Strings
	 * 
	 * @param theStr	String to parse.  If null or empty will return an empty set
	 * @param delimiter	String to split on.  If null or empty will return an empty set
	 * @return	A Set, possibly empty, never null
	 */
	public static final Set<String> delimitedStringToSet (String theStr, String delimiter)
	{
		Set<String>	results = new HashSet<> ();
		
		if (!isEmpty (theStr) && !isEmpty (delimiter))
		{
			String[]	subStrs = SplitFile.mySplit (theStr, delimiter, kReturnAll);
			
			for (String aStr : subStrs)
				results.add (aStr);
		}
		
		return results;
	}
	
	
	/**
	 * Test to see if a string is a number
	 * 
	 * @param item	String to test
	 * @return	True if a number (integer or floating point), else false
	 */
	public static final boolean isNumber (String item)
	{
		return isNumber (item, null);
	}
	
	
	/**
	 * Test to see if a string is a number, or possibly multiple numbers divided by {@code allowedSplitter}
	 * 
	 * @param item				String to test
	 * @param allowedSplitter	If not null or empty, holds a string whose presence doesn't 
	 * invalidate "numeric" property
	 * @return	True if a number (integer or floating point), else false
	 */
	public static final boolean isNumber (String item, String allowedSplitter)
	{
		if (isEmpty (item))
			return false;
		
		char	first = item.charAt (0);
		
		if ((first != '-') && (first != '.') && !Character.isDigit (first))
			return false;
		
		boolean	doDouble = item.indexOf ('.') >= 0;
		try
		{
			// Don't need or even want the value, just want to know if it's a number
			if (doDouble)
				Double.parseDouble (item);
			else
				Integer.parseInt (item);
			
			return true;
		}
		catch (NumberFormatException oops)
		{
			if (isEmpty (allowedSplitter))
				return false;
		}
		
		String[]	values = SplitFile.mySplit (item, allowedSplitter, kReturnAll);
		
		if (values.length <= 1)	// If didn't split, can't be a success
			return false;
		
		try
		{
			// Don't need or even want the value, just want to know if it's a number
			for (String value : values)
			{
				if (doDouble)
					Double.parseDouble (value);
				else
					Integer.parseInt (value);
			}
			
			return true;
		}
		catch (NumberFormatException oops)
		{
			return false;
		}
	}
	
	
	/**
	 * Get a String from stdin, without displaying the characters typed
	 * 
	 * @return	Null if got nothing, or else the string entered
	 */
	public static final String getPasswordFromStdIn ()
	{
		return getPasswordFromStdIn ("Please enter your password: ");
	}
	
	
	/**
	 * Get a String from stdin, without displaying the characters typed
	 * 
	 * @param prompt	Prompt to show user when requesting the password
	 * @return	Null if got nothing, or else the string entered
	 */
	public static final String getPasswordFromStdIn (String prompt)
	{
		try
		{
			Console	console = System.console ();
			char[]	password = console.readPassword ("%s", prompt);
			
			if ((password == null) || (password.length == 0))
				return null;
			
			return new String (password);
		}
		catch (IOError oops)
		{
			// Ignore
		}
		
		return null;
	}
	
	
	/**	
	 * Replace all occurrences of {@code find} in {@code theStr} with {@code replace}
	 * 
	 * @param theStr	{@link String} to modify
	 * @param find		{@link String} holding regular expression to find, must not be null or empty
	 * @param replace	{@link String} to replace with, must not be null
	 * @return	Null if {@code theStr} is null, or if {@code find} does not appear in {@code theStr}, 
	 * else {@link String} holding results.
	 */
	public static final String findAndReplace (String theStr, String find, String replace)
	{
		if (isEmpty (theStr))
			return null;
		
		String[]	parts = theStr.split (find, -1);
		
		if (parts.length < 2)
			return null;
		
		StringBuilder	results = new StringBuilder (theStr.length ());
		boolean			first = true;
		
		for (String part : parts)
		{
			if (first)
				first = false;
			else
				results.append (replace);
			
			results.append (part);
		}
		
		return results.toString ();
	}
	
	
	/**
	 * Find all the occurrences of {@code findString} in {@code stringList} elements, and replace 
	 * each occurrences of {@code findString} with {@code replaceStr}
	 * 
	 * @param stringList	{@link List} of {@link String}s to possibly update
	 * @param findStr		{@link String} to look for in each {@code stringList} element
	 * @param replaceStr	{@link String} to replace {@code findString} with, if found
	 */
	public static final void cleanStrings (List<String> stringList, String findStr, String replaceStr)
	{
		int	numStrs;
		
		if ((stringList == null) || ((numStrs = stringList.size ()) == 0))
			return;
		
		int	findLen = findStr.length ();
		int	replaceLen = replaceStr.length ();
		
		for (int i = 0; i < numStrs; ++i)
		{
			String	testStr = stringList.get (i);
			int		pos = testStr.indexOf (findStr);
			
			if (pos < 0)
				continue;
			
			StringBuilder	workspace = new StringBuilder (testStr);
			
			do
			{
				workspace.replace (pos, pos + findLen, replaceStr);
				pos = workspace.indexOf (findStr, pos + replaceLen);
			}
			while (pos >= 0);
			
			stringList.set (i, workspace.toString ());
		}
	}
	
	
	/**
	 * Use {@link java.util.Base64}'s {@link Encoder} class to Base64 encode a {@link String}
	 * 
	 * @param decoded	{@link String} to encode, if null will treat as ""
	 * @return	The encoding of the string.  If null the encoding of an empty string
	 */
	public static final String encodeBase64 (String decoded)
	{
		if (decoded == null)
			decoded = "";
		
		String	encoding = Base64.getEncoder ().encodeToString (decoded.getBytes ());
		return encoding;
	}
	
	
	/**
	 * Use {@link java.util.Base64}'s {@link Encoder} class to Base64 decode a {@link String}
	 * 
	 * @param encoded	{@link String} to decode, if null will return ""
	 * @return	The decoding of the string.  If null return an empty string
	 */
	public static final String decodeBase64 (String encoded)
	{
		if (encoded == null)
			return "";
		
		String	decoding = Base64.getEncoder ().encodeToString (encoded.getBytes ());
		return decoding;
	}
	
	
	/**
	 * Create and return the compliment of a {@link String} of bases, using the standard 
	 * DNA bases, replacing any unknown characters with {@value #kMissingChar}
	 * 
	 * @param theBases	{@link String} to decode, if null will return ""
	 * @return	The reverse compliment of the string.  If {@code theBases} was null return an empty string
	 */
	public static final String getDNAComplement (String theBases)
	{
		if (gDNADictionary == null)
			gDNADictionary = makeDictionary (kDNACompliments);
		
		return getComplement (theBases, gDNADictionary, kMissingChar);
	}
	
	
	/**
	 * Create and return the compliment of a {@link String} of bases, using the standard 
	 * DNA bases, replacing any unknown characters with {@code missingChar}
	 * 
	 * @param theBases		{@link String} to decode, if null will return ""
	 * @param missingChar	Char to use if a base doesn't have an entry in {@code dictionary}
	 * @return	The reverse compliment of the string.  If {@code theBases} was null return an empty string
	 */
	public static final String getDNAComplement (String theBases, char missingChar)
	{
		if (gDNADictionary == null)
			gDNADictionary = makeDictionary (kDNACompliments);
		
		return getComplement (theBases, gDNADictionary, missingChar);
	}
	
	
	/**
	 * Create and return the reverse compliment of a {@link String} of bases, using the standard 
	 * DNA bases, replacing any unknown characters with {@value #kMissingChar}
	 * 
	 * @param theBases	{@link String} to decode, if null will return ""
	 * @return	The reverse compliment of the string.  If {@code theBases} was null return an empty string
	 */
	public static final String getDNAReverseComplement (String theBases)
	{
		if (gDNADictionary == null)
			gDNADictionary = makeDictionary (kDNACompliments);
		
		return getReverseComplement (theBases, gDNADictionary, kMissingChar);
	}
	
	
	/**
	 * Create and return the reverse compliment of a {@link String} of bases, using the standard 
	 * DNA bases, replacing any unknown characters with {@code missingChar}
	 * 
	 * @param theBases		{@link String} to decode, if null will return ""
	 * @param missingChar	Char to use if a base doesn't have an entry in {@code dictionary}
	 * @return	The reverse compliment of the string.  If {@code theBases} was null return an empty string
	 */
	public static final String getDNAReverseComplement (String theBases, char missingChar)
	{
		if (gDNADictionary == null)
			gDNADictionary = makeDictionary (kDNACompliments);
		
		return getReverseComplement (theBases, gDNADictionary, missingChar);
	}
	
	
	/**
	 * Determine if a char is a DNA character
	 * 
	 * @param theChar	char to test
	 * @return	True if it's in the DNA dictionary, false if not
	 */
	public static final boolean isDNAChar (char theChar)
	{
		if (gDNADictionary == null)
			gDNADictionary = makeDictionary (kDNACompliments);
		
		return gDNADictionary.get (Character.valueOf (theChar)) != null;
	}
	
	
	/**
	 * Determine if a char is a RNA character
	 * 
	 * @param theChar	char to test
	 * @return	True if it's in the RNA dictionary, false if not
	 */
	public static final boolean isRNAChar (char theChar)
	{
		if (gRNADictionary == null)
			gRNADictionary = makeDictionary (kRNACompliments);
		
		return gRNADictionary.get (Character.valueOf (theChar)) != null;
	}
	
	
	/**
	 * Create and return the compliment of a {@link String} of bases, using the standard 
	 * RNA bases, replacing any unknown characters with {@value #kMissingChar}
	 * 
	 * @param theBases	{@link String} to decode, if null will return ""
	 * @return	The reverse compliment of the string.  If {@code theBases} was null return an empty string
	 */
	public static final String getRNAComplement (String theBases)
	{
		if (gRNADictionary == null)
			gRNADictionary = makeDictionary (kRNACompliments);
		
		return getComplement (theBases, gRNADictionary, kMissingChar);
	}
	
	
	/**
	 * Create and return the compliment of a {@link String} of bases, using the standard 
	 * RNA bases, replacing any unknown characters with {@code missingChar}
	 * 
	 * @param theBases		{@link String} to decode, if null will return ""
	 * @param missingChar	Char to use if a base doesn't have an entry in {@code dictionary}
	 * @return	The reverse compliment of the string.  If {@code theBases} was null return an empty string
	 */
	public static final String getRNAComplement (String theBases, char missingChar)
	{
		if (gRNADictionary == null)
			gRNADictionary = makeDictionary (kRNACompliments);
		
		return getComplement (theBases, gRNADictionary, missingChar);
	}
	
	
	/**
	 * Create and return the reverse compliment of a {@link String} of bases, using the standard 
	 * RNA bases, replacing any unknown characters with {@value #kMissingChar}
	 * 
	 * @param theBases	{@link String} to decode, if null will return ""
	 * @return	The reverse compliment of the string.  If {@code theBases} was null return an empty string
	 */
	public static final String getRNAReverseComplement (String theBases)
	{
		if (gRNADictionary == null)
			gRNADictionary = makeDictionary (kRNACompliments);
		
		return getReverseComplement (theBases, gRNADictionary, kMissingChar);
	}
	
	
	/**
	 * Create and return the reverse compliment of a {@link String} of bases, using the standard 
	 * RNA bases, replacing any unknown characters with {@code missingChar}
	 * 
	 * @param theBases		{@link String} to decode, if null will return ""
	 * @param missingChar	Char to use if a base doesn't have an entry in {@code dictionary}
	 * @return	The reverse compliment of the string.  If {@code theBases} was null return an empty string
	 */
	public static final String getRNAReverseComplement (String theBases, char missingChar)
	{
		if (gRNADictionary == null)
			gRNADictionary = makeDictionary (kRNACompliments);
		
		return getReverseComplement (theBases, gRNADictionary, missingChar);
	}
	
	
	/**
	 * Create and return the reverse compliment of a {@link String} of bases, using the provided 
	 * {@code dictionary}, replacing any unknown characters with {@value #kMissingChar}
	 * 
	 * @param theBases		{@link String} to decode, if null will return ""
	 * @param dictionary	{@link Map} from each {@link Character} to it's compliment
	 * @return	The reverse compliment of the string.  If {@code theBases} was null return an empty string
	 */
	public static final String getComplement (String theBases, Map<Character, Character> dictionary)
	{
		return getComplement (theBases, dictionary, kMissingChar);
	}
	
	
	/**
	 * Create and return the reverse compliment of a {@link String} of bases, using the provided 
	 * {@code dictionary}, replacing any unknown characters with {@value #kMissingChar}
	 * 
	 * @param theBases		{@link String} to decode, if null will return ""
	 * @param dictionary	{@link Map} from each {@link Character} to it's compliment
	 * @return	The reverse compliment of the string.  If {@code theBases} was null return an empty string
	 */
	public static final String getReverseComplement (String theBases, Map<Character, Character> dictionary)
	{
		return getReverseComplement (theBases, dictionary, kMissingChar);
	}
	
	
	/**
	 * Create and return the compliment of a {@link String} of bases, using the provided 
	 * {@code dictionary}, replacing any unknown letter characters with {@code missingChar}, and 
	 * leaving all other characters unchanged
	 * 
	 * @param theBases		{@link String} to decode, if null will return ""
	 * @param dictionary	{@link Map} from each {@link Character} to it's compliment
	 * @param missingChar	Char to use if a base doesn't have an entry in {@code dictionary}
	 * @return	The reverse compliment of the string.  If {@code theBases} was null return an empty string
	 */
	public static final String getComplement (String theBases, Map<Character, Character> dictionary, char missingChar)
	{
		if (theBases == null)
			return "";
		
		int	len = theBases.length ();
		
		if (len == 0)
			return "";
		
		StringBuilder	result = new StringBuilder (len);
		
		for (int i = 0; i < len; ++i)
		{
			char		baseChar = theBases.charAt (i);
			Character	theChar = dictionary.get (Character.valueOf (baseChar));
			
			if (theChar == null)
			{
				if (Character.isLetter (baseChar))
					result.append (missingChar);
				else
					result.append (baseChar);
			}
			else
				result.append (theChar.charValue ());
		}
		
		return result.toString ();
	}
	
	
	/**
	 * Create and return the reverse compliment of a {@link String} of bases, using the provided 
	 * {@code dictionary}, replacing any unknown letter characters with {@code missingChar}, keeping 
	 * any whitespace in it's relative position, and deleting anything else
	 * 
	 * @param theBases		{@link String} to decode, if null will return ""
	 * @param dictionary	{@link Map} from each {@link Character} to it's compliment
	 * @param missingChar	Char to use if a base doesn't have an entry in {@code dictionary}
	 * @return	The reverse compliment of the string.  If {@code theBases} was null return an empty string
	 */
	public static final String getReverseComplement (String theBases, Map<Character, Character> dictionary, char missingChar)
	{
		if (theBases == null)
			return "";
		
		int	len = theBases.length ();
		
		if (len == 0)
			return "";
		
		StringBuilder	result = new StringBuilder (len);
		
		for (int i = len - 1; i >= 0; --i)
		{
			char		baseChar = theBases.charAt (i);
			Character	theChar = dictionary.get (Character.valueOf (baseChar));
			
			if (theChar == null)
			{
				if (Character.isWhitespace (baseChar))
					result.append (baseChar);
				else if (Character.isLetter (baseChar))
					result.append (missingChar);
			}
			else
				result.append (theChar.charValue ());
		}
		
		return result.toString ();
	}
	
	
	/**
	 * Given a {@link String}, return a String[] with 1 length 1 entry per character in the String.
	 * If {@code theString} is null or empty, will return an empty String[]
	 * 
	 * @param theString	{@link String} to process.  Can be null
	 * @return	{@link String}[], length 0 if null, otherwise length of {@code theString}
	 */
	public static final String[] stringToStringArray (String theString)
	{
		if (isEmpty (theString))
			return new String[0];
		
		int			len = theString.length ();
		String[]	results = new String[len];
		
		for (int i = 0; i < len; ++i)
			results[i] = theString.substring (i, i + 1);
		
		return results;
	}
	
	
	/**
	 * Report if a {@link String} is null or empty
	 * 
	 * @param test	String to test
	 * @return	True if null or empty, else false
	 */
	public static final boolean isEmpty (String test)
	{
		return (test == null) || test.isEmpty ();
	}
	
	
	/**
	 * Utility routine for the common {@link String}[] test
	 * 
	 * @param tested	String to test
	 * @return	True if string is null or empty
	 */
	public static final boolean isEmpty (String[] tested)
	{
		return (tested == null) || (tested.length == 0);
	}
	
	
	/**
	 * Test if a {@link String} is null, empty, or if {@link String#trim ()} is empty
	 * 
	 * @param testStr	{@link String} to test
	 * @return	True if {@code testStr} is null, empty, or if {@link String#trim ()} is empty
	 */
	public static final boolean isBlank (String testStr)
	{
		if ((testStr == null) || testStr.isEmpty ())
			return true;
		
		return testStr.trim ().isEmpty ();
	}
	
	
	/**
	 * Use the first value if it isn't null, the second if the first is null
	 * 
	 * @param baseValue		Preferred value
	 * @param defaultValue	Default value
	 * @param <T>	Type we're comparing
	 * @return	{@code baseValue} if it isn't null, else {@code defaultValue}
	 */
	public static final <T> T ifNull (T baseValue, T defaultValue)
	{
		if (baseValue == null)
			return defaultValue;
		
		return baseValue;
	}
	
	
	/**
	 * Add to {@code builder} the first {@code maxLen} elements of {@code collection}
	 * 
	 * @param collection	{@link Collection} of items to turn into a String
	 * @param builder		{@link StringBuilder} to add to, must be valid
	 * @param maxLen		Maximum number of items to include in the String
	 */
	public static final <T> void toString (Collection<T> collection, StringBuilder builder, int maxLen)
	{
		int				i = 0;
		
		builder.append ("[");
		for (Iterator<T> iterator = collection.iterator (); iterator.hasNext () && (i < maxLen); ++i)
		{
			if (i > 0)
				builder.append (", ");
			
			T	next = iterator.next ();
			
			builder.append (next);
		}
		
		builder.append ("]");
	}
	
	
	/**
	 * Get the size of the String that would be created by concatenating together the first 
	 * {@code maxLen} elements of {@code theCollection}
	 * 
	 * @param theCollection	{@link Collection} of items to get a size from
	 * @param maxLen		Maximum number of items to include in the size
	 * @return	The size needed
	 */
	public static final <T> int getCollectionSize (Collection<T> theCollection, int maxLen)
	{
		int	i = 0;
		int	size = 2;	// Beginning and ending brackets
		
		for (Iterator<T> iterator = theCollection.iterator (); iterator.hasNext () && (i < maxLen); ++i)
		{
			if (i > 0)
				size += kSeparatorLen;
			
			T	next = iterator.next ();
			
			next.toString ().length ();
		}
		
		return size;
	}
	
	
	/**
	 * Given a {@code char[][]} with sub arrays whose length must be at least 2, make a dictionary 
	 * from {@code char[0]} to {@code char[1]} of each sub array
	 * 
	 * @param compliments	{@code char[][]} with sub arrays whose length must be at least 2.<br>
	 * Must not be null
	 * @return	{@link Map} holding translations, possibly empty, never null
	 */
	private static final Map<Character, Character> makeDictionary (char[][] compliments)
	{
		Map<Character, Character>	results = new HashMap<> (compliments.length);
		
		for (char[] translation : compliments)
		{
			results.put (Character.valueOf (translation[0]), Character.valueOf (translation[1]));
		}
		
		return results;
	}
	
}
