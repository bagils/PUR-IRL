/**
 * Utilities
 *
 * <p>@author Gregory Dougherty</p>
 * Copyright Mayo Clinic, 2014
 *
 */
package CRC_Prediction.Utils;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Class with utility routines for parsing and formatting Dates
 *
 * <p>@author Gregory Dougherty</p>
 */
public class DateUtils
{
	private static final Map<String, DateFormat>	formatMap = new HashMap<> ();
	
	private static final String	kMonthYearFormat = "MMMM, yyyy";
	private static final String	kDateFormat = "yyyy-MM-dd";
	private static final String	kDateTimeFormat = "yyyy-MM-dd HH:mm:ss";
	private static final String	kDateTimeUnixFormat = "yyyy-MM-dd'T'HH:mm:ss";
	private static final String	kDateTimeMSFormat = "yyyy-MM-dd'T'HH:mm:ss.SSS";
	private static final String	kHumanDateFormat = "EEEE, MMMM d, yyyy";
	private static final String[]	kDateFormats = {"yyyy-MM-dd", "MM/dd/yy", "MM/dd/yyyy", "MM/dd", "MM-dd-yy", "MM-dd-yyyy", "MM-dd", 
	                             	                "MMMM d, yyyy", "MMM d, yyyy", "EEEE, MMMM d, yyyy", kDateTimeMSFormat, kDateTimeFormat};
	
	private static final Calendar	theCalendar = new GregorianCalendar ();
	private static final DateFormat	monthYearFormat = new SimpleDateFormat (kMonthYearFormat);
	private static final DateFormat	dateFormat = new SimpleDateFormat (kDateFormat);
	private static final DateFormat	dateTimeFormat = new SimpleDateFormat (kDateTimeFormat);
	private static final DateFormat	dateTimeUnixFormat = new SimpleDateFormat (kDateTimeUnixFormat);
	private static final DateFormat	dateTimeMSFormat = new SimpleDateFormat (kDateTimeMSFormat);
	private static final DateFormat	longDateFormat = new SimpleDateFormat (kHumanDateFormat);
	private static DateFormat[]	dateFormats;
	private static DateFormat[]	unixDateFormats;
	private static final long	kOneSecond = 1000L;
	private static final long	kTenSeconds = kOneSecond * 10L;
	private static final long	kOneMinute = kOneSecond * 60L;
	private static final long	kOneHour = kOneMinute * 60L;
	/** Number of milliseconds in one day */
	public static final long	kOneDay = kOneHour * 24L;
	private static final int	kMSFormat = 0;
	private static final int	kNoMSFormat = kMSFormat + 1;
	private static final int	kNumUnixDateFormats = kNoMSFormat + 1;
	
	static {
		int	numFormats = kDateFormats.length;
		
		dateFormats = new DateFormat[numFormats];
		for (int i = 0; i < numFormats; ++i)
			dateFormats[i] = new SimpleDateFormat (kDateFormats[i]);
		
		unixDateFormats = new DateFormat[kNumUnixDateFormats];
		unixDateFormats[kNoMSFormat] = dateTimeUnixFormat;
		unixDateFormats[kMSFormat] = dateTimeMSFormat;
	}
	
	
	/**
	 * Type of date to add to a date
	 */
    public enum DateType {
        /** Days */ kDay, /** Weeks */ kWeek, /** Months */ kMonth, /** Years */ kYear
    };
    
    
	/**
	 * Parse a {@link String}, returning the matching {@link Date}
	 * 
	 * @param dateString	{@link String} to parse
	 * @return	A {@link Date}, or null if there was a problem parsing {@code dateString}
	 */
	public static final Date parse (String dateString)
	{
		if (StringUtils.isEmpty (dateString))
			return null;
		
		for (DateFormat format : dateFormats)
		{
			try
			{
				return format.parse (dateString);
			}
			catch (ParseException oops)
			{
				// Ignore
			}
		}
		
		return null;
	}
	
    
	/**
	 * Parse a {@link String} holding a Unix date-time string, ignoring anything after the milliseconds, 
	 * returning the matching {@link Date}
	 * 
	 * @param dateTimeString	{@link String} to parse, must be of format "yyyy-MM-dd HH:mm:ss[.SSS[0-9]*]"
	 * @return	A {@link Date}, or null if there was a problem parsing {@code dateString}
	 */
	public static final Date parseDateTime (String dateTimeString)
	{
		if (StringUtils.isEmpty (dateTimeString))
			return null;
		
		int	pos = dateTimeString.lastIndexOf ('.');
		int	len = dateTimeString.length ();
		int	goodLen;
		
		if ((pos > 0) && ((goodLen = pos + 3) < len))
			dateTimeString = dateTimeString.substring (0, goodLen);	// Trim off anything after milliseconds
		for (DateFormat format : unixDateFormats)
		{
			try
			{
				return format.parse (dateTimeString);
			}
			catch (ParseException oops)
			{
				// Ignore
			}
		}
		
		return null;
	}
	
	
	/**
	 * Format a {@link Date} into a {@link String}, using the format "yyyy-MM-dd"
	 * 
	 * @param theDate	{@link Date} to format
	 * @return	A {@link String}, or null if {@code theDate} was null
	 */
	public static final String format (Date theDate)
	{
		if (theDate == null)
			return null;
		
		return dateFormat.format (theDate);
	}
	
	
	/**
	 * Format a {@link Date} into a {@link String}, using the format "yyyy-MM-dd HH:mm:ss"
	 * 
	 * @param theDate	{@link Date} to format
	 * @return	A {@link String}, or null if {@code theDate} was null
	 */
	public static final String formatWithTime (Date theDate)
	{
		if (theDate == null)
			return null;
		
		return dateTimeFormat.format (theDate);
	}
	
	
	/**
	 * Format a {@link Date} into a {@link String}, using the format "yyyy-MM-dd HH:mm:ss.SSS"
	 * 
	 * @param theDate	{@link Date} to format
	 * @return	A {@link String}, or null if {@code theDate} was null
	 */
	public static final String formatWithTimeMS (Date theDate)
	{
		if (theDate == null)
			return null;
		
		return dateTimeMSFormat.format (theDate);
	}
	
	
	/**
	 * Format a {@link Date} into a {@link String}, using the format "Month, yyyy"
	 * 
	 * @param theDate	{@link Date} to format
	 * @return	A {@link String}, or null if {@code theDate} was null
	 */
	public static final String formatMonth (Date theDate)
	{
		if (theDate == null)
			return null;
		
		return monthYearFormat.format (theDate);
	}
	
  
	/**
	 * Parse a {@link String}, returning the matching {@link Date}
	 * 
	 * @param dateString	{@link String} to parse
	 * @param formatString	{@link String} providing the format to use
	 * @return	A {@link Date}, or null if there was a problem parsing {@code dateString}
	 */
	public static final Date parse (String dateString, String formatString)
	{
		if (StringUtils.isEmpty (dateString) || StringUtils.isEmpty (formatString))
			return null;
		
		try
		{
			DateFormat	format = formatMap.get (formatString);
			
			if (format == null)
			{
				format = new SimpleDateFormat (formatString);
				formatMap.put (formatString, format);
			}
			
			return format.parse (dateString);
		}
		catch (ParseException oops)
		{
			return null;
		}
	}
	
	
	/**
	 * Format a {@link Date} into a {@link String}, using the format {@code formatString}
	 * 
	 * @param theDate		{@link Date} to format
	 * @param formatString	{@link String} providing the format to use
	 * @return	A {@link String}, or null if there was a problem formatting it
	 */
	public static final String format (Date theDate, String formatString)
	{
		if ((theDate == null) || StringUtils.isEmpty (formatString))
			return null;
		
		DateFormat	format = formatMap.get (formatString);
		
		if (format == null)
		{
			format = new SimpleDateFormat (formatString);
			formatMap.put (formatString, format);
		}
		
		return format.format (theDate);
	}
	
	
	/**
	 * Given two {@link Date}s, compute the time difference between them.<br/>
	 * If < 1 second, report milliseconds, i.e. 234 ms<br/>
	 * If < 10 seconds, report seconds + milliseconds, i.e. 1.234 seconds<br/>
	 * If < 1 minute, report seconds, i.e. 15 seconds<br/>
	 * If < 1 hour, report minutes:seconds, i.e. 15:32<br/>
	 * If < 1 day, report hours:minutes:seconds, i.e. 5:15:32<br/>
	 * Else report days, hours:minutes:seconds, i.e. 2 days, 5:15:32
	 * 
	 * @param startTime	Start time, if null will return empty string
	 * @param endTime	Finish time, if null will return empty string, if else than {@code startTime}, 
	 * will flip the two (IOW, will always report a positive result)
	 * @return	{@link String}, empty if nothing to compute, else the difference
	 */
	public static final String getTimeDifference (Date startTime, Date endTime)
	{
		if ((startTime == null) || (endTime == null))
			return "";
		
		long			start = startTime.getTime ();
		long			end = endTime.getTime ();
		long			elapsed = start - end;
		StringBuilder	builder = new StringBuilder (100);
		
		if (elapsed < 0)
			elapsed = -elapsed;
		
		if (elapsed < kOneSecond)
		{
			builder.append (elapsed);
			builder.append (" ms");
		}
		else if (elapsed < kTenSeconds)
		{
			builder.append (elapsed / kOneSecond);
			builder.append ('.');
			builder.append (elapsed % kOneSecond);
			builder.append (" seconds");
		}
		else if (elapsed < kOneMinute)
		{
			builder.append (elapsed / kOneSecond);
			builder.append (" seconds");
		}
		else if (elapsed < kOneHour)
		{
			long	minutes = elapsed / kOneMinute;
			long	seconds = (elapsed - (minutes * kOneMinute)) / kOneSecond;
			
			builder.append (minutes);
			builder.append (':');
			addPadedStr (seconds, builder);
		}
		else
		{
			long	days = elapsed / kOneDay;
			long	used = days * kOneDay;
			long	hours = (elapsed - used) / kOneHour;
			long	minutes = (elapsed - (used += (hours * kOneHour))) / kOneMinute;
			long	seconds = (elapsed - (used += (minutes * kOneMinute))) / kOneSecond;
			
			if (days > 0)
			{
				builder.append (days);
				builder.append (" days, ");
			}
			
			builder.append (hours);
			builder.append (':');
			addPadedStr (minutes, builder);
			builder.append (':');
			addPadedStr (seconds, builder);
		}
		
		return builder.toString ();
	}
	
	
	/**
	 * Add a number to {@link StringBuilder}, padding it with a 0 if < 10
	 * 
	 * @param value		Number to add to {@code builder}.  Should be < 60
	 * @param builder	{@link StringBuilder} to add to, must be valid
	 */
	private static final void addPadedStr (long value, StringBuilder builder)
	{
		if (value < 10)
			builder.append ('0');
		
		builder.append (value);
	}
	
	
	/**
	 * Given a date, return its string representation, in the form YYYY-MM-DD or Day of week, Month Day, Year
	 * 
	 * @param theDate		Date to turn into a String
	 * @param shortFormat	True if want YYYY-MM-DD, false if want Day of week, Month Day, Year
	 * @return	The Date in the specified format, or null if theDate is null
	 */
	public static final String getString (Date theDate, boolean shortFormat)
	{
		if (shortFormat)
			return dateFormat.format (theDate);
		
		return longDateFormat.format (theDate);
	}
	
	
	/**
	 * Given two {@link Date} objects, get the number of milliseconds that {@code firstDate} came 
	 * before {@code secondDate}
	 * 
	 * @param firstDate		Starting {@link Date}
	 * @param secondDate	Ending {@link Date}
	 * @return	number of milliseconds from {@code firstDate} to {@code secondDate}.<br/>
	 * If {@code firstDate} is after {@code secondDate}, result will be negative.<br/>
	 * If either is null, will replace with the current Date / time
	 */
	public static final long getElapsedTime (Date firstDate, Date secondDate)
	{
		if (firstDate == null)
			firstDate = new Date ();
		if (secondDate == null)
			secondDate = new Date ();
		
		return secondDate.getTime () - firstDate.getTime ();
	}
	
	
	/**
	 * Given a number of milliseconds, return a human readable string with significant days / hours / 
	 * minutes, and seconds and milliseconds.  Examples:<br/>
	 * 0.123<br/>
	 * 55.123<br/>
	 * 22:05.123<br/>
	 * 6:44:55.123<br/>
	 * 5 days 8:08:11:234
	 * 
	 * @param milliseconds	Number of milliseconds to turn into a time
	 * @return	The Date in the specified format, or null if theDate is null
	 */
	public static final String msToTimeString (long milliseconds)
	{
		long	millis = milliseconds % 1000;
		long	seconds = milliseconds / 1000;
		long	minutes = seconds / 60;
		long	hours = minutes / 60;
		long	days = hours / 24;
		boolean	started = false;
		
		StringBuilder	result = new StringBuilder ();
		
		if (days > 0)
		{
			result.append (days);
			result.append (" days ");
			started = true;
		}
		
		if (started || (hours > 0))
		{
			hours = hours % 24;
			result.append (hours);
			result.append (':');
			started = true;
		}
		
		if (started || (minutes > 0))
		{
			minutes = minutes % 60;
			if (started && (minutes < 10))
				result.append ('0');
			result.append (minutes);
			result.append (':');
			started = true;
		}
		
		seconds = seconds % 60;
		if (started && (seconds < 10))
			result.append ('0');
		result.append (seconds);
		result.append ('.');
		if (millis < 10)
			result.append ("00");
		else if (millis < 100)
			result.append ('0');
		result.append (millis);
		
		return result.toString ();
	}
	
	
	/**
	 * Given a @link Date} object, add the specified number and type of units to the Date
	 * 
	 * @param theDate	@link Date} to add to
	 * @param units		Type of date element to add (days / weeks / months / years)
	 * @param count		How many to add
	 * @return	A new date with the offset, or null if theDate was null
	 */
	public static final Date addToDate (Date theDate, DateType units, int count)
	{
		if (theDate == null)
			return null;
		
		if (count == 0)
			return theDate;
		
		int	calendarUnits;
		
		switch (units)
		{
			case kDay:
				calendarUnits = Calendar.DAY_OF_YEAR;
				break;
				
			case kWeek:
				calendarUnits = Calendar.WEEK_OF_YEAR;
				break;
				
			case kMonth:
				calendarUnits = Calendar.MONTH;
				break;
				
			case kYear:
				default:
				calendarUnits = Calendar.YEAR;
				break;
				
		}
		
		theCalendar.setTime (theDate);
		theCalendar.add (calendarUnits, count);
		
		return theCalendar.getTime ();
	}
	
	
	/**
	 * @return	Get the standard DateFormat object
	 */
	public static final DateFormat getDateFormat ()
	{
		return dateFormat;
	}
	
	
	/**
	 * Compare two dates with a resolution of seconds, not milliseconds <br/>
	 * Same as {@code baseDate.compareTo (otherDate)} if both {@code baseDate} and {@code otherDate} 
	 * are not null, and if they are not within one second of each other
	 * 
	 * @param baseDate	Base {@link Date}. If {@code baseDate < otherDate}, will return negative number
	 * @param otherDate	Comparing {@link Date}. If {@code baseDate < otherDate}, will return negative number
	 * @return	A negative integer, zero, or a positive integer as {@code baseDate} is less than, equal to, 
	 * or greater than {@code otherDate}
	 */
	public static final int compareDates (Date baseDate, Date otherDate)
	{
		if (baseDate == null)
		{
			if (otherDate == null)
				return 0;
			
			return 1;
		}
		else if (otherDate == null)
			return -1;
		
		long	baseTime = baseDate.getTime ();
		long	otherTime = otherDate.getTime ();
		
		if (baseTime > otherTime)
		{
			if (baseTime > (otherTime + kOneSecond))
				return 1;
		}
		else
		{
			if (baseTime < (otherTime - kOneSecond))
				return -1;
		}
		
		return 0;
	}
	
	
	/**
	 * Get the year.<br/>
	 * Gets the actual year.  Does <b>not</b> emulated {@link Date#getYear ()} and get year - 1900
	 * 
	 * @param theDate	{@link Date} to process.  If null will return -1
	 * @return	Integer year
	 */
	public static final int getYear (Date theDate)
	{
		if (theDate == null)
			return -1;
		
		theCalendar.setTime (theDate);
		
		return theCalendar.get (Calendar.YEAR);
	}
	
	
	/**
	 * Get the month, as a number from 1 to 12
	 * 
	 * @param theDate	{@link Date} to process.  If null will return 0
	 * @return	Integer month, 1 - 12 if {@code theDate} is not null, 0 if it is null
	 */
	public static final int getMonth (Date theDate)
	{
		if (theDate == null)
			return 0;
		
		theCalendar.setTime (theDate);
		
		return theCalendar.get (Calendar.MONTH) + 1;
	}
	
	
	/**
	 * Get the day, as a number from 1 to 31
	 * 
	 * @param theDate	{@link Date} to process.  If null will return 0
	 * @return	Integer day, 1 - 31 if {@code theDate} is not null, 0 if it is null
	 */
	public static final int getDay (Date theDate)
	{
		if (theDate == null)
			return 0;
		
		theCalendar.setTime (theDate);
		
		return theCalendar.get (Calendar.DAY_OF_MONTH);
	}
	
	
	/**
	 * Given a {@code year}, {@code month}, and {@code day}, make a {@link Date} for 12:00 AM of 
	 * that day
	 * 
	 * @param year	Year of interest, will not add 1900 to it.  Reciprocal of {@link #getYear (Date)}
	 * @param month	Month of interest.  Reciprocal of {@link #getMonth (Date)}, i.e. a value from 1 - 12
	 * @param day	Day of interest.  Reciprocal of {@link #getDay (Date)}
	 * @return	a {@link Date}
	 */
	public static final Date makeDate (int year, int month, int day)
	{
		theCalendar.clear ();
		theCalendar.set (year, month - 1, day);
		
		return theCalendar.getTime ();
	}
	
	
	/**
	 * Given a {@link Date}, make a {@link Date} for 12:00 AM of that day
	 * 
	 * @param theDate	{@link Date} of interest.  If null with use current date
	 * @return	a {@link Date}
	 */
	public static final Date getTheDate (Date theDate)
	{
		if (theDate == null)
			theDate = new Date ();
		
		theCalendar.setTime (theDate);
		
		int	day = theCalendar.get (Calendar.DAY_OF_MONTH);
		int	month = theCalendar.get (Calendar.MONTH);
		int	year = theCalendar.get (Calendar.YEAR);
		
		theCalendar.clear ();
		theCalendar.set (year, month, day);
		
		return theCalendar.getTime ();
	}
	
	
	/**
	 * Given a {@link Date}, make a {@link Date} for the hour of that day
	 * 
	 * @param theDate	{@link Date} of interest.  If null with use current date
	 * @return	a {@link Date}
	 */
	public static final Date getTheDateAndHour (Date theDate)
	{
		if (theDate == null)
			theDate = new Date ();
		
		theCalendar.setTime (theDate);
		
		boolean	am = theCalendar.get (Calendar.AM_PM) == Calendar.AM;
		int		hour = theCalendar.get (Calendar.HOUR) + (am ? 0 : 12);
		int		day = theCalendar.get (Calendar.DAY_OF_MONTH);
		int		month = theCalendar.get (Calendar.MONTH);
		int		year = theCalendar.get (Calendar.YEAR);
		
		theCalendar.clear ();
		theCalendar.set (year, month, day);
		theCalendar.add (Calendar.HOUR, hour);
		
		return theCalendar.getTime ();
	}
	
}
