
package CRC_Prediction.Utils;


import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.io.Closeables;
import java.io.*;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Pattern;


/**
 * This class parses a VCF file.
 *
 * @author Mark Woon
 */
public class vcfparser implements Closeable
{
	
	private static final Pattern	sf_tabSplitter			= Pattern.compile ("\t");
	private static final Pattern	sf_commaSplitter		= Pattern.compile (",");
//	private static final Pattern	sf_colonSplitter		= Pattern.compile (":");	// GTD Not used
	private static final Pattern	sf_semicolonSplitter	= Pattern.compile (";");
	
	private BufferedReader			m_reader;
	
	private int						m_lineNumber;
	private boolean					m_alreadyFinished;
	
	
	protected vcfparser (BufferedReader reader)
	{
		m_reader = reader;
	}
	
	
	/**
	 * Parses the entire VCF file (including the metadata).
	 *
	 * This is the preferred way to read a VCF file.
	 */
	public void parse () throws IOException
	{
		boolean hasNext = true;
		while (hasNext)
		{
			hasNext = parseNextLine ();
		}
		Closeables.closeQuietly (m_reader);
	}
	
	
	/**
	 * Parses just the next data line available, also reading all the metadata if it has not been
	 * read.
	 * This is a specialized method; in general calling {@link #parse()} to parse the entire stream
	 * is preferred.
	 *
	 * @return Whether another line may be available to read; false only if and only if this is the
	 *         last line available
	 * @throws IllegalStateException If the stream was already fully parsed
	 */
	public boolean parseNextLine () throws IOException
	{
		
		String line = m_reader.readLine ();
		if (line == null)
		{
			m_alreadyFinished = true;
			return false;
		}
		
		if (m_alreadyFinished)
		{
			// prevents user errors from causing infinite loops
			throw new IllegalStateException ("Already finished reading the stream");
		}
		
		m_lineNumber++;
		
		try
		{
			
			List<String> data = toList (sf_tabSplitter, line);
			
			// CHROM
			// String chromosome = data.get(0); // GTD Not used
			
			// POS
			@SuppressWarnings ("unused") // Has side-effect
			long position;
			try
			{
				position = Long.parseLong (data.get (1));
			}
			catch (NumberFormatException e)
			{
				throw new IllegalArgumentException ("Error parsing VCF data line #" + m_lineNumber + ": POS " + data.get (1) + " is not a number");
			}
			
			// ID
//			List<String> ids = null; // GTD Not used
			
//			ids = toList(sf_semicolonSplitter, data.get(2)); // GTD Not used
			
			// REF
//			String ref = data.get (3);	// GTD Not used
			
			// ALT
//			List<String> alt = null;	// GTD Not used
//			if (!data.get (7).isEmpty () && !data.get (4).equals ("."))
//			{
//				alt = toList (sf_commaSplitter, data.get (4));
//			}
			
			// QUAL
			@SuppressWarnings ("unused")
			BigDecimal quality = null;
			if (!data.get (5).isEmpty () && !data.get (5).equals ("."))
			{
				try
				{
					quality = new BigDecimal (data.get (5));
				}
				catch (NumberFormatException e)
				{
					throw new IllegalArgumentException ("Error parsing VCF data line #"
							+ m_lineNumber + ": QUAL " + data.get (5) + " is not a number");
				}
			}
			
			// FILTER
//			List<String> filters = null;	// GTD Not used
//			if (!data.get (6).equals ("PASS"))
//			{
//				filters = toList (sf_semicolonSplitter, data.get (6));
//			}
			
			// INFO
			ListMultimap<String, String> info = null;
			if (!data.get (7).equals ("") && !data.get (7).equals ("."))
			{
				info = ArrayListMultimap.create ();
				List<String> props = toList (sf_semicolonSplitter, data.get (7));
				for (String prop : props)
				{
					int idx = prop.indexOf ('=');
					if (idx == -1)
					{
						info.put (prop, "");
					}
					else
					{
						String key = prop.substring (0, idx);
						String value = prop.substring (idx + 1);
						info.putAll (key, toList (sf_commaSplitter, value));
					}
				}
			}
			
			// FORMAT
//			List<String> format = null;	// GTD Not used
//			if (data.size () >= 9 && data.get (8) != null)
//			{
//				format = toList (sf_colonSplitter, data.get (8));
//			}
			
			m_lineNumber++;
			
		}
		catch (IllegalArgumentException ex)
		{
			throw ex;
		}
		catch (RuntimeException e)
		{
			throw new IllegalArgumentException (
					"Error parsing VCF data line #" + m_lineNumber + ": " + line, e);
		}
		return true;
	}
	
	
	@Override
	public void close ()
	{
		Closeables.closeQuietly (m_reader);
	}
	
	
	private List<String> toList (Pattern pattern, String string)
	{
		String[] array = pattern.split (string);
		List<String> list = new ArrayList<> (array.length);
		Collections.addAll (list, array);
		return list;
	}
	
	
	public int getLineNumber ()
	{
		return m_lineNumber;
	}
	
	
	public static class Builder
	{
		private BufferedReader	m_reader;
		private Path			m_vcfFile;
		
		
		/**
		 * Provides the {@link Path} to the VCF file to parse.
		 */
		public Builder fromFile (Path dataFile)
		{
			Preconditions.checkNotNull (dataFile);
			if (m_reader != null)
			{
				throw new IllegalStateException ("Already loading from reader");
			}
			if (!dataFile.toString ().endsWith (".vcf"))
			{
				throw new IllegalArgumentException (
						"Not a VCF file (doesn't end with .vcf extension");
			}
			m_vcfFile = dataFile;
			return this;
		}
		
		
		/**
		 * Provides a {@link BufferedReader} to the beginning of the VCF file to parse.
		 */
		public Builder fromReader (BufferedReader reader)
		{
			Preconditions.checkNotNull (reader);
			if (m_vcfFile != null)
			{
				throw new IllegalStateException ("Already loading from file");
			}
			m_reader = reader;
			return this;
		}
		
		
		public vcfparser build () throws IOException
		{
			
			if (m_vcfFile != null)
			{
				m_reader = Files.newBufferedReader (m_vcfFile);
			}
			if (m_reader == null)
			{
				throw new IllegalStateException ("Must specify either file or reader to parse");
			}
			return new vcfparser (m_reader);
		}
	}
}
