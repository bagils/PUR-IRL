/**
 * 
 */

package CRC_Prediction;


import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;


/**
 * @author John Kalantari
 *
 *         Implements the byte-based MDP factory to generate an MDP model initialized with specific
 *         parameters.
 */
public class MDPFactory
{
	
	String _outputDirectoryNameStr;
	
	
	public MDPFactory (String outputDirName)
	{
		_outputDirectoryNameStr = outputDirName;
	}
	
	
	/**
	 * 
	 * Deserialize and instantiate MDP from file (containing serialized MDP object)
	 */
	public MDP get (File serializedMDP)
	{
		try
		{
			System.out.println ("Trying to get() an MDP from file:" + serializedMDP);
			ObjectInputStream ois = null;
			MDP otox = null;
			
			// from the input URL create a new ObjectInputStram object
			// ..url argument should be in the form:-otomurl
			// file:///users/john/documents/workspace/Henri/serializedLearnedOtomaton.txt
			// ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(serializedMDP)));
			ois = new ObjectInputStream (new BufferedInputStream (new GZIPInputStream (new FileInputStream (serializedMDP))));
			
			// reads an object from ois object and casts it as an MDP class object
			// ....NOTE: this method SHOULD call MDP's own readObject() class method for proper deserialization!
			otox = (MDP) ois.readObject ();
			
			ois.close ();
			
			// return the MDP 'otox' to be used as the model with its already existing parameters
			return otox;
			
		}
		catch (ClassNotFoundException ex)
		{
			Logger.getLogger (MDPFactory.class.getName ()).log (Level.SEVERE, null, ex);
		}
		catch (IOException ex)
		{
			Logger.getLogger (MDPFactory.class.getName ()).log (Level.SEVERE, null, ex);
		}
		
		return null;
	}
	
	
	public void write (MDP mdpEnvironmentObj) throws Exception, FileNotFoundException, IOException, IllegalArgumentException
	{
		String timeStamp = new SimpleDateFormat ("yyyy.MM.dd.HH.mm").format (new Date ());
		
		String outputFileNameofSerializedMDP = _outputDirectoryNameStr + "/" + timeStamp + ".mdp.serialized";
		// ObjectOutputStream oos2 = new ObjectOutputStream(new BufferedOutputStream(new
		// FileOutputStream(outputFileNameofSerializedMDP)));
		// with GZIPOUtputstream
		ObjectOutputStream oos2 = 
				new ObjectOutputStream (new BufferedOutputStream (new GZIPOutputStream (new FileOutputStream (outputFileNameofSerializedMDP))));
		
		try
		{
			
			oos2.writeObject (mdpEnvironmentObj);
			System.out.println ("Stored the MDP object : " + outputFileNameofSerializedMDP);
			oos2.flush ();
			oos2.close ();
		}
		catch (IOException ioe)
		{
			System.err.println ("Error writing MDP object file -- probably corrupt -- try again");
			throw ioe;
		}
	}
	
}
