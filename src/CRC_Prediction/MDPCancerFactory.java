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
 * Copyright 2019, Mayo Foundation for Medical Education and Research
 * 
 * Implements the byte-based MDPCancer factory to generate an MDPCancer model initialzed with specific parameters.
*/
public class MDPCancerFactory  {

	String _outputDirectoryNameStr;
	
	public MDPCancerFactory(String outputDirName) {
		_outputDirectoryNameStr = outputDirName;
	}

    /**
     * 
     * Deserialize and instantiate MDPCancer from file (containing serialized MDPCancer object)
     */
    public MDPCancer get(File serializedMDPCancer) {
            try {
            	System.out.println("Trying to get() an MDPCancer from file:"+ serializedMDPCancer);
                ObjectInputStream ois = null;
                MDPCancer otox = null;
                
                //from the input URL create a new ObjectInputStram object
                //..url argument should be in the form:-otomurl file:///users/john/documents/workspace/Henri/serializedLearnedOtomaton.txt
                //ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(serializedMDPCancer)));
                ois = new ObjectInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(serializedMDPCancer))));

                
                //reads an object from ois object and casts it as an MDPCancer class object
                //....NOTE: this method SHOULD call MDPCancer's own readObject() class method for proper deserialization!
                otox = (MDPCancer) ois.readObject();
                
                ois.close();

 
                //return the MDPCancer 'otox' to be used as the model with its already existing parameters
                return otox;
                
            } catch (ClassNotFoundException ex) {
                Logger.getLogger(MDPCancerFactory.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(MDPCancerFactory.class.getName()).log(Level.SEVERE, null, ex);
            }
        
        return null;
    }
    
    public void write(MDPCancer mdpCancerEnvironmentObj) throws Exception, FileNotFoundException, IOException, IllegalArgumentException { 
		String timeStamp = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss.SSS").format(new Date());


        String outputFileNameofSerializedMDP = _outputDirectoryNameStr+"/"+timeStamp+".mdpcancer.serialized";
       // ObjectOutputStream oos2 = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(outputFileNameofSerializedMDP)));
        //with GZIPOUtputstream
        ObjectOutputStream oos2 = new ObjectOutputStream(new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(outputFileNameofSerializedMDP))));

        try{

        	oos2.writeObject(mdpCancerEnvironmentObj);
            System.out.println("Stored the MDPCancer object : "+outputFileNameofSerializedMDP);
            oos2.flush();
            oos2.close();
        }
        catch (IOException ioe) {
            System.err.println("Error writing MDPCancer object file -- probably corrupt -- try again");
            throw ioe;
        }
    }
    

    
    
}