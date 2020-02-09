package CRC_Prediction;

public class IRLAlgorithm {
	//irlopts
	public  String _algoName;
	public  LikelihoodFunction _likelihood;
	public  Prior _prior;
	//public static boolean _messages = False; 
	//public static maximumTime = 3600
	public  double _lowerRewardBounds = -1;
	public  double _upperRewardBounds = 1;
	public  int _algType = 0;
	
	public  int _numberOfTables=0;  //in non-Bayesian IRL applications the number of tables must be specified a priori
	public  int _maxNumberIterations=0;
	public  int _iterationsForTableAssignmentUpdate= 0;
	public  int _iterationsForRewardUpdate=0;
	public  int _iterationsForTransfer =0;
	public  int _numRestarts =1;
	public  double _alpha = 1.0;
	public 	double _discountHyperParameter = 0.0;

	
	public  double [] _rlist;  //reward list of reward values; used by normal-gammar prior
	public  double [] _rdist; //reward distribution; used by normal-gammar prior
	
	
	
	/**
	 * IRL algorithm constructor for EM algorithm
	 * @param name
	 * @param likelihoodFunct
	 * @param algtype
	 * @param numTables
	 * @param maxIterations
	 * @param numRestarts
	 */
	public IRLAlgorithm(String name, LikelihoodFunction likelihoodFunct, Prior prior, int algtype, int numTables, int maxIterations, int numRestarts){
		setAlgoName(name);
		setLikelihood(likelihoodFunct);
		setPrior(prior);
		setAlgoType(algtype);
		setNumTables(numTables);
		setMaxIterations(maxIterations);
		setNumRestarts(numRestarts);
	}
	
	/**
	 * IRL algorithm constructor for BIRL algorithm using Dirichlet Process Prior
	 * @param name
	 * @param likelihoodFunct
	 * @param algtype
	 * @param alpha
	 * @param maxIterations
	 * @param iterationsForTableAssignments
	 * @param iterationsForReward
	 * @param iterationsForTransfer
	 */
	public IRLAlgorithm(String name, LikelihoodFunction likelihoodFunct, Prior prior, int algtype, double alpha, double discount, int maxIterations, int iterationsForTableAssignments, int iterationsForReward, int iterationsForTransfer){
		setAlgoName(name);
		setLikelihood(likelihoodFunct);
		setPrior(prior);
		setAlgoType(algtype);
		setAlpha(alpha);
		setDiscountHyperparamter(discount);
		setMaxIterations(maxIterations);
		setTableAssignmentUpdateIterations(iterationsForTableAssignments);
		setRewardUpdateIterations(iterationsForReward);
		setTransferIterations(iterationsForTransfer);
		setInitialRewardDistribution(prior);

	}
	
	
	
	
	
	
	public String getAlgoName() {
		return _algoName;
	};
	public LikelihoodFunction  getLikelihood() {
		return _likelihood;
	};
	
	public Prior getPrior() {
		return _prior;
	}
	public int getAlgoType() {
		return _algType;
	};
	public int getNumTables() {
		return _numberOfTables ;
	};
	public int getNumRestarts() {
		return _numRestarts ;
	};
	public double getAlpha() {
		return _alpha;
	};
	public double getDiscountHyperparameter() {
		return _discountHyperParameter;
	};
	public int getMaxIterations() {
		return _maxNumberIterations;
	};
	public int getTableAssignmentUpdateIterations() {
		return _iterationsForTableAssignmentUpdate ;
	};
	public int getRewardUpdateIterations() {
		return _iterationsForRewardUpdate;
	};
	public int getTransferIterations() {
		return _iterationsForTransfer;
	};
	public double getRewardUpperBounds() {
		return _upperRewardBounds;
	}
	
	public double getRewardLowerBounds() {
		return _lowerRewardBounds;
	}
	public double [] getRewardArray() {
		return _rlist;
	}
	public double [] getRewardDistro() {
		return _rdist;
	}

	
	
	public void setAlgoName(String name) {
		_algoName= name;
	};
	public void  setLikelihood(LikelihoodFunction likelihoodFunct) {
		_likelihood = likelihoodFunct;
	};
	public void setPrior(Prior prior) {
		_prior = prior;
	}
	public void setAlgoType(int algtype) {
		_algType = algtype;
	};
	public void setNumTables(int numTables) {
		_numberOfTables = numTables;
	};
	public void setNumRestarts(int numRestarts) {
		_numRestarts = numRestarts;
	};
	public void setAlpha(double alpha) {
		_alpha = alpha;
	};
	public void setDiscountHyperparamter(double discount) {
		_discountHyperParameter = discount;
	};
	public void setMaxIterations(int maxIterations) {
		_maxNumberIterations = maxIterations;
	};
	public void setTableAssignmentUpdateIterations(int iterationsForTableAssignments) {
		_iterationsForTableAssignmentUpdate = iterationsForTableAssignments;
	};
	public void setRewardUpdateIterations(int iterationsForReward) {
		_iterationsForRewardUpdate = iterationsForReward;
	};
	public void setTransferIterations(int iterationsForTransfer) {
		_iterationsForTransfer = iterationsForTransfer;
	};
	public void setRewardUpperBounds(double upperRewardBounds) {
		_upperRewardBounds = upperRewardBounds;
	}
	
	public void setRewardLowerBounds(double lowerRewardBounds) {
		_lowerRewardBounds = lowerRewardBounds;
	}
	
	public void setRewardArray(double [] rArr) {
		_rlist = rArr;
	}
	public void setRewardDistro(double [] rArr) {
		_rdist = rArr;
	}
	
	public void setInitialRewardDistribution(Prior prior) {
		if(prior.get_identifier()==1) { //normal-gamma
			//if selected prior is normal-gamam, build distribution for 1-dim reward value
			//rlist = (opts.lb(1): opts.delta: opts.ub(1)); //creates a vector from lowerbound to upperbound where each element is incremented by delta
		    //rdist = 2*opts.gamma(2) + opts.beta.*rlist.^2./(1 + opts.beta);
		    //rdist = rdist.^(-opts.gamma(1)-0.5);
		    //rdist = rdist./sum(rdist);
		}
		if(prior.get_identifier()==2) { //beta-gamma
//			 	mlist = betarnd(opts.beta, 1 - opts.beta, [opts.nmu, 1]);
//			    vlist = gamrnd(opts.gamma(1), 1/opts.gamma(2), size(mlist));
//			    a = mlist.*vlist;
//			    b = (1 - mlist).*vlist;
//			    
//			    rlist = betarnd(a, b, size(a));
//			    rlist = rlist(rlist > opts.lb(1) & rlist < opts.ub(1));
//			    rdist = ones(size(rlist));
//			    rdist = rdist./sum(rdist);
//			    
//			    opts.beta1 = a;
//			    opts.beta2 = b;
		}
		//else if(prior.get_identifier()==3) { //Gaussian
			//not necessary?
		//}
		else {
			_rlist = null;		//new double [];  //to what dimension do we initialize size to set for our reward list and distributions??
			_rdist = null; //new double [];
		}
	}


}
