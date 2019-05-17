package kriging;

import linktolinkBPR.LinkToLinks;

/**
 * This class will calculate the distance between two input of dimension nxt where n is the number of link2link and t is the time slice
 * @author Ashraf
 *
 */
public class Variogram {

	/**
	 * 
	 * @param a first tensor
	 * @param b second tensor
	 * @param l2ls Network info
	 * @param n link in calculation
	 * @param t time in calculation
	 * @param ka number of connected l2l to consider
	 * @param kt number of connected time to consider
	 * @param beta parameters for the variogram
	 * @return
	 */
	public double calcDistance(double[][] a,double[][]b,LinkToLinks l2ls,int n,int t,double[][]beta, int ka,int kt) {
		double[][] weight= new double[a.length][a[0].length];
		
		return 0;
	}
	
}
