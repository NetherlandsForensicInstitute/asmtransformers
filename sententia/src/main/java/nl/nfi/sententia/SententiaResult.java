package nl.nfi.sententia;

public class SententiaResult {
		private String functionName;
		private Double functionScore;
		
		public SententiaResult(String name, Double score) {
			this.functionScore = score;
			this.functionName = name;
		}
		
		public String getFunctionName() {
			return functionName;
		}
		
		public Double getFunctionScore() {
			return functionScore;
		}

}
