package nl.nfi.sententia;

import java.util.List;

import org.json.simple.JSONObject;

import ghidra.program.model.listing.Function;
import ghidra.util.exception.CancelledException;
import ghidra.util.exception.InvalidInputException;



public class FunctionDescriptor {
	public String name;
	public String binaryName;
	List<List<Object>> functionCFG;
	public String binarySHA256;
	
	public FunctionDescriptor(Function function) throws InvalidInputException, CancelledException {
		this(function.getName(), SententiaUtil.getFunctionCFG(function), function.getProgram().getName(), function.getProgram().getExecutableSHA256());
	}
	
	
	public FunctionDescriptor(String functionName, List<List<Object>> functionCFG, String binaryName, String binarySHA256) {
		this.name = functionName;
		this.functionCFG = functionCFG;
		this.binaryName = binaryName;
		this.binarySHA256 = binarySHA256;
	}
	
	public JSONObject toJson() {
		JSONObject jsonFunctionDescriptor = new JSONObject();
		jsonFunctionDescriptor.put("name", this.name);
		
		jsonFunctionDescriptor.put("cfg", this.functionCFG);
		jsonFunctionDescriptor.put("binary_name", this.binaryName);
		jsonFunctionDescriptor.put("binary_sha256", this.binarySHA256);
		return jsonFunctionDescriptor;
	}

}
