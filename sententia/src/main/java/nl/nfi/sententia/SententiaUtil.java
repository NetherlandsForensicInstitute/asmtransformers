package nl.nfi.sententia;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import ghidra.program.model.address.AddressRange;
import ghidra.program.model.address.AddressRangeIterator;
import ghidra.program.model.address.AddressSet;
import ghidra.program.model.block.CodeBlock;
import ghidra.program.model.block.SimpleBlockModel;
import ghidra.program.model.listing.CodeUnit;
import ghidra.program.model.listing.Function;
import ghidra.util.exception.CancelledException;
import ghidra.util.exception.InvalidInputException;

public class SententiaUtil {
	public static List<List<Object>> getFunctionCFG(Function function) 
			throws InvalidInputException, CancelledException {
		
		List<List<Object>> functionCFG = new ArrayList<List<Object>>();
		
		SimpleBlockModel simpleBlockModel = new SimpleBlockModel(function.getProgram());
		
		for (CodeBlock block: simpleBlockModel.getCodeBlocksContaining(function.getBody(), null)) {
			List<String> disassembly = new ArrayList<>();
			
			AddressRangeIterator rangeIter = block.getAddressRanges();
			AddressRange range = rangeIter.next();
			AddressSet addressSet =  new AddressSet(range);
			
			for (CodeUnit codeUnit: function.getProgram().getListing().getCodeUnits(addressSet, true)) {
				disassembly.add(codeUnit.toString());
			}
			
			if (rangeIter.hasNext()) {
				// A basic block should never have multiple address ranges, as it's supposed to be a contiguos block of instructions 
				throw new InvalidInputException(String.format("Basic block in function %s at 0x%x has multiple address ranges, while only one is to be expected!", function.getName(), block.getFirstStartAddress().getOffset()));
			}
			
			functionCFG.add(Arrays.asList(block.getFirstStartAddress().getOffsetAsBigInteger(), disassembly));
		}
		
		return functionCFG;
	}
}
