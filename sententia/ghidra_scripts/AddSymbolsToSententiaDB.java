//Scan the currently active program for annotated functions and add them to the remote Sententia DB.
//@author Jeffrey Rongen (NFI)
//@category Analyzers
//@keybinding
//@menupath
//@toolbar

import java.net.URI;
import ghidra.app.script.GhidraScript;
import ghidra.framework.options.ToolOptions;
import ghidra.framework.plugintool.PluginTool;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.FunctionManager;
import nl.nfi.sententia.SententiaAPI;
import nl.nfi.sententia.FunctionDescriptor;

public class AddSymbolsToSententiaDB extends GhidraScript {
	
	SententiaAPI api;
	
	@Override
	protected void run() throws Exception {
		PluginTool tool = state.getTool();
		ToolOptions options = tool.getOptions("SententiaPlugin");
		
		URI endpoint = new URI(options.getString("Endpoint", null));
		api = new SententiaAPI(currentProgram, endpoint);
		
		FunctionManager functionManager = currentProgram.getFunctionManager();
				
		FunctionIterator functions = functionManager.getFunctions(false);
		monitor.initialize(functionManager.getFunctionCount());
				
		for (Function function : functions) {
		
			monitor.incrementProgress();
			
			if (function == null) {
				continue;
			}
						
			FunctionDescriptor functionDescriptor = new FunctionDescriptor(function);
			api.addSignatureToDB(functionDescriptor);
			
		}
	}
}
