/* ###
 * IP: GHIDRA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package nl.nfi.sententia;

import java.io.IOException;

import java.net.URI;
import java.net.URISyntaxException;

import org.json.simple.parser.ParseException;

import ghidra.app.services.AbstractAnalyzer;
import ghidra.app.services.AnalysisPriority;
import ghidra.app.services.AnalyzerType;
import ghidra.app.util.importer.MessageLog;
import ghidra.framework.options.Options;
import ghidra.program.model.address.AddressRange;
import ghidra.program.model.address.AddressSetView;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Program;
import ghidra.util.Msg;
import ghidra.util.exception.CancelledException;
import ghidra.util.exception.InvalidInputException;
import ghidra.util.task.TaskMonitor;


/**
 * TODO: Provide class-level documentation that describes what this analyzer does.
 */
public class SententiaAnalyzer extends AbstractAnalyzer {
	private SententiaAPI sententiaAPI;

	public SententiaAnalyzer() {
		super("Sententia", "Recover function names with neural network based signatures.", AnalyzerType.FUNCTION_SIGNATURES_ANALYZER);
		setPriority(AnalysisPriority.FUNCTION_ID_ANALYSIS);
	}

	@Override
	public boolean getDefaultEnablement(Program program) {

		// TODO: Return true if analyzer should be enabled by default

		return true;
	}
	
	@Override
	public boolean canAnalyze(Program program) {
		// TODO: Check if architecture is compatible.
		return true;
	}

	@Override
	public void registerOptions(Options options, Program program) {

		options.registerOption("Server URL", SententiaAPI.SENTENTIA_DEFAULT_URL.toString(), null,
			"Sententia endpoint URL");
//		options.registerOption("Threshold", 0.8, null,
//				"Threshold above which a function name is considered a match");
	}
	
	@Override
	public void optionsChanged(Options options, Program program) {
		
		try {
			if (this.sententiaAPI == null) {
				this.sententiaAPI = new SententiaAPI(program, new URI(options.getString("Server URL", SententiaAPI.SENTENTIA_DEFAULT_URL)));
			}
			this.sententiaAPI.setServerURL(new URI(options.getString("Server URL", SententiaAPI.SENTENTIA_DEFAULT_URL)));
		} catch(URISyntaxException e) {
			Msg.showError(this, null, "Invalid endpoint URL", "The endpoint URL is invalid, please enter a valid endpoint URL.", e);
		}	
	}

	@Override
	public boolean added(Program program, AddressSetView set, TaskMonitor monitor, MessageLog log)
			throws CancelledException {
						
		FunctionManager functionManager = program.getFunctionManager();
		
		for (AddressRange functionRange: set) { 
			Function currentFunction = functionManager.getFunctionAt(functionRange.getMinAddress());
			if (currentFunction == null) {
				continue;
			}
			
			// Todo actually do something with the results
			try {
				this.sententiaAPI.getMatchingFunctions(new FunctionDescriptor(currentFunction), 25);
			} catch (IOException | ParseException | InvalidInputException | URISyntaxException | InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		return true;
	
	}
	
}
