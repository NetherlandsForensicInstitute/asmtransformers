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

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Iterator;
import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;

import org.json.simple.parser.ParseException;

import docking.ActionContext;
import docking.ComponentProvider;
import docking.action.DockingAction;
import docking.action.ToolBarData;
import ghidra.app.CorePluginPackage;
import ghidra.app.plugin.PluginCategoryNames;
import ghidra.app.plugin.ProgramPlugin;
import ghidra.app.services.ConsoleService;
import ghidra.framework.model.DomainObjectChangeRecord;
import ghidra.program.util.ProgramChangeRecord;
import ghidra.framework.model.DomainObjectChangedEvent;
import ghidra.framework.model.DomainObjectListener;
import ghidra.framework.options.OptionsChangeListener;
import ghidra.framework.options.ToolOptions;
import ghidra.framework.plugintool.*;
import ghidra.framework.plugintool.util.PluginStatus;
import ghidra.program.database.symbol.FunctionSymbol;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Program;
import ghidra.program.model.symbol.SourceType;
import ghidra.program.util.ProgramEvent;
import ghidra.program.util.ProgramLocation;
import ghidra.util.HelpLocation;
import ghidra.util.Msg;
import ghidra.util.exception.CancelledException;
import ghidra.util.exception.InvalidInputException;
import resources.Icons;

/**
 * TODO: Provide class-level documentation that describes what this plugin does.
 */
//@formatter:off
@PluginInfo(
	status = PluginStatus.UNSTABLE,
	packageName = CorePluginPackage.NAME,
	category = PluginCategoryNames.ANALYSIS,
	shortDescription = "Suggest function names from a database of deep learning based function signatures, based on similarity.",
	description = "Sententia is a frontend for the ASM Transformers based model and server. It allows the user to generate and store neural network based function signatures in a database, which are coupled to the user supplied function name. The accompanyin server then allows the user to add and recall these combinations in a user-friendly interface."
)
//@formatter:on
public class SententiaPlugin extends ProgramPlugin implements DomainObjectListener {

	SententiaSimilarFunctions displayComponent;
	private Function currentFunction;
	private SententiaAPI sententiaAPI;
	private OptionsChangeListener optionsChangeListener;
	private ConsoleService consoleService;
	
	/**
	 * Plugin constructor.
	 * 
	 * @param tool The plugin tool that this plugin is added to.
	 * @throws URISyntaxException 
	 */
	public SententiaPlugin(PluginTool tool) throws URISyntaxException {
		super(tool);

		// TODO: Customize provider (or remove if a provider is not desired)
		String pluginName = getName();
		displayComponent = new SententiaSimilarFunctions(this, pluginName);

		// TODO: Customize help (or remove if help is not desired)
		String topicName = this.getClass().getPackage().getName();
		String anchorName = "HelpAnchor";
		displayComponent.setHelpLocation(new HelpLocation(topicName, anchorName));
		
		ToolOptions options = tool.getOptions(pluginName);
		options.registerOption("Endpoint", SententiaAPI.SENTENTIA_DEFAULT_URL, null, "The URL for the Sententia server endpoint");
		
		try {
			sententiaAPI = new SententiaAPI(this.currentProgram, tool);
		} catch (URISyntaxException e) {
			// TODO Actually handle this.
			sententiaAPI = new SententiaAPI(this.currentProgram, new URI(SententiaAPI.SENTENTIA_DEFAULT_URL));
		}
		
		optionsChangeListener = new OptionsChangeListener() {
			  @Override
			  public void optionsChanged(ToolOptions changedOptions, String optionName, Object oldValue, Object newValue) {
	            if (optionName.equals("Endpoint")) {
	            	if (sententiaAPI != null ) {
	            		try {
							sententiaAPI.setServerURL(new URI(newValue.toString()));
						} catch (URISyntaxException e) {
							Msg.showError(this, null, "Invalid endpoint URL", "The endpoint URL is invalid, please enter a valid endpoint URL.", e);
						}
	            	}
	            }
	        }
					
		};
		
		options.addOptionsChangeListener(optionsChangeListener);
	}
	
	@Override
	public void init() {
		super.init();
		// TODO: Acquire services if necessary
	}
	
	/**
	 * Overridden in order to add ourselves as a {@link DomainObjectListener}
	 * to the current program.
	 * 
	 * @param program The activated program.
	 */
	@Override
	protected void programActivated(Program program) {
		if (program == null) {
			return;
		}
		
		this.consoleService = tool.getService(ConsoleService.class);
		
		currentProgram = program;
		program.addListener(this);
	}
	
	//TODO we probably need dispose() for this: https://ghidra.re/ghidra_docs/api/ghidra/framework/plugintool/Plugin.html#plugin-life-cycle-heading
	@Override
	protected void programClosed (Program program) {
		if (program == null) {
			return;
		}
		program.removeListener(this);
	}
	
	public Function getCurrentFunction() {
		return currentFunction;
	}
	
	@Override
	public void domainObjectChanged(DomainObjectChangedEvent event) {
		System.out.println(event);
		Iterator<DomainObjectChangeRecord> it = event.iterator();
		while (it.hasNext()) {
			DomainObjectChangeRecord evt = it.next();
			// Check if we're dealing with a rename and the is applied to a function
			if (evt.getEventType().getId() == ProgramEvent.SYMBOL_RENAMED.getId() && ((ProgramChangeRecord)evt).getObject() instanceof FunctionSymbol) {				
				
				Function changedFunction = currentProgram.getFunctionManager().getFunctionAt(((ProgramChangeRecord)evt).getStart());
				
				try {
					sententiaAPI.addSignatureToDB(new FunctionDescriptor(changedFunction));
				} catch (InvalidInputException | CancelledException | IOException | URISyntaxException e) {
					String errMsg = "Failed to add function to database. Is the server running? If so, check the endpoint URL!";
					logError(errMsg, e);
					return;
				}	
				
			}
		}
			
	}	

	@Override
	protected void locationChanged(ProgramLocation loc) {
		if (currentProgram == null || loc == null) {
			return;
		}

		
		currentFunction = currentProgram.getFunctionManager().getFunctionContaining(loc.getAddress());
		
		if (currentFunction == null) {
			return;
		}
		
		try {
			FunctionDescriptor functionDescriptor = new FunctionDescriptor(currentFunction);
			ArrayList<SententiaResult> results = sententiaAPI.getMatchingFunctions(functionDescriptor, 25);
			displayComponent.updateSimilarFunctions(results);
		} catch (InvalidInputException | CancelledException | IOException | ParseException | URISyntaxException e) {
			String errMsg = "Failed to get similar functions. Is the server running? If so, check the endpoint URL!";
			logError(errMsg, e);
			return;
		}	

	}
	
	protected SententiaAPI getSententiaAPI() {
		return this.sententiaAPI;
	}
	
	protected void logError(String errMsg, Exception e) {
		Msg.error(this, errMsg, e);
		if (this.consoleService != null) {
			this.consoleService.addErrorMessage(this.name, e.getMessage() + ": " + errMsg);
		}
	}

	private class SententiaSimilarFunctions extends ComponentProvider {

		private JPanel panel;
		private JTable similarFunctionsTable;
		private DockingAction action;
		public Plugin plugin;

		public SententiaSimilarFunctions(Plugin plugin, String owner) {
			super(plugin.getTool(), "Sententia Similar Functions", plugin.getName());
			this.plugin = plugin;
			buildPanel();
			createActions();
		}

		// Customize GUI
		private void buildPanel() {
			panel = new JPanel(new BorderLayout());
			
//			DefaultTableModel model = new DefaultTableModel();
			
			similarFunctionsTable = new JTable(new DefaultTableModel(new Object[]{"Function name", "Score"}, 25));
			
			// Set row color based on similarity score
			similarFunctionsTable.setDefaultRenderer(Object.class, new DefaultTableCellRenderer(){

				@Override
			    public Component getTableCellRendererComponent(JTable table,
			            Object value, boolean isSelected, boolean hasFocus, int row, int col) {

			        super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, col);

			        Double score = (double) 0;
			        try {
			        	score = (double) table.getModel().getValueAt(row, 1);
			        } catch (NullPointerException | ClassCastException e) {
						return this;
					}
			        if (score > 0.0) {
			            setBackground(Color.getHSBColor((float)(128*score)/360, 1.0f, 0.6f));
			            setForeground(Color.WHITE);
			        } else {
			            setBackground(table.getBackground());
			            setForeground(table.getForeground());
			        }       
			        return this;
			    }
			});
			
			similarFunctionsTable.addMouseListener(new MouseAdapter() {
				public void mouseClicked(MouseEvent me) {
					if (me.getClickCount() == 2) {     
		               JTable target = (JTable)me.getSource();
		               int row = target.getSelectedRow();
		             
		               String newName = similarFunctionsTable.getValueAt(row, 0).toString();
		               @SuppressWarnings("static-access")
					   Integer transactionId = currentProgram.startTransaction("Set name of function from %s to %s".format(currentFunction.getName(), newName));
		               try {
						currentFunction.setName(newName, SourceType.ANALYSIS);
						} catch (Exception e) {
							currentProgram.endTransaction(transactionId, false);
							e.printStackTrace();
						}
		                currentProgram.endTransaction(transactionId, true);
					}
				}
			});
			
			similarFunctionsTable.setDefaultEditor(Object.class, null);
			panel.add(new JScrollPane(similarFunctionsTable));
			setVisible(true);
		}
		
		public SententiaAPI getApi() {
			return ((SententiaPlugin)this.plugin).getSententiaAPI();
		}

		// TODO: Customize actions
		private void createActions() {
			action = new DockingAction("Sententia similar function names", getName()) {
				@Override
				public void actionPerformed(ActionContext context) {
					// Make this run in the background and at activation of a program
					Msg.showInfo(getClass(), panel, "Add functions to database", "Adding functions to DB!");
					FunctionManager functionManager = currentProgram.getFunctionManager();
					FunctionIterator functions = functionManager.getFunctions(false);
					
					SententiaAPI api = ((SententiaSimilarFunctions)context.getComponentProvider()).getApi();
					while (functions.hasNext()) {
						Function function = functions.next();
						String functionName = function.getName();
						if ((!functionName.toLowerCase().startsWith("fun_")) && (!functionName.toLowerCase().startsWith("thunk"))) {
							try {
								api.addSignatureToDB(new FunctionDescriptor(function));
							} catch (InvalidInputException|CancelledException|IOException|URISyntaxException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
						}
					}
				}
			};
			action.setToolBarData(new ToolBarData(Icons.ADD_ICON, null));
			action.setEnabled(true);
			action.markHelpUnnecessary();
			dockingTool.addLocalAction(this, action);
		}

		@Override
		public JComponent getComponent() {
			return panel;
		}
		
		public void updateSimilarFunctions(ArrayList<SententiaResult> results) {
			DefaultTableModel model = (DefaultTableModel) similarFunctionsTable.getModel();
			model.setRowCount(0);
				
			for (SententiaResult result: results) {
				model.addRow(new Object[]{result.getFunctionName(), result.getFunctionScore()});
			}
			
		}
	}
}
