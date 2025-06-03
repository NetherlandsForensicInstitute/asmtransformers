package nl.nfi.sententia;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.ProtocolException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;


import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.JSONArray;
import org.json.simple.parser.ParseException;

import ghidra.framework.plugintool.PluginTool;
import ghidra.program.model.listing.Program;

public class SententiaAPI {
	public static final String SENTENTIA_DEFAULT_URL = "http://localhost:5000";
	public static final String API_URL = "/api/v1";
	private URI serverURL;
	
	public SententiaAPI(Program currentProgram, PluginTool sententiaTool) throws URISyntaxException {
		this(currentProgram, new URI(sententiaTool.getOptions("SententiaPlugin").getString("Endpoint", SENTENTIA_DEFAULT_URL)));
	}
	
	public SententiaAPI(Program currentProgram, URI url) {
		this.serverURL = url;
	}
	
	public void setServerURL(URI url) {
		this.serverURL = url;
	}
	
	
	public ArrayList<SententiaResult> getMatchingFunctions(FunctionDescriptor descriptor, int top) 
			throws IOException, ProtocolException, ParseException, URISyntaxException {
		
		ArrayList<SententiaResult> results = new ArrayList<SententiaResult>();
			
		URLConnection serverConnection;
		

		serverConnection = new URI(this.serverURL + API_URL + "/search").toURL().openConnection();
		
		HttpURLConnection serverHTTP = (HttpURLConnection) serverConnection;
		
		serverHTTP.setRequestMethod("POST");
		serverHTTP.setDoOutput(true);
		
		serverHTTP.setFixedLengthStreamingMode(descriptor.toJson().toJSONString().length());
		serverHTTP.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
		serverHTTP.connect();
		
		try(OutputStream os = serverHTTP.getOutputStream()) {
		    os.write(descriptor.toJson().toJSONString().getBytes(StandardCharsets.UTF_8));
		}
		
		BufferedReader reader = new BufferedReader(new InputStreamReader(serverConnection.getInputStream()));
		
		JSONParser parser = new JSONParser();

		JSONArray jsonResponse = (JSONArray) parser.parse(reader);
		
		
		 for (Object result : jsonResponse) {
			 JSONObject jsonResult = (JSONObject) result;
			 if (jsonResult.get("function") instanceof String && jsonResult.get("similarity") instanceof Double) {
                 results.add(new SententiaResult(
                     (String) jsonResult.get("function"),
                     (Double) jsonResult.get("similarity")
                 ));
			 } else {
				throw new IOException("Malformed json result!");
			 }
		 }
		 
		
		// Sort the list by score, descending
		results.sort(Comparator.comparing(res -> res.getFunctionScore(), Comparator.nullsLast(Comparator.reverseOrder())));
		
		//TODO: Always return only top n results
		return results;
		
	}
	
	public void addSignatureToDB(FunctionDescriptor descriptor) throws IOException, URISyntaxException {
					
		URLConnection serverConnection = new URI(serverURL.toString() + API_URL + "/add").toURL().openConnection();	 
		
		HttpURLConnection serverHTTP = (HttpURLConnection) serverConnection;
		
		serverHTTP.setRequestMethod("POST");
		serverHTTP.setDoOutput(true);
		
		final byte[] payload = descriptor.toJson().toJSONString().getBytes(StandardCharsets.UTF_8);
		
		serverHTTP.setFixedLengthStreamingMode(payload.length);
		serverHTTP.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
		serverHTTP.connect();
		
		try(OutputStream os = serverHTTP.getOutputStream()) {
		    os.write(payload);
		}
		
		serverHTTP.disconnect();
		
	}
}
