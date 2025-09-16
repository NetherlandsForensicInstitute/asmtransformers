package nl.nfi.sententia;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpHeaders;
import java.net.http.HttpRequest.BodyPublishers;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import ghidra.framework.plugintool.PluginTool;
import ghidra.program.model.listing.Program;

public class SententiaAPI {
    public static final String SENTENTIA_DEFAULT_URL = "http://localhost:5000";
    public static final String API_URL = "/api/v1";
    private URI serverURL;
    private HttpClient client;

    public SententiaAPI(Program currentProgram, PluginTool sententiaTool) throws URISyntaxException {
        this(currentProgram, new URI(sententiaTool.getOptions("SententiaPlugin").getString("Endpoint", SENTENTIA_DEFAULT_URL)));
    }

    public SententiaAPI(Program currentProgram, URI url) {
        this.serverURL = url;
        // Initialize HttpClient with HTTP/2 enabled and connection reuse
        this.client = HttpClient.newBuilder()
                                .version(HttpClient.Version.HTTP_2)  // Force HTTP/2
                                .build();
    }

    public void setServerURL(URI url) {
        this.serverURL = url;
    }

    public ArrayList<SententiaResult> getMatchingFunctions(FunctionDescriptor descriptor, int top) 
            throws IOException, ParseException, URISyntaxException, InterruptedException {
        
        ArrayList<SententiaResult> results = new ArrayList<>();
        
        // Construct the full URL
        URI targetURI = new URI(this.serverURL.toString() + API_URL + "/search");

        // Prepare the JSON payload
        String jsonPayload = descriptor.toJson().toJSONString();

        // Create HTTP request
        HttpRequest request = HttpRequest.newBuilder()
                .uri(targetURI)
                .header("Content-Type", "application/json; charset=UTF-8")
                .POST(BodyPublishers.ofString(jsonPayload, StandardCharsets.UTF_8))  // Send POST request with the JSON body
                .build();

        // Send the request and get the response
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        
        // If server doesn't support HTTP/2, fall back to HTTP1.1
        if (response.statusCode() == 422) {
        	this.client = HttpClient.newBuilder()
                    .version(HttpClient.Version.HTTP_1_1)  // Downgrade to HTTP1.1
                    .build();
        	response = client.send(request, HttpResponse.BodyHandlers.ofString());
        }

        // Parse the response
        JSONParser parser = new JSONParser();
        JSONArray jsonResponse = (JSONArray) parser.parse(response.body());

        // Process the response
        for (Object result : jsonResponse) {
            JSONObject jsonResult = (JSONObject) result;
            if (jsonResult.get("function") instanceof String && jsonResult.get("similarity") instanceof Double) {
                results.add(new SententiaResult(
                    (String) jsonResult.get("function"),
                    (Double) jsonResult.get("similarity")
                ));
            } else {
                throw new IOException("Malformed JSON result!");
            }
        }

        // Sort the results by similarity score in descending order
        results.sort(Comparator.comparing(SententiaResult::getFunctionScore, Comparator.nullsLast(Comparator.reverseOrder())));

        // Return the top N results if required (default is returning all)
        return results;

    }

    public void addSignatureToDB(FunctionDescriptor descriptor) throws IOException, URISyntaxException, InterruptedException {
        URI targetURI = new URI(serverURL.toString() + API_URL + "/add");

        // Prepare the JSON payload
        String jsonPayload = descriptor.toJson().toJSONString();

        // Create HTTP request
        HttpRequest request = HttpRequest.newBuilder()
                .uri(targetURI)
                .header("Content-Type", "application/json; charset=UTF-8")
                .POST(BodyPublishers.ofString(jsonPayload, StandardCharsets.UTF_8))  // Send POST request with the JSON body
                .build();

        // Send the request
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

        // If server doesn't support HTTP/2, fall back to HTTP1.1
        if (response.statusCode() == 422) {
        	this.client = HttpClient.newBuilder()
                    .version(HttpClient.Version.HTTP_1_1)  // Downgrade to HTTP1.1
                    .build();
        	response = client.send(request, HttpResponse.BodyHandlers.ofString());
        }
        
        // Check if the request was successful (you can add more robust error handling here)
        if (response.statusCode() != 200) {
            throw new IOException("Failed to add signature to database. HTTP status code: " + response.statusCode());
        }
    }
}
