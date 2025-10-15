package nl.nfi.sententia;

import java.io.IOException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandler;

public class DowngradingHTTPClient {
    private HttpClient client;

    public DowngradingHTTPClient() {
        // Default client with HTTP/2
        this.client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_2)
                .build();
    }

    public HttpResponse<String> send(HttpRequest request, BodyHandler<String> responseBodyHandler) throws InterruptedException, IOException {
        HttpResponse<String> response = client.send(request, responseBodyHandler);

        // If server doesn't support HTTP/2, fall back to HTTP1.1
        if (response.statusCode() == 422) {
            // Downgrade to HTTP/1.1 and retry
            this.client = HttpClient.newBuilder()
                    .version(HttpClient.Version.HTTP_1_1)
                    .build();

            response = client.send(request, responseBodyHandler);
        }

        return response;
    }

    // Optional: add other convenience methods if needed
}
