package com.corecrew.coreaiassistant;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.web.WebView;
import org.fxmisc.richtext.CodeArea;
import javafx.scene.control.Button;
import com.vladsch.flexmark.html.HtmlRenderer;
import com.vladsch.flexmark.parser.Parser;
import okhttp3.*;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;

public class HelloController {

    @FXML
    private WebView chatView;
    @FXML
    private CodeArea codeInput;
    @FXML
    private Button sendButton;

    private final Parser mdParser = Parser.builder().build();
    private final HtmlRenderer mdRenderer = HtmlRenderer.builder().build();
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    private static final String API_URL = "https://openrouter.ai/api/v1/chat/completions";
    private static final String API_KEY = System.getenv("OPENROUTER_API_KEY");

    private final OkHttpClient client = new OkHttpClient();

    @FXML
    private void initialize() {
        String baseHtml = """
                <html>
                <head>
                    <style>
                        body {
                            margin: 0;
                            padding: 0;
                            font-family: Arial, sans-serif;
                            background: #fff;
                            color: #000;
                        }
                        .chat-container {
                            display: flex;
                            flex-direction: column;
                            gap: 16px;
                            padding: 16px;
                        }
                        .message {
                            max-width: 80%;
                            padding: 12px 16px;
                            border-radius: 12px;
                            word-wrap: break-word;
                            white-space: pre-wrap;
                        }
                        .user {
                            align-self: flex-end;
                            background-color: #007aff;
                            color: #fff;
                            border-bottom-right-radius: 4px;
                        }
                        .ai {
                            align-self: flex-start;
                            background-color: #f1f1f2;
                            color: #000;
                            border-bottom-left-radius: 4px;
                        }
                        pre {
                            background: #2d2d2d;
                            color: #f8f8f2;
                            padding: 10px;
                            border-radius: 6px;
                            overflow-x: auto;
                            font-family: Consolas, monospace;
                            white-space: pre-wrap;
                            word-break: break-word;
                        }
                        code {
                            font-family: Consolas, monospace;
                            background: rgba(0,0,0,0.05);
                            padding: 2px 4px;
                            border-radius: 4px;
                            word-wrap: break-word;
                        }
                    </style>
                </head>
                <body>
                    <div class="chat-container" id="chat"></div>
                </body>
                </html>
                """;

        chatView.getEngine().loadContent(baseHtml);
    }

    @FXML
    private void onSendClicked() {
        String userMessage = codeInput.getText().trim();
        if (userMessage.isEmpty()) return;

        appendMessage("user", userMessage);
        codeInput.clear();

        String payload = new JSONObject()
                .put("model", "openai/gpt-3.5-turbo")
                .put("messages", new JSONArray()
                        .put(new JSONObject().put("role", "system").put("content", "You are a helpful assistant."))
                        .put(new JSONObject().put("role", "user").put("content", userMessage)))
                .put("max_tokens", 8192)
                .put("temperature", 0.7)
                .toString();

        Request request = new Request.Builder()
                .url(API_URL)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .post(RequestBody.create(payload, JSON))
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Platform.runLater(() -> appendMessage("ai", "Network error: " + e.getMessage()));
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                String body = response.body().string();
                if (!response.isSuccessful()) {
                    Platform.runLater(() -> appendMessage("ai", "HTTP error: " + response.code()));
                    return;
                }

                String reply = new JSONObject(body)
                        .getJSONArray("choices")
                        .getJSONObject(0)
                        .getJSONObject("message")
                        .getString("content");

                Platform.runLater(() -> appendMessage("ai", reply));
            }
        });
    }

    private void appendMessage(String sender, String markdown) {
        String html = mdRenderer.render(mdParser.parse(markdown))
                .replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "");

        String script = String.format("""
                    var chat = document.getElementById('chat');
                    var msg = document.createElement('div');
                    msg.className = 'message %s';
                    msg.innerHTML = \"%s\";
                    chat.appendChild(msg);
                    window.scrollTo(0, document.body.scrollHeight);
                """, sender, html);

        chatView.getEngine().executeScript(script);
    }
}