package com.corecrew.coreaiassistant.utils;

import com.vladsch.flexmark.html.HtmlRenderer;
import com.vladsch.flexmark.parser.Parser;
import javafx.application.Platform;
import javafx.scene.web.WebView;
import org.json.JSONArray;
import org.json.JSONObject;

public class AssistantUtils {

    private static final Parser mdParser = Parser.builder().build();
    private static final HtmlRenderer mdRenderer = HtmlRenderer.builder().build();

    public static String buildPayload(String userMessage) {
        return new JSONObject()
                .put("model", "openai/gpt-3.5-turbo")
                .put("messages", new JSONArray()
                        .put(new JSONObject().put("role", "system").put("content", "You are a helpful assistant."))
                        .put(new JSONObject().put("role", "user").put("content", userMessage)))
                .put("max_tokens", 8192)
                .put("temperature", 0.7)
                .toString();
    }

    public static void appendMessage(WebView chatView, String sender, String markdown) {
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
            msg.innerHTML = "%s";
            chat.appendChild(msg);
            window.scrollTo(0, document.body.scrollHeight);
        """, sender, html);

        Platform.runLater(() -> chatView.getEngine().executeScript(script));
    }
}
