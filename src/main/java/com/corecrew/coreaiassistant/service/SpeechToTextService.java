package com.corecrew.coreaiassistant.service;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

import javafx.application.Platform;

public class SpeechToTextService {
    private Process process;
    private BufferedReader reader;
    private OnSpeechResultListener listener;

    public interface OnSpeechResultListener {
        void onSpeechRecognized(String text);
    }

    public void startListening(OnSpeechResultListener listener) throws IOException {
        this.listener = listener;

        String projectRoot = System.getProperty("user.dir");

        // **POINT TO YOUR pyenv PYTHON** here:
        String pythonPath = "/Users/chandan/.pyenv/versions/3.10.11/bin/python3";

        // path to your listener script
        String scriptPath = projectRoot
                + "/src/main/java/com/corecrew/coreaiassistant/service/facebook/hf_voice_listener.py";

        // -u = unbuffered stdout, so we see Python prints in real time
        ProcessBuilder pb = new ProcessBuilder(
                pythonPath, "-u", scriptPath
        );
        pb.directory(new File(projectRoot));
        pb.redirectErrorStream(true);

        process = pb.start();
        reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8)
        );

        Thread listenerThread = new Thread(() -> {
            try {
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) {
                        continue;  // skip blanks
                    }

                    // Strip any emoji prefix (🗣️ ) from Python output
                    if (line.startsWith("🗣️")) {
                        int idx = line.indexOf(' ');
                        if (idx != -1) {
                            line = line.substring(idx).trim();
                        }
                    }

                    final String recognizedText = line;
                    // Callback onto JavaFX application thread
                    Platform.runLater(() -> listener.onSpeechRecognized(recognizedText));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }, "STT-Listener-Thread");
        listenerThread.setDaemon(true);
        listenerThread.start();
    }

    public void stopListening() throws IOException {
        if (process != null) {
            process.destroy();  // SIGTERM
            try {
                if (!process.waitFor(1, TimeUnit.SECONDS)) {
                    process.destroyForcibly();
                }
            } catch (InterruptedException e) {
                process.destroyForcibly();
                Thread.currentThread().interrupt();
            }
        }
        if (reader != null) {
            reader.close();
        }
    }
}
