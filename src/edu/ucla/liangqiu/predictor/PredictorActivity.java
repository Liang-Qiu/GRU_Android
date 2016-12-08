/*
 * Copyright 2016 LiangKlausQiu. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Entrance Activity

package edu.ucla.liangqiu.predictor;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.widget.Toast;
import android.content.Intent;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.view.View;
import android.widget.ImageButton;
import android.widget.TextView;
import android.util.Log;


import java.util.List;
import java.util.ArrayList;
import java.io.IOException;

public class PredictorActivity extends Activity {
    private static final int PERMISSIONS_REQUEST = 1;
    private static final String PERMISSION_RECORD_AUDIO = Manifest.permission.RECORD_AUDIO;
    private static final String PERMISSION_INTERNET = Manifest.permission.INTERNET;
    private static final String PERMISSION_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;
    private static final String TAG = "PredictorActvity";

    private TextView txtSpeechInput;
    private ResultsView resultsView;
    private ImageButton btnSpeak;
    private SpeechRecognizer sr;

    private TensorFlowWordPredictor predictor = new TensorFlowWordPredictor();
    private static final int NUM_CLASSES = 10000;
    private static final int INPUT_MAX_SIZE = 20;
    //TODO not sure about the name
    private static final String INPUT_NAME = "Test/input:0";
    private static final String OUTPUT_NAME = "Test/Model/result:0";
    private static final String MODEL_FILE = "file:///android_asset/gru_graph.pb";
    private static final String VOCAB_FILE = "file:///android_asset/gru_graph_vocab.txt";


    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.e(TAG,  "On Create!");
        setContentView(R.layout.main);
        txtSpeechInput = (TextView) findViewById(R.id.txtSpeechInput);
        resultsView = (ResultsView) findViewById(R.id.results);
        btnSpeak = (ImageButton) findViewById(R.id.btnSpeak);

        if (hasPermission()) {
            if (null == savedInstanceState) {
                initialize();
            }
        } else {
            requestPermission();
        }

    }

    private void initialize() {
        // tensorflow initialize
        try {
            //TODO Initialize parameters review
            predictor.initializeTensorFlow(getAssets(), MODEL_FILE, VOCAB_FILE, NUM_CLASSES, INPUT_MAX_SIZE, INPUT_NAME, OUTPUT_NAME);
        } catch (final IOException e) {
            Log.e(TAG, "initializeTensorFlow IO Exception!");
        }
        // voice recognition initialize
        sr = SpeechRecognizer.createSpeechRecognizer(this);
        sr.setRecognitionListener(new listener());
        btnSpeak.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                promptSpeechInput();
            }
        });
    }

    private class listener implements RecognitionListener
    {
        public void onReadyForSpeech(Bundle params)
        {
            Log.i(TAG, "onReadyForSpeech");
        }
        public void onBeginningOfSpeech()
        {
            Log.i(TAG, "onBeginningOfSpeech");
        }
        public void onRmsChanged(float rmsdB)
        {
            Log.i(TAG, "onRmsChanged");
        }
        public void onBufferReceived(byte[] buffer)
        {
            Log.i(TAG, "onBufferReceived");
        }
        public void onEndOfSpeech()
        {
            Log.i(TAG, "onEndofSpeech");
        }
        public void onError(int error)
        {
            Log.e(TAG,  "error: " +  error);
            txtSpeechInput.setText("error: " + error);
        }
        public void onResults(Bundle results)
        {
            String str = new String();
            ArrayList<String> data = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
            str = data.get(0);
            Log.e(TAG, "onResults: " + str);
            txtSpeechInput.setText(str); // The first recognition result
            //TODO
            final List<Predictor.Prediction> predictedWords = predictor.predictWord(str);
            resultsView.setResults(predictedWords);
        }
        public void onPartialResults(Bundle partialResults)
        {
            Log.i(TAG, "onPartialResults");
        }
        public void onEvent(int eventType, Bundle params)
        {
            Log.i(TAG, "onEvent " + eventType);
        }
    }

    private void promptSpeechInput() {
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        intent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS,5);
        sr.startListening(intent);
        //TODO not sure
        Log.i(TAG, "listening ...");
    }

    @Override
    public void onRequestPermissionsResult(
            final int requestCode, final String[] permissions, final int[] grantResults) {
        switch (requestCode) {
            case PERMISSIONS_REQUEST: {
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED
                        && grantResults[1] == PackageManager.PERMISSION_GRANTED
                        && grantResults[2] == PackageManager.PERMISSION_GRANTED) {
                    initialize();
                } else {
                    requestPermission();
                }
            }
        }
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED && checkSelfPermission(PERMISSION_INTERNET) == PackageManager.PERMISSION_GRANTED
                && checkSelfPermission(PERMISSION_STORAGE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_RECORD_AUDIO) || shouldShowRequestPermissionRationale(PERMISSION_INTERNET)
                    || shouldShowRequestPermissionRationale(PERMISSION_STORAGE)) {
                Toast.makeText(PredictorActivity.this, "Record_video, Internet and storage permission are required for this demo", Toast.LENGTH_LONG).show();
            }
            requestPermissions(new String[] {PERMISSION_RECORD_AUDIO, PERMISSION_INTERNET, PERMISSION_STORAGE}, PERMISSIONS_REQUEST);
        }
    }
}
