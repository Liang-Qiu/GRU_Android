/* Copyright 2016 LiangKlausQiu. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Core code, using tensorflow GRU model to predict

package edu.ucla.liangqiu.predictor;

import android.content.res.AssetManager;
//import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;
import java.util.HashMap;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/** A predictor specialized to predict words using TensorFlow. */
public class TensorFlowWordPredictor implements Predictor {
  static {
    //TODO
    System.loadLibrary("tensorflow_demo"); // load liangqiu_preditor.so
  }

  private static final String TAG = "TensorFlowWordPredictor";

  // Only return this many results with at least this confidence.
  private static final int MAX_RESULTS = 3;

  //TODO may change the THRESHOLD
  private static final float THRESHOLD = 0.1f;

  // Config values.
  private String inputName;
  private String outputName;
  private int input_max_Size;

  // Pre-allocated buffers.
  private Vector<String> id_to_word = new Vector<String>();
  private HashMap<String, Integer> word_to_id = new HashMap<String, Integer>();
  private int[] intValues;   // a serial of word ids
  private float[] outputs;   // confidence corresponding to vocab
  private String[] outputNames; // names of output nodes

  private TensorFlowInferenceInterface inferenceInterface;

  /**
   * Initializes a native TensorFlow session for predicting words.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param vocabFilename The filepath of vocab file for classes.
   * @param numClasses The number of classes output by the model.
   * @param numSteps The input max numsteps. A longest sentence of size numSteps is assumed.
   * @param inputName The label of the text input node.
   * @param outputName The label of the output node.
   * @return The native return value, 0 indicating success.
   * @throws IOException
   */
  public int initializeTensorFlow(
      AssetManager assetManager,
      String modelFilename,
      String vocabFilename,
      int numClasses,
      int input_max_Size,
      String inputName,
      String outputName) throws IOException {
    this.inputName = inputName;
    this.outputName = outputName;

    // Read the label names into memory.
    // TODO(andrewharp): make this handle non-assets.
    String actualFilename = vocabFilename.split("file:///android_asset/")[1];
    Log.i(TAG, "Reading vocab from: " + actualFilename);
    BufferedReader br = null;
    br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
    String line;
    int index = 0;
    while ((line = br.readLine()) != null) {
      id_to_word.add(line);
      word_to_id.put(line, index);
      index++;
    }
    br.close();
    Log.e(TAG, "Read " + id_to_word.size() + ", " + numClasses + " specified");

    this.input_max_Size = input_max_Size;

    //TODO Pre-allocate buffers. this.intValues?
    outputNames = new String[] {outputName};
    intValues = new int[input_max_Size];
    outputs = new float[numClasses];

    inferenceInterface = new TensorFlowInferenceInterface();

    return inferenceInterface.initializeTensorFlow(assetManager, modelFilename);
  }

  @Override
  public List<Prediction> predictWord(final String string) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("predictWord");

    Trace.beginSection("preprocessText");
    Log.e(TAG, "inut_string:" + string);
    //TODO 将string转化为int[numsteps] !!!!!!
    String[] input_words = string.split(" ");
    //intValues = new int[input_words.length];
    if (input_words.length < input_max_Size) {
      for (int i = 0; i < input_words.length; ++i) {
        Log.e(TAG, "input_word: " + input_words[i]);
        if (word_to_id.containsKey(input_words[i])) intValues[i] = word_to_id.get(input_words[i]);
        else intValues[i] = 1; //rare words, <unk> in the vocab
        Log.e(TAG, "input_id: " + intValues[i]);
      }
      for (int i = input_words.length; i < input_max_Size; ++i) {
        intValues[i] = 2; //padding using <eos>
      }
    }
    else {
      Log.e(TAG, "input out of max Size allowed!");
      return null;
    }
    Trace.endSection();
    // Copy the input data into TensorFlow.
    Trace.beginSection("fillNodeFloat");
    // TODO
    inferenceInterface.fillNodeInt(inputName, new int[] {1, input_max_Size}, intValues);
    Log.e(TAG, "fillNodeInt success!");
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("runInference");
    inferenceInterface.runInference(outputNames);
    Log.e(TAG, "runInference success!");
    Trace.endSection();

    // Copy the output Tensor back into the output array.
    Trace.beginSection("readNodeFloat");
    inferenceInterface.readNodeFloat(outputName, outputs);
    Log.e(TAG, "readNodeFloat success!");
    Trace.endSection();

    // Find the best predictions.
    PriorityQueue<Prediction> pq = new PriorityQueue<Prediction>(3,
        new Comparator<Prediction>() {
          @Override
          public int compare(Prediction lhs, Prediction rhs) {
            // Intentionally reversed to put high confidence at the head of the queue.
            return Float.compare(rhs.getConfidence(), lhs.getConfidence());
          }
        });
    for (int i = 0; i < outputs.length; ++i) {
      if (outputs[i] > THRESHOLD) {
        pq.add(new Prediction("" + i, id_to_word.get(i), outputs[i]));
      }
    }
    final ArrayList<Prediction> predictions = new ArrayList<Prediction>();
    for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
      predictions.add(pq.poll());
    }
    for (int i = 0; i < predictions.size(); ++i) {
      Log.e(TAG, predictions.get(i).toString());
    }
    Trace.endSection(); // "predict word"
    return predictions;
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }
}
