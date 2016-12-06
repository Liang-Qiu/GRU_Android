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
  private int inputSize;

  // Pre-allocated buffers.
  private Vector<String> vocab = new Vector<String>();
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
      int inputSize,
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
    while ((line = br.readLine()) != null) {
      vocab.add(line);
    }
    br.close();
    Log.e(TAG, "Read " + vocab.size() + ", " + numClasses + " specified");

    this.inputSize = inputSize;

    //TODO Pre-allocate buffers. this.intValues?
    outputNames = new String[] {outputName};
    intValues = new int[inputSize];
    outputs = new float[numClasses];

    inferenceInterface = new TensorFlowInferenceInterface();

    return inferenceInterface.initializeTensorFlow(assetManager, modelFilename);
  }

  @Override
  public List<Prediction> predictWord(final String string) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("predictWord");

    Trace.beginSection("preprocessText");
    //TODO 将string转化为int[numsteps] !!!!!!

    for (int i = 0; i < intValues.length; ++i) {
      intValues[i] = 1;
    }
    Trace.endSection();

    // Copy the input data into TensorFlow.
    Trace.beginSection("fillNodeFloat");
    // TODO
    inferenceInterface.fillNodeInt(inputName, new int[] {1, inputSize, 1, 1}, intValues);
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
        pq.add(new Prediction("" + i, vocab.get(i), outputs[i]));
      }
    }
    final ArrayList<Prediction> predictions = new ArrayList<Prediction>();
    for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
      predictions.add(pq.poll());
    }
    Trace.endSection(); // "predict word"
    return predictions;
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }
}
