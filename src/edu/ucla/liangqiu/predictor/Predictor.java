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
// Interface for TensorFlowWordPredictor to implement

package edu.ucla.liangqiu.predictor;

//import android.graphics.Bitmap;
//import android.graphics.RectF;
import java.util.List;

/**
 * Generic interface for interacting with different prediction engines.
 */
public interface Predictor {
  /**
   * An immutable result returned by a Predictor describing what was recognized.
   */
  public class Prediction {
    /**
     * A unique identifier for what has been predicted. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /**
     * Display name for the prediction.
     */
    private final String title;

    /**
     * A sortable score for how good the prediction is relative to others. Higher should be better.
     */
    private final Float confidence;


    public Prediction(
        final String id, final String title, final Float confidence) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
    }

    public String getId() {
      return id;
    }

    public String getTitle() {
      return title;
    }

    public Float getConfidence() {
      return confidence;
    }

    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "[" + id + "] ";
      }

      if (title != null) {
        resultString += title + " ";
      }

      if (confidence != null) {
        resultString += String.format("(%.1f%%) ", confidence * 100.0f);
      }

      return resultString.trim();
    }
  }

  //TODO "text" type may not be right

  List<Prediction> predictWord(String string);

  void close();
}
