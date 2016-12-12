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
//Implement ResultsView to show the prediction results implements

package edu.ucla.liangqiu.predictor;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.View;
import android.util.Log;

import java.util.List;

import edu.ucla.liangqiu.predictor.Predictor.Prediction;

//TODO the whole view needs to be revised
public class PredictionScoreView extends View implements ResultsView {
  private static final float TEXT_SIZE_DIP = 24;
  private List<Prediction> results;
  private final float textSizePx;
  private final Paint fgPaint;
  private final Paint bgPaint;
  private static final String TAG = "TensorFlowWordPredictor";

  public PredictionScoreView(final Context context, final AttributeSet set) {
    super(context, set);

    textSizePx = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics()); // 24 DIP
    fgPaint = new Paint();
    fgPaint.setTextSize(textSizePx);

    bgPaint = new Paint();
    bgPaint.setColor(0xffeeeeee);
  }

  @Override
  public void setResults(final List<Prediction> results) {
    this.results = results;
    postInvalidate(); // refresh UI
    Log.e(TAG, "set results!");
  }

  @Override
  public void onDraw(final Canvas canvas) {
    final int x = 10;
    int y = (int) (fgPaint.getTextSize() * 1.5f); // position

    canvas.drawPaint(bgPaint); // background color

    if (results != null) {
      for (final Prediction predic : results) {
        canvas.drawText(predic.getTitle() + ": " + String.format("%.1f%% ", predic.getConfidence() * 100.0f), x, y, fgPaint);
        y += fgPaint.getTextSize() * 1.5f;
      }
    }
    Log.e(TAG, "Drawed the view!");
  }
}
