import * as tf from "@tensorflow/tfjs";
import { renderBoxes } from "./renderBox";
import labels from "./labels.json";
import { counting } from "./counting";

var numClass = labels.length;
/**
 * Preprocess image / frame before forwarded into the model
 * @param {HTMLVideoElement|HTMLImageElement} source
 * @param {Number} modelWidth
 * @param {Number} modelHeight
 * @returns input tensor, xRatio and yRatio
 */
var preprocess = (source, modelWidth, modelHeight) => {
  let xRatio, yRatio; // ratios for boxes

  var input = tf.tidy(() => {
    var img = tf.browser.fromPixels(source);

    // padding image to square => [n, m] to [n, n], n > m
    var [h, w] = img.shape.slice(0, 2); // get source width and height
    console.log("size anh goc",[h,w]);
    var maxSize = Math.max(w, h); // get max size
    // console.log('max size:',maxSize)
    var imgPadded = img.pad([
      [0, maxSize - h], // padding y [bottom only]
      [0, maxSize - w], // padding x [right only]
      [0, 0],
    ]);

    xRatio = maxSize / w; // update xRatio
    yRatio = maxSize / h; // update yRatio

    return tf.image
      .resizeBilinear(imgPadded, [modelWidth, modelHeight]) // resize frame to the size of the model, this case 640x640
      // .resizeBilinear(img, [modelWidth, modelHeight]) // resize frame to the size of the model, this case 640x640
      .div(255.0) // normalize
      .expandDims(0); // add batch // the model input is (1,640,640,3), therefore batch dimension mustbe added
  });

  return [input, xRatio, yRatio]; //input is 640x640
};

/**
 * Function run inference and do detection from source.
 * @param {HTMLImageElement|HTMLVideoElement} source
 * @param {tf.GraphModel} model loaded YOLOv8 tensorflow.js model
 * @param {HTMLCanvasElement} canvasRef canvas reference
 * @param {VoidFunction} callback function to run after detection process
 */

export var detect = async (source, model, canvasRef, callback = () => {}) => {
  // console.time('doSomething')
  var [modelWidth, modelHeight] = model.inputShape.slice(1, 3); // get model width and height
  // the input shape of the model is (1,640,640,3) with slice(1,3), element 1 and 2 will be select which are 640 and 640

  tf.engine().startScope(); // start scoping tf engine
  var [input, xRatio, yRatio] = preprocess(source, modelWidth, modelHeight); // preprocess image
  console.log("input: ",input); 
  var res = await model.net.executeAsync(input); // inference model
  var transRes = res.transpose([0, 2, 1]); // transpose result [b, det, n] => [b, n, det]
  // b: batch
  // det: x,y,wh
  // n: number of detection 
  var boxes = tf.tidy(() => {
    var w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
    // starting point at index 0 of dimension 0, index 0 of dimension 1 and index 2 of dimension 2
    // [x,x,1] means take only 1 channel
    var h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
    // starting point at index 0 of dimension 0, index 0 of dimension 1 and index 3 of dimension 2
    // console.log([w,h])
    var x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
    // tf.div : divide the tensor to 2 : [2,4,6] / 2 = [1,2,3]
    // take the coordinate of the center of the box respect to the whole image minus the w/2
    var y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
    // take the coordinate of the center of the box respect to the whole image minus the h/2
    return tf
      .concat(
        [
          y1,
          x1,
          tf.add(y1, h), //y2 = y1 + h
          tf.add(x1, w), //x2 = x1 + w
        ],
        2 // concat on the dimenson 2 (channel)
      )
      .squeeze();
  }); // process boxes [y1, x1, y2, x2]

  var [scores, classes] = tf.tidy(() => { // take the value for variable scores and claseses
    var rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(); // class scores
    // starting point at index 0 of dimenson 0(height), index 0 of dimension 1 (width),index 4 of dimension 2(channels)
    // the number of channels select = numclass because each predict will contain score for all class
    return [rawScores.max(1), rawScores.argMax(1)]; 
  }); // get max scores and classes index

  var nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); // NMS to filter boxes
  // console.log("nms:",nms);
  // tf.print(nms);
  // nms.print()

  var boxes_data = boxes.gather(nms, 0).dataSync(); // indexing boxes by nms index
  console.log("boxes_data",boxes_data)
  // dataSync will transfer tensor to an array
  // gather is use to pick boxes base of the result of nms
  var scores_data = scores.gather(nms, 0).dataSync(); // indexing scores by nms index
  var classes_data = classes.gather(nms, 0).dataSync(); // indexing classes by nms index
  // tf.print(classes_data)
  renderBoxes(canvasRef, boxes_data, scores_data, classes_data, [xRatio, yRatio]); // draw boxes
  var obj_num = counting(canvasRef, boxes_data, scores_data, classes_data, [xRatio, yRatio]);
  // console.log(obj_num);
  tf.dispose([res, transRes, boxes, scores, classes, nms]); // clear memory

  callback();

  tf.engine().endScope(); // end of scoping
  // console.timeEnd('doSomething')
};


/**
 * Function to detect video from every source.
 * @param {HTMLVideoElement} vidSource video source
 * @param {tf.GraphModel} model loaded YOLOv8 tensorflow.js model
 * @param {HTMLCanvasElement} canvasRef canvas reference
 */
export var detectVideo = (vidSource, model, canvasRef) => {
  /**
   * Function to detect every frame from video
   */
  var detectFrame = async () => {
    if (vidSource.videoWidth === 0 && vidSource.srcObject === null) {
      var ctx = canvasRef.getContext("2d");
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas
      return; // handle if source is closed
    }

    detect(vidSource, model, canvasRef, () => {
      requestAnimationFrame(detectFrame); // get another frame
      // detectFrame(); =>> this create error in video
      //The line of code you provided is a callback mechanism used to achieve real-time object detection in a video stream
      //() => {...}: This is an anonymous callback function that is passed as the fourth argument to the detect function. 
      //It contains code to be executed after the detection process is complete. 
      //In this case, it's an arrow function without any explicit parameters.
    });
  };

  detectFrame(); // initialize to detect every frame
};
