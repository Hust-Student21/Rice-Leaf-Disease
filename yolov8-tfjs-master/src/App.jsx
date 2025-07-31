import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // set backend to webgl
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import { detect, detectVideo } from "./utils/detect";
import "./style/App.css";

const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 }); // loading state
  // loading.loading = true
  // loading.progress = 0
  const [model, setModel] = useState({
    net: null,
    inputShape: [1, 0, 0, 3], // batch_size,h,w,rgb channels
  }); // init model & input shape

  // model.net = null
  // model.inputshape = [1,0,0,3]

  // references
  // ref does not rerender your component when you change 
  const imageRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
// for the variable that not need to show on UI we use useRef otherwise we will use useState
  // model configs
  const modelName = "Normal";

  useEffect(() => { // this will only be run once when the model is first load
    tf.ready().then(async () => {
      const yolov8 = await tf.loadGraphModel(
        `${window.location.href}/${modelName}_tfjs_model/model.json`,
        {
          onProgress: (fractions) => { // in the tf.loadGraphModel
            setLoading({ loading: true, progress: fractions }); // set loading fractions
          },
        }
      ); // load model

      // warming up model
      const dummyInput = tf.ones(yolov8.inputs[0].shape);
      const warmupResults = yolov8.execute(dummyInput);
      // warm up the model part can be ignore how ever 
      //it will make the web a little bit laggy at first

      setLoading({ loading: false, progress: 1 }); // finnished loading model
      setModel({
        net: yolov8,
        inputShape: yolov8.inputs[0].shape,
      }); // set model & input shape

      tf.dispose([warmupResults, dummyInput]); // cleanup memory
    });
  }, []);

  return (
    <div className="App">
      {loading.loading && <Loader>Loading model... {(loading.progress * 100).toFixed(2)}%</Loader>}

      {/* {loading.loading}: This part is a condition that checks if the loading.loading property is true. 
      If it's true, the subsequent JSX code after the && will be rendered; otherwise, it will not be rendered. */}

      {/* {(loading.progress * 100).toFixed(2)}%: This part calculates the loading progress and displays it as a percentage in the 
      loading indicator. loading.progress is a number representing the progress fraction (e.g., 0.5 for 50% progress). 
      It multiplies this value by 100 to convert it to a percentage. 
      The toFixed(2) method is then used to round the percentage to two decimal places. */}
      <div className="header">
        <h1>📷 YOLOv8 Live Detection App</h1>
        <p>
          YOLOv8 live detection application on browser powered by <code>tensorflow.js</code>
        </p>
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
      </div>

      <div className="content">
        <img
          src="#" // this line is useless
          ref={imageRef}
          //by accessing dom element we will show the changes on the screen without causing re-render
          //In a controlled <input /> you'd use value and onChange. In an uncontrolled <input /> you'd use defaultValue and ref
          onLoad={() => {
            detect(imageRef.current, model, canvasRef.current)
            // console.log("app:",imageRef.current)
          }}
          // onLoad={() => ...}: This part of the code is an event handler for the onLoad event of the <img> element. 
          // The onLoad event is triggered when the image has finished loading in the browser.
          //The reason the onLoad event handler inside the img element works is that the onLoad event is a DOM event and not a React-specific event. 
          //It is triggered when the image has successfully loaded in the browser, regardless of whether React re-renders the component or not.
          //So, when the image is loaded, the onLoad event is fired, and the provided function inside the event handler 
          //(i.e., the detect function call) will be executed, regardless of whether the component re-renders or not.
        />
        <video
          autoPlay
          muted
          ref={cameraRef}
          onPlay={() => detectVideo(cameraRef.current, model, canvasRef.current)}
        />
        <video
          autoPlay
          muted
          ref={videoRef}
          onPlay={() => detectVideo(videoRef.current, model, canvasRef.current)}
        />
        <canvas width={model.inputShape[1]} height={model.inputShape[2]} ref={canvasRef} />
        {/* this line is to draw the bounding box out of the screen */}
      </div>

      <ButtonHandler imageRef={imageRef} cameraRef={cameraRef} videoRef={videoRef} />
    </div>
  );
};

export default App;
