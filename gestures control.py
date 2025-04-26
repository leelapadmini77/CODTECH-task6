import React, { useRef, useEffect } from "react";
import Webcam from "react-webcam";
import * as handpose from "@tensorflow-models/handpose";
import * as tf from "@tensorflow/tfjs";
function App() {
  const webcamRef = useRef(null);

  const prevGesture = useRef(null);

  const playPause = () => {
    const video = document.getElementById("test-video");

    if (video) {
      video.paused ? video.play() : video.pause();

      console.log("Toggled Play/Pause");
    }
  };

  const adjustVolume = (up) => {
    const video = document.getElementById("test-video");

    if (video) {
      video.volume = Math.min(1, Math.max(0, video.volume + (up ? 0.1 : -0.1)));

      console.log("Volume " + (up ? "Up" : "Down"));
    }
  };

  useEffect(() => {
    const runHandpose = async () => {
      const net = await handpose.load();

      console.log("Handpose model loaded.");

      setInterval(() => detect(net), 200);
    };

    const detect = async (net) => {
      if (webcamRef.current && webcamRef.current.video.readyState === 4) {
        const video = webcamRef.current.video;

        const hand = await net.estimateHands(video);

        if (hand.length > 0) {
          const fingers = hand[0].annotations;

          const indexTip = fingers.indexFinger[3];

          const thumbTip = fingers.thumb[3];

          const pinchDistance = Math.sqrt(
            Math.pow(indexTip[0] - thumbTip[0], 2) +
              Math.pow(indexTip[1] - thumbTip[1], 2)
          );

          if (pinchDistance < 40) {
            if (prevGesture.current !== "playPause") {
              playPause();

              prevGesture.current = "playPause";
            }
          } else if (indexTip[1] < thumbTip[1]) {
            if (prevGesture.current !== "volumeUp") {
              adjustVolume(true);

              prevGesture.current = "volumeUp";
            }
          } else if (indexTip[1] > thumbTip[1]) {
            if (prevGesture.current !== "volumeDown") {
              adjustVolume(false);

              prevGesture.current = "volumeDown";
            }
          } else {
            prevGesture.current = null;
          }
        } else {
          prevGesture.current = null;
        }
      }
    };

    runHandpose();
  }, []);

  return (
    <div className="App">
      <h1>Gesture Control Web App</h1>

      <p>
        Try pinching to toggle play/pause or move your index finger up/down
        relative to your thumb to adjust volume.
      </p>
      <div className="videoplay" style={{ display: "flex",alignItems:"center",justifyContent:"center" }}>
        <div className="normalai" style={{ width: "50%" }}>
          <video
            id="test-video"
            width="640"
            height="360"
            controls
            style={{ marginBottom: "20px" }}
          >
            <source
              src="https://www.w3schools.com/html/mov_bbb.mp4"
              type="video/mp4"
            />
            Your browser does not support the video tag.
          </video>
        </div>
        <div className="webai" style={{ width: "50%" }}>
          <Webcam
            className="videoplayai"
            ref={webcamRef}
            style={{
              position: "relative",

              zIndex: 9,

              width: 640,

              height: 480,
            }}
          />
        </div>
      </div>
    </div>
  );
}

export default App;