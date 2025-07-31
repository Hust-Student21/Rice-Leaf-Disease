/**
 * Class to handle webcam
 */
export class Webcam {
  /**
   * Open webcam and stream it through video tag.
   * @param {HTMLVideoElement} videoRef video tag reference
   */
  open = (videoRef) => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) { // check if the browser availble for streaming
      navigator.mediaDevices
        .getUserMedia({  // set up the camera and video properties, no audio,etc
          // getUserMedia will ask the user permission to access webcam
          audio: false,
          video: {
            facingMode: "environment",
          },
        })
        .then((stream) => {
          videoRef.srcObject = stream;
        });
    } else alert("Can't open Webcam!");
  };

  /**
   * Close opened webcam.
   * @param {HTMLVideoElement} videoRef video tag reference
   */
  close = (videoRef) => {
    if (videoRef.srcObject) {
      videoRef.srcObject.getTracks().forEach((track) => {
        track.stop();
      }); // stop streaming
      videoRef.srcObject = null;
    } else alert("Please open Webcam first!");
  };
}
