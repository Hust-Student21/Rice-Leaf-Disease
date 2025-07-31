import labels from "./labels.json";

/**
 * Render prediction boxes
 * @param {HTMLCanvasElement} canvasRef canvas tag reference
 * @param {Array} boxes_data boxes array
 * @param {Array} scores_data scores array
 * @param {Array} classes_data class array
 * @param {Array[Number]} ratios boxes ratio [xRatio, yRatio]
 */
export var renderBoxes = (canvasRef, boxes_data, scores_data, classes_data, ratios) => {
  var ctx = canvasRef.getContext("2d");
  //Đối tượng getContext(“2d”) trong HTML5
  // sở hữu nhiều hàm dành cho vẽ hình ảnh như hình hộp, hình tròn, chữ
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas

  var colors = new Colors();

  // font configs
  var font = `${Math.max(
    Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40),
    14
  )}px Arial`; // create font
  ctx.font = font;
  ctx.textBaseline = "top";
  for (let i = 0; i < scores_data.length; ++i) {
    // filter based on class threshold
    var klass = labels[classes_data[i]];
    var color = colors.get(classes_data[i]); // get a hex value
    var score = (scores_data[i] * 100).toFixed(1);
     
    let [y1, x1, y2, x2] = boxes_data.slice(i * 4, (i + 1) * 4);
    console.log("ngoc automation",[y1, x1, y2, x2])

    x1 *= ratios[0]; // take back the ratio to drawn bounding boxes
    x2 *= ratios[0];
    y1 *= ratios[1];
    y2 *= ratios[1];
    console.log("ngoc automation2",[y1, x1, y2, x2])
    var width = x2 - x1;
    console.log("check width",width)
    var height = y2 - y1;
    console.log("check hiehgt",height)

    // draw box.
    ctx.fillStyle = Colors.hexToRgba(color, 0.5); // select color
    // this line can be set to ctx.fillStyle=color how ever we need to tune the alpha value
    //ctx.fillStyle = "rgba(255, 0, 0, 0.5)";
    console.log(Colors.hexToRgba(color, 0.2))
    ctx.fillRect(x1, y1, width, height); // draw bounding box

    // draw border box.
    ctx.strokeStyle = color; // color for the line of the border, hex value
    ctx.lineWidth = Math.max(Math.min(ctx.canvas.width, ctx.canvas.height) / 200, 2.5); //?
    //This code snippet is a common technique used to ensure that the line width remains 
    //appropriate for drawing regardless of the canvas dimensions. 
    //It ensures that the line width is at least 2.5 but is also scaled proportionally to the canvas size. 
    //This is particularly useful for responsive designs where the canvas size may change based on the screen or container dimensions.

    //ctx.canvas.width =640
    //ctx.canvas.height=640
    // console.log("ngoc automation",Math.min(ctx.canvas.width, ctx.canvas.height))
    ctx.strokeRect(x1, y1, width, height);

    // Draw the label background.
    ctx.fillStyle = color;
    var textWidth = ctx.measureText(klass + " - " + score + "%").width;
    var textHeight = parseInt(font, 10); // base 10
    var yText = y1 - (textHeight + ctx.lineWidth);
    ctx.fillRect(
      x1 - 1,
      yText < 0 ? 0 : yText, // handle overflow label box, in case where y box label near 0 
      textWidth + ctx.lineWidth,
      textHeight + ctx.lineWidth
    );

    // Draw labels
    ctx.fillStyle = "#ffffff"; //white
    ctx.fillText(klass + " - " + score + "%", x1 - 1, yText < 0 ? 0 : yText);
  }
};

class Colors {
  // ultralytics color palette https://ultralytics.com/
  constructor() {
    //Constructor: The constructor function is called when you create an instance of the Colors class. 
    //It initializes the palette array with a list of hexadecimal color codes and 
    //calculates the length of the palette to store in the property n.

    this.palette = [
      "#FF3838",
      "#FF9D97",
      "#FF701F",
      "#FFB21D",
      "#CFD231",
      "#48F90A",
      "#92CC17",
      "#3DDB86",
      "#1A9334",
      "#00D4BB",
      "#2C99A8",
      "#00C2FF",
      "#344593",
      "#6473FF",
      "#0018EC",
      "#8438FF",
      "#520085",
      "#CB38FF",
      "#FF95C8",
      "#FF37C7",
    ];
    this.n = this.palette.length;
  }

  get = (i) => this.palette[Math.floor(i) % this.n];

  static hexToRgba = (hex, alpha) => { // the alpha value is for how sharp the color
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    //For example, let's say hex is "Hello, #AABBCC, how are you?". With the original regular expression, 
    //it would not match because the color code is not the only content in the string.
    //However, if you remove the ^ and $, the regular expression will find the color code "#AABBCC" within the string.

    //The i flag in the regular expression makes the pattern case-insensitive, so it will match both uppercase and lowercase letters.
    //#?: Matches an optional "#" character at the beginning of the string. The ? quantifier makes the "#" character optional.
    //    this means that you dont need to write #AABBCC, but instead you can write AABBCC
    //([a-f\d]{2}): This part of the expression captures two hexadecimal characters (0-9, a-f) and places them in a capturing group. 
    //              This is repeated three times, capturing the red, green, and blue components of the color.
    return result

      ? `rgba(${[parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)].join(
          ", "
        )}, ${alpha})`
      : null;
  };
}
//parseInt(result[1], 16):In JavaScript (and many other programming languages), the parseInt function is used to convert strings to integers. 
//                        The second argument of parseInt specifies the base to use for the conversion.
//                        When converting hexadecimal (base-16) strings to integers, you pass 16 as the base argument.

//The entire expression is wrapped in backticks (`) to create a template string.

//The ? symbol is used as a ternary operator to check whether result is truthy (i.e., not null or undefined). 
//If result is truthy (meaning the regular expression match was successful), the code following the ? is executed. 
//If result is falsy (meaning the regular expression match was not successful), the code following the : is executed.
//If result is truthy:
//The rgba color representation is generated based on the captured components and the provided alpha value.
//This generated rgba color string is returned.
//If result is falsy:
//The function returns null.