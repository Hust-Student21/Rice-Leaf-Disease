import labels from "./labels.json";
import Tracker from "./tracker"
/**
 * Render prediction boxes
 * @param {HTMLCanvasElement} canvasRef canvas tag reference
 * @param {Array} boxes_data boxes array
 * @param {Array} scores_data scores array
 * @param {Array} classes_data class array
 * @param {Array[Number]} ratios boxes ratio [xRatio, yRatio]
 * @param {Boolean} clear0bj
 */
const cy1 = 320;
const cy2 = 340;
const offset = 6;
var down = {};
var up = {};
var class_0 = 0;
var class_1 = 0;
var class_2 = 0;
var obj = 0;
export const counting = (canvasRef, boxes_data, scores_data, classes_data, ratios) => {
    let lst = []
    for (let i = 0; i < scores_data.length; ++i) {
        const klass = labels[classes_data[i]];

        let [y1, x1, y2, x2] = boxes_data.slice(i * 4, (i + 1) * 4);
        lst.push([y1, x1, y2, x2,klass]); //oke
        // console.log(lst);
    }

    let bbox_id_ = new Tracker();
    let bbox_id = bbox_id_.update(lst);

    for (let i = 0; i < bbox_id.length;i++){
        let [x3,y3,x4,y4,cls,id] = bbox_id[i];
        let cx = Math.floor((x3 + x4)/ 2);
        let cy = Math.floor((y3 + y4)/ 2);
        console.log([cx,cy]);

        if (cy1 < (cy + offset) && (cy1 > (cy - offset))){
            down[id] = cy;
        }
        if (id.toString() in down){
            if (cy2 < (cy + offset) && (cy2 > (cy-offset))){
                obj +=1;
            }
        }

        if (cy2 < (cy + offset) && (cy2 > (cy - offset))){
            up[id] = cy;
        }
        if (id.toString() in up){
            if (cy1 < (cy + offset) && (cy1 > (cy-offset))){
                obj+=1;
            }
        }
    }
    return obj;
}