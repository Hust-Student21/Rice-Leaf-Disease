import labels from "./labels.json";
import * as tf from "@tensorflow/tfjs";

class Tracker {
    constructor() {
      // Store the center positions of the objects
      this.centerPoints = {};
      // Keep the count of the IDs
      // each time a new object id is detected, the count will increase by one
      this.idCount = 0;
    }
    /**
  * Render prediction boxes
  * @param {Array} data boxes array
  */
    update = (data) => {
        let objects_bbs_ids = [];
        for(let i=0;i<data.length;i++){
            let [x, y, w, h, cls] = data[i];
            // console.log([x, y, w, h, cls]) //oke
            let cx = Math.floor((x + x + w) / 2); 
            let cy = Math.floor((y + y + h) / 2);
            // console.log(cx); //oke
            // console.log(cy);

            let same_object_detected = false;
            for(let id in this.centerPoints){
                let pt = this.centerPoints[id];
                let dist = Math.hypot(cx - pt[0], cy - pt[1]);

                if (dist < 35){
                    this.centerPoints[id] = [cx, cy];
                    objects_bbs_ids.push([x, y, w, h, cls, id]);
                    same_object_detected = true;
                    break;
                }
            }
            if(same_object_detected==false){
                this.centerPoints[this.idCount] = [cx, cy];
                // console.log(this.centerPoints);
                objects_bbs_ids.push([x, y, w, h, cls, this.idCount]);
                // console.log(objects_bbs_ids);
                this.idCount += 1;
            }
        }

        let new_center_points = {};
        for(let j=0;j < objects_bbs_ids.length;j++){
            let [a,b,c,d,e, object_id] = objects_bbs_ids[j];
            let center = this.centerPoints[object_id];
            // console.log(center);
            new_center_points[object_id] = center;
        }
        this.centerPoints = {...new_center_points};
        return objects_bbs_ids;
    }
}
export default Tracker