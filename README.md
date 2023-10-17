# Estimate pedestrian speed

This project is to detect pedestrian walking speed in crosswalk zone using YOLOv8 and Deepsort.

Each bounding boxes are colored to blue (no motion), red (slow walker), yellow (average walker), and green (fast walker) to classify each pedestrian's walking speed. According to the real-time walking speed, the ETS will be shown in the last line in the bounding box. 


### output video

![123](https://github.com/chaelin0722/Estimate_pedestrian_speed/assets/53431568/2353b4d5-5c76-4bdf-857b-3ab072f9a7be)


### Pipeline

<img width="1633" alt="pipeline" src="https://github.com/chaelin0722/Estimate_pedestrian_speed/assets/53431568/a1ef2ce0-5461-4ed8-aa38-e2e1317968e9">

### Environment
~~~
python 3.10.12
ultralytics 8.0.3
torch 2.0.1+cu118
~~~

or you can import speed.yaml file

### Inference
~~~
cd YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect
python speed_detect.py model="model.pt" source="inputfile.avi"
~~~


### reference
[1] https://github.com/ultralytics/ultralytics

[2] https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking


