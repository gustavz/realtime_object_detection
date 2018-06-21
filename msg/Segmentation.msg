std_msgs/Header header

# Bounding boxes in pixels
sensor_msgs/RegionOfInterest[] boxes

# String class Name for each bounding box
string[] class_names

# Segmentation map as Image
sensor_msgs/Image seg_map

# Current FPS Performance
float32 fps
