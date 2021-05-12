# object-based-video-synopsis
An Object-based Video Synopsis processor using Tracking and Image processing.
Tracking: DeepSORT and YOLOv4
Image processing: Background subtraction

# Usage of video synopsis processor
```
python main.py --video /path/to/input/video --output /path/to/output --frame_cut True
```

# Demo
![alt text](https://github.com/iamyugachang/object-based-video-synopsis/blob/master/demo/demo.gif)

# Default file structure
input videos in `./data/videos/`,
output videos in `./outputs/`,
output frame_cut in `./outputs/frame_cut`
