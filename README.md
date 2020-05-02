# heartsight
Vision-based Heart Rate Monitoring

## Data-flow

### Image processing

|Step|Output size|Description|
|----|-----------|-----------|
|Video Input|480x640x3|Raw RGB image coming from a webcam or video file|
|Extract Face|?x?x3|Extract using face detection function a box containing the face|
|Center Face|?x?x3|Rotate and translate face using landmarks in order to have always the same skin pixel at roughly the same position|
|Resize Face|32x32x3|Resize (downscale or upscale) to have fixed input size for the neural network|

### 
