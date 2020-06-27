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

### Real-time loop

|Step|Duration|Description|
|----|--------|-----------|
|Acquisition|Enough time to capture a few heartbeats|Frame-by-frame recording of the image w/o any processing to have the highest FPS as possible|
|Sequence processing|TBD|Sequence processing to estimate heart-rate value|

## Pi setup

### Steps to have headless working raspberry pi

1. Install raspbian-lite distrib
2. In the boot folder, copy/paste the following files:

* *ssh* (emptyp file) enables ssh connection (otherwise blocked)
* *wpa_supplicant.conf* is for WiFi connection setup

### Steps to enable camera

1. Plug-in the camera module
2. In *sudo raspi-config*, enable camera usage
3. Test taking a picture using *raspistill -o test.jpg*

### Steps to install jupyter

1. install pip3 *sudo apt install python3-pip*
2. install jupyter *sudo pip3 install jupyter*
3. Create jupyter config *jupyter notebook --generate-config*
4. Listen to all ip: *c.NotebookApp.ip = '*'*
5. Disable browser launching: *c.NotebookApp.open_browser = False*
6. Set a port to listen on: *c.NotebookApp.port = 5555*
7. Install matplotlib, numpy

### Steps to read pulse sensor from rpi

1. install spidev through pip3
2. Enable SPI interface in raspi-config