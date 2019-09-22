# Vehicle tracking using background subtraction and Kalman filter
Using background subtraction to detect vehicles which move across the camera. Once vehicles are detected, use Kalman Filter to track them, and then estimate their velocities.

## Build project
Build project with cmake:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Run project
Clone and copy test data to build folder:
```
cd ../../
git clone https://github.com/nqoptik/computer_vision_data.git
cd vehicle_tracking/build/
cp -r ../../computer_vision_data/vehicle_tracking/build/* .
```

Run vehicle tracking:
```
./vehicle_tracking <video_file>
```
