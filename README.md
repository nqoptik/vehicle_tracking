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
Copy test data to build folder:
```
cp -r ../../computer_vision_basics_data/vehicle_tracking/build/* .
```

Run vehicle tracking:
```
./vehicle_tracking <video_file>
```
