# chainer_wrapper_cpp

Preparation (for Ubuntu16.04, Python2.7)
- sudo apt install libboost-all-dev python-pip python-sklearn
- pip install pillow chainer cupy --user
- git clone https://github.com/ndarray/Boost.NumPy.git
    - cd Boost.NumPy
    - cmake .
    - make && sudo make install
    - echo /usr/local/lib | sudo tee /etc/ld.so.conf.d/usr-local-lib.conf
    - echo /usr/local/lib64 | sudo tee /etc/ld.so.conf.d/usr-local-lib64.conf
    - sudo ldconfig

Build
```
git clone https://github.com/kiyoshiiriemon/chainer_wrapper_cpp.git
cd chainer_wrapper_cpp
mkdir build
cd build
cmake ../
cd ..
```

Exec
```
./build/wl_cpp predict_wl.py *.jpg
eog *.png
```
