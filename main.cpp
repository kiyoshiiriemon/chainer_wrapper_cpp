
#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB

#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <list>
#include <chrono>
#include <opencv2/opencv.hpp>


namespace np = boost::numpy;

static const int IMH = 240;
static const int IMW = 320;

int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cout << "spacify python script" << std::endl;
        return 0;
    }

    Py_Initialize();
    np::initialize();

    auto main_ns = boost::python::import("__main__").attr("__dict__");

    std::ifstream ifs(argv[1]);
    std::string script((std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>());

    // load model
    boost::python::exec(script.c_str(), main_ns);

    // predict
    auto func = main_ns["predict"];

    boost::python::tuple shape = boost::python::make_tuple(IMH, IMW, 3);
    np::ndarray A = np::zeros(shape, np::dtype::get_builtin<unsigned char>());
    //for(int k=0; k < 10; ++k) {
    for(int k=2; k < argc; ++k) {
        std::string fname = argv[k];
        cv::Mat mat = cv::imread(fname);

        auto start = std::chrono::system_clock::now();
        cv::resize(mat, mat, cv::Size(IMW, IMH));
        unsigned char *data = mat.data;
        unsigned char *adata = reinterpret_cast<unsigned char*>(A.get_data());
        memcpy(adata, data, sizeof(unsigned char) * IMH*IMW*3);
#if 0
        for (int i = 0; i < IMH; i++) {
            for (int j = 0; j < IMW; j++) {
                for (int c = 0; c < 3; c++) {
                    //A[i][j][c] = data[i * mat.step + j * 3 + c];
                    adata[i * mat.step + j * 3 + c] = data[i * mat.step + j * 3 + c];
                }
            }
        }
#endif

        auto start_call = std::chrono::system_clock::now();
        auto ret = func(A);
        np::ndarray arr = boost::python::extract<np::ndarray>(ret);
        //std::cout << arr.get_shape() << std::endl;

        auto end = std::chrono::system_clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        double elapsed_py = std::chrono::duration_cast<std::chrono::milliseconds>(end-start_call).count();
        std::cout << "python time ms : " << elapsed_py << std::endl;
        std::cout << "elapsed ms : " << elapsed_ms << std::endl;

        int *p = reinterpret_cast<int *>(arr.get_data());
        for (int i = 0; i < IMH; i++) {
            for (int j = 0; j < IMW; j++) {
                for (int c = 0; c < 3; c++) {
                    int v = p[i*IMW+j];
                    if (v > 0) {
                        data[i * mat.step + j * 3 + c] = 255;
                    } else {
                        data[i * mat.step + j * 3 + c] = 0;
                    }
                }
            }
        }
        cv::imwrite(fname + "_out.png", mat);
    }

    return 0;
}

