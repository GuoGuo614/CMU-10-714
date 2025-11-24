#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (size_t i = 0; i < m; i += batch) {
        const float *X_batch = X + i * n;
        const unsigned char *y_batch = y + i;
        size_t batch_size = (m - i < batch) ? (m - i) : batch;

        float Z[batch_size * k];
        mul_martix(X_batch, theta, Z, batch_size, n, k);

        float softmax[batch_size * k];
        for (size_t row = 0; row < batch_size; row++) {
            float sum = 0.0f;
            for (size_t col = 0; col < k; col++) {
                softmax[row * k + col] = expf(Z[row * k + col]);
                sum += softmax[row * k + col];
            }

            for (size_t col = 0; col < k; col++) {
                softmax[row * k + col] /= sum;
            }

            softmax[row * k + y_batch[row]] -= 1.0f;
        }

        float X_batch_T[n * batch_size];
        T_martix(X_batch, X_batch_T, batch_size, n);
        float gradient[n * k];
        mul_martix(X_batch_T, softmax, gradient, n, batch_size, k);
        for (size_t j = 0; j < n * k; i++) {
            theta[j] -= lr * (gradient[j] / batch_size);
        }
    }
    /// END YOUR CODE
}

void T_martix(const float *src, float *dst, size_t m, size_t n) {
    for (size_t row = 0; row < m; row++) {
        for (size_t col = 0; col < n; col++) {
            dst[col * m + row] = src[row * n + col];
        }
    }
}

void mul_martix(const float *src1, const float *src2, float *dst, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m * n; i++) {
        dst[i] = 0.0f;
    }
    
    for (size_t i = 0; i < m; i++) {
        for (size_t l = 0; l < k; l++) {
            float a = src1[i * k + l];
            for (size_t j = 0; j < n; j++) {
                dst[i * n + j] += a * src2[l * n + j];
            }
        }
    }
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
