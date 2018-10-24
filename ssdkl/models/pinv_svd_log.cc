#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "Eigen/SVD"
#include <cmath>

REGISTER_OP("PseudoInverse")
    .Input("mat: float")
    .Output("mat_inv: float")
    .Output("mat_det: float");
REGISTER_OP("PseudoInverseNoDet")
    .Input("mat: float")
    .Output("mat_inv: float");

using namespace tensorflow;

class PseudoInverseOp : public OpKernel {
    public:
        explicit PseudoInverseOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& input_tensor = context->input(0);
            auto input = input_tensor.matrix<float>();
            Eigen::MatrixXf matrix_input(input_tensor.shape().dim_size(0), input_tensor.shape().dim_size(1));

            //copy brute
            for(int i = 0; i < input_tensor.shape().dim_size(0); i++) {
                for(int j = 0; j < input_tensor.shape().dim_size(1); j++) {
                    matrix_input(i,j) = input(i,j);
                }
            }

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                             &output_tensor));
            auto output = output_tensor->template matrix<float>();

            Eigen::JacobiSVD<Eigen::MatrixXf> svd(matrix_input, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const int N = svd.singularValues().size();
            float log_determinant = 0.0;
            Eigen::VectorXf inv_singularValues(N);
            for(int i = 0; i < N; i++) {
                float s_i = svd.singularValues()(i);
                if(s_i < 0.000001) {
                    inv_singularValues(i) = 0.0;
                }
                else {
                    log_determinant += log(s_i);
                    inv_singularValues(i) = 1.0 / s_i;
                }
            }

            // by convention, if all singular values are 0, then
            // determinant is 1
            if(log_determinant == 0.0) {
                log_determinant = 1.0;
            }


            Eigen::MatrixXf matrix_output = svd.matrixV() * inv_singularValues.asDiagonal() * svd.matrixU().transpose();
            for(int i = 0; i < input_tensor.shape().dim_size(0); i++) {
                for(int j = 0; j < input_tensor.shape().dim_size(1); j++) {
                    output(i,j) = matrix_output(i,j);
                }
            }

            // create an output tensor
            Tensor* det_tensor = NULL;
            TensorShape shape; //default is 1 element
            shape.AddDim(1);
            shape.AddDim(1);
            OP_REQUIRES_OK(context, context->allocate_output(1, shape, &det_tensor));
            auto det = det_tensor->template matrix<float>();
            det(0,0) = log_determinant;
        }
    };

class PseudoInverseOpNoDet : public OpKernel {
    public:
        explicit PseudoInverseOpNoDet(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& input_tensor = context->input(0);
            auto input = input_tensor.matrix<float>();
            Eigen::MatrixXf matrix_input(input_tensor.shape().dim_size(0), input_tensor.shape().dim_size(1));

            //copy brute
            for(int i = 0; i < input_tensor.shape().dim_size(0); i++) {
                for(int j = 0; j < input_tensor.shape().dim_size(1); j++) {
                    matrix_input(i,j) = input(i,j);
                }
            }

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                             &output_tensor));
            auto output = output_tensor->template matrix<float>();

            Eigen::JacobiSVD<Eigen::MatrixXf> svd(matrix_input, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const int N = svd.singularValues().size();
            Eigen::VectorXf inv_singularValues(N);
            for(int i = 0; i < N; i++) {
                float s_i = svd.singularValues()(i);
                if(s_i < 0.000001) {
                    inv_singularValues(i) = 0.0;
                }
                else {
                    inv_singularValues(i) = 1.0 / s_i;
                }
            }

            Eigen::MatrixXf matrix_output = svd.matrixV() * inv_singularValues.asDiagonal() * svd.matrixU().transpose();
            for(int i = 0; i < input_tensor.shape().dim_size(0); i++) {
                for(int j = 0; j < input_tensor.shape().dim_size(1); j++) {
                    output(i,j) = matrix_output(i,j);
                }
            }
        }
    };
REGISTER_KERNEL_BUILDER(Name("PseudoInverse").Device(DEVICE_CPU), PseudoInverseOp);
REGISTER_KERNEL_BUILDER(Name("PseudoInverseNoDet").Device(DEVICE_CPU), PseudoInverseOpNoDet);
