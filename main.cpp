#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]){
    if(argc != 2){
        std::cout << "Please provide input image path" << std::endl;
        return 0;
    }

    const std::string img_path(argv[1]);

    cv::Mat img = cv::imread(img_path);
    std::cout << "img width: " << img.size().width << std::endl;
    std::cout << "img height: " << img.size().height << std::endl;
    std::cout << "img channels: " << img.channels() << std::endl;

    // calculate optimal chroma threshold
    int chromas_sum = 0;
    uint8_t* pixelPtr = img.data;
    std::vector<int> U(img.rows * img.cols);
    std::vector<int> C(img.rows * img.cols);
    std::vector<int> V(img.rows * img.cols);
    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.cols; ++j){
            int blue  = pixelPtr[i * img.cols * img.channels() + j * img.channels() + 0];
            int green = pixelPtr[i * img.cols * img.channels() + j * img.channels() + 1];
            int red   = pixelPtr[i * img.cols * img.channels() + j * img.channels() + 2];

            int u = std::min(std::min(blue, green), red);
            U[i * img.cols + j] = u;

            int v = std::max(std::max(blue, green), red);
            V[i * img.cols + j] = v;

            int c = v - u;
            C[i * img.cols + j] = c;
            chromas_sum += c;
       }
    }

    int total_pixel_count = img.size().width * img.size().height;
    float chromas_mean = static_cast<float>(chromas_sum) / total_pixel_count;

    float optimal_threshold = std::max(255.0 / 8, 255.0 / 4 - chromas_mean);
    std::cout << optimal_threshold << std::endl;

    // filtering rgb image
    cv::Mat output(img.size(), img.type());
    for(int i = 0; i < img.rows; ++i){
        for (int j = 0; j < img.cols; ++j){
            int blue  = pixelPtr[i * img.cols * img.channels() + j * img.channels() + 0];
            int green = pixelPtr[i * img.cols * img.channels() + j * img.channels() + 1];
            int red   = pixelPtr[i * img.cols * img.channels() + j * img.channels() + 2];
            int mean_intensity = (blue + green + red) / 3;

            if(C[i * img.cols + j] < optimal_threshold){
                red = green = blue = mean_intensity;
            }
            else if(C[i * img.cols + j] >= optimal_threshold){
                // update red channel
                if(red == V[i * img.cols + j]){
                    red = 255;
                }
                else if(red == U[i * img.cols + j]){
                    red = 0;
                }
                else{
                    red = (red - U[i * img.cols + j]) / std::abs(green - blue);
                }

                // update green channels
                if(green == V[i * img.cols + j]){
                    green = 255;
                }
                else if(green == U[i * img.cols + j]){
                    green = 0;
                }
                else{
                    green = (green - U[i * img.cols + j]) / std::abs(red - blue);
                }

                // update blue channel
                if(blue == V[i * img.cols + j]){
                    blue = 255;
                }
                else if(blue == U[i * img.cols + j]){
                    blue = 0;
                }
                else{
                    blue = (blue - U[i * img.cols + j]) / std::abs(red - green);
                }
            }

            cv::Vec3b pixel(blue, green, red);
            output.at<cv::Vec3b>(i, j) = pixel;
        }
    }

    cv::imwrite("CCE.png", output);
    return 0;
}