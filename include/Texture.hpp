//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        if (u < 0) u = 0;
        if (u > 1) u = 1;
        if (v < 0) v = 0;
        if (v > 1) v = 1;
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    Eigen::Vector3f getColorBilinear(float u, float v) {
        float u_img = u * width;
        float v_img = (1 - v) * height;
        // center point
        float cu = (u_img - floor(u_img)) > 0.5 ? std::ceil(u_img) : std::floor(u_img);
        float cv = (v_img - floor(v_img)) > 0.5 ? std::ceil(v_img) : std::floor(v_img);
        //
        float u0 = std::max(0.f, cu - 0.5f), v0 = std::max(0.f, cv - 0.5f);
        float u1 = std::min((float)width, cu + 0.5f), v1 = std::min((float)height, cv + 0.5f);
        //
        auto color00 = image_data.at<cv::Vec3b>(v0, u0);
        auto color01 = image_data.at<cv::Vec3b>(v1, u0);
        auto color10 = image_data.at<cv::Vec3b>(v0, u1);
        auto color11 = image_data.at<cv::Vec3b>(v1, u1);
        //
        float s = u_img - u0, t = v_img - v0;
        //
        Eigen::Vector3f color0 = (1 - s) * Eigen::Vector3f(color00[0], color00[1], color00[2]) + s * Eigen::Vector3f(color10[0], color10[1], color10[2]);
        Eigen::Vector3f color1 = (1 - s) * Eigen::Vector3f(color01[0], color01[1], color01[2]) + s * Eigen::Vector3f(color11[0], color11[1], color11[2]);
        Eigen::Vector3f color = (1 - t) * color0 + t * color1;
        return color;
    }

};
#endif //RASTERIZER_TEXTURE_H
