#include "yolo_openVino.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <algorithm>
#include <cctype> // 添加这个头文件用于isspace
// 在src/main.cpp开头添加
#include <opencv2/core/utils/logger.hpp>
// 从 TXT 文件读取配置
// 先定义trim函数（关键修改点）
// 安全的 trim 函数实现

static inline std::string trim(const std::string& s) {
    auto start = s.begin();
    auto end = s.end();

    // 找到第一个非空格字符
    while (start != end && std::isspace(*start)) {
        start++;
    }

    // 如果是空字符串，直接返回
    if (start == end) {
        return "";
    }

    // 找到最后一个非空格字符
    do {
        end--;
    } while (end != start && std::isspace(*end));

    return std::string(start, end + 1);
}

bool loadConfigFromTxt(const std::string& filePath, InitDataInfo& config) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open config file: " << filePath << std::endl;
        return false;
    }

    std::map<std::string, std::string> configMap;
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        size_t equalPos = line.find('=');
        if (equalPos != std::string::npos) {
            std::string key = trim(line.substr(0, equalPos));
            std::string value = trim(line.substr(equalPos + 1));
            if (!key.empty()) {
                configMap[key] = value;
            }
        }
    }

    try {
        // 使用count检查键是否存在，并提供默认值
        if (configMap.count("log_Path")) {
            config.base_infos.log_Path = configMap["log_Path"];
        }
        else {
            std::cerr << "Warning: image_path not found, using default" << std::endl;
            config.base_infos.log_Path = "./log.txt";
         
        }
        // 使用count检查键是否存在，并提供默认值
        if (configMap.count("model_path")) {
            config.updata_infos.Model_Path = configMap["model_path"];
   
        }
        else {
            std::cerr << "Warning: image_path not found, using default" << std::endl;
            config.updata_infos.Model_Path = "";

        }

        if (configMap.count("image_path")) {
            config.updata_infos.Image_Path = configMap["image_path"];
        }
        else {
            std::cerr << "Warning: image_path not found, using default" << std::endl;
            config.updata_infos.Image_Path = "";

        }

        if (configMap.count("save_path")) {
            config.updata_infos.Save_Path = configMap["save_path"];
   
        }
        else {
            std::cerr << "Warning: save_path not found, using default" << std::endl;
            config.updata_infos.Save_Path = "";

        }

        if (configMap.count("yolo_detect_styles")) {
            config.base_infos.yolo_detect_styles = configMap["yolo_detect_styles"];
        }
        else {
            std::cerr << "Warning: detect_style not found, using default" << std::endl;
            config.base_infos.yolo_detect_styles = "";
           
        }

        if (configMap.count("GPU_Style")) {
            config.base_infos.GPU_Styles = configMap["GPU_Style"];
        }
        else {
            std::cerr << "Warning: GPU_Style not found, using default" << std::endl;
            config.base_infos.GPU_Styles = "CPU";
     
        }

        // 数值转换
        if (configMap.count("config_threshold")) {
            try {
                config.updata_infos.config = std::stod(configMap["config_threshold"]);
            }
            catch (const std::exception& e) {
                std::cerr << "Error parsing config_threshold, using default: " << e.what() << std::endl;
                config.updata_infos.config = 0.5;
            }
        }
        else {
            std::cerr << "Warning: config_threshold not found, using default" << std::endl;
            config.updata_infos.config = 0.5;
       
        }

        if (configMap.count("nms_threshold")) {
            try {
                config.updata_infos.Nms_Threshold = std::stod(configMap["nms_threshold"]);
            }
            catch (const std::exception& e) {
                std::cerr << "Error parsing nms_threshold, using default: " << e.what() << std::endl;
                config.updata_infos.Nms_Threshold = 0.45;
            }
        }
        else {
            std::cerr << "Warning: nms_threshold not found, using default" << std::endl;
            config.updata_infos.Nms_Threshold = 0.45;
       
        }

        if (configMap.count("target_size")) {
            try {
                config.updata_infos.targetSize = std::stoi(configMap["target_size"]);
            }
            catch (const std::exception& e) {
                std::cerr << "Error parsing target_size, using default: " << e.what() << std::endl;
                config.updata_infos.targetSize = 1280;
            }
        }
        else {
            std::cerr << "Warning: target_size not found, using default" << std::endl;
            config.updata_infos.targetSize = 1280;
     
        }
        if (configMap.count("yolo_model_Size")) {
            try {
                config.updata_infos.yolo_model_Size = std::stoi(configMap["yolo_model_Size"]);
            }
            catch (const std::exception& e) {
                std::cerr << "Error parsing yolo_detect_styles, using default: " << e.what() << std::endl;
                config.updata_infos.yolo_model_Size = 1280;
            }
        }
        else {
            std::cerr << "Warning: yolo_detect_styles not found, using default" << std::endl;
            config.updata_infos.yolo_model_Size = 1280;
        }

        if (configMap.count("model_size_h")) {
            try {
                config.base_infos.yolo_model_Size_h = std::stoi(configMap["model_size_h"]);
            }
            catch (const std::exception& e) {
                std::cerr << "Error parsing model_size_h, using default: " << e.what() << std::endl;
                config.base_infos.yolo_model_Size_h = config.updata_infos.yolo_model_Size ;
            }
        }
        else {
            std::cerr << "Warning: model_size_h not found, using default" << std::endl;
            config.base_infos.yolo_model_Size_h = config.updata_infos.yolo_model_Size ;
       
        }

        if (configMap.count("model_size_w")) {
            try {
                config.base_infos.yolo_model_Size_w = std::stoi(configMap["model_size_w"]);
            }
            catch (const std::exception& e) {
                std::cerr << "Error parsing model_size_w, using default: " << e.what() << std::endl;
                config.base_infos.yolo_model_Size_w = config.updata_infos.yolo_model_Size ;
            }
        }
        else {
            std::cerr << "Warning: model_size_w not found, using default" << std::endl;
            config.base_infos.yolo_model_Size_w = config.updata_infos.yolo_model_Size ;
        }


        if (configMap.count("scaleFactor")) {
            try {
                config.updata_infos.scaleFactor = std::stof(configMap["scaleFactor"]);
            }
            catch (const std::exception& e) {
                std::cerr << "Error parsing yolo_detect_styles, using default: " << e.what() << std::endl;
                config.updata_infos.yolo_model_Size = 1.0;
            }
        }
        else {
            std::cerr << "Warning: scaleFactor not found, using default" << std::endl;
            config.updata_infos.scaleFactor = 1.0;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error parsing config values: " << e.what() << std::endl;
        return false;
    }

    return true;
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR); // 只显示 ERROR
    // 创建模型实例
    IYoloModel* yolomodel = CreateYoloModel();
    // 初始化参数
    bool is_Video = 0;
    cv::Scalar paddingColor = cv::Scalar(255, 255, 255);
  
    InitDataInfo info;
    std::filesystem::path current_path = std::filesystem::current_path();
    std::cout << "当前路径: " << current_path << std::endl;
    if (!loadConfigFromTxt("../../sysparam/config.txt", info)) {
        std::cerr << "Failed to load config. Using default values." << std::endl;
        return 1;
    }
    if (info.base_infos.yolo_detect_styles == "od")
    {
        bool initDetectflag = yolomodel->LoadModel(info);
    }
    else if (info.base_infos.yolo_detect_styles == "cls")
    {
     bool initClsflag = yolomodel->LoadModel(info);
    }
    else if (info.base_infos.yolo_detect_styles == "pose")
    {
        bool initPoseflag = yolomodel->LoadModel(info);
    }
    else if (info.base_infos.yolo_detect_styles == "seg")
    {
        bool initSegflag = yolomodel->LoadModel(info);
    }
    else {
        std::cerr << "Invalid task: " << info.base_infos.yolo_detect_styles << "\n";
        return 1;
    }

    if (!is_Video)
    {
        std::vector<std::string> m_vecFileList; // 改为成员变量
        // 先检查路径是否存在
        if (!std::filesystem::exists(info.updata_infos.Image_Path)) {
            std::cerr << "Error: Path does not exist: " << info.updata_infos.Image_Path << std::endl;
            return 1 ;
        }
        cv::glob(info.updata_infos.Image_Path, m_vecFileList, false);
        // 添加路径检查
        if (m_vecFileList.empty()) {
            std::cerr << "Error: No files found in path: " << info.updata_infos.Image_Path
                << " (or path does not exist)" << std::endl;
            return 1;  // 或者返回错误码，根据你的函数返回类型决定
        }

        for (size_t i = 0; i < m_vecFileList.size(); ++i) {
            std::string image_name = m_vecFileList[i];
            std::filesystem::path p(image_name);
            info.updata_infos.New_Id = i;
            info.updata_infos.Real_RST_Name = p.stem().string(); ;
            std::cout << image_name << std::endl;
            cv::Mat resizedImg = cv::imread(image_name);
            if (resizedImg.empty()) {
                throw std::runtime_error("无法读取图像: " + image_name);
            }
            try {
                if (info.base_infos.yolo_detect_styles == "od")
                {
                    bool InferDetectflag = yolomodel->YoloDetectInfer(resizedImg, info);
                    std::cout<<"目标检测"<<std::endl;
                    // 创建可调整大小的窗口
                    cv::namedWindow("Instance od Results", cv::WINDOW_NORMAL);
                    cv::imshow("Instance od Results", info.result_data_od.resultImage);
                    cv::waitKey(10);
                }
                else if (info.base_infos.yolo_detect_styles == "cls") {
                    bool InferClsflag = yolomodel->YoloClsInfer(resizedImg, info);
                    std::cout<<"分类检测"<<std::endl;
                    // 创建可调整大小的窗口
                    cv::namedWindow("Instance Cls Results", cv::WINDOW_NORMAL);
                    cv::imshow("Instance Segmentation Results", info.result_seg.resultImage);
                    cv::waitKey(10);
                }
                else if (info.base_infos.yolo_detect_styles == "pose") {
                    bool InferPoseflag = yolomodel->YoloPoseInfer(resizedImg, info);
                    std::cout<<"姿态检测"<<std::endl;
                    // 创建可调整大小的窗口
                    cv::namedWindow("Instance Pose Results", cv::WINDOW_NORMAL);
                    cv::imshow("Instance Segmentation Results", info.result_seg.resultImage);
                    cv::waitKey(10);
                }
                else if (info.base_infos.yolo_detect_styles == "seg") {
                    bool InferSegflag = yolomodel->YoloSegInfer(resizedImg, info);
                    std::cout<<"语义分割"<<std::endl;
                    // 创建可调整大小的窗口
                    cv::namedWindow("Instance Segmentation Results", cv::WINDOW_NORMAL);
                    cv::imshow("Instance Segmentation Results", info.result_seg.resultImage);
                    cv::waitKey(10);
                }
                else {
                    std::cerr << "Invalid task: " << info.base_infos.yolo_detect_styles << "\n";
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
                return 1;
            }
        }
    }
    cv::waitKey(0);
    // 销毁模型
    DestroyYoloModel(yolomodel);
    return 0;
}