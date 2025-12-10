#pragma once
#include "export_macro.h"
#if defined(_MSC_VER) && _MSC_VER >= 1600
#pragma execution_character_set("utf-8")
#endif

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "base64.h"
#include <chrono>
namespace fs = std::filesystem;
using json = nlohmann::json;

using cv::Mat;
using cv::Rect;
using cv::Scalar;
using std::string;
using std::vector;

constexpr size_t MAX_CHAR_NUM = 256;
namespace Constants {
    const std::vector<cv::Scalar> COLOR = {
            Scalar(255, 0, 0), Scalar(255, 0, 255), Scalar(170, 0, 255), Scalar(255, 0, 85),
            Scalar(255, 0, 170), Scalar(85, 255, 0), Scalar(255, 170, 0), Scalar(0, 255, 0),
            Scalar(255, 255, 0), Scalar(0, 255, 85), Scalar(170, 255, 0), Scalar(0, 85, 255),
            Scalar(0, 255, 170), Scalar(0, 0, 255), Scalar(0, 255, 255), Scalar(85, 0, 255)
    };
    //const std::vector<std::string> CLASS_NAMES = {
    //	"0lq_md", "1lq_zd", "2lq_bs", "3lq_zm", "4jq_chl", "5jq_qs", "6jq_md", "7jq_zm", "8hq_zd", "9hq_ym",
    //	"10logo_ycqs", "11logo_ycd", "12logo_ycxh", "13logo_ztqy", "14logo_zd", "15logo_qx", "16logo_xy", "17yj", "18logo_xjdc", "19hh",
    //	"20logo_ty", "21logo_ztyc", "22Lq_zm", "23x", "24x1", "25x2", "26md", "27slt", "28hongd", "29ms_bq",
    //	"30ms_lb", "31md_yj", "32ms_qj", "33x3", "34zheheng","35x4", "36heid", "37hswz","38pk", "39slt_fz", "40tm_yj", "41", "42", "43", "44", "45", "46", "47",
    //	"48", "49", "50", "51", "52", "53", "54", "55", "56", "57",
    //	"58", "59", "60", "61", "62tv", "63laptop", "64mouse", "65remote", "66keyboard", "67cell",
    //	"68microwave", "69oven", "70toaster", "71sink", "72refrigerator", "73book", "74clock", "75vase", "76scissors", "77teddy bear",
    //	"78drier", "79toothbrush"
    //};
    const std::vector<std::string> CLASS_NAMES = {
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "s snowboard", "sports ball", "kite", "baseball bat", "baseboard", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };
}

struct YOLO_API CharRect {
    int x, y, width, height;
};
struct YOLO_API CharRect_txt {
    float x, y, width, height;
};
struct YOLO_API RowInfo {
    int count;
    float first_x, first_y;
};
struct YOLO_API Result_Data_Od {
    int char_num = 0;
    int index_list[MAX_CHAR_NUM]{};
    CharRect char_region[MAX_CHAR_NUM]{};
    CharRect_txt char_region_txt[MAX_CHAR_NUM]{};
    float cls_score[MAX_CHAR_NUM]{};
    std::vector<RowInfo> outputs;
    std::string dTime;
    cv::Mat resultImage;
};
struct YOLO_API Result_Seg {
    std::string dTime;
    cv::Rect box;
    float score = 0.0f;
    int class_id = 1;
    std::string class_name;
    float box_area = 0.0f;
    float mask_area = 0.0f;
    cv::Mat resultImage;
};
struct YOLO_API Updata_Infos {
    float config = 0.25f;
    float Nms_Threshold = 0.45f;
    int yolo_model_Size = 1280;
    int targetSize = 640;
    int FirstPixelValue = 0;
    float scaleFactor = 1.0f;
    int rotation = 0;
    int Cut_x = 0, Cut_y = 0, Cut_w = 0, Cut_h = 0;
    int Top_Update = 0, Bottom_Update = 0, Right_Update = 0, Left_Update = 0;
    int id = 0;
    int New_Id = 0;
    cv::Rect cropRect;
    std::string Image_Path = "";
    std::string Real_RST_Name = "";
    std::string Save_Path = "";
    std::string Model_Path = "";
    std::string Sub_Directory = "";
    cv::Scalar fillColor = cv::Scalar(0, 0, 0);
    bool Src_cut_Img = false;
    bool Src_Gray = false;
    bool Save_Gray = false;
    bool Draw_Save = false;
    bool Src_Save = false;
    bool Txt_Save = false;
    bool Samll_Img_Save = false;
    bool Use_Old_Name = false;
    bool Use_Sub_Directory = true;
    bool Show_Conf_Details = false;
    bool ID_Forced_Update = false;
    bool Name_Forced_Update = false;
    bool cropImage = false;
    bool convertToGray = false;
};
struct YOLO_API Base_infos {
    int yolo_model_Size_h;
    int yolo_model_Size_w = yolo_model_Size_h;
    std::vector<cv::Scalar> color = {
            Scalar(255, 0, 0), Scalar(255, 0, 255), Scalar(170, 0, 255), Scalar(255, 0, 85),
            Scalar(255, 0, 170), Scalar(85, 255, 0), Scalar(255, 170, 0), Scalar(0, 255, 0),
            Scalar(255, 255, 0), Scalar(0, 255, 85), Scalar(170, 255, 0), Scalar(0, 85, 255),
            Scalar(0, 255, 170), Scalar(0, 0, 255), Scalar(0, 255, 255), Scalar(85, 0, 255)
    };
    std::vector<std::string> class_names = {
            "0lq_md", "1lq_zd", "2lq_bs", "3lq_zm", "4jq_chl", "5jq_qs", "6jq_md", "7jq_zm", "8hq_zd", "9hq_ym",
            "10logo_ycqs", "11logo_ycd", "12logo_ycxh", "13logo_ztqy", "14logo_zd", "15logo_qx", "16logo_xy", "17yj", "18logo_xjdc", "19hh",
            "20logo_ty", "21logo_ztyc", "22Lq_zm", "23x", "24x1", "25x2", "26md", "27slt", "28hongd", "29ms_bq",
            "30ms_lb", "31md_yj", "32ms_qj", "33x3", "34zheheng", "35x4", "36heid", "37hswz", "38pk", "39slt_fz", "40tm_yj", "41", "42", "43", "44", "45", "46", "47",
            "48", "49", "50", "51", "52", "53", "54", "55", "56", "57",
            "58", "59", "60", "61", "62tv", "63laptop", "64mouse", "65remote", "66keyboard", "67cell",
            "68microwave", "69oven", "70toaster", "71sink", "72refrigerator", "73book", "74clock", "75vase", "76scissors", "77teddy bear",
            "78drier", "79toothbrush"
    };
    std::string yolo_detect_styles;
    std::string GPU_Styles;
    std::string log_Path = "";
};
struct YOLO_API InitDataInfo {
    Updata_Infos updata_infos;
    Result_Data_Od result_data_od;
    Result_Seg result_seg;
    Base_infos base_infos;
};

class YOLO_API IYoloModel {


public:
    virtual ~IYoloModel() = default;
    ov::Core core;
    ov::InferRequest infer_request_Detect;
    ov::CompiledModel compiled_model_Detect;
    ov::InferRequest infer_request_Cls;
    ov::CompiledModel compiled_model_Detect_Cls;
    ov::InferRequest infer_request_Seg;
    ov::CompiledModel compiled_model_Seg;
    ov::InferRequest infer_request_Pose;
    ov::CompiledModel compiled_model_Pose;
    //virtual bool LoadModel(const std::string& xmlName, const std::string& device, const std::string& task) = 0;
    virtual bool LoadModel(const InitDataInfo& info) = 0;

    virtual bool YoloDetectInfer( Mat& src,  InitDataInfo& info) = 0;
    virtual bool YoloPoseInfer( Mat& src,  InitDataInfo& info) = 0;
    virtual bool YoloClsInfer( Mat& src,  InitDataInfo& info) = 0;
    virtual bool YoloSegInfer(Mat& src, InitDataInfo& info) = 0;
};

extern "C" YOLO_API IYoloModel* CreateYoloModel();
extern "C" YOLO_API void DestroyYoloModel(IYoloModel* model);

