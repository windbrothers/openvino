#include "yolo_openVino.h"
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
class YoloModelImpl : public IYoloModel {
public:
    YoloModelImpl() = default;
    ~YoloModelImpl() override = default;
    bool LoadModel(const InitDataInfo& info)override;
    bool YoloDetectInfer( Mat& src,  InitDataInfo& info)override;
    bool YoloPoseInfer( Mat& src,  InitDataInfo& info)override;
    bool YoloClsInfer( Mat& src,  InitDataInfo& info)override;
    bool YoloSegInfer(Mat& src, InitDataInfo& info) override;
};
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
void saveimg(InitDataInfo& info, const std::string& input_img_name, const Mat& img, int index) {
    fs::path save_path(info.updata_infos.Save_Path);
    save_path /= info.updata_infos.Sub_Directory;
    if (!fs::exists(save_path)) {fs::create_directories(save_path);
    }
    std::string img_path = save_path.string() + "/" + info.updata_infos.Real_RST_Name + ".png";
    imwrite(img_path, img);
}
static Mat letterbox(const Mat& source, const cv::Size& target_size) {
    const int col = source.cols;
    const int row = source.rows;
    const float scale = std::min(
            static_cast<float>(target_size.width) / col,
            static_cast<float>(target_size.height) / row);
    const int new_width = static_cast<int>(col * scale);
    const int new_height = static_cast<int>(row * scale);
    Mat resized;
    resize(source, resized, cv::Size(new_width, new_height));
    Mat result = Mat::zeros(target_size.height, target_size.width, CV_8UC3);
    const int x_offset = (target_size.width - new_width) / 2;
    const int y_offset = (target_size.height - new_height) / 2;
    resized.copyTo(result(Rect(x_offset, y_offset, new_width, new_height)));
    return result;
}

inline bool fileExists(const fs::path& filePath) {
    return fs::exists(filePath) && fs::is_regular_file(filePath);
}
inline float float_format(float num) {
    std::stringstream ss;
    ss.precision(6);
    ss.setf(std::ios::fixed);
    ss << num;
    return std::stof(ss.str());
}
void writeDetectionLog(const fs::path& savePath,
                       const fs::path& subDir,
                       const std::string& filename,
                       int label,
                       float x, float y, float width, float height,
                       const std::string& index) {
    fs::path logDir = savePath / subDir /"txt";
    fs::create_directories(logDir);
    std::string finalName = filename;
    /*  if (!index.empty()) {
          finalName += "_" + index;
      }*/
    finalName += ".txt";
    std::ofstream ofs(logDir / finalName, std::ios::app);
    if (!ofs) {
        throw std::runtime_error("Failed to open log file: " + (logDir / finalName).string());
    }
    ofs << label << " "
        << std::fixed << std::setprecision(6) << x << " "
        << y << " " << width << " " << height << "\n";
}
std::vector<cv::Point> simplifyContour(const std::vector<cv::Point>& contour, double epsilon_factor = 0.005) {
    if (contour.size() < 3) return {};
    std::vector<cv::Point> approx;
    double epsilon = epsilon_factor * arcLength(contour, true);
    approxPolyDP(contour, approx, epsilon, true);
    return approx;
}
fs::path prepareSavePath(const std::string& base_path) {
    fs::path save_path(base_path);
    save_path /= "src";
    if (!fs::exists(save_path)) {
        fs::create_directories(save_path);
    }
    return save_path;
}
void saveAllResults(InitDataInfo& info,
                    const Mat& final_result,
                    const Mat& mask_overlay,
                    const Mat& contour_image,
                    const Mat& original_src) {
    std::vector<std::pair<std::string, Mat>> save_images = {
            {"seg", final_result},
            {"seg_mask", mask_overlay},
            {"contour_image", contour_image},
            {"src", original_src}
    };
    int index = 0;
    for (const auto& [dir, img] : save_images) {
        info.updata_infos.Sub_Directory = dir;
        saveimg(info, info.updata_infos.Real_RST_Name, img, index++);
    }
}

static void draw_seg(Mat& img,
                     const Rect& box,
                     float score,
                     int class_id,
                     const Scalar& color,
                     float box_area,
                     InitDataInfo& info
) {
    std::string label;
    if (info.base_infos.yolo_detect_styles == "seg")
    {
        label = Constants::CLASS_NAMES[class_id] + ":"
                + std::to_string(score).substr(0, 4) + " area:" + std::to_string(static_cast<int>(box_area));
    }
    else
    {
        label = Constants::CLASS_NAMES[class_id];
    }

    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
    Rect text_box(box.tl().x, box.tl().y - 30, text_size.width + 10, text_size.height + 10);
    putText(img, label, cv::Point(box.tl().x + 5, box.tl().y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
}

static void  drawDetection( cv::Mat& img,
                            cv::Rect& box,
                            float score,
                            int  classId,
                            const  cv::Scalar& color
) {
    //  cv::Mat result = img.clone();
    cv::rectangle(img, box, color, 1);
    std::ostringstream label;
    label << std::to_string(classId) << ":" << std::fixed << std::setprecision(2) << score;
    /*   cv::putText(result, label.str(), cv::Point(box.tl().x + 5, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX,
                   1, Scalar(155, 10, 10), 1);*/
    cv::Size text_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
    Rect text_box(box.tl().x, box.tl().y - 30, text_size.width + 10, text_size.height + 10);
    putText(img, label.str(), cv::Point(box.tl().x + 5, box.tl().y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
}
std::string matToBase64(const Mat& image) {
    std::vector<uchar> buffer;
    imencode(".png", image, buffer);
    return base64_encode(buffer.data(), buffer.size());
}
bool shapeExists(const json& shapes, const json& new_shape) {
    for (const auto& shape : shapes) {
        if (shape["label"] == new_shape["label"]) {
            return true;
        }
    }
    return false;
}
void generateLabelMeJSONFromOverlay(bool is_add,
                                    const Mat& mask_overlay,
                                    const Mat& original_image,
                                    const std::string& output_json_path,
                                    const std::string& label_name) {
    json json_data;
    std::ifstream in_file(output_json_path);
    if (in_file.good()) {
        is_add = true;
        try {
            in_file >> json_data;
        }
        catch (...) {
            json_data = json::object();
        }
        in_file.close();
    }
    Mat gray, binary;
    cvtColor(mask_overlay, gray, cv::COLOR_BGR2GRAY);
    threshold(gray, binary, 1, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    json shapes;
    if (is_add && json_data.contains("shapes")) {
        shapes = json_data["shapes"];
    }
    for (const auto& contour : contours) {
        auto simplified = simplifyContour(contour);
        if (simplified.size() < 3) continue;
        json shape{
                {"label", label_name},
                {"group_id", nullptr},
                {"shape_type", "polygon"},
                {"flags", json::object()}
        };
        json points = json::array();
        for (const auto& pt : simplified) {
            points.push_back({ pt.x, pt.y });
        }
        shape["points"] = points;
        shapes.push_back(shape);
    }
    if (!is_add) {
        json_data["version"] = "5.3.1";
        json_data["flags"] = json::object();
        json_data["imagePath"] = output_json_path.substr(
                output_json_path.find_last_of("/\\") + 1);
        json_data["imageData"] = matToBase64(original_image);
        json_data["imageHeight"] = original_image.rows;
        json_data["imageWidth"] = original_image.cols;
    }
    json_data["shapes"] = shapes;
    std::ofstream out_file(output_json_path);
    out_file << json_data.dump(2);
    out_file.close();
}
cv::Mat processMask(const Mat& mask_conf, const Mat& proto,
                    const Rect& box, const cv::Size& img_size, float scale) {
    Mat mask = mask_conf * proto;
    for (int j = 0; j < mask.cols; j++) {
        mask.at<float>(0, j) = sigmoid(mask.at<float>(0, j));
    }
    const int base_mask_size = 160;
    const int mask_height = base_mask_size;
    Mat mask_resized;
    resize(mask, mask_resized, cv::Size(base_mask_size * 2 * scale, mask_height));
    const int x1 = std::max(0, box.x);
    const int y1 = std::max(0, box.y);
    const int x2 = std::min(img_size.width - 1, box.br().x);
    const int y2 = std::min(img_size.height - 1, box.br().y);
    const int roi_width = x2 - x1;
    const int roi_height = y2 - y1;
    if (roi_width <= 0 || roi_height <= 0)
        return Mat();
    float coord_scale = scale;
    int start_row = static_cast<int>(y1 * coord_scale);
    int end_row = static_cast<int>(y2 * coord_scale);
    int start_col = static_cast<int>(x1 * coord_scale);
    int end_col = static_cast<int>(x2 * coord_scale);
    start_row = std::max(0, std::min(start_row, mask_resized.rows));
    end_row = std::max(start_row, std::min(end_row, mask_resized.rows));
    start_col = std::max(0, std::min(start_col, mask_resized.cols));
    end_col = std::max(start_col, std::min(end_col, mask_resized.cols));
    if (start_row >= end_row || start_col >= end_col) {
        return Mat();
    }
    Mat mask_roi = mask_resized(cv::Range(start_row, end_row), cv::Range(start_col, end_col));
    Mat resized_mask;
    resize(mask_roi, resized_mask, cv::Size(roi_width, roi_height));
    threshold(resized_mask, resized_mask, 0.5, 255, cv::THRESH_BINARY);
    resized_mask.convertTo(resized_mask, CV_8U);
    return resized_mask;
}
cv::Mat preprocessImage(const cv::Mat& src, const InitDataInfo& params) {
    CV_Assert(!src.empty() && params.updata_infos.targetSize > 0);
    CV_Assert(params.updata_infos.scaleFactor > 0);
    cv::Mat processed = params.updata_infos.cropImage ? src(params.updata_infos.cropRect) : src.clone();
    int maxDim = std::max(processed.rows, processed.cols);
    cv::Mat square(maxDim, maxDim, processed.type(), params.updata_infos.fillColor);
    cv::Rect roi((maxDim - processed.cols) / 2, (maxDim - processed.rows) / 2,
                 processed.cols, processed.rows);
    processed.copyTo(square(roi));
    double scale = static_cast<double>(params.updata_infos.targetSize) / maxDim * params.updata_infos.scaleFactor;
    cv::Mat scaled;
    cv::resize(square, scaled, cv::Size(), scale, scale, cv::INTER_LINEAR);
    int cropW = std::min(scaled.cols, params.updata_infos.targetSize);
    int cropH = std::min(scaled.rows, params.updata_infos.targetSize);
    int cropX = (scaled.cols - cropW) / 2;
    int cropY = (scaled.rows - cropH) / 2;
    cv::Mat cropped = scaled(cv::Rect(cropX, cropY, cropW, cropH));
    cv::Mat result(params.updata_infos.targetSize, params.updata_infos.targetSize, processed.type(), params.updata_infos.fillColor);
    cv::Rect dstRoi((params.updata_infos.targetSize - cropW) / 2, (params.updata_infos.targetSize - cropH) / 2, cropW, cropH);
    cropped.copyTo(result(dstRoi));
    switch (params.updata_infos.rotation) {
        case 90: cv::rotate(result, result, cv::ROTATE_90_CLOCKWISE); break;
        case 180: cv::rotate(result, result, cv::ROTATE_180); break;
        case 270: {
            cv::rotate(result, result, cv::ROTATE_90_COUNTERCLOCKWISE);
            cv::flip(result, result, 1);
            break;
        }
    }
    if (params.updata_infos.convertToGray) {
        cv::Mat gray;
        cv::cvtColor(result, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(gray, result, cv::COLOR_GRAY2BGR);
    }
    return result;
}
bool YoloModelImpl::LoadModel(const InitDataInfo& params) {
    try {
        if (params.base_infos.yolo_detect_styles == "od") {
            compiled_model_Detect = core.compile_model(params.updata_infos.Model_Path, params.base_infos.GPU_Styles);
            infer_request_Detect = compiled_model_Detect.create_infer_request();
        }
        else if (params.base_infos.yolo_detect_styles == "cls") {
            compiled_model_Detect_Cls = core.compile_model(params.updata_infos.Model_Path, params.base_infos.GPU_Styles);
            infer_request_Cls = compiled_model_Detect_Cls.create_infer_request();
        }
        else if (params.base_infos.yolo_detect_styles == "seg") {
            compiled_model_Seg = core.compile_model(params.updata_infos.Model_Path, params.base_infos.GPU_Styles);
            infer_request_Seg = compiled_model_Seg.create_infer_request();
        }
        else if (params.base_infos.yolo_detect_styles == "pose") {
            compiled_model_Pose = core.compile_model(params.updata_infos.Model_Path, params.base_infos.GPU_Styles);
            infer_request_Pose = compiled_model_Pose.create_infer_request();
        }
        else {
            std::cerr << "Invalid task: " << params.base_infos.yolo_detect_styles << std::endl;
            return false;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
    return true;
}
void logDetection(InitDataInfo& info, int class_id, float score, const Rect& box) {
    std::ofstream log_file("log.txt", std::ios::app);
    if (log_file.is_open()) {
        log_file << "名称: " << info.updata_infos.Real_RST_Name
                 << " | ID: " << class_id
                 << " | 置信度: " << score
                 << " | 坐标: ("
                 << box.x << "," << box.y << ","
                 << box.width << "," << box.height << ")"
                 << std::endl;
        log_file.close();
    }
}
bool YoloModelImpl::YoloSegInfer(Mat& src, InitDataInfo& info)
{
    auto start = std::chrono::high_resolution_clock::now();
    int index = 0;
    Mat letterbox_img = preprocessImage(src, info);
    src = letterbox_img.clone();
    float scale = static_cast<float>(letterbox_img.cols) / info.base_infos.yolo_model_Size_h;
    cv::Mat blob = cv::dnn::blobFromImage(letterbox_img, 1.0 / 255.0, cv::Size(info.base_infos.yolo_model_Size_h, info.base_infos.yolo_model_Size_h), Scalar(), true);
    ov::Tensor input_tensor(compiled_model_Seg.input().get_element_type(),
                            compiled_model_Seg.input().get_shape(),
                            blob.ptr(0));
    infer_request_Seg.set_input_tensor(input_tensor);
    infer_request_Seg.infer();
    auto outputs = compiled_model_Seg.outputs();
    if (outputs.size() < 2)
    {
        std::cout << "not seg model,please check 20251105" << std::endl;
        return 1;
    }
    auto output0 = infer_request_Seg.get_output_tensor(0);
    auto output1 = infer_request_Seg.get_output_tensor(1);
    auto output0_shape = output0.get_shape();
    auto output1_shape = output1.get_shape();
    Mat detections(output0.get_shape()[1], output0.get_shape()[2], CV_32F, output0.data<float>());
    transpose(detections, detections);
    const Mat proto(32, output1.get_shape()[2] * output1.get_shape()[2], CV_32F, output1.data<float>());
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<Rect> boxes;
    std::vector<Mat> mask_confs;
    for (int i = 0; i < detections.rows; i++) {
        Mat classes_scores = detections.row(i).colRange(4, 84);
        cv::Point class_id;
        double max_score;
        minMaxLoc(classes_scores, nullptr, &max_score, nullptr, &class_id);
        if (max_score > info.updata_infos.config) {
            class_scores.push_back(static_cast<float>(max_score));
            class_ids.push_back(class_id.x);
            const float cx = detections.at<float>(i, 0);
            const float cy = detections.at<float>(i, 1);
            const float w = detections.at<float>(i, 2);
            const float h = detections.at<float>(i, 3);
            boxes.emplace_back(
                    static_cast<int>((cx - 0.5f * w) * scale),
                    static_cast<int>((cy - 0.5f * h) * scale),
                    static_cast<int>(w * scale),
                    static_cast<int>(h * scale)
            );
            mask_confs.push_back(detections.row(i).colRange(84, 116));
        }
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, class_scores, info.updata_infos.config, info.updata_infos.Nms_Threshold, indices);
    Mat result_img = letterbox_img.clone();
    Mat mask_overlay = Mat::zeros(letterbox_img.size(), letterbox_img.type());
    Mat contour_image = Mat::zeros(letterbox_img.size(), CV_8UC3);
    cv::RNG rng(12345);
    std::cout << "indices.size(): " << indices.size() << std::endl;
    int test_cont = 0;
    for (int idx : indices) {
        const int class_id = class_ids[idx];
        info.updata_infos.id = class_id;
        const Scalar color = Constants::COLOR[class_id % Constants::COLOR.size()];
        const Rect& box = boxes[idx];
        float box_area = box.area();
        draw_seg(result_img, box, class_scores[idx], class_id, color, box_area,info);
        if (box_area > 4) {
            Mat mask = mask_confs[idx] * proto;
            for (int j = 0; j < mask.cols; j++) {
                mask.at<float>(0, j) = sigmoid(mask.at<float>(0, j));
            }
            mask = mask.reshape(1, 160 * info.base_infos.yolo_model_Size_h / 640);
            const int x1 = std::max(0, box.x);
            const int y1 = std::max(0, box.y);
            const int x2 = std::min(letterbox_img.cols, box.br().x);
            const int y2 = std::min(letterbox_img.rows, box.br().y);
            const int roi_width = x2 - x1;
            const int roi_height = y2 - y1;
            if (roi_width <= 0 || roi_height <= 0) continue;
            Mat mask_roi = mask(
                    cv::Range(static_cast<int>(y1 * 0.25f / scale), static_cast<int>(y2 * 0.25f / scale)),
                    cv::Range(static_cast<int>(x1 * 0.25f / scale), static_cast<int>(x2 * 0.25f / scale))
            );
            Mat resized_mask;
            if (!mask_roi.empty())
            {
                resize(mask_roi, resized_mask, cv::Size(roi_width, roi_height));
                threshold(resized_mask, resized_mask, 0.5, 255, cv::THRESH_BINARY);
                resized_mask.convertTo(resized_mask, CV_8U);
                Scalar mask_color(rng.uniform(0, 180), rng.uniform(0, 180), rng.uniform(0, 180));
                mask_overlay(Rect(x1, y1, roi_width, roi_height)).setTo(mask_color, resized_mask);
                int current_mask_area = countNonZero(resized_mask);
                fs::path json_save_path(info.updata_infos.Save_Path);
                json_save_path /= "json";
                if (!fs::exists(json_save_path)) {
                    fs::create_directories(json_save_path);
                }
                fs::path fullPath = json_save_path / (info.updata_infos.Real_RST_Name + ".json");
                bool is_add = fileExists(fullPath);
                std::ofstream log_file(info.base_infos.log_Path, std::ios::app);
                if (log_file.is_open()) {
                    log_file << "名字: " << info.updata_infos.Real_RST_Name << ", ID: " << class_id <<
                             "面积: " << box_area <<
                             ", 置信度: " << class_scores[idx] << ", 坐标 : (" << box.x << ", "
                             << box.y << ", " << box.width << ", " << box.height << ")\n";
                    log_file.close();
                }
                test_cont = test_cont + 1;
                std::cout << " ID: " << class_id << ", 名称: " << class_scores[idx] << ", 面积: (" << box.x << ", "
                          << box.y << ", " << box.width << ", " << box.height << ")" << std::endl;
                vector<vector< cv::Point>> contours;
                vector< cv::Vec4i> hierarchy;
                findContours(resized_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                int t = 0;
                for (const auto& contour : contours) {
                    t = t + 1;
                    vector< cv::Point> global_contour;
                    for (const auto& pt : contour) {
                        global_contour.emplace_back(pt.x + x1, pt.y + y1);
                    }
                    Mat json_img = Mat::zeros(letterbox_img.size(), CV_8UC3);
                    drawContours(json_img, vector<vector< cv::Point>>{global_contour}, -1, Scalar(0, 255, 0), 2);
                    generateLabelMeJSONFromOverlay(is_add,
                                                   json_img,
                                                   src,
                                                   json_save_path.string() + "/" + info.updata_infos.Real_RST_Name + ".json",
                                                   Constants::CLASS_NAMES[info.updata_infos.id]
                    );
                    drawContours(contour_image, vector<vector< cv::Point>>{global_contour}, -1, color, 2);
                }
                info.updata_infos.Sub_Directory = "contour_image";
                cv::addWeighted(mask_overlay, 1, contour_image, 0.5, 0, mask_overlay);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto sec_d = std::chrono::duration<double>(end - start);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    info.result_seg.dTime = std::to_string(sec_d.count());
    std::cout << "detect counts=" << test_cont << "  耗时=" << info.result_seg.dTime << "s" << std::endl;
    Mat final_result;
    cv::addWeighted(result_img, 1, mask_overlay, 0.3, 0, final_result);
    info.result_seg.resultImage = final_result;
    info.updata_infos.Sub_Directory = "seg";
    saveimg(info, info.updata_infos.Real_RST_Name, final_result, index);
    info.updata_infos.Sub_Directory = "seg_mask";
    info.updata_infos.Sub_Directory = "src";
    saveimg(info, info.updata_infos.Real_RST_Name, src, index);
    // std::cout << "第" << info.updata_infos.New_Id << "张，存储路径" + info.updata_infos.Save_Path  << std::endl<<"\n";;
    return true;
}
bool YoloModelImpl::YoloDetectInfer( Mat& src,  InitDataInfo& info) {
    Mat letterbox_img = preprocessImage(src, info);
    src = letterbox_img.clone();

    //  Mat letterbox_img = letterbox(src, cv::Size(info.base_infos.yolo_model_Size_w, info.base_infos.yolo_model_Size_h));
    float scale = static_cast<float>(letterbox_img.cols) / info.updata_infos.yolo_model_Size;
    Mat blob = cv::dnn::blobFromImage(letterbox_img, 1.0 / 255.0,
                                      cv::Size(info.updata_infos.yolo_model_Size, info.updata_infos.yolo_model_Size), cv::Scalar(), true);
    ov::Tensor input_tensor(compiled_model_Detect.input().get_element_type(),
                            compiled_model_Detect.input().get_shape(),
                            blob.ptr(0));
    infer_request_Detect.set_input_tensor(input_tensor);
    infer_request_Detect.infer();
    auto output = infer_request_Detect.get_output_tensor(0);
    Mat output_buffer(output.get_shape()[1], output.get_shape()[2], CV_32F, output.data<float>());
    transpose(output_buffer, output_buffer);
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<Rect> boxes;
    for (int i = 0; i < output_buffer.rows; i++) {
        Mat classes_scores = output_buffer.row(i).colRange(4, 84);
        cv::Point class_id;
        double max_score;
        minMaxLoc(classes_scores, nullptr, &max_score, nullptr, &class_id);
        if (max_score > info.updata_infos.config) {
            class_scores.push_back(static_cast<float>(max_score));
            class_ids.push_back(class_id.x);
            const float cx = output_buffer.at<float>(i, 0);
            const float cy = output_buffer.at<float>(i, 1);
            const float w = output_buffer.at<float>(i, 2);
            const float h = output_buffer.at<float>(i, 3);
            boxes.emplace_back(
                    static_cast<int>((cx - 0.5f * w) * scale),
                    static_cast<int>((cy - 0.5f * h) * scale),
                    static_cast<int>(w * scale),
                    static_cast<int>(h * scale)
            );
        }
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, class_scores, info.updata_infos.config, info.updata_infos.Nms_Threshold, indices);
    Mat result_img = src.clone();
    int tmp_K = 0;
    for (int idx : indices) {
        tmp_K = tmp_K + 1;
        std::cout << "id="
                  << class_ids[idx]<< " ,x=" << boxes[idx].x  << " ,y=" << boxes[idx].y
                  <<  " ,w=" << boxes[idx].width << " ,h=" << boxes[idx].height << std::endl;

          cv::Rect roi(boxes[idx].x , boxes[idx].y, boxes[idx].width, boxes[idx].height );

        // 根据ROI裁剪图像
        cv::Mat croppedImg = src(roi);
        info.updata_infos.Sub_Directory = "cut/"+std::to_string(class_ids[idx]);
        saveimg(info, info.updata_infos.Real_RST_Name, croppedImg, tmp_K);
        const int color_idx = class_ids[idx] % Constants::CLASS_NAMES.size();
        drawDetection(result_img, boxes[idx], class_scores[idx], class_ids[idx], Constants::COLOR[color_idx] );
        writeDetectionLog(info.updata_infos.Save_Path,"", info.updata_infos.Real_RST_Name,class_ids[idx],
                          boxes[idx].x*1.0/info.updata_infos.targetSize,
                          boxes[idx].y * 1.0 / info.updata_infos.targetSize,
                          boxes[idx].width * 1.0 / info.updata_infos.targetSize,
                          boxes[idx].height * 1.0 / info.updata_infos.targetSize,std::to_string(tmp_K)
        );
        info.updata_infos.Sub_Directory = "src";
        saveimg(info, info.updata_infos.Real_RST_Name, src, tmp_K);
        info.updata_infos.Sub_Directory = "draw";
        saveimg(info, info.updata_infos.Real_RST_Name, result_img, tmp_K);
    }
    info.result_data_od.resultImage=result_img;
    return true;
}
bool YoloModelImpl::YoloPoseInfer( Mat& src,  InitDataInfo& info) {
    Mat letterbox_img = letterbox(src, cv::Size(info.base_infos.yolo_model_Size_h, info.base_infos.yolo_model_Size_h));
    float scale = static_cast<float>(letterbox_img.cols) / info.updata_infos.targetSize;
    Mat blob = cv::dnn::blobFromImage(letterbox_img, 1.0 / 255.0, cv::Size(info.updata_infos.targetSize, info.updata_infos.targetSize), Scalar(), true);
    ov::Tensor input_tensor(compiled_model_Pose.input().get_element_type(),
                            compiled_model_Pose.input().get_shape(),
                            blob.ptr(0));
    infer_request_Pose.set_input_tensor(input_tensor);
    infer_request_Pose.infer();
    auto output = infer_request_Pose.get_output_tensor(0);
    Mat output_buffer(output.get_shape()[1], output.get_shape()[2], CV_32F, output.data<float>());
    transpose(output_buffer, output_buffer);
    std::vector<float> class_scores;
    std::vector<Rect> boxes;
    std::vector<std::vector<cv::Point>> keypoints_list;
    for (int i = 0; i < output_buffer.rows; i++) {
        const float score = output_buffer.at<float>(i, 4);
        if (score > info.updata_infos.config) {
            class_scores.push_back(score);
            const float cx = output_buffer.at<float>(i, 0);
            const float cy = output_buffer.at<float>(i, 1);
            const float w = output_buffer.at<float>(i, 2);
            const float h = output_buffer.at<float>(i, 3);
            boxes.emplace_back(
                    static_cast<int>((cx - 0.5f * w) * scale),
                    static_cast<int>((cy - 0.5f * h) * scale),
                    static_cast<int>(w * scale),
                    static_cast<int>(h * scale)
            );
            std::vector<cv::Point> keypoints;
            for (int j = 0; j < 17; j++) {
                const float x = output_buffer.at<float>(i, 5 + j * 3) * scale;
                const float y = output_buffer.at<float>(i, 6 + j * 3) * scale;
                keypoints.emplace_back(static_cast<int>(x), static_cast<int>(y));
            }
            keypoints_list.push_back(keypoints);
        }
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, class_scores, info.updata_infos.config, info.updata_infos.Nms_Threshold, indices);
    Mat result_img = src.clone();
    for (int idx : indices) {
        for (const auto& kp : keypoints_list[idx]) {
            circle(result_img, kp, 5, Scalar(0, 255, 0), -1);
        }
    }
    imshow("Pose Estimation Results", result_img);
    cv::waitKey(1);
    return true;
}
bool YoloModelImpl::YoloClsInfer( Mat& src,  InitDataInfo& info) {
    float scale = src.size[0] / 640.0;
    Mat blob = cv::dnn::blobFromImage(src, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(), true);
    ov::Tensor input_tensor(compiled_model_Detect_Cls.input().get_element_type(),
                            compiled_model_Detect_Cls.input().get_shape(),
                            blob.ptr(0));
    infer_request_Cls.set_input_tensor(input_tensor);
    infer_request_Cls.infer();
    auto output = infer_request_Cls.get_output_tensor(0);
    auto output_shape = output.get_shape();
    float* output_buffer = output.data<float>();
    std::vector<float> result(output_buffer, output_buffer + output_shape[1]);
    auto max_idx = std::max_element(result.begin(), result.end());
    int class_id = max_idx - result.begin();
    float score = *max_idx;
    std::cout << "Class ID:" << class_id << " Score:" << score << std::endl;
    return true;
}

extern "C" YOLO_API IYoloModel* CreateYoloModel() {
    return new YoloModelImpl();
}
extern "C" YOLO_API void DestroyYoloModel(IYoloModel* model) {
    delete model;
}
