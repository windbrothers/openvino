// export_macro.h
#pragma once

#ifdef _WIN32
#ifdef INFERENCE_LIB_EXPORT
#define YOLO_API __declspec(dllexport)
#else
#define YOLO_API __declspec(dllimport)
#endif
#else
#define YOLO_API
#endif