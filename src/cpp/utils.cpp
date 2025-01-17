#include <experimental/filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "utils.hpp"
#include "NvInfer.h"
#include <filesystem>
#include "trt_model.hpp"
#include "trt_logger.hpp"

// namespace fs = std::filesystem;
namespace fs = std::experimental::filesystem;
using namespace std;

bool fileExists(const string fileName) {
    if (!experimental::filesystem::exists(
            experimental::filesystem::path(fileName))){
        return false;
    }else{
        return true;
    }
}

bool fileRead(const string &path, vector<unsigned char> &data, size_t &size){
    stringstream trtModelStream;
    ifstream cache(path);

    /* 将engine的内容写入trtModelStream中*/
    trtModelStream.seekg(0, trtModelStream.beg);
    trtModelStream << cache.rdbuf();
    cache.close();

    /* 计算model的大小*/
    trtModelStream.seekg(0, ios::end);
    size = trtModelStream.tellg();

    vector<uint8_t> tmp;
    trtModelStream.seekg(0, ios::beg);
    tmp.resize(size);

    /* 将trtModelStream中的stream通过read函数写入modelMem中*/
    trtModelStream.read((char*)data[0], size);
    return true;
}

vector<unsigned char> loadFile(const string &file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);
        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

vector<string> loadDataList(const string& file){
    vector<string> list;
    auto *f = fopen(file.c_str(), "r");
    if (!f) LOGE("Failed to open %s", file.c_str());

    char str[512];
    while (fgets(str, 512, f) != NULL) {
        for (int i = 0; str[i] != '\0'; ++i) {
            if (str[i] == '\n'){
                str[i] = '\0';
                break;
            }
        }
        list.push_back(str);
    }
    fclose(f);
    return list;
}

string printDims(const nvinfer1::Dims dims) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dims.nbDims; i ++) {
        oss << std::fixed << dims.d[i];
        if (i != dims.nbDims - 1) {
            oss << " x ";
        }
    }
    oss << "]";
    return oss.str();

}

string printTensor(float* tensor, int size){
    std::ostringstream oss;
    oss << "[ ";
    for (int i = 0; i < size; i ++) {
        oss << std::fixed << std::setprecision(4) << tensor[i];
        if (i != size - 1) {
            oss << ", ";
        }
    }
    oss << " ]";
    return oss.str();
}


string printTensorShape(nvinfer1::ITensor* tensor){
    string str;
    str += "[";
    auto dims = tensor->getDimensions();
    for (int j = 0; j < dims.nbDims; j++) {
        str += to_string(dims.d[j]);
        if (j != dims.nbDims - 1) {
            str += " x ";
        }
    }
    str += "]";
    return str;
}


string changePath(string srcPath, string relativePath, 
                  string postfix, string tag){
    fs::path sourcePath(srcPath);
    fs::path basePath = sourcePath.parent_path();
    fs::path filename = sourcePath.stem();

    fs::path newPath = basePath / relativePath / filename;

    if (!tag.empty()) {
        newPath += "-" + tag;
    }

    newPath += postfix;

    return newPath.string();
}


string getOutputPath(string srcPath, string postfix){
    int pos = srcPath.rfind(".");
    string tarPath;
    tarPath = srcPath.substr(0, pos);
    tarPath += "_" + postfix + ".png";
    return tarPath;
}

string getFileType(string filePath){
    int pos = filePath.rfind(".");
    string suffix;
    suffix = filePath.substr(pos, filePath.length());
    return suffix;
}

string getFileName(string filePath){
    int pos = filePath.rfind("/");
    string suffix;
    suffix = filePath.substr(pos + 1, filePath.length());
    return suffix;
}

string getPrecision(nvinfer1::DataType type) {
    switch(type) {
        case nvinfer1::DataType::kFLOAT:  return "FP32";
        case nvinfer1::DataType::kHALF:   return "FP16";
        case nvinfer1::DataType::kINT32:  return "INT32";
        case nvinfer1::DataType::kINT8:   return "INT8";
        case nvinfer1::DataType::kUINT8:  return "UINT8";
        default:                          return "unknown";
    }
}

