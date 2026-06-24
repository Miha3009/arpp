#include "../include/patcher.h"
#include <filesystem>
#include <stdexcept>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

Context::Context(const std::string& directory, size_t num_threads) {
    fs::path p(directory);
    if (!fs::exists(p)) throw std::runtime_error("Directory does not exist: " + directory);
    this->directory = p;
    load_config();
    threadPool = std::make_shared<ThreadPoolLoader>(this, num_threads);
}

Context::~Context() {
    threadPool.reset();
    config.clear();
    index.clear();
    dicts.clear();
    for (auto [key, ddict] : ddicts) ZSTD_freeDDict(ddict);
    ddicts.clear();
}

fs::path Context::get_directory() const {
    if (directory.empty()) throw std::runtime_error("Directory not set");
    return directory;
}

std::shared_ptr<ThreadPoolLoader> Context::get_thread_pool() const {
    return threadPool;
}

void Context::set_config_param(const std::string& name, float value) {
    config[name] = value;
}

float Context::get_config_param(const std::string& name) const {
    auto it = config.find(name);
    if (it == config.end()) throw std::runtime_error("Config param " + name + " not found");
    return it->second;
}

bool Context::has_config_param(const std::string& name) const {
    return config.find(name) != config.end();
}

void Context::set_config_list(const std::string& name, const std::vector<float>& values) {
    config[name + "_count"] = static_cast<float>(values.size());
    for (size_t i = 0; i < values.size(); ++i) config[name + "_" + std::to_string(i)] = values[i];
}

std::vector<float> Context::get_config_list(const std::string& name) const {
    int count = static_cast<int>(get_config_param(name + "_count"));
    std::vector<float> result(count);
    for (int i = 0; i < count; ++i) result[i] = get_config_param(name + "_" + std::to_string(i));
    return result;
}

void Context::save_config() const {
    fs::path path = directory / "config.bin";
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot save config");

    for (const auto& [name, value] : config) {
        if (name.find('@') != std::string::npos) continue;
        size_t len = name.size();
        out.write((char*)&len, sizeof(len));
        out.write(name.c_str(), len);
        out.write((char*)&value, sizeof(value));
    }
}

void Context::load_config() {
    fs::path path = directory / "config.bin";
    if (!fs::exists(path)) {
        config = {
            {"dict_size", 32768},
            {"compression_level", 5},
            {"patch_size_h", 32},
            {"patch_size_w", 32},
            {"patch_size_t", 4},
            {"grid_sizes", 20},
        };
        save_config();
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot load config");

    while (in.peek() != EOF) {
        size_t len;
        in.read((char*)&len, sizeof(len));
        if (in.eof()) break;

        std::string name(len, '\0');
        in.read(name.data(), len);

        float value;
        in.read((char*)&value, sizeof(value));

        config[name] = value;
    }
}

std::vector<uint8_t> Context::get_dict(const std::string& element) const {
    auto it = dicts.find(element);
    if (it == dicts.end()) throw std::runtime_error("Dict for " + element + " not found");
    return it->second;
}

ZSTD_DDict* Context::get_ddict(const std::string& element) const {
    auto it = ddicts.find(element);
    if (it == ddicts.end()) throw std::runtime_error("DDict for " + element + " not found");
    return it->second;
}

void Context::save_dict(const std::string& element, const std::vector<uint8_t>& dict) {
    dicts[element] = dict;
    std::unordered_map<PatchKey, uint64_t> index;
    save_index(element, index);
}

void Context::save_index(const std::string& element, const std::unordered_map<PatchKey, uint64_t>& index) {
    fs::path path = directory / (element + ".idx");
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot create index: " + path.string());

    out.write(reinterpret_cast<const char*>(dicts[element].data()), dicts[element].size());

    std::vector<std::pair<std::string, float>> keys;
    std::string prefix = element + "@";
    size_t size = 0;
    for (const auto& [name, value] : config) if (name.find(prefix) == 0) size++;
    out.write((char*)&size, sizeof(size));
    for (const auto& [name, value] : config) {
        if (name.find(prefix) == 0) {
            std::string key = name.substr(prefix.size());
            size_t len = key.size();
            out.write((char*)&len, sizeof(len));
            out.write(key.c_str(), len);
            out.write((char*)&value, sizeof(value));
        }
    }

    size = index.size();
    out.write((char*)&size, sizeof(size));
    
    for (const auto& [key, offset] : index) {
        out.write((char*)&key, sizeof(PatchKey));
        out.write((char*)&offset, sizeof(uint64_t));
    }
}

void Context::load_index(const std::string& element) {
    fs::path path = directory / (element + ".idx");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open index: " + path.string());

    size_t dictSize = static_cast<size_t>(get_config_param("dict_size"));
    std::vector<uint8_t> dict(dictSize);
    in.read(reinterpret_cast<char*>(dict.data()), dictSize);
    dicts[element] = dict;

    ZSTD_DDict* ddict = ZSTD_createDDict(dict.data(), dictSize);
    if (!ddict) throw std::runtime_error("Failed to create decompression dict");
    ddicts[element] = ddict;

    std::string prefix = element + "@";
    size_t size;
    in.read((char*)&size, sizeof(size));

    for (size_t i = 0; i < size; ++i) {
        size_t len;
        in.read((char*)&len, sizeof(len));
        std::string key(len, '\0');
        in.read(key.data(), len);

        float value;
        in.read((char*)&value, sizeof(value));

        config[prefix + key] = value;
    }

    in.read((char*)&size, sizeof(size));
    auto index = std::make_shared<std::unordered_map<PatchKey, uint64_t>>();
    index->reserve(size);

    for (size_t i = 0; i < size; ++i) {
        PatchKey key;
        uint64_t offset;
        in.read((char*)&key, sizeof(PatchKey));
        in.read((char*)&offset, sizeof(uint64_t));
        (*index)[key] = offset;
    }

    this->index[element] = std::move(index);
}

const std::shared_ptr<std::unordered_map<PatchKey, uint64_t>>& Context::get_index(const std::string& element) const {
    auto it = index.find(element);
    if (it == index.end()) throw std::runtime_error("Index not loaded: " + element);
    return it->second;
}

ThreadPoolLoader::ThreadPoolLoader(Context* context, size_t num_threads): stop(false), context(context) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [this] { return stop || !tasks.empty(); });
                    if (stop && tasks.empty()) return;
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                task();
            }
        });
    }
}

ThreadPoolLoader::~ThreadPoolLoader() {
    stop = true;
    cv.notify_all();
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
}

std::future<torch::Tensor> ThreadPoolLoader::enqueue(Request req) {
    auto promise = std::make_shared<std::promise<torch::Tensor>>();
    auto future = promise->get_future();
    {
        std::unique_lock<std::mutex> lock(mutex);
        tasks.emplace([this, req, promise]() {
            try {
                promise->set_value(process(req));
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        });
    }
    cv.notify_one();
    return future;
}

