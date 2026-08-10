#pragma once

#include <torch/torch.h>
#include <zstd.h>
#include <zdict.h>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <filesystem>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <functional>
#include <fstream>

namespace fs = std::filesystem;

struct PatchKey {
    uint16_t x0;
    uint16_t y0;
    uint16_t t0;
    uint16_t xyStep;
    uint16_t tStep;
    uint16_t tag;

    bool operator==(const PatchKey& other) const {
        return x0 == other.x0 && y0 == other.y0 && t0 == other.t0 && xyStep == other.xyStep && tStep == other.tStep && tag == other.tag;
    }
};

std::ostream& operator<<(std::ostream& os, const PatchKey& key);

namespace std {
    template <>
    struct hash<PatchKey> {
        size_t operator()(const PatchKey& k) const {
            return (k.x0 * 73856093) ^ (k.y0 * 19349663) ^ (k.t0 * 83492791) ^ (k.xyStep * 35791394) ^ (k.tStep * 28629357) ^ (k.tag * 303700049);
        }
    };
}

struct Request {
    std::string element;
    int x0;
    int y0;
    std::string t0;
    int xSize;
    int ySize;
    int tSize;
    int xyStep;
    int tStep;
    int tag;

    Request(const std::string& element, int x0, int y0, const std::string& t0, int xSize, int ySize, int tSize, int xyStep, int tStep)
        : element(element), x0(x0), y0(y0), t0(t0), xSize(xSize), ySize(ySize), tSize(tSize), xyStep(xyStep), tStep(tStep), tag(0) {}

    Request(const std::string& element, int x0, int y0, const std::string& t0, int xSize, int ySize, int tSize, int xyStep, int tStep, int tag)
        : element(element), x0(x0), y0(y0), t0(t0), xSize(xSize), ySize(ySize), tSize(tSize), xyStep(xyStep), tStep(tStep), tag(tag) {}
};

class Context;

class ThreadPoolLoader {
public:
    ThreadPoolLoader(Context* context, size_t num_threads);
    ~ThreadPoolLoader();
    std::future<torch::Tensor> enqueue(Request req);

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mutex;
    std::condition_variable cv;
    bool stop;
    Context* context;

    torch::Tensor process(const Request& req);
};

struct PatchMapping {
    int tgtX0, tgtX1;
    int tgtY0, tgtY1;
    int srcX0, srcX1;
    int srcY0, srcY1;
    uint16_t x0, y0;
};

class Context {
public:
    Context(const std::string& directory, size_t num_threads);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
 
    fs::path get_directory() const;
    std::shared_ptr<ThreadPoolLoader> get_thread_pool() const;
    void set_config_param(const std::string& name, float value);
    float get_config_param(const std::string& name) const;
    bool has_config_param(const std::string& name) const;
    void save_config() const;
    void load_config();
    std::vector<uint8_t> get_dict(const std::string& element) const;
    ZSTD_DDict* get_ddict(const std::string& element) const;
    void save_dict(const std::string& element, const std::vector<uint8_t>& dict);
    const std::shared_ptr<std::unordered_map<PatchKey, uint64_t>>& get_index(const std::string& element) const;
    void save_index(const std::string& element, const std::unordered_map<PatchKey, uint64_t>& index);
    void load_index(const std::string& element);
    void set_config_list(const std::string& name, const std::vector<float>& values);
    std::vector<float> get_config_list(const std::string& name) const;

private:
    fs::path directory;
    std::unordered_map<std::string, float> config;
    std::unordered_map<std::string, std::shared_ptr<std::unordered_map<PatchKey, uint64_t>>> index;
    std::unordered_map<std::string, std::vector<uint8_t>> dicts;
    std::unordered_map<std::string, ZSTD_DDict*> ddicts;
    std::shared_ptr<ThreadPoolLoader> threadPool;
};

void train_dict(Context& context, std::vector<torch::Tensor>& data, const std::string& element, float precision, bool timeInvariant);
void save(Context& context, std::vector<torch::Tensor>& data, const std::vector<std::string> &dates, const std::string &element, uint16_t tag);
std::vector<torch::Tensor> load(Context& context, const std::vector<Request>& requests);
std::vector<torch::Tensor> load_climate(Context& context, const std::vector<Request>& requests);
void aggregate(Context& context, const std::string& element, const std::string& climateStartStr, const std::string& climateEndStr);
