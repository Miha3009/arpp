#include "../include/patcher.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <iomanip>

#ifdef __linux__
#include <malloc.h>
#endif

namespace fs = std::filesystem;

std::ostream& operator<<(std::ostream& os, const PatchKey& key) {
    os << "PatchKey{x0=" << key.x0 
       << ", y0=" << key.y0 
       << ", t0=" << key.t0 
       << ", xyStep=" << key.xyStep 
       << ", tStep=" << key.tStep
       << ", tag=" << key.tag << "}";
    return os;
}

std::vector<uint16_t> encode(
    torch::Tensor& patch,
    float precision
) {
    auto sizes = patch.sizes();
    size_t E = sizes[0];
    size_t T = sizes[1];
    size_t H = sizes[2];
    size_t W = sizes[3];

    std::vector<uint16_t> quantized(E * T * H * W + 2);
    if (!patch.is_cpu() || patch.scalar_type() != torch::kFloat32 || !patch.is_contiguous()) {
        patch = patch.contiguous().cpu().to(torch::kFloat32);
    }
    float* ptr = patch.data_ptr<float>();

    float minValue = std::numeric_limits<float>::max();
    for (size_t i = 0; i < E * T * H * W; ++i) {
        if (!std::isnan(ptr[i])) minValue = std::min(minValue, ptr[i]);
    }
    if (minValue == std::numeric_limits<float>::max()) minValue = NAN;
    else minValue = std::floor(minValue / precision) * precision;

    std::memcpy(&quantized[0], &minValue, sizeof(float));
    for (size_t i = 0; i < E * T * H * W; ++i) {
        if (std::isnan(ptr[i])) {
            quantized[i + 2] = 65535;
            continue;
        }
        float val = (ptr[i] - minValue) / precision;
        quantized[i + 2] = static_cast<uint16_t>(std::clamp(val + 0.5f, 0.0f, 65535.0f));
    }

    for (size_t i = E * T * H * W + 1; i > 2; --i) quantized[i] -= quantized[i - 1];

    return quantized;
}

void encode(
        std::ostream& out,
        torch::Tensor& patch,
        float precision,
        PatchKey key,
        ZSTD_CCtx* cctx,
        ZSTD_CDict* cdict,
        std::unordered_map<PatchKey, uint64_t>& index,
        uint64_t& totalBytes) {
    std::vector<uint8_t> compressed;
    auto patchEncoded = encode(patch, precision);
    size_t rawSize = patchEncoded.size() * sizeof(uint16_t);
    size_t bound = ZSTD_compressBound(rawSize);
    compressed.resize(bound);
    size_t compressedSize = ZSTD_compress_usingCDict(
        cctx, compressed.data(), bound,
        reinterpret_cast<const uint8_t*>(patchEncoded.data()), rawSize, cdict
    );

    if (ZSTD_isError(compressedSize)) {
        ZSTD_freeCCtx(cctx);
        ZSTD_freeCDict(cdict);
        throw std::runtime_error("Compression failed: " + std::string(ZSTD_getErrorName(compressedSize)));
    }

    out.write(reinterpret_cast<const char*>(&compressedSize), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(compressed.data()), compressedSize);
    index[key] = totalBytes;
    totalBytes += compressedSize + sizeof(uint32_t);
}

torch::Tensor decode(
    std::vector<uint16_t> data,
    float precision,
    size_t E,
    size_t T,
    size_t H,
    size_t W
) {
    float minValue;
    memcpy(&minValue, &data[0], sizeof(float));

    torch::Tensor patch = torch::zeros({static_cast<long>(E), static_cast<long>(T), static_cast<long>(H), static_cast<long>(W)}, torch::kFloat32);
    float* ptr = patch.data_ptr<float>();

    for (size_t i = 1; i < E * T * H * W; ++i) data[i + 2] += data[i + 1];
    for (size_t i = 0; i < E * T * H * W; ++i) {
         if (data[i + 2] == 65535) {
             ptr[i] = NAN;
             continue;
         }
         ptr[i] = minValue + static_cast<float>(data[i + 2]) * precision;
    }

    return patch;
}

torch::Tensor decode(
    std::istream& in,
    uint64_t offset,
    ZSTD_DCtx* dctx,
    ZSTD_DDict* ddict,
    float precision,
    size_t E,
    size_t T,
    size_t H,
    size_t W
) {
    size_t originalSize = (E * T * H * W + 2) * sizeof(uint16_t);

    in.seekg(static_cast<std::streamoff>(offset));

    uint32_t compressedSize;
    in.read(reinterpret_cast<char*>(&compressedSize), sizeof(uint32_t));    

    std::vector<uint8_t> compressed(compressedSize);
    in.read(reinterpret_cast<char*>(compressed.data()), compressedSize);

    std::vector<uint16_t> decompressed(originalSize / sizeof(uint16_t));
    size_t decompressedSize = ZSTD_decompress_usingDDict(
        dctx, decompressed.data(), originalSize,
        compressed.data(), compressedSize, ddict
    );
    if (ZSTD_isError(decompressedSize)) {
        throw std::runtime_error("ZSTD decompression error: " + std::string(ZSTD_getErrorName(decompressedSize)));
    }

    return decode(decompressed, precision, E, T, H, W);
}

torch::Tensor round_patch(torch::Tensor& tensor, float precision) {
    float* ptr = tensor.data_ptr<float>();
    float minValue = std::numeric_limits<float>::max();
    size_t N = tensor.numel();
    for (size_t i = 0; i < N; ++i) {
        if (!std::isnan(ptr[i])) minValue = std::min(minValue, ptr[i]);
    }
    if (minValue == std::numeric_limits<float>::max()) return tensor;
    else minValue = std::floor(minValue / precision) * precision;

    for (size_t i = 0; i < N; ++i) {
        if (!std::isnan(ptr[i])) {
            float val = (ptr[i] - minValue) / precision;
            ptr[i] = static_cast<uint16_t>(std::clamp(val + 0.5f, 0.0f, 65535.0f)) * precision + minValue;
        }
    }
    return tensor;
}

namespace DateUtils {
    constexpr std::array<int, 13> month_offsets = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};
    constexpr std::array<int, 13> month_offsets_leap = {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366};

    inline bool is_leap(int year) {
        return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    }

    static const auto year_offset = []() {
        std::array<int, 130> arr{};
        int days = 0;
        for (int y = 1970; y < 2100; ++y) {
            arr[y - 1970] = days;
            days += is_leap(y) ? 366 : 365;
        }
        return arr;
    }();
    
    inline int get_day_number(int year, int month, int day) {
        const auto& offsets = is_leap(year) ? month_offsets_leap : month_offsets;
        return year_offset[year - 1970] + offsets[month - 1] + day - 1;
    }

    inline int day_number_to_date(int day_number) {
        int year = 1970;
        for (; year < 2099; ++year) {
            if (day_number < year_offset[year + 1 - 1970]) break;
        }
        day_number -= year_offset[year - 1970];
        const auto& offsets = is_leap(year) ? month_offsets_leap : month_offsets;
        int month = 0;
        while (month < 12 && day_number >= offsets[month + 1]) ++month;
        day_number -= offsets[month];
        return year * 10000 + (month + 1) * 100 + day_number + 1;
    }
}

uint16_t date_to_day(const std::string& date) {
    int year = std::stoi(date.substr(0, 4));
    int month = std::stoi(date.substr(4, 2));
    int day = std::stoi(date.substr(6, 2));
    return static_cast<uint16_t>(DateUtils::get_day_number(year, month, day));
}

std::vector<uint16_t> dates_to_days(const std::vector<std::string>& dates) {
    std::vector<uint16_t> result;
    result.reserve(dates.size());
    for (const std::string& date : dates) result.push_back(date_to_day(date));
    return result;
}

void train_dict(
    Context& context,
    std::vector<torch::Tensor>& data,
    const std::string& element,
    float precision,
    bool timeInvariant
) {
    if (data.empty()) throw std::runtime_error("Empty data");

    size_t patchH = context.get_config_param("patch_size_h");
    size_t patchW = context.get_config_param("patch_size_w");
    size_t dictSize = context.get_config_param("dict_size");

    for (auto& tensor : data) {
        if (tensor.dim() == 2) tensor = tensor.unsqueeze(0).unsqueeze(0);
        else if (tensor.dim() == 3) tensor = tensor.unsqueeze(1);
    }

    auto sizes = data[0].sizes();
    size_t mapE = sizes[0];
    size_t mapH = sizes[2];
    size_t mapW = sizes[3];
    context.set_config_param(element + "@min_value", std::numeric_limits<float>::max());
    context.set_config_param(element + "@max_value", std::numeric_limits<float>::lowest());
    context.set_config_param(element + "@map_size_e", mapE);
    context.set_config_param(element + "@map_size_h", mapH);
    context.set_config_param(element + "@map_size_w", mapW);
    context.set_config_param(element + "@precision", precision);
    context.set_config_param(element + "@time_invariant", timeInvariant ? 1 : 0);
    if (mapE != 1) context.set_config_list(element + "@map_size_e", {0, static_cast<float>(mapE)});

    std::vector<uint8_t> samplesBuffer;
    std::vector<size_t> sampleSizes;
    for (size_t t = 0; t < data.size(); ++t) {
        for (size_t y0 = 0; y0 + patchH <= mapH; y0 += patchH) {
            for (size_t x0 = 0; x0 + patchW <= mapW; x0 += patchW) {
                auto patch = data[t].slice(2, y0, y0 + patchH)
                                 .slice(3, x0, x0 + patchW);
                auto sample = encode(patch, precision);
                size_t sampleSizeBytes = sample.size() * sizeof(uint16_t);
                sampleSizes.push_back(sampleSizeBytes);
                const uint8_t* rawData = reinterpret_cast<const uint8_t*>(sample.data());
                samplesBuffer.insert(samplesBuffer.end(), rawData, rawData + sampleSizeBytes);
            }
        }
    }

    std::vector<uint8_t> dictBuffer(dictSize);
    size_t actualDictSize = ZDICT_trainFromBuffer(
        dictBuffer.data(), dictBuffer.size(),
        samplesBuffer.data(),
        sampleSizes.data(), sampleSizes.size()
    );

    if (ZDICT_isError(actualDictSize)) {
        throw std::runtime_error("ZSTD dict training failed: " + std::string(ZDICT_getErrorName(actualDictSize)));
    }

    context.save_dict(element, dictBuffer);
    context.load_index(element);
}

void save(
    Context& context,
    std::vector<torch::Tensor>& data,
    const std::vector<std::string> &dates,
    const std::string &element,
    uint16_t tag
) {
    if (data.empty()) throw std::runtime_error("Empty data");
    if (data.size() != dates.size()) throw std::runtime_error("Data size != dates size");

    fs::path binPath = context.get_directory() / (element + ".bin");
    std::ifstream in(binPath, std::ios::binary);
    std::ofstream out(binPath, std::ios::binary | std::ios::app);
    if (!out) throw std::runtime_error("Cannot create file: " + binPath.string());
    out.seekp(0, std::ios::end);

    float precision = context.get_config_param(element + "@precision");
    size_t mapE = context.get_config_param(element + "@map_size_e");
    size_t mapH = context.get_config_param(element + "@map_size_h");
    size_t mapW = context.get_config_param(element + "@map_size_w");
    size_t patchT = context.get_config_param("patch_size_t");
    size_t patchH = context.get_config_param("patch_size_h");
    size_t patchW = context.get_config_param("patch_size_w");
    float globalMin = context.get_config_param(element + "@min_value");
    float globalMax = context.get_config_param(element + "@max_value");
    int compressionLevel = (int)context.get_config_param("compression_level");
    bool timeInvariant = context.get_config_param(element + "@time_invariant") > 0;
    if (timeInvariant) patchT = 1;
    std::vector<uint16_t> days = dates_to_days(dates);
    context.load_index(element);

    if (mapE != 1) {
        auto mapEVec = context.get_config_list(element + "@map_size_e");
        mapE = static_cast<size_t>(mapEVec.back());
    }
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i].dim() == 2) data[i] = data[i].unsqueeze(0).unsqueeze(0);
        else if (data[i].dim() == 3) data[i] = data[i].unsqueeze(1);
        if (mapH != (size_t)data[i].size(2) || mapW != (size_t)data[i].size(3)) {
            throw std::runtime_error("Data size mismatch with config");
        }
        if (mapE != (size_t)data[i].size(0)) {
            if (mapE == 1 || i != 0) throw std::runtime_error("Data size mismatch with config");
            auto mapEVec = context.get_config_list(element + "@map_size_e");
            mapEVec.push_back(days[i]);
            mapEVec.push_back(data[i].size(0));
            mapE = static_cast<size_t>(data[i].size(0));
            context.set_config_list(element + "@map_size_e", mapEVec);
        }
        auto valid = data[i].masked_select(~torch::isnan(data[i]));
        if (valid.numel() == 0) continue;
        globalMin = std::min(globalMin, valid.min().item<float>());
        globalMax = std::max(globalMax, valid.max().item<float>());
    }
    context.set_config_param(element + "@min_value", globalMin);
    context.set_config_param(element + "@max_value", globalMax);

    std::vector<uint8_t> dictRaw = context.get_dict(element);
    ZSTD_CDict* cdict = ZSTD_createCDict(dictRaw.data(), dictRaw.size(), compressionLevel);
    if (!cdict) throw std::runtime_error("Failed to create compression dict");

    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    if (!cctx) {
        ZSTD_freeCDict(cdict);
        throw std::runtime_error("Failed to create compression context");
    }

    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    if (!dctx) {
        ZSTD_freeCDict(cdict);
        ZSTD_freeCCtx(cctx);
        throw std::runtime_error("Failed to create decompression context");
    }
    ZSTD_DDict* ddict = context.get_ddict(element);

    std::map<uint16_t, std::vector<size_t>> dateGroups;
    for (size_t i = 0; i < days.size(); ++i) {
        uint16_t groupKey = days[i] / patchT;
        dateGroups[groupKey].push_back(i);
    }

    auto index = context.get_index(element);

    uint64_t totalBytes = out.tellp();
    for (size_t y0 = 0; y0 < mapH; y0 += patchH) {
        for (size_t x0 = 0; x0 < mapW; x0 += patchW) {
            size_t ySize = std::min(y0 + patchH, mapH) - y0;
            size_t xSize = std::min(x0 + patchW, mapW) - x0;
            for (auto& [date, indices] : dateGroups) {
                uint16_t t0 = timeInvariant ? 0 : static_cast<uint16_t>(date * patchT);
                PatchKey key = {static_cast<uint16_t>(x0), static_cast<uint16_t>(y0), t0, 1, 1, tag};
                auto it = index->find(key);
                torch::Tensor patch = it == index->end()
                    ? torch::full({static_cast<long>(mapE), static_cast<long>(patchT), static_cast<long>(patchH), static_cast<long>(patchW)}, NAN)
                    : decode(in, it->second, dctx, ddict, precision, mapE, patchT, patchH, patchW);
                for (size_t idx : indices) {
                    uint16_t day = days[idx];
                    patch.slice(1, day % patchT, day % patchT + 1).slice(2, 0, ySize).slice(3, 0, xSize) =
                    data[idx].slice(2, y0, y0 + ySize).slice(3, x0, x0 + xSize);
                    if (xSize != patchW) {
                        patch.slice(1, day % patchT, day % patchT + 1).slice(2, 0, ySize).slice(3, xSize, patchW) =
                        data[idx].slice(2, y0, y0 + ySize).slice(3, 0, (x0 + patchW) % mapW);
                    }
                }
                encode(out, patch, precision, key, cctx, cdict, *index, totalBytes);
                if (timeInvariant) break;
            }
        }
    }

    out.close();
    ZSTD_freeCCtx(cctx);
    ZSTD_freeCDict(cdict);
    ZSTD_freeDCtx(dctx);
    ZSTD_freeDDict(ddict);

    context.save_index(element, *index);
    context.load_index(element);
}

std::vector<std::tuple<int, int, int>> gen_times(int start, int length, int T, bool timeInvariant) {
    std::vector<std::tuple<int, int, int>> result;
    std::vector<std::pair<int, int>> tmp;

    if (timeInvariant) {
        result.emplace_back(0, 1, 0);
        return result;
    }

    int pos = start;
    int remaining = length;
    while (remaining > 0) {
        int max_size = 4;
        while ((pos % max_size) == 0 && max_size <= remaining) {
            max_size <<= 2;
        }

        max_size >>= 2;
        tmp.emplace_back(pos, max_size);
        pos += max_size;
        remaining -= max_size;
    }

    for (auto p : tmp) {
        result.emplace_back(p.first / (p.second * T) * p.second * T, p.second, (p.first / p.second) % T);
    }

    return result;
}

int pos_mod(int a, int b) {
    return ((a % b) + b) % b;
}

PatchMapping get_patch_mapping(int x, int y, int x0, int y0, int xSize, int ySize, int xyStep, int mapW, int patchH, int patchW) {
    size_t patchHStep = patchH * xyStep;
    size_t patchWStep = patchW * xyStep;
    int padW = mapW + (xyStep - mapW % xyStep) % xyStep;
    int xMod = pos_mod(pos_mod(x, padW), patchWStep);

    PatchMapping m;
    m.tgtX0 = (x - x0) / xyStep;
    m.tgtX1 = (std::min(x - xMod + (int)patchWStep, x0 + xSize * xyStep) - x0) / xyStep;
    m.tgtY0 = (y - y0) / xyStep;
    m.tgtY1 = (std::min(y - (int)(y % patchHStep) + (int)patchHStep, y0 + ySize * xyStep) - y0) / xyStep;
    m.srcX0 = xMod / xyStep;
    m.srcX1 = m.srcX0 + (m.tgtX1 - m.tgtX0);
    m.srcY0 = y % patchHStep / xyStep;
    m.srcY1 = m.srcY0 + (m.tgtY1 - m.tgtY0);
    m.x0 = static_cast<uint16_t>(pos_mod(x, padW) - xMod),
    m.y0 = static_cast<uint16_t>(y - (y % patchHStep));
    return m;
}

void fill_missing_y(torch::Tensor& tensor, int y0, int ySize, int xyStep, size_t mapH, int dim) {
    if (y0 < 0) {
        int missingTop = (-y0 + xyStep - 1) / xyStep;
        for (int i = 0; i < missingTop; ++i) {
            auto target = tensor.slice(dim, i, i+1);
            auto source = tensor.slice(dim, missingTop, missingTop+1);
            target.copy_(source);
        }
    }
    
    int yEnd = y0 + ySize * xyStep;
    if (yEnd > (int)mapH) {
        int missingBottom = (yEnd - mapH + xyStep - 1) / xyStep;
        int lastValidRow = ySize - missingBottom - 1;
        for (int i = 0; i < missingBottom; ++i) {
            auto target = tensor.slice(dim, ySize - missingBottom + i, ySize - missingBottom + i + 1);
            auto source = tensor.slice(dim, lastValidRow, lastValidRow + 1);
            target.copy_(source);
        }
    }
}

size_t find_map_e(const std::vector<float>& mapEVec, int t0) {
    size_t mapE = 1;
    for (size_t i = 0; i < mapEVec.size(); i += 2) {
        if (mapEVec[i] <= t0) mapE = static_cast<size_t>(mapEVec[i+1]);
    }
    return mapE;
}

torch::Tensor ThreadPoolLoader::process(const Request& req) {
    std::ifstream in(context->get_directory() / (req.element + ".bin"), std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open data file for element: " + req.element);
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    if (!dctx) throw std::runtime_error("Failed to create decompression context");
    ZSTD_DDict* ddict = context->get_ddict(req.element);
    auto index = context->get_index(req.element);
    float precision = context->get_config_param(req.element + "@precision");
    size_t mapE = context->get_config_param(req.element + "@map_size_e");
    size_t mapH = context->get_config_param(req.element + "@map_size_h");
    size_t mapW = context->get_config_param(req.element + "@map_size_w");
    size_t patchT = context->get_config_param("patch_size_t");
    size_t patchH = context->get_config_param("patch_size_h");
    size_t patchW = context->get_config_param("patch_size_w");
    bool timeInvariant = context->get_config_param(req.element + "@time_invariant") > 0;
    if (timeInvariant) patchT = 1;
    int t0 = date_to_day(req.t0);

    if (mapE != 1) {
        auto mapEVec = context->get_config_list(req.element + "@map_size_e");
        mapE = find_map_e(mapEVec, t0);
    }

    torch::Tensor result = torch::full({static_cast<long>(mapE), req.tSize, req.ySize, req.xSize}, NAN);

    int maxY = 0;
    for (int y = req.y0; maxY < req.ySize;) {
        if (y >= static_cast<int>(mapH)) break;
        for (int x = req.x0; x < req.x0 + req.xSize * req.xyStep;) {
            PatchMapping m = get_patch_mapping(x, y, req.x0, req.y0, req.xSize, req.ySize, req.xyStep, mapW, patchH, patchW);
            PatchKey key = {m.x0, m.y0, 0, static_cast<uint16_t>(req.xyStep), 1, static_cast<uint16_t>(req.tag)};

            std::unordered_map<PatchKey, std::pair<bool, torch::Tensor>> requested;
            for (int t = t0, tResult = 0; t < t0 + req.tSize * req.tStep; t += req.tStep, tResult++) {
                std::vector<torch::Tensor> patches;
                std::vector<std::tuple<int, int, int>> times = gen_times(t, req.tStep, patchT, timeInvariant);
                for (auto time : times) {
                    key.t0 = std::get<0>(time);
                    key.tStep = std::get<1>(time);
                    int tIdx = std::get<2>(time);
                    auto reqIt = requested.find(key);
                    if (reqIt != requested.end()) {
                        if(!reqIt->second.first) break;
                        patches.push_back(reqIt->second.second.slice(1, tIdx, tIdx + 1));
                        continue;
                    }

                    auto indexIt = index->find(key);
                    if (indexIt == index->end()) {
                        requested[key] = std::make_pair(false, torch::Tensor());
                        break;
                    }
                    auto patch = decode(in, indexIt->second, dctx, ddict, precision, mapE, patchT, patchH, patchW);
                    patches.push_back(patch.slice(1, tIdx, tIdx + 1));
                    requested[key] = std::make_pair(true, patch);
                }
                if (times.size() != patches.size()) continue;

                auto target = result.slice(1, tResult, tResult + 1).slice(2, m.tgtY0, m.tgtY1).slice(3, m.tgtX0, m.tgtX1);
                auto source = [&](const torch::Tensor& patch) {
                    return patch.slice(2, m.srcY0, m.srcY1).slice(3, m.srcX0, m.srcX1);
                };
                if (patches.size() == 1) {
                    target.copy_(source(patches[0]));
                } else {
                    torch::Tensor sum = torch::zeros_like(target);
                    for (size_t i = 0; i < patches.size(); ++i) sum += source(patches[i]) * std::get<1>(times[i]);
                    target.copy_(sum / req.tStep);
                }
            }
            x += (m.tgtX1 - m.tgtX0) * req.xyStep;
            maxY = std::max(maxY, m.tgtY1);
        }
        y = req.y0 + maxY * req.xyStep;
    }

    fill_missing_y(result, req.y0, req.ySize, req.xyStep, mapH, 2);

    if (mapE == 1) result = result.squeeze(0);

    ZSTD_freeDCtx(dctx);
    return result;
}

std::vector<torch::Tensor> load(Context& context, const std::vector<Request>& requests) {
    std::shared_ptr<ThreadPoolLoader> threadPool = context.get_thread_pool();
    for (const auto& req : requests) {
        if (!context.has_config_param(req.element + "@precision")) context.load_index(req.element);
    }
    std::vector<std::future<torch::Tensor>> futures;
    for (const auto& req : requests) futures.push_back(threadPool->enqueue(req));
    std::vector<torch::Tensor> results;
    for (auto& f : futures) results.push_back(f.get());
    return results;
}

torch::Tensor process_climate(Context& context, const Request& req) {
    bool timeInvariant = context.get_config_param(req.element + "@time_invariant") > 0;
    if (timeInvariant) {
        return torch::full({req.tSize, req.ySize, req.xSize}, 0);
    }

    std::ifstream in(context.get_directory() / (req.element + ".bin"), std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open data file for element: " + req.element);
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    if (!dctx) throw std::runtime_error("Failed to create decompression context");
    ZSTD_DDict* ddict = context.get_ddict(req.element);
    auto index = context.get_index(req.element);
    float precision = context.get_config_param(req.element + "@precision");
    size_t mapH = context.get_config_param(req.element + "@map_size_h");
    size_t mapW = context.get_config_param(req.element + "@map_size_w");
    size_t patchH = context.get_config_param("patch_size_h");
    size_t patchW = context.get_config_param("patch_size_w");

    torch::Tensor result = torch::full({req.tSize, req.ySize, req.xSize}, NAN);
    int t0 = date_to_day(req.t0);

    int maxY = 0;
    for (int y = req.y0; maxY < req.ySize;) {
        if (y >= static_cast<int>(mapH)) break;
        for (int x = req.x0; x < req.x0 + req.xSize * req.xyStep;) {
            PatchMapping m = get_patch_mapping(x, y, req.x0, req.y0, req.xSize, req.ySize, req.xyStep, mapW, patchH, patchW);
            PatchKey key = {m.x0, m.y0, 0, static_cast<uint16_t>(req.xyStep), 0, static_cast<uint16_t>(req.tag)};
            for (int t = 0; t < req.tSize * req.tStep; ++t) {
                key.t0 = static_cast<uint16_t>(DateUtils::day_number_to_date(t0 + t) % 10000);
                auto indexIt = index->find(key);
                if (indexIt == index->end()) continue;
                auto patch = decode(in, indexIt->second, dctx, ddict, precision, 1, 1, patchH, patchW).squeeze(0).squeeze(0);
                if (t % req.tStep == 0) {
                    result.slice(0, t / req.tStep, t / req.tStep + 1).slice(1, m.tgtY0, m.tgtY1).slice(2, m.tgtX0, m.tgtX1) =
                    patch.slice(0, m.srcY0, m.srcY1).slice(1, m.srcX0, m.srcX1);
                } else {
                    result.slice(0, t / req.tStep, t / req.tStep + 1).slice(1, m.tgtY0, m.tgtY1).slice(2, m.tgtX0, m.tgtX1) +=
                    patch.slice(0, m.srcY0, m.srcY1).slice(1, m.srcX0, m.srcX1);
                }
            }
            x += (m.tgtX1 - m.tgtX0) * req.xyStep;
            maxY = std::max(maxY, m.tgtY1);
        }
        y = req.y0 + maxY * req.xyStep;
    }
    if (req.tStep > 1) result = result / req.tStep;
    fill_missing_y(result, req.y0, req.ySize, req.xyStep, mapH, 1);

    ZSTD_freeDCtx(dctx);
    return result;
}

std::vector<torch::Tensor> load_climate(Context& context, const std::vector<Request>& requests) {
    std::vector<torch::Tensor> results;
    for (const auto& req : requests) {
        if (!context.has_config_param(req.element + "@precision")) context.load_index(req.element);
    }
    for (const Request& req : requests) results.push_back(process_climate(context, req));
    return results;
}

void print_progress(const std::string& msg, int progress, int total) {
    int width = std::to_string(total).length();
    std::cout << "\r" << msg << ": " << std::right << std::setw(width) << progress << "/" << total << std::flush;
}

void aggregate(Context& context, const std::string& element, const std::string& climateStartStr, const std::string& climateEndStr) {
    fs::path binPath = context.get_directory() / (element + ".bin");
    fs::path tmpPath = context.get_directory() / (element + ".tmp");
    std::ifstream in(binPath, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open data file for element: " + element);
    std::ofstream out(tmpPath, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open tmp for element: " + element);

    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    if (!dctx) throw std::runtime_error("Failed to create decompression context");
    context.load_index(element);
    ZSTD_DDict* ddict = context.get_ddict(element);
    auto index = context.get_index(element);
    float precision = context.get_config_param(element + "@precision");
    size_t mapE = context.get_config_param(element + "@map_size_e");
    size_t mapH = context.get_config_param(element + "@map_size_h");
    size_t mapW = context.get_config_param(element + "@map_size_w");
    float globalMin = context.get_config_param(element + "@min_value");
    float globalMax = context.get_config_param(element + "@max_value");
    size_t patchT = context.get_config_param("patch_size_t");
    size_t patchH = context.get_config_param("patch_size_h");
    size_t patchW = context.get_config_param("patch_size_w");
    uint16_t climateStart = climateStartStr.empty() ? 0 : date_to_day(climateStartStr);
    uint16_t climateEnd = climateEndStr.empty() ? 0 : date_to_day(climateEndStr);
    int grid_sizes = static_cast<int>(context.get_config_param("grid_sizes"));
    int compressionLevel = (int)context.get_config_param("compression_level");
    bool timeInvariant = context.get_config_param(element + "@time_invariant") > 0;
    if (timeInvariant) patchT = 1;

    std::vector<uint8_t> dictRaw = context.get_dict(element);
    ZSTD_CDict* cdict = ZSTD_createCDict(dictRaw.data(), dictRaw.size(), compressionLevel);
    if (!cdict) throw std::runtime_error("Failed to create compression dict");

    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    if (!cctx) {
        ZSTD_freeCDict(cdict);
        throw std::runtime_error("Failed to create compression context");
    }

    std::map<std::tuple<uint16_t, uint16_t, uint16_t>, std::vector<uint16_t>> spatialGroups;

    for (const auto& [key, offset] : *index) {
        if (key.tStep != 1 && key.xyStep != 1) continue;
        auto spatialKey = std::make_tuple(key.x0, key.y0, key.tag);
        spatialGroups[spatialKey].emplace_back(key.t0);
    }

    std::vector<float> mapEVec = mapE != 1 ? context.get_config_list(element + "@map_size_e") : std::vector<float>();

    uint64_t totalBytes = 0;
    std::unordered_map<PatchKey, uint64_t> newIndex;
    int spatialProgress = 0;
    print_progress("Spatial", 0, spatialGroups.size());
    for (const auto& [spatialKey, times] : spatialGroups) {
        auto [x0, y0, tag] = spatialKey;
        std::vector<uint16_t> sortedTimes(times.begin(), times.end());
        std::sort(sortedTimes.begin(), sortedTimes.end());

        std::unordered_map<uint16_t, torch::Tensor> climate;
        std::unordered_map<uint16_t, int> counter;
        for (uint16_t t : sortedTimes) {
            if (t < climateStart || t > climateEnd) continue;
            mapE = find_map_e(mapEVec, t + patchT - 1);
            PatchKey key = {x0, y0, t, 1, 1, tag};
            auto it = index->find(key);
            if (it == index->end()) continue;
            torch::Tensor patch = decode(in, it->second, dctx, ddict, precision, mapE, patchT, patchH, patchW);
            patch = patch.mean(0, true);
            for (int tIdx = 0; tIdx < (int)patchT; ++tIdx)    {
                uint16_t climateIdx = static_cast<uint16_t>(DateUtils::day_number_to_date(t + tIdx) % 10000);
                auto slice = patch.slice(1, tIdx, tIdx + 1);
                if (slice.isnan().all().item<bool>()) continue;
                if (climate.find(climateIdx) != climate.end()) {
                    climate[climateIdx] += slice;
                    counter[climateIdx]++;
                } else {
                    climate[climateIdx] = slice.clone();
                    counter[climateIdx] = 1;
                }
            }
        }
        if (climateStart != climateEnd) {
            for (auto& [idx, tensor] : climate) {
                climate[idx] = tensor / counter[idx];
                PatchKey key = {x0, y0, idx, 1, 0, tag};
                encode(out, climate[idx], precision, key, cctx, cdict, newIndex, totalBytes);
                climate[idx] = round_patch(climate[idx], precision);
            }
        }

        struct BlockInfo {
            torch::Tensor sum;
            torch::Tensor block;
            int sumStart;
            int sumCount;
            int blockStart;
        };

        std::unordered_map<int, BlockInfo> blocks;
        mapE = find_map_e(mapEVec, sortedTimes[0] + patchT - 1);
        for (size_t i = 0; i < sortedTimes.size(); ++i) {
            PatchKey key = {x0, y0, sortedTimes[i], 1, 1, tag};
            auto it = index->find(key);
            if (it == index->end()) continue;
            size_t newMapE = find_map_e(mapEVec, sortedTimes[i] + patchT - 1);
            if (mapE != newMapE) {
                blocks.clear();
                mapE = newMapE;
            }
            torch::Tensor patch = decode(in, it->second, dctx, ddict, precision, mapE, patchT, patchH, patchW);

            if (climateStart != climateEnd) {
                for (int tIdx = 0; tIdx < (int)patchT; ++tIdx) {
                    uint16_t climateIdx = static_cast<uint16_t>(DateUtils::day_number_to_date(sortedTimes[i] + tIdx) % 10000);
                    if (climate.find(climateIdx) == climate.end()) continue;
                    patch.slice(1, tIdx, tIdx + 1) -= climate[climateIdx];
                }
            }
            encode(out, patch, precision, {x0, y0, sortedTimes[i], 1, 1, tag}, cctx, cdict, newIndex, totalBytes);
            if (timeInvariant) continue;

            int stage = 1;
            for (int tIdx = 0; tIdx < (int)patchT; ++tIdx) {
                auto slice = patch.slice(1, tIdx, tIdx + 1);
                int time = sortedTimes[i] + tIdx;
                while (true) {
                    int sumSize = 1 << (2 * stage);
                    int blockSize = sumSize * patchT;
                    int sumStart = time / sumSize * sumSize;
                    int blockStart = time / blockSize * blockSize;
                    auto itBlock = blocks.find(stage);
                    if (itBlock == blocks.end() || itBlock->second.sumStart != sumStart) {
                        BlockInfo info;
                        info.sum = slice.clone();
                        info.sumStart = sumStart;
                        info.sumCount = 1;
                        info.blockStart = blockStart;
                        if (itBlock != blocks.end()) {
                            if (itBlock->second.blockStart != blockStart) {
                                PatchKey key = {x0, y0, static_cast<uint16_t>(itBlock->second.blockStart), 1, static_cast<uint16_t>(sumSize), tag};
                                encode(out, itBlock->second.block, precision, key, cctx, cdict, newIndex, totalBytes);
                                info.block = torch::full({static_cast<long>(mapE), static_cast<long>(patchT), static_cast<long>(patchH), static_cast<long>(patchW)}, NAN);
                            } else {
                                info.block = itBlock->second.block;
                            }
                        } else {
                            info.block = torch::full({static_cast<long>(mapE), static_cast<long>(patchT), static_cast<long>(patchH), static_cast<long>(patchW)}, NAN);
                        }
                        blocks[stage] = info;
                        break;
                    }

                    BlockInfo& info = itBlock->second;
                    info.sum += slice;
                    info.sumCount += 1;
                    if (info.sumCount != 4) break;

                    info.sum /= 4;
                    int blockIdx = (time - blockStart) / sumSize;
                    info.block.slice(1, blockIdx, blockIdx + 1) = info.sum;
                    slice = info.sum;
                    stage++;
                }
            }
            for (auto& [stg, info] : blocks) {
                int sumSize = 1 << (2 * stg);
                PatchKey key = {x0, y0, static_cast<uint16_t>(info.blockStart), 1, static_cast<uint16_t>(sumSize), tag};
                encode(out, info.block, precision, key, cctx, cdict, newIndex, totalBytes);
            }
        }
        for (auto [stage, info] : blocks) {
            PatchKey key = {x0, y0, static_cast<uint16_t>(info.blockStart), 1, static_cast<uint16_t>(1 << (2 * stage)), tag};
            encode(out, info.block, precision, key, cctx, cdict, newIndex, totalBytes);
        }
        spatialProgress++;
        print_progress("Spatial", spatialProgress, spatialGroups.size());
    }
    std::cout << std::endl;
    context.save_index(element, newIndex);
    in.close();
    out.close();

    fs::remove(binPath);
    fs::rename(tmpPath, binPath);

    in = std::ifstream(binPath, std::ios::binary);
    out = std::ofstream(binPath, std::ios::binary | std::ios::app);
    if (!in || !out) throw std::runtime_error("Cannot open data file for element: " + element);

    std::map<std::tuple<uint16_t, uint16_t, uint16_t>, std::vector<PatchKey>> temporalGroups;
    for (const auto& [key, offset] : newIndex) temporalGroups[{key.t0, key.tStep, key.tag}].push_back(key);

    int temporalProgress = 0;
    print_progress("Temporal", temporalProgress, temporalGroups.size());
    for (const auto& [temporalKey, patchKeys] : temporalGroups) {
        auto [t0, tStep, tag] = temporalKey;
        mapE = tStep == 0 ? 1 : find_map_e(mapEVec, t0 + tStep * patchT - 1);
        auto map = torch::full({static_cast<long>(mapE), static_cast<long>(tStep == 0 ? 1 : patchT), static_cast<long>(mapH), static_cast<long>(mapW*2)}, NAN);
        if (tStep == 0) map = torch::full({1, 1, static_cast<long>(mapH), static_cast<long>(mapW*2)}, NAN);
        for (const PatchKey& key : patchKeys) {
            auto patch = decode(in, newIndex[key], dctx, ddict, precision, map.size(0), map.size(1), patchH, patchW);
            size_t xSize = std::min(key.x0 + patchW, mapW) - key.x0;
            size_t ySize = std::min(key.y0 + patchH, mapH) - key.y0;
            if (xSize == 0 || ySize == 0) continue;
            map.slice(2, key.y0, key.y0 + ySize).slice(3, key.x0, key.x0 + xSize) =
            patch.slice(2, 0, ySize).slice(3, 0, xSize).clone();
        }
        map.slice(3, mapW, mapW * 2) = map.slice(3, 0, mapW);

        for (int step = 2; step <= grid_sizes; step <<= 1) {
            if ((step & grid_sizes) == 0) continue;

            int padH = (step - mapH % step) % step;
            int padW = (step - mapW % step) % step;
            auto padded = torch::nn::functional::pad(map,
                     torch::nn::functional::PadFuncOptions({0, padW, 0, padH}).mode(torch::kConstant).value(NAN));

            torch::Tensor pooledMap = padded.unfold(2, step, step).unfold(3, step, step).mean({4, 5});
            int pooledH = pooledMap.size(2);
            int pooledW = pooledMap.size(3);

            for (int y0 = 0; y0 < pooledH; y0 += patchH) {
                for (int x0 = 0; x0 < pooledW; x0 += patchW) {
                    if (x0 * step >= (int)mapW) continue;
                    size_t ySize = std::min(y0 + (int)patchH, pooledH) - y0;
                    auto patch = torch::full({map.size(0), map.size(1), static_cast<long>(patchH), static_cast<long>(patchW)}, NAN);
                    patch.slice(2, 0, ySize) = pooledMap.slice(2, y0, y0 + ySize).slice(3, x0, x0 + patchW);
                    PatchKey key = {static_cast<uint16_t>(x0 * step), static_cast<uint16_t>(y0 * step), t0, static_cast<uint16_t>(step), tStep, tag};
                    encode(out, patch, precision, key, cctx, cdict, newIndex, totalBytes);
                }
            }
        }
        temporalProgress++;
        print_progress("Temporal", temporalProgress, temporalGroups.size());
#ifdef __linux__
        if (temporalProgress % 10 == 0) malloc_trim(0);
#endif
    }
    std::cout << std::endl;

    context.save_index(element, newIndex);
    ZSTD_freeCCtx(cctx);
    ZSTD_freeCDict(cdict);
    ZSTD_freeDCtx(dctx);
}
