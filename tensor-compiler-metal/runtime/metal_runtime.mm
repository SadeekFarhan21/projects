// Metal runtime for the tcm tensor compiler.
//
// Objective-C++ (compiled as C++20 with -fobjc-arc) exposing a small API to
// Python via pybind11:
//   Device.compile(source, function)  -> Pipeline   (runtime MSL compilation, cached)
//   Device.alloc(nbytes)              -> Buffer     (shared storage, unified memory)
//   Device.run(dispatches, repeat, mode, profile) -> timing dict
//
// All dispatches passed to one run() call are encoded into a single command
// buffer (batching). Timing comes from GPU timestamps: either the command
// buffer GPUStartTime/GPUEndTime pair, or per-dispatch stage-boundary counter
// samples when profile=True.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

struct Buffer {
  id<MTLBuffer> buf = nil;
  size_t nbytes = 0;
};
using BufferPtr = std::shared_ptr<Buffer>;

struct Pipeline {
  id<MTLComputePipelineState> pso = nil;
  std::string name;
  int max_threads = 0;
  int exec_width = 0;
  int static_tg_mem = 0;
};
using PipelinePtr = std::shared_ptr<Pipeline>;

struct Dispatch {
  PipelinePtr pipeline;
  std::vector<BufferPtr> buffers;
  std::array<uint32_t, 3> groups{1, 1, 1};
  std::array<uint32_t, 3> threads{1, 1, 1};
};

class Device {
 public:
  Device() {
    @autoreleasepool {
      dev_ = MTLCreateSystemDefaultDevice();
      if (!dev_) throw std::runtime_error("no Metal device available");
      queue_ = [dev_ newCommandQueue];
      counters_ok_ = [dev_ supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
      if (counters_ok_) {
        for (id<MTLCounterSet> cs in dev_.counterSets) {
          if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) ts_set_ = cs;
        }
        if (!ts_set_) counters_ok_ = false;
      }
    }
  }

  std::string name() const { return std::string([dev_.name UTF8String]); }
  size_t max_buffer_length() const { return dev_.maxBufferLength; }
  size_t max_threadgroup_memory() const { return dev_.maxThreadgroupMemoryLength; }
  bool counters_supported() const { return counters_ok_; }
  size_t recommended_working_set() const { return dev_.recommendedMaxWorkingSetSize; }

  PipelinePtr compile(const std::string& src, const std::string& fn, bool fast_math) {
    std::string key = fn + (fast_math ? "\x01" : "\x00") + src;
    {
      std::lock_guard<std::mutex> g(mu_);
      auto it = cache_.find(key);
      if (it != cache_.end()) {
        cache_hits_++;
        return it->second;
      }
    }
    auto t0 = std::chrono::steady_clock::now();
    auto p = std::make_shared<Pipeline>();
    @autoreleasepool {
      NSError* err = nil;
      MTLCompileOptions* opts = [MTLCompileOptions new];
      if (@available(macOS 15.0, *)) {
        opts.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
      } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = fast_math;
#pragma clang diagnostic pop
      }
      NSString* s = [NSString stringWithUTF8String:src.c_str()];
      id<MTLLibrary> lib = [dev_ newLibraryWithSource:s options:opts error:&err];
      if (!lib) {
        std::string msg = err ? [[err localizedDescription] UTF8String] : "unknown";
        throw std::runtime_error("MSL compile failed for " + fn + ":\n" + msg);
      }
      id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn.c_str()]];
      if (!f) throw std::runtime_error("function not found in library: " + fn);
      id<MTLComputePipelineState> pso = [dev_ newComputePipelineStateWithFunction:f error:&err];
      if (!pso) {
        std::string msg = err ? [[err localizedDescription] UTF8String] : "unknown";
        throw std::runtime_error("pipeline creation failed for " + fn + ": " + msg);
      }
      p->pso = pso;
      p->name = fn;
      p->max_threads = int(pso.maxTotalThreadsPerThreadgroup);
      p->exec_width = int(pso.threadExecutionWidth);
      p->static_tg_mem = int(pso.staticThreadgroupMemoryLength);
    }
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> g(mu_);
    compiles_++;
    compile_seconds_ += dt;
    cache_[key] = p;
    return p;
  }

  BufferPtr alloc(size_t nbytes) {
    auto b = std::make_shared<Buffer>();
    size_t n = nbytes == 0 ? 16 : nbytes;
    b->buf = [dev_ newBufferWithLength:n options:MTLResourceStorageModeShared];
    if (!b->buf) throw std::runtime_error("buffer allocation failed");
    std::memset([b->buf contents], 0, n);
    b->nbytes = nbytes;
    return b;
  }

  void upload(Buffer& b, py::array arr) {
    py::buffer_info info = arr.request();
    if (!(arr.flags() & py::array::c_style)) throw std::runtime_error("upload needs a C-contiguous array");
    size_t n = size_t(info.size) * size_t(info.itemsize);
    if (n > b.nbytes) throw std::runtime_error("upload larger than buffer");
    std::memcpy([b.buf contents], info.ptr, n);
  }

  void download(Buffer& b, py::array out) {
    py::buffer_info info = out.request(true);
    if (!(out.flags() & py::array::c_style)) throw std::runtime_error("download needs a C-contiguous array");
    size_t n = size_t(info.size) * size_t(info.itemsize);
    if (n > b.nbytes) throw std::runtime_error("download larger than buffer");
    std::memcpy(info.ptr, [b.buf contents], n);
  }

  // mode "batched": one command buffer, one encoder, all dispatches (default).
  // mode "per_dispatch": one command buffer per dispatch, each committed and waited.
  py::dict run(const std::vector<Dispatch>& ds, int repeat, const std::string& mode, bool profile) {
    py::dict out;
    if (ds.empty()) {
      out["gpu_s"] = 0.0;
      out["wall_s"] = 0.0;
      return out;
    }
    if (repeat < 1) repeat = 1;
    auto w0 = std::chrono::steady_clock::now();
    double gpu_total = 0.0;
    std::vector<double> per_kernel;
    {
      py::gil_scoped_release nogil;
      @autoreleasepool {
        if (mode == "per_dispatch") {
          for (int r = 0; r < repeat; ++r) {
            for (const auto& d : ds) {
              id<MTLCommandBuffer> cb = [queue_ commandBuffer];
              id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
              encode(enc, d);
              [enc endEncoding];
              [cb commit];
              [cb waitUntilCompleted];
              check(cb);
              gpu_total += cb.GPUEndTime - cb.GPUStartTime;
            }
          }
        } else if (profile && counters_ok_) {
          size_t n = ds.size() * size_t(repeat);
          MTLCounterSampleBufferDescriptor* sd = [MTLCounterSampleBufferDescriptor new];
          sd.counterSet = ts_set_;
          sd.storageMode = MTLStorageModeShared;
          sd.sampleCount = n * 2;
          NSError* err = nil;
          id<MTLCounterSampleBuffer> sb = [dev_ newCounterSampleBufferWithDescriptor:sd error:&err];
          if (!sb) throw std::runtime_error("counter sample buffer creation failed");
          MTLTimestamp c0, g0, c1, g1;
          [dev_ sampleTimestamps:&c0 gpuTimestamp:&g0];
          id<MTLCommandBuffer> cb = [queue_ commandBuffer];
          size_t idx = 0;
          for (int r = 0; r < repeat; ++r) {
            for (const auto& d : ds) {
              MTLComputePassDescriptor* pd = [MTLComputePassDescriptor computePassDescriptor];
              pd.sampleBufferAttachments[0].sampleBuffer = sb;
              pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = idx * 2;
              pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = idx * 2 + 1;
              id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDescriptor:pd];
              encode(enc, d);
              [enc endEncoding];
              idx++;
            }
          }
          [cb commit];
          [cb waitUntilCompleted];
          check(cb);
          [dev_ sampleTimestamps:&c1 gpuTimestamp:&g1];
          gpu_total = cb.GPUEndTime - cb.GPUStartTime;
          // sampleTimestamps reports the CPU side in nanoseconds on macOS, so the
          // ratio below converts GPU timestamp ticks into nanoseconds.
          double gpu_tick_ns = (g1 > g0) ? double(c1 - c0) / double(g1 - g0) : 1.0;
          NSData* data = [sb resolveCounterRange:NSMakeRange(0, n * 2)];
          const MTLCounterResultTimestamp* ts = (const MTLCounterResultTimestamp*)data.bytes;
          per_kernel.resize(ds.size(), 0.0);
          for (size_t i = 0; i < n; ++i) {
            uint64_t a = ts[2 * i].timestamp, b = ts[2 * i + 1].timestamp;
            double dt = (b > a && a != MTLCounterErrorValue && b != MTLCounterErrorValue)
                            ? double(b - a) * gpu_tick_ns * 1e-9
                            : 0.0;
            per_kernel[i % ds.size()] += dt / repeat;
          }
        } else {
          id<MTLCommandBuffer> cb = [queue_ commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          for (int r = 0; r < repeat; ++r) {
            for (const auto& d : ds) encode(enc, d);
          }
          [enc endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
          check(cb);
          gpu_total = cb.GPUEndTime - cb.GPUStartTime;
        }
      }
    }
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
    out["gpu_s"] = gpu_total;
    out["wall_s"] = wall;
    out["repeat"] = repeat;
    if (!per_kernel.empty()) out["per_kernel_s"] = per_kernel;
    return out;
  }

  py::dict stats() const {
    py::dict d;
    d["compiles"] = compiles_;
    d["cache_hits"] = cache_hits_;
    d["compile_seconds"] = compile_seconds_;
    d["cached_pipelines"] = cache_.size();
    return d;
  }

  void clear_cache() {
    std::lock_guard<std::mutex> g(mu_);
    cache_.clear();
  }

 private:
  void encode(id<MTLComputeCommandEncoder> enc, const Dispatch& d) {
    [enc setComputePipelineState:d.pipeline->pso];
    for (size_t i = 0; i < d.buffers.size(); ++i) {
      [enc setBuffer:d.buffers[i]->buf offset:0 atIndex:i];
    }
    MTLSize g = MTLSizeMake(d.groups[0], d.groups[1], d.groups[2]);
    MTLSize t = MTLSizeMake(d.threads[0], d.threads[1], d.threads[2]);
    [enc dispatchThreadgroups:g threadsPerThreadgroup:t];
  }

  static void check(id<MTLCommandBuffer> cb) {
    if (cb.status == MTLCommandBufferStatusError) {
      std::string msg = cb.error ? [[cb.error localizedDescription] UTF8String] : "unknown";
      throw std::runtime_error("command buffer failed: " + msg);
    }
  }

  id<MTLDevice> dev_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLCounterSet> ts_set_ = nil;
  bool counters_ok_ = false;
  std::mutex mu_;
  std::unordered_map<std::string, PipelinePtr> cache_;
  size_t compiles_ = 0, cache_hits_ = 0;
  double compile_seconds_ = 0.0;
};

PYBIND11_MODULE(_metal, m) {
  m.doc() = "Metal runtime for tcm";
  py::class_<Buffer, BufferPtr>(m, "Buffer").def_readonly("nbytes", &Buffer::nbytes);
  py::class_<Pipeline, PipelinePtr>(m, "Pipeline")
      .def_readonly("name", &Pipeline::name)
      .def_readonly("max_threads", &Pipeline::max_threads)
      .def_readonly("exec_width", &Pipeline::exec_width)
      .def_readonly("static_tg_mem", &Pipeline::static_tg_mem);
  py::class_<Dispatch>(m, "Dispatch")
      .def(py::init([](PipelinePtr p, std::vector<BufferPtr> bufs, std::array<uint32_t, 3> groups,
                       std::array<uint32_t, 3> threads) {
             Dispatch d;
             d.pipeline = std::move(p);
             d.buffers = std::move(bufs);
             d.groups = groups;
             d.threads = threads;
             return d;
           }),
           py::arg("pipeline"), py::arg("buffers"), py::arg("groups"), py::arg("threads"))
      .def_readonly("groups", &Dispatch::groups)
      .def_readonly("threads", &Dispatch::threads);
  py::class_<Device>(m, "Device")
      .def(py::init<>())
      .def_property_readonly("name", &Device::name)
      .def_property_readonly("max_buffer_length", &Device::max_buffer_length)
      .def_property_readonly("max_threadgroup_memory", &Device::max_threadgroup_memory)
      .def_property_readonly("counters_supported", &Device::counters_supported)
      .def_property_readonly("recommended_working_set", &Device::recommended_working_set)
      .def("compile", &Device::compile, py::arg("source"), py::arg("function"), py::arg("fast_math") = false)
      .def("alloc", &Device::alloc, py::arg("nbytes"))
      .def("upload", &Device::upload)
      .def("download", &Device::download)
      .def("run", &Device::run, py::arg("dispatches"), py::arg("repeat") = 1, py::arg("mode") = "batched",
           py::arg("profile") = false)
      .def("stats", &Device::stats)
      .def("clear_cache", &Device::clear_cache);
}
