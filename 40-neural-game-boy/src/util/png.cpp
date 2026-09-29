#include "util/png.h"

#include <zlib.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace gb::png {

namespace {
using u8 = std::uint8_t;
using u32 = std::uint32_t;

u32 be32(const u8* p) { return (u32(p[0]) << 24) | (u32(p[1]) << 16) | (u32(p[2]) << 8) | p[3]; }

void put32(std::vector<u8>& out, u32 v) {
  for (int s = 24; s >= 0; s -= 8) out.push_back(u8(v >> s));
}

void chunk(std::vector<u8>& out, const char* type, const std::vector<u8>& data) {
  put32(out, u32(data.size()));
  size_t start = out.size();
  out.insert(out.end(), type, type + 4);
  out.insert(out.end(), data.begin(), data.end());
  u32 crc = u32(crc32(0, out.data() + start, uInt(out.size() - start)));
  put32(out, crc);
}

u8 paeth(int a, int b, int c) {
  int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
  if (pa <= pb && pa <= pc) return u8(a);
  return pb <= pc ? u8(b) : u8(c);
}
}  // namespace

Gray read_gray(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<u8> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  static const u8 sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  if (d.size() < 8 || std::memcmp(d.data(), sig, 8) != 0) throw std::runtime_error("not a PNG");

  int w = 0, h = 0, depth = 0, ctype = 0;
  std::vector<u8> idat, plte;
  size_t pos = 8;
  while (pos + 8 <= d.size()) {
    u32 len = be32(&d[pos]);
    std::string type(reinterpret_cast<const char*>(&d[pos + 4]), 4);
    const u8* body = &d[pos + 8];
    if (type == "IHDR") {
      w = int(be32(body));
      h = int(be32(body + 4));
      depth = body[8];
      ctype = body[9];
      if (body[12] != 0) throw std::runtime_error("interlaced PNG not supported");
    } else if (type == "PLTE") {
      plte.assign(body, body + len);
    } else if (type == "IDAT") {
      idat.insert(idat.end(), body, body + len);
    } else if (type == "IEND") {
      break;
    }
    pos += 12 + len;
  }
  int channels;
  switch (ctype) {
    case 0: channels = 1; break;
    case 2: channels = 3; break;
    case 3: channels = 1; break;
    case 4: channels = 2; break;
    case 6: channels = 4; break;
    default: throw std::runtime_error("unsupported PNG color type");
  }
  if (depth > 8) throw std::runtime_error("16-bit PNG not supported");
  const size_t bpp_bits = size_t(channels) * depth;
  const size_t stride = (size_t(w) * bpp_bits + 7) / 8;
  const size_t bpp = std::max<size_t>(1, bpp_bits / 8);

  std::vector<u8> raw((stride + 1) * size_t(h));
  uLongf raw_len = uLongf(raw.size());
  if (uncompress(raw.data(), &raw_len, idat.data(), uLong(idat.size())) != Z_OK)
    throw std::runtime_error("PNG inflate failed");

  std::vector<u8> cur(stride), prev(stride, 0);
  Gray g;
  g.width = w;
  g.height = h;
  g.pixels.resize(size_t(w) * h);
  for (int y = 0; y < h; ++y) {
    const u8* line = &raw[size_t(y) * (stride + 1)];
    u8 filter = line[0];
    for (size_t i = 0; i < stride; ++i) {
      int a = i >= bpp ? cur[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
      u8 x = line[1 + i];
      switch (filter) {
        case 0: cur[i] = x; break;
        case 1: cur[i] = u8(x + a); break;
        case 2: cur[i] = u8(x + b); break;
        case 3: cur[i] = u8(x + ((a + b) >> 1)); break;
        case 4: cur[i] = u8(x + paeth(a, b, c)); break;
        default: throw std::runtime_error("bad PNG filter");
      }
    }
    for (int x = 0; x < w; ++x) {
      auto sample = [&](int ch) -> int {
        size_t bit = (size_t(x) * channels + ch) * depth;
        int v = (cur[bit / 8] >> (8 - depth - bit % 8)) & ((1 << depth) - 1);
        return v;
      };
      int lum;
      if (ctype == 3) {
        int i = sample(0);
        lum = (plte[i * 3] * 299 + plte[i * 3 + 1] * 587 + plte[i * 3 + 2] * 114) / 1000;
      } else {
        int scale = 255 / ((1 << depth) - 1);
        if (channels >= 3)
          lum = (sample(0) * 299 + sample(1) * 587 + sample(2) * 114) * scale / 1000;
        else
          lum = sample(0) * scale;
      }
      g.pixels[size_t(y) * w + x] = u8(lum);
    }
    std::swap(cur, prev);
  }
  return g;
}

void write_gray(const std::string& path, int width, int height, const std::uint8_t* pixels) {
  std::vector<u8> raw;
  raw.reserve(size_t(width + 1) * height);
  for (int y = 0; y < height; ++y) {
    raw.push_back(0);
    raw.insert(raw.end(), pixels + size_t(y) * width, pixels + size_t(y + 1) * width);
  }
  uLongf zlen = compressBound(uLong(raw.size()));
  std::vector<u8> z(zlen);
  compress2(z.data(), &zlen, raw.data(), uLong(raw.size()), 9);
  z.resize(zlen);

  std::vector<u8> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  std::vector<u8> ihdr;
  put32(ihdr, u32(width));
  put32(ihdr, u32(height));
  ihdr.insert(ihdr.end(), {8, 0, 0, 0, 0});
  chunk(out, "IHDR", ihdr);
  chunk(out, "IDAT", z);
  chunk(out, "IEND", {});
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
}

}  // namespace gb::png
