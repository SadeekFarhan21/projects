#include "minikafka/net.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <system_error>

#include "minikafka/bytes.h"
#include "minikafka/protocol.h"

namespace mk {

const char* error_name(ErrorCode e) {
  switch (e) {
    case ErrorCode::None: return "NONE";
    case ErrorCode::UnknownTopic: return "UNKNOWN_TOPIC";
    case ErrorCode::OffsetOutOfRange: return "OFFSET_OUT_OF_RANGE";
    case ErrorCode::CorruptMessage: return "CORRUPT_MESSAGE";
    case ErrorCode::InvalidRequest: return "INVALID_REQUEST";
    case ErrorCode::TopicExists: return "TOPIC_EXISTS";
    case ErrorCode::UnknownMember: return "UNKNOWN_MEMBER";
    case ErrorCode::IllegalGeneration: return "ILLEGAL_GENERATION";
    case ErrorCode::RebalanceInProgress: return "REBALANCE_IN_PROGRESS";
    case ErrorCode::MessageTooLarge: return "MESSAGE_TOO_LARGE";
    case ErrorCode::InternalError: return "INTERNAL_ERROR";
    case ErrorCode::UnknownApi: return "UNKNOWN_API";
  }
  return "UNKNOWN_ERROR";
}

namespace {
[[noreturn]] void throw_errno(const char* what) {
  throw std::system_error(errno, std::generic_category(), what);
}

sockaddr_in resolve(const std::string& host, uint16_t port) {
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (host.empty() || host == "0.0.0.0") {
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
  } else if (host == "localhost") {
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  } else if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    if (::getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res)
      throw std::runtime_error("cannot resolve " + host);
    addr.sin_addr = reinterpret_cast<sockaddr_in*>(res->ai_addr)->sin_addr;
    ::freeaddrinfo(res);
  }
  return addr;
}
}  // namespace

void set_socket_options(int fd) {
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
  ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
  int buf = 4 << 20;
  ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
  ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
}

int listen_tcp(const std::string& host, uint16_t port, uint16_t* bound_port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw_errno("socket");
  int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr = resolve(host, port);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    throw_errno("bind");
  }
  if (::listen(fd, 128) != 0) {
    ::close(fd);
    throw_errno("listen");
  }
  if (bound_port) {
    socklen_t len = sizeof(addr);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
    *bound_port = ntohs(addr.sin_port);
  }
  return fd;
}

int connect_tcp(const std::string& host, uint16_t port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw_errno("socket");
  sockaddr_in addr = resolve(host, port);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    int e = errno;
    ::close(fd);
    errno = e;
    throw_errno("connect");
  }
  set_socket_options(fd);
  return fd;
}

bool read_full(int fd, void* buf, size_t n) {
  auto* p = static_cast<uint8_t*>(buf);
  size_t got = 0;
  while (got < n) {
    ssize_t r = ::recv(fd, p + got, n - got, 0);
    if (r < 0) {
      if (errno == EINTR) continue;
      throw_errno("recv");
    }
    if (r == 0) {
      if (got == 0) return false;
      throw std::runtime_error("connection closed mid-message");
    }
    got += static_cast<size_t>(r);
  }
  return true;
}

void write_full(int fd, const void* buf, size_t n) {
  auto* p = static_cast<const uint8_t*>(buf);
  while (n > 0) {
    ssize_t w = ::send(fd, p, n, 0);
    if (w < 0) {
      if (errno == EINTR) continue;
      throw_errno("send");
    }
    p += w;
    n -= static_cast<size_t>(w);
  }
}

bool read_frame(int fd, std::vector<uint8_t>& out, uint32_t max_size) {
  uint8_t hdr[4];
  if (!read_full(fd, hdr, 4)) return false;
  const uint32_t len = load_u32(hdr);
  if (len > max_size) throw std::runtime_error("frame too large");
  out.resize(len);
  if (len > 0 && !read_full(fd, out.data(), len)) throw std::runtime_error("connection closed mid-frame");
  return true;
}

}  // namespace mk
