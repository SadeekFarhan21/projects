// Thin blocking-socket helpers shared by the broker and the clients.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mk {

// Listens on 127.0.0.1-or-any:port (port 0 picks a free port). Returns the fd.
int listen_tcp(const std::string& host, uint16_t port, uint16_t* bound_port);
int connect_tcp(const std::string& host, uint16_t port);
void set_socket_options(int fd);  // TCP_NODELAY, SO_NOSIGPIPE, larger buffers

// Returns false on clean EOF before any byte was read; throws on errors or EOF mid-message.
bool read_full(int fd, void* buf, size_t n);
void write_full(int fd, const void* buf, size_t n);

// Reads one u32-length-prefixed frame into `out`. Returns false on clean EOF.
bool read_frame(int fd, std::vector<uint8_t>& out, uint32_t max_size);

}  // namespace mk
