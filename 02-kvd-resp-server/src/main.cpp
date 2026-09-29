// kvd: the server binary.
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "server.h"

namespace {

kv::Server* g_server = nullptr;

void on_signal(int) {
    if (g_server) g_server->request_stop();
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--port N] [--bind ADDR] [--aof PATH] [--appendonly yes|no]\n"
                 "          [--appendfsync always|everysec|no] [--hz N] [--verbose]\n",
                 argv0);
}

}  // namespace

int main(int argc, char** argv) {
    kv::ServerConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--port") cfg.port = std::atoi(next().c_str());
        else if (a == "--bind") cfg.bind = next();
        else if (a == "--aof") cfg.aof_path = next();
        else if (a == "--appendonly") cfg.aof_enabled = next() == "yes";
        else if (a == "--appendfsync") {
            if (!kv::parse_fsync_policy(next(), cfg.fsync)) {
                usage(argv[0]);
                return 2;
            }
        } else if (a == "--hz") cfg.hz = std::atoi(next().c_str());
        else if (a == "--verbose") cfg.verbose = true;
        else {
            usage(argv[0]);
            return a == "--help" || a == "-h" ? 0 : 2;
        }
    }

    std::signal(SIGPIPE, SIG_IGN);
    kv::Server server(cfg);
    std::string err;
    if (!server.start(&err)) {
        std::fprintf(stderr, "kvd: %s\n", err.c_str());
        return 1;
    }
    g_server = &server;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::fprintf(stderr, "kvd ready on %s:%d (aof=%s, fsync=%s)\n", cfg.bind.c_str(), server.port(),
                 cfg.aof_enabled ? cfg.aof_path.c_str() : "off",
                 cfg.fsync == kv::FsyncPolicy::Always     ? "always"
                 : cfg.fsync == kv::FsyncPolicy::EverySec ? "everysec"
                                                          : "no");
    server.run();
    auto st = server.stats();
    std::fprintf(stderr, "kvd shutting down: %llu connections, %llu commands, %llu keys expired actively\n",
                 (unsigned long long)st.connections_accepted, (unsigned long long)st.commands,
                 (unsigned long long)st.expired_active);
    return 0;
}
