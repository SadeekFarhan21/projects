// Windowed frontend. Keys: arrows = d-pad, Z = A, X = B, Enter = Start,
// Backspace or Right Shift = Select, Tab (hold) = fast forward, Esc = quit.
#include <SDL.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#include "core/gameboy.h"

int run_sdl(const std::string& rom_path, int scale) {
  auto gb = gb::GameBoy::from_file(rom_path);
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
    return 1;
  }
  std::string title = "neural-game-boy  " + gb->bus.cart().header().title;
  SDL_Window* win = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                     gb::kScreenW * scale, gb::kScreenH * scale, 0);
  SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
  SDL_Texture* tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
                                       gb::kScreenW, gb::kScreenH);
  // A light green-grey LCD palette for play; datasets store raw shades.
  const uint32_t palette[4] = {0xFFE0F0D0, 0xFF88A878, 0xFF346856, 0xFF081820};

  using clock = std::chrono::steady_clock;
  const auto frame_time = std::chrono::nanoseconds(int64_t(1e9 / gb::kFramesPerSecond));
  auto next = clock::now();
  bool running = true;
  uint32_t argb[gb::kScreenW * gb::kScreenH];
  while (running) {
    SDL_Event ev;
    while (SDL_PollEvent(&ev))
      if (ev.type == SDL_QUIT || (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_ESCAPE)) running = false;
    const Uint8* k = SDL_GetKeyboardState(nullptr);
    uint8_t b = 0;
    if (k[SDL_SCANCODE_Z]) b |= gb::kBtnA;
    if (k[SDL_SCANCODE_X]) b |= gb::kBtnB;
    if (k[SDL_SCANCODE_BACKSPACE] || k[SDL_SCANCODE_RSHIFT]) b |= gb::kBtnSelect;
    if (k[SDL_SCANCODE_RETURN]) b |= gb::kBtnStart;
    if (k[SDL_SCANCODE_RIGHT]) b |= gb::kBtnRight;
    if (k[SDL_SCANCODE_LEFT]) b |= gb::kBtnLeft;
    if (k[SDL_SCANCODE_UP]) b |= gb::kBtnUp;
    if (k[SDL_SCANCODE_DOWN]) b |= gb::kBtnDown;
    gb->set_buttons(b);
    bool fast = k[SDL_SCANCODE_TAB];

    int frames = fast ? 8 : 1;
    for (int i = 0; i < frames; ++i) gb->run_frame();
    const auto& fb = gb->framebuffer();
    for (int i = 0; i < gb::kScreenW * gb::kScreenH; ++i) argb[i] = palette[fb[i] & 3];
    SDL_UpdateTexture(tex, nullptr, argb, gb::kScreenW * 4);
    SDL_RenderClear(ren);
    SDL_RenderCopy(ren, tex, nullptr, nullptr);
    SDL_RenderPresent(ren);

    next += frame_time;
    auto now = clock::now();
    if (next > now) std::this_thread::sleep_until(next);
    else next = now;
  }
  SDL_DestroyTexture(tex);
  SDL_DestroyRenderer(ren);
  SDL_DestroyWindow(win);
  SDL_Quit();
  return 0;
}
