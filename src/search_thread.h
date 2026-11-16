#pragma once

// A thread for the search, started with a stack of SEARCH_THREAD_STACK bytes whatever the platform's default
// for secondary threads is (1 MiB on Windows, 512 KiB on macOS). A search thread's entry holds a Position by
// value (about 220 KB: the state stack and the game history) and every search frame holds 4 KiB move lists;
// with 512 KiB the first search overflows. The size is a reservation: pages are committed only as the search
// touches them. If the stack cannot be had, the job runs on a default thread rather than not at all.

#include <cstddef>
#include <functional>
#include <memory>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#endif

namespace luminex {

inline constexpr std::size_t SEARCH_THREAD_STACK = std::size_t(32) * 1024 * 1024;

class SearchThread {
public:
    SearchThread() = default;
    explicit SearchThread(std::function<void()> job) { start(std::move(job)); }
    SearchThread(const SearchThread&) = delete;
    SearchThread& operator=(const SearchThread&) = delete;
    SearchThread(SearchThread&& other) noexcept { take(other); }
    SearchThread& operator=(SearchThread&& other) noexcept {
        if (this != &other) { join(); take(other); }
        return *this;
    }
    ~SearchThread() { join(); }

    bool joinable() const { return running_ || fallback_.joinable(); }

    void join() {
        if (fallback_.joinable()) fallback_.join();
        if (!running_) return;
#ifdef _WIN32
        WaitForSingleObject(handle_, INFINITE);
        CloseHandle(handle_);
        handle_ = nullptr;
#else
        pthread_join(thread_, nullptr);
#endif
        running_ = false;
    }

private:
    using Job = std::function<void()>;

    static void run(void* job) {
        std::unique_ptr<Job> owned(static_cast<Job*>(job));
        (*owned)();
    }

    void start(Job job) {
        Job* owned = new Job(std::move(job));
#ifdef _WIN32
        handle_ = reinterpret_cast<HANDLE>(_beginthreadex(
            nullptr, unsigned(SEARCH_THREAD_STACK),
            [](void* p) -> unsigned { run(p); return 0; },
            owned, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr));
        running_ = handle_ != nullptr;
#else
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, SEARCH_THREAD_STACK);
        running_ = pthread_create(&thread_, &attr, [](void* p) -> void* { run(p); return nullptr; }, owned) == 0;
        pthread_attr_destroy(&attr);
#endif
        if (!running_) fallback_ = std::thread([owned] { run(owned); });
    }

    void take(SearchThread& other) {
#ifdef _WIN32
        handle_ = other.handle_;
        other.handle_ = nullptr;
#else
        thread_ = other.thread_;
#endif
        running_ = other.running_;
        other.running_ = false;
        fallback_ = std::move(other.fallback_);
    }

#ifdef _WIN32
    HANDLE handle_ = nullptr;
#else
    pthread_t thread_{};
#endif
    bool running_ = false;
    std::thread fallback_;
};

} // namespace luminex
