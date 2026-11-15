// large_pages.h — memory for the big tables the search reads at random: the
// transposition table and the network's feature weights.
//
// Their pages are 2 MiB where the system allows. A random read into a table
// of tens of megabytes otherwise lands on a 4 KiB page whose address
// translation is rarely cached (the second-level TLB holds a few thousand
// pages: 8 to 16 MB of 4 KiB ones), and the read waits for a page walk. On
// Linux the memory is aligned to 2 MiB and advised MADV_HUGEPAGE before its
// first write, so the faults of that write take transparent huge pages when
// THP is "always" or "madvise" (Ubuntu's default, under which nothing else
// gets them). Elsewhere, or where the system declines, the pages stay 4 KiB
// and nothing else changes.
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#if defined(__linux__)
#include <sys/mman.h>
#endif
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace luminex {

inline constexpr std::size_t LARGE_PAGE = std::size_t(2) << 20;

// The engine builds without exceptions, so a failed allocation ends it, as
// std::vector's did.
[[noreturn]] inline void large_alloc_failed(std::size_t bytes) {
    std::fprintf(stderr, "luminex: cannot allocate %zu MB\n", (bytes + (1u << 20) - 1) >> 20);
    std::abort();
}

// bytes of zeroed memory, aligned to LARGE_PAGE and rounded up to whole large
// pages (on Windows aligned to 64 bytes)
inline void* large_alloc(std::size_t bytes) {
    void* p = nullptr;
#if defined(_WIN32)
    p = _aligned_malloc(bytes, 64);
    if (!p) large_alloc_failed(bytes);
#else
    bytes = (bytes + LARGE_PAGE - 1) / LARGE_PAGE * LARGE_PAGE;
    if (posix_memalign(&p, LARGE_PAGE, bytes) != 0) large_alloc_failed(bytes);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    madvise(p, bytes, MADV_HUGEPAGE);   // advice: refused, the pages stay 4 KiB
#endif
#endif
    std::memset(p, 0, bytes);
    return p;
}

inline void large_free(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

// n zeroed elements of T in large_alloc memory, not copyable: the part of
// std::vector these tables used, with reset() for resize().
template <class T>
class LargeArray {
    static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>,
                  "the elements are zeroed bytes, never constructed or destroyed");
public:
    LargeArray() = default;
    ~LargeArray() { large_free(p_); }
    LargeArray(const LargeArray&) = delete;
    LargeArray& operator=(const LargeArray&) = delete;

    // n zero elements in place of the old ones, which are freed first; none for n = 0
    void reset(std::size_t n) {
        large_free(p_);
        p_ = nullptr;
        n_ = 0;
        if (n) {
            p_ = static_cast<T*>(large_alloc(n * sizeof(T)));
            n_ = n;
        }
    }
    std::size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }
    T* data() { return p_; }
    const T* data() const { return p_; }
    T& operator[](std::size_t i) { return p_[i]; }
    const T& operator[](std::size_t i) const { return p_[i]; }

private:
    T* p_ = nullptr;
    std::size_t n_ = 0;
};

} // namespace luminex
