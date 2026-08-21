#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>

#include "uti.h"
#include "fastbit/bitvector.h"
#include "ewah/ewah.h"
#include "croaring/roaring.hh"

namespace dense_btv_delivery {

inline uint32_t reverse_low_bits(uint32_t value, unsigned width) {
    value = ((value & 0x55555555U) << 1) | ((value >> 1) & 0x55555555U);
    value = ((value & 0x33333333U) << 2) | ((value >> 2) & 0x33333333U);
    value = ((value & 0x0F0F0F0FU) << 4) | ((value >> 4) & 0x0F0F0F0FU);
    value = ((value & 0x00FF00FFU) << 8) | ((value >> 8) & 0x00FF00FFU);
    value = (value << 16) | (value >> 16);
    return width == 0 ? 0 : value >> (32 - width);
}

inline void set_ones(uint64_t* words, uint64_t begin, uint64_t end) {
    if (begin >= end) return;
    const size_t first = static_cast<size_t>(begin >> 6);
    const size_t last = static_cast<size_t>((end - 1) >> 6);
    const unsigned first_offset = static_cast<unsigned>(begin & 63);
    const unsigned end_offset = static_cast<unsigned>(end & 63);
    if (first == last) {
        const uint64_t low = ~uint64_t{0} << first_offset;
        const uint64_t high = end_offset == 0
            ? ~uint64_t{0}
            : (uint64_t{1} << end_offset) - 1;
        words[first] |= low & high;
        return;
    }
    words[first] |= ~uint64_t{0} << first_offset;
    if (last > first + 1) {
        std::memset(words + first + 1, 0xFF,
                    (last - first - 1) * sizeof(uint64_t));
    }
    words[last] |= end_offset == 0
        ? ~uint64_t{0}
        : (uint64_t{1} << end_offset) - 1;
}

inline void write_low_bits(uint64_t* words, uint64_t position,
                           uint64_t value, unsigned width) {
    if (width == 0) return;
    const size_t word = static_cast<size_t>(position >> 6);
    const unsigned offset = static_cast<unsigned>(position & 63);
    words[word] |= value << offset;
    if (offset + width > 64) {
        words[word + 1] |= value >> (64 - offset);
    }
}

inline roaring::api::bitset_t* from_wah(const ibis::bitvector& source,
                                         uint32_t logical_size) {
    auto* result = roaring::api::bitset_create_with_capacity(logical_size);
    if (result == nullptr) throw std::bad_alloc();
    uint64_t position = 0;
    for (auto it = source.m_vec.begin();
         it != source.m_vec.end() && position < logical_size; ++it) {
        const uint32_t encoded = *it;
        if (encoded > ibis::bitvector::ALLONES) {
            const uint64_t run_bits = std::min<uint64_t>(
                uint64_t(encoded & ibis::bitvector::MAXCNT) *
                    ibis::bitvector::MAXBITS,
                uint64_t(logical_size) - position);
            if ((encoded & ibis::bitvector::FILLBIT) != 0) {
                set_ones(result->array, position, position + run_bits);
            }
            position += run_bits;
        } else {
            const unsigned width = static_cast<unsigned>(std::min<uint64_t>(
                ibis::bitvector::MAXBITS,
                uint64_t(logical_size) - position));
            write_low_bits(
                result->array, position,
                reverse_low_bits(encoded, ibis::bitvector::MAXBITS), width);
            position += width;
        }
    }
    if (position < logical_size && source.active.nbits > 0) {
        const unsigned width = static_cast<unsigned>(std::min<uint64_t>(
            source.active.nbits, uint64_t(logical_size) - position));
        write_low_bits(
            result->array, position,
            reverse_low_bits(source.active.val, source.active.nbits), width);
    }
    const unsigned tail = logical_size & 63U;
    if (tail != 0 && result->arraysize != 0) {
        result->array[result->arraysize - 1] &= (uint64_t{1} << tail) - 1;
    }
    return result;
}

inline roaring::api::bitset_t* from_ewah(
        const ewah::EWAHBoolArray<uint64_t>& source,
        uint32_t logical_size) {
    auto* result = roaring::api::bitset_create_with_capacity(logical_size);
    if (result == nullptr) throw std::bad_alloc();
    size_t position = 0;
    auto iterator = source.raw_iterator();
    while (iterator.hasNext() && position < result->arraysize) {
        auto& rlw = iterator.next();
        const size_t run_words = std::min<size_t>(
            rlw.getRunningLength(), result->arraysize - position);
        if (rlw.getRunningBit() && run_words != 0) {
            std::memset(result->array + position, 0xFF,
                        run_words * sizeof(uint64_t));
        }
        position += run_words;
        const size_t literal_words = std::min<size_t>(
            rlw.getNumberOfLiteralWords(), result->arraysize - position);
        if (literal_words != 0) {
            std::memcpy(result->array + position, iterator.dirtyWords(),
                        literal_words * sizeof(uint64_t));
            position += literal_words;
        }
    }
    const unsigned tail = logical_size & 63U;
    if (tail != 0 && result->arraysize != 0) {
        result->array[result->arraysize - 1] &= (uint64_t{1} << tail) - 1;
    }
    return result;
}

inline double wah_time(const ibis::bitvector& source, uint32_t logical_size) {
    Timer timer;
    timer.reset();
    auto* result = from_wah(source, logical_size);
    roaring::api::bitset_free(result);
    return timer.elapsed_ms();
}

inline double ewah_time(const ewah::EWAHBoolArray<uint64_t>& source,
                        uint32_t logical_size) {
    Timer timer;
    timer.reset();
    auto* result = from_ewah(source, logical_size);
    roaring::api::bitset_free(result);
    return timer.elapsed_ms();
}

}  // namespace dense_btv_delivery
