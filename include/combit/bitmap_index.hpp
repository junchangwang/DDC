#ifndef BITMAP_INDEX_HPP
#define BITMAP_INDEX_HPP

#include <unordered_map>
#include <string>
#include <vector>
#include <cstdint>
#include <combit/bitmap_vector.hpp>

namespace combit {

/**
 * BitmapIndex
 *
 * A simple in-memory equality-encoding bitmap index.
 *
 * For each distinct key value, there is one BitmapVector. When a row at
 * a given position has value V, the bit at that position is set in V's
 * bitmap vector.
 *
 * Template parameter K is the key type (must be hashable and equality-
 * comparable).
 */
template <typename K>
class BitmapIndex {
public:
    /// Insert: record that key `value` appears at `row_position`.
    void insert(const K& value, uint64_t row_position) {
        vectors_[value].set_bit(row_position);
    }

    /// Lookup: return the bitmap vector for the given key, or nullptr.
    const BitmapVector* lookup(const K& value) const {
        auto it = vectors_.find(value);
        return it != vectors_.end() ? &it->second : nullptr;
    }

    /// Return all positions where `value` appears.
    std::vector<uint64_t> get_positions(const K& value) const {
        std::vector<uint64_t> result;
        auto it = vectors_.find(value);
        if (it == vectors_.end()) return result;

        const BitmapVector& bv = it->second;
        for (uint64_t i = 0; i < bv.size(); ++i) {
            if (bv.get_bit(i)) {
                result.push_back(i);
            }
        }
        return result;
    }

    /// Return the compressed encoding for a given key.
    ComBitEncoding get_encoding(const K& value) const {
        auto it = vectors_.find(value);
        if (it == vectors_.end()) return {};
        return it->second.encode();
    }

    /// Return the number of distinct values indexed.
    size_t cardinality() const { return vectors_.size(); }

    /// Access all vectors (read-only).
    const std::unordered_map<K, BitmapVector>& vectors() const {
        return vectors_;
    }

private:
    std::unordered_map<K, BitmapVector> vectors_;
};

} // namespace combit

#endif // BITMAP_INDEX_HPP