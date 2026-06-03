#include <sstream>
#include <types.hpp>

namespace ddc {

size_t DDCEncoding::decompressed_bit_count() const {
    size_t total = 0;
    for (size_t i = 0; i < content.size(); ++i) {
        if (header[i]) {
            // Compressed word: fill_length gives # of words compressed
            total += static_cast<size_t>(fill_length(content[i])) * WORD_SIZE;
        } else {
            // Literal word: represents exactly WORD_SIZE bits
            total += WORD_SIZE;
        }
    }
    return total;
}

std::string DDCEncoding::to_string() const {
    std::ostringstream oss;

    oss << "header section:   ";
    for (bool b : header) {
        oss << (b ? '1' : '0');
    }

    oss << "\ncontent section:  ";
    for (size_t i = 0; i < content.size(); ++i) {
        if (i > 0) oss << ' ';
        for (int bit = static_cast<int>(WORD_SIZE) - 1; bit >= 0; --bit) {
            oss << ((content[i] >> bit) & 1);
        }
    }
    oss << '\n';
    return oss.str();
}

} // namespace ddc