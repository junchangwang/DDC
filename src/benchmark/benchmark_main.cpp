#include <iostream>
#include <string>
#include <vector>
#include "bitmap_backend.h"
#include "backends/wah/wah_backend.h"
#include "backends/croaring/croaring_backend.h"
#include "backends/combit/combit_backend.h"

void print_array(const std::string& name, const std::vector<uint32_t>& arr) {
    std::cout << name << ": [";
    for(size_t i=0; i<arr.size(); ++i) {
        std::cout << arr[i] << (i==arr.size()-1 ? "" : ", ");
    }
    std::cout << "]\n";
}

int main(int argc, char** argv) {
    std::cout << " Bitmap Correctness Verification Tests \n";
    WahBackend wah;
    CroaringBackend croaring;

    auto w_btv1 = wah.Create();
    auto c_btv1 = croaring.Create();
    auto w_btv2 = wah.Create();
    auto c_btv2 = croaring.Create();

    // add 1, 0, 1, 1 (want decode have 0, 2, 3)
    std::vector<bool> seq1 = {true, false, true, true};
    for(bool b : seq1) {
        wah.Append(*w_btv1, b);
        croaring.Append(*c_btv1, b);
    }
    
    auto dec_w1 = wah.Decode(*w_btv1);
    auto dec_c1 = croaring.Decode(*c_btv1);
    
    std::cout << "\n--- Test 1: Append(1,0,1,1) ---\n";
    print_array("WAH Decode", dec_w1);
    if(dec_w1 == dec_c1 && dec_w1.size() == 3) {
        std::cout << "[PASS] Append & Decode are exactly correct and identical!\n";
    } else {
        std::cout << "[FAIL] Data mismatch!\n";
    }

    // btv2 add 0, 1, 1, 0 (want decode have 1, 2)
    std::vector<bool> seq2 = {false, true, true, false};
    for(bool b : seq2) {
        wah.Append(*w_btv2, b);
        croaring.Append(*c_btv2, b);
    }

    auto w_or = wah.bitOr(*w_btv1, *w_btv2);
    auto c_or = croaring.bitOr(*c_btv1, *c_btv2);

    std::cout << "\n--- Test 2: bitOr Result ---\n";
    auto dec_wor = wah.Decode(*w_or);
    auto dec_cor = croaring.Decode(*c_or);
    print_array("WAH OR Result", dec_wor);
    if(dec_wor == dec_cor) {
        std::cout << "[PASS] WAH and CRoaring OR results are identical!\n";
    } else {
        std::cout << "[FAIL] OR logic mismatch!\n";
    }

    std::cout << "\n--- Test 3: Cardinality ---\n";
    uint64_t w_card = wah.Cardinality(*w_or);
    uint64_t c_card = croaring.Cardinality(*c_or);
    std::cout << "WAH Cardinality: " << w_card << ", CRoaring: " << c_card << "\n";
    if(w_card == c_card) {
        std::cout << "[PASS] Cardinality calculation matches!\n";
    }

    //Serialize & Load
    std::cout << "\n--- Test 4: Serialize & Load ---\n";
    wah.Serialize(*w_or, "correctness_test.bin");
    auto w_loaded = wah.Load("correctness_test.bin");
    auto dec_loaded = wah.Decode(*w_loaded);
    
    if(dec_loaded == dec_wor) {
        std::cout << "[PASS] Serialize and Load is consistent without data loss!\n";
    }

    std::cout << "\n=======================================\n";
    std::cout << " All Baseline Correctness Tests Finished!\n";
    return 0;
}