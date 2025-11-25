// ================================================================
// CHƯƠNG TRÌNH DEMO: FAST GAUSSIAN BLUR VỚI SONG SONG HÓA
// ================================================================
// Chương trình này minh họa việc sử dụng thuật toán Fast Gaussian Blur
// được tối ưu với song song hóa (parallelization) sử dụng OpenMP.
//
// Chương trình sẽ chạy và so sánh cả hai phiên bản:
// - Có OpenMP (song song hóa)
// - Không có OpenMP (single-threaded, dùng OMP_NUM_THREADS=1)
//
// ================================================================

#include <iostream>
#include <chrono>  // Để đo thời gian thực thi
#include <iomanip>  // Để format output
#include <string>
#include <cmath>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

// ================================================================
// THƯ VIỆN XỬ LÝ ẢNH
// ================================================================
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"               // Thư viện dùng để load (đọc) ảnh từ file (header-only)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"         // Thư viện dùng để lưu ảnh ra file (header-only)

// ================================================================
// THUẬT TOÁN FAST GAUSSIAN BLUR
// ================================================================
// Include header với OpenMP bật (mặc định)
#define USE_OPENMP 1
#include "fast_gaussian_blur_template.h"

typedef unsigned char uchar;         // Đặt alias uchar = unsigned char (giá trị 0–255)

// ================================================================
// HÀM HIỂN THỊ THỜI GIAN CHI TIẾT
// ================================================================
void print_detailed_time(const std::string& label, 
                         const std::chrono::high_resolution_clock::time_point& start,
                         const std::chrono::high_resolution_clock::time_point& end) {
    auto duration = end - start;
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    
    // Tính toán với độ chính xác cao
    double ms = milliseconds + (microseconds % 1000) / 1000.0;
    double us = microseconds + (nanoseconds % 1000) / 1000.0;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  " << std::setw(35) << std::left << label << ": ";
    std::cout << std::setw(12) << std::right << ms << " ms  ";
    std::cout << "(" << std::setw(12) << std::right << us << " µs)";
    std::cout << std::endl;
}

// ================================================================
// HÀM HIỂN THỊ KẾT QUẢ SO SÁNH
// ================================================================
void print_comparison_table(double time_with_omp, double time_without_omp) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              KẾT QUẢ SO SÁNH HIỆU NĂNG                                ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Phiên bản                    │  Thời gian (ms)  │  Tốc độ tăng     ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════╣\n";
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "║  " << std::setw(27) << std::left << "Có OpenMP (Multi-thread)" 
              << "│  " << std::setw(15) << std::right << time_with_omp << "  │  " 
              << std::setw(15) << std::right << "1.00x" << "  ║\n";
    
    std::cout << "║  " << std::setw(27) << std::left << "Không có OpenMP (Single)" 
              << "│  " << std::setw(15) << std::right << time_without_omp << "  │  " 
              << std::setw(15) << std::right << (time_without_omp / time_with_omp) << "x" << "  ║\n";
    
    double speedup = time_without_omp / time_with_omp;
    double improvement = ((time_without_omp - time_with_omp) / time_without_omp) * 100.0;
    double time_saved = time_without_omp - time_with_omp;
    
    std::cout << "╠═══════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Tăng tốc: " << std::setw(10) << std::right << speedup << "x  ";
    std::cout << "│  Cải thiện: " << std::setw(8) << std::right << improvement << "%  ";
    std::cout << "│  Tiết kiệm: " << std::setw(8) << std::right << time_saved << " ms  ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

// ================================================================
// HÀM CHÍNH
// ================================================================
int main(int argc, char * argv[])
{   
    // Kiểm tra số lượng tham số truyền vào
    if( argc < 4 )
    {
        // In hướng dẫn sử dụng
        printf("%s [input] [output] [sigma] [order - optional] [border - optional]\n", argv[0]);
        printf("\n");
        printf("- input:  file ảnh input (jpg/png/bmp/...)\n");
        printf("- output: file ảnh output muốn lưu (.png/.jpg/.bmp)\n");
        printf("- sigma:  độ mờ Gaussian (float, > 0)\n");
        printf("- order:  số lần blur (bộ lọc box đa cấp), mặc định = 3\n");
        printf("- border: cách xử lý biên ảnh [mirror, extend, crop, wrap]\n");
        printf("\n");
        exit(1);                     // Thoát chương trình vì thiếu tham số
    }

    // =====================
    // 1) LOAD ẢNH
    // =====================
    int width, height, channels;

    // stbi_load đọc file ảnh và trả về mảng pixel 1D (uchar*)
    uchar * image_data = stbi_load(argv[1], &width, &height, &channels, 0);

    if (!image_data) {
        printf("Lỗi: Không thể load ảnh từ file %s\n", argv[1]);
        exit(1);
    }

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║          FAST GAUSSIAN BLUR - SO SÁNH HIỆU NĂNG                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Source image: %s\n", argv[1]);
    printf("Kích thước: %dx%d pixels (%d channels)\n", width, height, channels);
    printf("Tổng số pixels: %d\n", width * height);
    printf("Tổng kích thước: %.2f MB\n", (width * height * channels) / (1024.0 * 1024.0));
    
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    printf("OpenMP: Có sẵn (Max threads: %d)\n", max_threads);
#else
    printf("OpenMP: Không có sẵn (sẽ chạy single-threaded)\n");
#endif
    printf("\n");

    // =====================
    // 2) ĐỌC THAM SỐ
    // =====================

    const float sigma = std::atof(argv[3]);   // Độ mờ Gaussian
    const int passes = argc > 4 ? std::atoi(argv[4]) : 3;  // Số lần blur (mặc định 3)

    const std::string policy = argc > 5
                                ? std::string(argv[5])
                                : "mirror";   // Cách xử lý biên

    Border border;
    if (policy == "mirror")         border = Border::kMirror;
    else if (policy == "extend")    border = Border::kExtend;
    else if (policy == "crop")      border = Border::kKernelCrop;
    else if (policy == "wrap")      border = Border::kWrap;
    else                            border = Border::kMirror; // Default

    printf("Tham số xử lý:\n");
    printf("  - Sigma: %.2f\n", sigma);
    printf("  - Passes: %d\n", passes);
    printf("  - Border policy: %s\n", policy.c_str());
    printf("\n");

    // =====================
    // 3) TẠO BỘ ĐỆM (BUFFER) CHO CẢ HAI PHIÊN BẢN
    // =====================

    std::size_t size = width * height * channels; // số phần tử pixel tổng cộng

    // Buffer cho phiên bản có OpenMP (multi-threaded)
    uchar * new_image_omp = new uchar[size];
    uchar * old_image_omp = new uchar[size];
    
    // Buffer cho phiên bản không có OpenMP (single-threaded)
    uchar * new_image_no_omp = new uchar[size];
    uchar * old_image_no_omp = new uchar[size];

    // =====================
    // 4) COPY DỮ LIỆU ẢNH VÀO BUFFER
    // =====================

    for(std::size_t i = 0; i < size; ++i)
    {
        old_image_omp[i] = image_data[i];
        old_image_no_omp[i] = image_data[i];
    }

    // =====================
    // 5) CHẠY PHIÊN BẢN CÓ OPENMP (MULTI-THREADED)
    // =====================
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  PHIÊN BẢN CÓ OPENMP (Song song hóa - Multi-threaded)                ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    
#ifdef _OPENMP
    // Đảm bảo sử dụng tất cả threads có sẵn
    omp_set_num_threads(omp_get_max_threads());
    printf("Số threads: %d\n", omp_get_max_threads());
#else
    printf("OpenMP không có sẵn, chạy single-threaded\n");
#endif
    printf("\n");
    
    auto start_omp = std::chrono::high_resolution_clock::now();
    
    fast_gaussian_blur(old_image_omp, new_image_omp,
                       width, height, channels,
                       sigma, passes, border);
    
    auto end_omp = std::chrono::high_resolution_clock::now();
    auto duration_omp = end_omp - start_omp;
    double time_omp_ms = std::chrono::duration_cast<std::chrono::microseconds>(duration_omp).count() / 1000.0;
    
    print_detailed_time("Tổng thời gian xử lý", start_omp, end_omp);
    printf("\n");

    // =====================
    // 6) CHẠY PHIÊN BẢN KHÔNG CÓ OPENMP (SINGLE-THREADED)
    // =====================
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  PHIÊN BẢN KHÔNG CÓ OPENMP (Single-threaded)                         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    
#ifdef _OPENMP
    // Đặt số threads = 1 để mô phỏng single-threaded
    omp_set_num_threads(1);
    printf("Số threads: 1 (single-threaded)\n");
#else
    printf("OpenMP không có sẵn, đã chạy single-threaded\n");
#endif
    printf("\n");
    
    auto start_no_omp = std::chrono::high_resolution_clock::now();
    
    fast_gaussian_blur(old_image_no_omp, new_image_no_omp,
                       width, height, channels,
                       sigma, passes, border);
    
    auto end_no_omp = std::chrono::high_resolution_clock::now();
    auto duration_no_omp = end_no_omp - start_no_omp;
    double time_no_omp_ms = std::chrono::duration_cast<std::chrono::microseconds>(duration_no_omp).count() / 1000.0;
    
    print_detailed_time("Tổng thời gian xử lý", start_no_omp, end_no_omp);
    printf("\n");

    // =====================
    // 7) HIỂN THỊ KẾT QUẢ SO SÁNH
    // =====================
    print_comparison_table(time_omp_ms, time_no_omp_ms);

    // =====================
    // 8) COPY KẾT QUẢ BLUR VỀ image_data ĐỂ LƯU FILE
    // (Sử dụng kết quả từ phiên bản có OpenMP)
    // =====================

    for(std::size_t i = 0; i < size; ++i)
    {
        image_data[i] = (uchar)(new_image_omp[i]);
    }

    // =====================
    // 9) LƯU ẢNH RA FILE
    // =====================
    std::string file(argv[2]);
    std::string ext = file.substr(file.size()-3);  // Lấy phần .png/.jpg...

    if( ext == "bmp" )
        stbi_write_bmp(argv[2], width, height, channels, image_data);
    else if( ext == "jpg" )
        stbi_write_jpg(argv[2], width, height, channels, image_data, 90); // chất lượng 90%
    else
    {
        // Nếu không phải png thì chuyển về png
        if( ext != "png" )
        {
            printf("Image format '%s' not supported, writing default png\n",
                   ext.c_str()); 
            file = file.substr(0, file.size()-4) + std::string(".png");
        }
        stbi_write_png(file.c_str(), width, height, channels,
                       image_data, channels * width); // stride = width*channels
    }
    
    printf("Đã lưu ảnh kết quả vào: %s\n", argv[2]);
    printf("\n");

    // =====================
    // 10) GIẢI PHÓNG BỘ NHỚ
    // =====================

    stbi_image_free(image_data);   // Giải phóng ảnh load từ file
    delete[] new_image_omp;         // Giải phóng buffer kết quả
    delete[] old_image_omp;         // Giải phóng buffer input
    delete[] new_image_no_omp;      // Giải phóng buffer kết quả
    delete[] old_image_no_omp;      // Giải phóng buffer input

    return 0;
}
