// ================================================================
// CHƯƠNG TRÌNH DEMO: FAST GAUSSIAN BLUR VỚI SONG SONG HÓA
// ================================================================
// Chương trình này minh họa việc sử dụng thuật toán Fast Gaussian Blur
// được tối ưu với song song hóa (parallelization) sử dụng OpenMP.
//
// SONG SONG HÓA: 
// - Sử dụng OpenMP để tận dụng tất cả CPU cores
// - Song song hóa theo hàng trong horizontal blur
// - Song song hóa theo block trong transpose
// - Tăng tốc đáng kể trên CPU đa nhân
//
// ================================================================

#include <iostream>
#include <chrono>  // Để đo thời gian thực thi

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
// Include header chứa implementation của Fast Gaussian Blur
// Thuật toán này sử dụng nhiều lần box blur để xấp xỉ Gaussian blur
// và được tối ưu với song song hóa (parallelization) sử dụng OpenMP
#include "fast_gaussian_blur_template.h"

typedef unsigned char uchar;         // Đặt alias uchar = unsigned char (giá trị 0–255)

// #define USE_FLOAT                // Uncomment nếu muốn xử lý ảnh dạng float (0.0–1.0) thay vì uchar (0–255)

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

    printf("Source image: %s %dx%d (%d)\n",
            argv[1], width, height, channels);
    // width  = chiều rộng
    // height = chiều cao
    // channels = 3 (RGB), 4 (RGBA), 1 (GRAY)...

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

    // =====================
    // 3) TẠO BỘ ĐỆM (BUFFER)
    // =====================

    std::size_t size = width * height * channels; // số phần tử pixel tổng cộng

#ifdef USE_FLOAT
    // Nếu xử lý bằng float
    float * new_image = new float[size];
    float * old_image = new float[size];
#else
    // Nếu xử lý 8-bit (uchar)
    uchar * new_image = new uchar[size];
    uchar * old_image = new uchar[size];
#endif

    // =====================
    // 4) COPY DỮ LIỆU ẢNH VÀO old_image
    // =====================

    for(std::size_t i = 0; i < size; ++i)
    {
#ifdef USE_FLOAT
        // Chuyển pixel uchar (0–255) về float (0–1)
        old_image[i] = (float)image_data[i] / 255.f;
#else
        // Dùng thẳng giá trị pixel 8-bit
        old_image[i] = image_data[i];
#endif
    }

    // =====================
    // 5) BẮT ĐẦU ĐO THỜI GIAN
    // =====================
    auto start = std::chrono::system_clock::now(); 

    // =====================
    // 6) THỰC HIỆN GAUSSIAN BLUR
    // =====================

    // Hàm fast_gaussian_blur thực hiện Fast Gaussian Blur với song song hóa (parallelization):
    // 
    // Thuật toán chính:
    // - Tính bán kính box từ sigma (sử dụng công thức tối ưu)
    // - Blur ngang nhiều lần (N passes) - MỖI PASS ĐƯỢC SONG SONG HÓA
    // - Transpose ảnh (chuyển vị) - CŨNG ĐƯỢC SONG SONG HÓA
    // - Blur ngang lần nữa (thực chất là blur dọc trên ảnh gốc) - MỖI PASS ĐƯỢC SONG SONG HÓA
    // - Transpose lại để trả về dạng ban đầu - CŨNG ĐƯỢC SONG SONG HÓA
    //
    // SONG SONG HÓA (PARALLELIZATION):
    // - Việc song song hóa được thực hiện BÊN TRONG hàm fast_gaussian_blur
    // - Sử dụng OpenMP (#pragma omp parallel for) để song song hóa:
    //   * horizontal_blur: mỗi hàng (row) được xử lý bởi một thread riêng
    //   * flip_block: mỗi block được xử lý song song với collapse(2)
    // - Tự động tận dụng tất cả CPU cores có sẵn
    // - Tăng tốc đáng kể trên CPU đa nhân (multi-core)
    // - Để sử dụng song song hóa, cần compile với flag -fopenmp (GCC/Clang) hoặc /openmp (MSVC)
    //
    // Lưu ý: Các buffer old_image và new_image sẽ được swap nhiều lần trong quá trình xử lý
    fast_gaussian_blur(old_image, new_image,
                       width, height, channels,
                       sigma, passes, border);

    // =====================
    // 7) KẾT THÚC ĐO THỜI GIAN
    // =====================
    auto end = std::chrono::system_clock::now();
    float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>
                        (end-start).count();

    printf("Time %.4fms\n", elapsed);   // In thời gian xử lý blur

    // =====================
    // 8) COPY KẾT QUẢ BLUR VỀ image_data ĐỂ LƯU FILE
    // =====================

    for(std::size_t i = 0; i < size; ++i)
    {
#ifdef USE_FLOAT
        // scale float 0–1 về lại 0–255
        image_data[i] = (uchar)(new_image[i] * 255.f);
#else
        image_data[i] = (uchar)(new_image[i]);  // copy trực tiếp
#endif
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

    // =====================
    // 10) GIẢI PHÓNG BỘ NHỚ
    // =====================

    stbi_image_free(image_data);   // Giải phóng ảnh load từ file
    delete[] new_image;            // Giải phóng buffer kết quả
    delete[] old_image;            // Giải phóng buffer input

    return 0;
}
