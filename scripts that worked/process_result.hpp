#include <fstream>
#include <opencv2/opencv.hpp>
#include <vitis/ai/classification.hpp>

cv::Mat process_result(cv::Mat& image, vitis::ai::ClassificationResult& result, 
                       bool save_result = false, 
                       const std::string& output_image_path = "result.jpg",
                       const std::string& log_file_path = "") {
    int baseLine = 0;
    double fontScale = 1.0;
    cv::Scalar color = cv::Scalar(0, 0, 255);
    int thickness = 1;
    int y_offset = 20; // Initial text offset from the top

    // Open the log file for writing results if a path is provided
    std::ofstream log_file;
    if (!log_file_path.empty()) {
        log_file.open(log_file_path, std::ios::app);
        log_file << "Top-K Classification Results:\n";
    }

    for (size_t i = 0; i < result.scores.size(); i++) {
        std::ostringstream label;
        label << "Class: " << result.scores[i].index << ", Score: " << result.scores[i].score;

        // Write to log file
        if (log_file.is_open()) {
            log_file << "Class: " << result.scores[i].index 
                     << ", Score: " << result.scores[i].score << "\n";
        }

        // Display on image
        cv::Size textSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseLine);
        
        while (textSize.width > (image.cols - 10)) {
            fontScale -= 0.1;
            textSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseLine);
        }

        cv::putText(image, label.str(), cv::Point(10, y_offset),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, color, thickness);
        y_offset += (textSize.height + 10);
    }

    // Close the log file if it was opened
    if (log_file.is_open()) {
        log_file.close();
    }

    if (save_result) {
        cv::imwrite(output_image_path, image);
    }

    return image;
}

