

#include "mainwindow.h"
#include "ui_mainwindow.h"

//理想变换
cv::Mat ideal_low_kernel(cv::Mat &scr, float sigma);
cv::Mat ideal_low_pass_filter(cv::Mat &src, float sigma);
cv::Mat ideal_high_kernel(cv::Mat &scr, float sigma);
cv::Mat ideal_high_pass_filter(cv::Mat &src, float sigma);
//高斯变换
cv::Mat gaussian_low_pass_kernel(cv::Mat scr, float sigma);
cv::Mat gaussian_low_pass_filter(cv::Mat &src, float d0);
cv::Mat gaussian_high_pass_kernel(cv::Mat scr, float sigma);
cv::Mat gaussian_high_pass_filter(cv::Mat &src, float d0);
//巴特沃斯变换
cv::Mat butterworth_low_kernel(cv::Mat &scr, float sigma, int n);
cv::Mat butterworth_low_pass_filter(cv::Mat &src, float d0, int n);
cv::Mat butterworth_high_kernel(cv::Mat &scr, float sigma, int n);
cv::Mat butterworth_high_pass_filter(cv::Mat &src, float d0, int n);
cv::Mat frequency_filter(cv::Mat &scr, cv::Mat &blur);
cv::Mat image_make_border(cv::Mat &src);
void fftshift(cv::Mat &plane0, cv::Mat &plane1);
void getcart(int rows, int cols, cv::Mat &x, cv::Mat &y);
Mat powZ(cv::InputArray src, double power);
Mat sqrtZ(cv::InputArray src);

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}
MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::on_open_clicked()
{
    inimg = QFileDialog::getOpenFileName(this,"请选择任意格式的图片","","");
    QImage img;
    img.load(inimg);
    ui->input->setPixmap(QPixmap::fromImage(img).scaled(ui->input->width(),ui->input->height()));
}
void MainWindow::on_sobel_clicked()
{
    opt = 's';
}
void MainWindow::on_process_clicked()
{
    if(opt=='h'){
        switch (type){
        case 'l':processImg(1);
            break;
        case 'g':processImg(2);
            break;
        case 'b':processImg(3);
            break;
        default:
            break;
        }
    }
    else if(opt=='l'){
        switch (type){
        case 'l':processImg(4);
            break;
        case 'g':processImg(5);
            break;
        case 'b':processImg(6);
            break;
        default:
            break;
        }

    }
    switch (opt) {
    case 's':
        sobel();
        break;
    case 'c':
        canny();
        break;
    case 't':
        smooth();
        break;
    case 'u':
        sharpen();
        break;
    default:
        break;
    }

}
void MainWindow::sobel()
{
    Mat srcImage = imread(inimg.toLocal8Bit().toStdString());

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;


    cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
    //    计算x方向的梯度
    Sobel(srcImage, grad_x,CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);

    //    计算y方向的梯度
    Sobel(srcImage, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);

    //    合并梯度
    Mat dstImage;
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dstImage);
    imwrite("D://sobel.jpg", dstImage);
    output.load("D://sobel.jpg");
    remove("D://sobel.jpg");
    ui->output->setPixmap(QPixmap::fromImage(output).scaled(ui->output->width(),ui->output->height()));
}
QImage MainWindow::mat2qim(Mat  mat)
{
    cvtColor(mat, mat, COLOR_BGR2RGB);
    QImage qim((const unsigned char*)mat.data, mat.cols, mat.rows, mat.step,
               QImage::Format_RGB888);
    return qim;
}
Mat MainWindow::qim2mat(QImage  qim)
{
    Mat mat = Mat(qim.height(), qim.width(),
                  CV_8UC3,(void*)qim.constBits(),qim.bytesPerLine());
    return mat;
}
void MainWindow::showImg(Mat result)
{
    Mat Rgb;
    cvtColor(result, Rgb, CV_BGR2RGB);//颜色空间转换
    QImage Img = QImage((const uchar*)(Rgb.data), Rgb.cols, Rgb.rows, Rgb.cols * Rgb.channels(), QImage::Format_RGB888);
    ui->output->setPixmap(QPixmap::fromImage(Img).scaled(ui->output->width(),ui->output->height()));
}
void MainWindow::on_canny_clicked()
{
    opt = 'c';
}
void MainWindow::canny()
{
    Mat srcImage = imread(inimg.toLocal8Bit().toStdString());
    Mat grayImage;
    Mat srcImage1 = srcImage.clone();
    cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
    Mat dstImage, edge;
    blur(grayImage, grayImage, Size(3,3));
    Canny(grayImage, edge, 150, 100, 3);
    dstImage.create(srcImage1.size(), srcImage1.type());
    dstImage = Scalar::all(0);
    srcImage1.copyTo(dstImage, edge);
    showImg(dstImage);

}
void MainWindow::on_smooth_clicked()
{
    opt ='t';
}
void MainWindow::smooth()
{
    /*-------中值滤波medianBlur-----------*/
    Mat src = imread(inimg.toLocal8Bit().toStdString());
    Mat MedianBlurImg;
    medianBlur(src, MedianBlurImg, 5);
    showImg(MedianBlurImg);

}
void MainWindow::on_sharpen_clicked()
{
    opt='u';
}
void MainWindow::sharpen()
{
    Mat src = imread(inimg.toLocal8Bit().toStdString());
    Mat s = src.clone();
    Mat kernel;
        kernel = (Mat_<int>(3, 3) <<
                      0, -1, 0,
                  -1, 4, -1,
                  0, -1, 0
                  );

    filter2D(s, s, s.depth(), kernel);
    result = src + s * 0.01 * 100;
    showImg(result);

}
void MainWindow::on_save_clicked()
{

    QString savepath = QFileDialog::getExistingDirectory(this,"请选择保存路径","");
//    output.save("D://output.jpg");
    imwrite(savepath.toLocal8Bit().toStdString()+"//output.jpg",result);

}
void MainWindow::on_histogram_clicked()
{
    Mat src = imread(inimg.toLocal8Bit().toStdString());
    //1、创建3个矩阵来处理每个通道输入图像通道。
    //我们用向量类型变量来存储每个通道，并用split函数将输入图像划分成3个通道。
    vector<Mat>bgr;
    split(src, bgr);

    //2、定义直方图的区间数 int numbers = 256;

    //3、定义变量范围并创建3个矩阵来存储每个直方图
    float range[] = { 0,256 };
    const float* histRange = { range };
    Mat b_hist, g_hist, r_hist;

    //4、使用calcHist函数计算直方图
    int numbins = 256;
    calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &numbins, &histRange);
    calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, &numbins, &histRange);
    calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, &numbins, &histRange);

    //5、创建一个512*300像素大小的彩色图像，用于绘制显示
    int width = 512;
    int height = 300;
    Mat histImage(height, width, CV_8UC3, Scalar(20, 20, 20));

    //6、将最小值与最大值标准化直方图矩阵
    normalize(b_hist, b_hist, 0, height, NORM_MINMAX);
    normalize(g_hist, g_hist, 0, height, NORM_MINMAX);
    normalize(r_hist, r_hist, 0, height, NORM_MINMAX);

    //7、使用彩色通道绘制直方图
    int binStep = cvRound((float)width / (float)numbins);  //通过将宽度除以区间数来计算binStep变量

    for (int i = 1; i < numbins; i++)
    {
        line(histImage,
             Point(binStep * (i - 1), height - cvRound(b_hist.at<float>(i - 1))),
             Point(binStep * (i), height - cvRound(b_hist.at<float>(i))),
             Scalar(255, 0, 0)
             );
        line(histImage,
             Point(binStep * (i - 1), height - cvRound(g_hist.at<float>(i - 1))),
             Point(binStep * (i), height - cvRound(g_hist.at<float>(i))),
             Scalar(0, 255, 0)
             );
        line(histImage,
             Point(binStep * (i - 1), height - cvRound(r_hist.at<float>(i - 1))),
             Point(binStep * (i), height - cvRound(r_hist.at<float>(i))),
             Scalar(0, 0, 255)
             );
    }
    showImg(histImage);


}
void MainWindow::on_junheng_clicked()
{
    // 读入图片
    Mat src = imread(inimg.toLocal8Bit().toStdString());
    // 色彩空间转换HSV色彩空间转换
    Mat HSV;
    cvtColor(src, HSV, COLOR_BGR2HSV);
    // 图像通道分离
    vector<Mat>channels;
    split(HSV, channels);
    // 图像直方图均衡化
    //equalizeHist(channels[0], channels[0]);
    //equalizeHist(channels[1], channels[1]);
    equalizeHist(channels[2], channels[2]);
    // 通道合并
    merge(channels, HSV);
    //色彩空间转换BGR
    cvtColor(HSV, result, COLOR_HSV2BGR);
    // 显示图形
    showImg(result);

}
void MainWindow::on_spin_clicked()
{
    Mat src = imread(inimg.toLocal8Bit().toStdString());
    transpose(src,src);
    flip(src, src, 1);
    showImg(src);

}
void MainWindow::on_cut_clicked()
{
    Mat src = imread(inimg.toLocal8Bit().toStdString());
    Rect rect(300, 200, 600, 400);
    Mat image_roi = src(rect);
    showImg(image_roi);
}
void MainWindow::on_task102_clicked()
{
    //素材图像尺寸
    const int step_x = 20;
    const int step_y = 20;
    Mat src = imread(inimg.toLocal8Bit().toStdString());
    cv::resize(src, src, Size(step_x*30, step_y*30), 1, 1, INTER_CUBIC);
    vector<Mat>images;
    string filename = "C:\\Users\\jachin\\Desktop\\pic\\";
    vector<String> imagePathList;
    glob(filename, imagePathList);
    int num = 15;
    for (int i = 0; i < num; i++)
    {
        Mat img = cv::imread(imagePathList[i]);

        cv::resize(img, img, Size(step_x, step_y), 1, 1, INTER_AREA);

        images.push_back(img);

    }
    int rows = src.rows;
    int cols = src.cols;
    //height:表示生成的蒙太奇图像需要多少张素材图像填充rows
    //width:表示生成的蒙太奇图像需要多少张素材图像填充cols
    int height = rows / step_y, width = cols / step_x;
    Mat dst = Mat(src.size(), CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            //index表示当前素材图像的索引
            int index = i * width + j;

            //将图像赋值给需要生成的蒙太奇图像对应区域
            images[index].copyTo(dst(Rect(j * step_x, i * step_y, step_x, step_y)));
        }
    }
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            //像素RGB值修改
            dst.at<Vec3b>(i, j)[0] = 0.312*dst.at<Vec3b>(i, j)[0] + 0.698*src.at<Vec3b>(i, j)[0];
            dst.at<Vec3b>(i, j)[1] = 0.312*dst.at<Vec3b>(i, j)[1] + 0.698*src.at<Vec3b>(i, j)[1];
            dst.at<Vec3b>(i, j)[2] = 0.312*dst.at<Vec3b>(i, j)[2] + 0.698*src.at<Vec3b>(i, j)[2];
        }
    }
    showImg(dst);

}
void MainWindow::on_task101_clicked()
{
    Mat src = imread(inimg.toLocal8Bit().toStdString());
    cv::resize(src, src, Size(640, 480), 1, 1, INTER_CUBIC);
    cvtColor( src, result, COLOR_BGR2GRAY );
    imwrite("D://out.jpg", result);
    output.load("D://out.jpg");
    remove("D://out.jpg");
    ui->output->setPixmap(QPixmap::fromImage(output).scaled(ui->output->width(),ui->output->height()));
}
void MainWindow::on_pinlvyu_clicked()
{
    Mat inputImag_1 = imread(inimg.toLocal8Bit().toStdString());
    Mat resultImage;

    Mat srcGray;
    cvtColor(inputImag_1, srcGray, CV_RGB2GRAY);

    int m = getOptimalDFTSize(srcGray.rows);
    int n = getOptimalDFTSize(srcGray.cols);

    Mat padded;

    //【1】把灰度图像放扩展到最佳尺寸,在右边和下边扩展图像,扩展边界部分填充为0;
    copyMakeBorder(srcGray, padded, 0, m - srcGray.rows,0, n - srcGray.cols, BORDER_CONSTANT, Scalar::all(0));
    cout<<padded.size()<<endl;
    //【2】为傅里叶变换的结果分配存储空间。
    //这里是获取了两个Mat,一个用于存放dft变换的实部，一个用于存放虚部,初始的时候,实部就是图像本身,虚部全为零
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(),CV_32F)};
    Mat complexImg;
    //将几个单通道的mat融合成一个多通道的mat,这里融合的complexImg既有实部又有虚部
    merge(planes,2,complexImg);
    // 【3】进行离散傅里叶变换
    //对上边合成的mat进行傅里叶变换,支持原地操作,傅里叶变换结果为复数.通道1存的是实部,通道二存的是虚部
    dft(complexImg,complexImg);
    //把变换后的结果分割到两个mat,一个实部,一个虚部,方便后续操作
    // 【4】复数转换为幅值
    split(complexImg,planes);
    // 【5】尺度缩放
    //这一部分是为了计算dft变换后的幅值，傅立叶变换的幅度值范围大到不适合在屏幕上显示。高值在屏幕上显示为白点，而低值为黑点，高低值的变化无法有效分辨。为了在屏幕上凸显出高低变化的连续性，我们可以用对数尺度来替换线性尺度,以便于显示幅值,计算公式如下:
    //=> log(1 + sqrt(Re(DFT(I))^2 +Im(DFT(I))^2))
    magnitude(planes[0],planes[1],planes[0]);
    Mat mag = planes[0];
    mag += Scalar::all(1);
    log(mag, mag);
    // 【6】剪切和重分布幅度图象限
    //crop the spectrum, if it has an odd number of rows or columns
    //修剪频谱,如果图像的行或者列是奇数的话,那其频谱是不对称的,要修剪
    mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
    Mat _magI = mag.clone();
    //这一步的目的仍然是为了显示,但是幅度值仍然超过可显示范围[0,1],我们使用 normalize() 函数将幅度归一化到可显示范围。
    normalize(_magI, _magI, 0, 1, CV_MINMAX);
    // imshow("before rearrange", _magI);

    //rearrange the quadrants of Fourier image
    //so that the origin is at the image center
    // 【7】重新分配象限，使（0,0）移动到图像中心，
    //在《数字图像处理》中，傅里叶变换之前要对源图像乘以（-1）^(x+y)进行中心化。
    //这是是对傅里叶变换结果进行中心化
    int cx = mag.cols/2;
    int cy = mag.rows/2;

    Mat tmp;
    Mat q0(mag, Rect(0, 0, cx, cy));   //Top-Left - Create a ROI per quadrant
    Mat q1(mag, Rect(cx, 0, cx, cy));  //Top-Right
    Mat q2(mag, Rect(0, cy, cx, cy));  //Bottom-Left
    Mat q3(mag, Rect(cx, cy, cx, cy)); //Bottom-Right

    //swap quadrants(Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    // swap quadrant (Top-Rightwith Bottom-Left)
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    // 【8】归一化，用0到1之间的浮点值将矩阵变换为可视的图像格式
    normalize(mag,mag, 0, 1, NORM_MINMAX);
    // imshow("Input Image", srcGray);
    imshow("spectrum magnitude Image",mag);

    //傅里叶的逆变换
    Mat ifft;
    idft(complexImg,ifft,DFT_REAL_OUTPUT);
    normalize(ifft,ifft,0,1,CV_MINMAX);
    // imshow("inverse fft Image",ifft);
}
void MainWindow::on_highpass_clicked()
{
    opt ='h';
}
void MainWindow::on_lowpass_clicked()
{
    opt='l';



}
void MainWindow::on_ideal_clicked()
{
    type='l';
}
void MainWindow::on_gaosi_clicked()
{
    type='g';
}
void MainWindow::on_batewosi_clicked()
{
    type='b';
}
void MainWindow::processImg(int function)
{
    Mat test = imread(inimg.toLocal8Bit().toStdString(),0);
    float D;
    switch(function){
    case 1:
        D = 5.0f;
        result = ideal_high_pass_filter(test, D);
        break;
    case 2:
        D = 5.0f;
        result = gaussian_high_pass_filter(test, D);
        break;
    case 3:
        D = 5.0f;
        result = butterworth_high_pass_filter(test, D, 2);
        break;
    case 4:
        D = 50.0f;
        result = ideal_low_pass_filter(test, D);
        break;
    case 5:
        D = 50.0f;
        result = gaussian_low_pass_filter(test, D);
        break;
    case 6:
        D = 50.0f;
        result = butterworth_low_pass_filter(test, D,2);
        break;
    }
    imwrite("D://sobel.jpg", result);
    output.load("D://sobel.jpg");
    remove("D://sobel.jpg");
    ui->output->setPixmap(QPixmap::fromImage(output).scaled(ui->output->width(),ui->output->height()));

}

void MainWindow::on_task41_clicked()
{
    Mat test = imread(inimg.toLocal8Bit().toStdString(),IMREAD_GRAYSCALE);
    result = ideal_low_pass_filter(test, 14.99f);

    Mat s = result.clone();
    Mat kernel;
    kernel = (Mat_<int>(3, 3) <<
                  0, -1, 0,
              -1, 4, -1,
              0, -1, 0
              );

    filter2D(s, s, s.depth(), kernel);
    result = result + s * 0.1 * 100;

    imwrite("D://sobel.jpg", result);
    output.load("D://sobel.jpg");
    remove("D://sobel.jpg");
    ui->output->setPixmap(QPixmap::fromImage(output).scaled(ui->output->width(),ui->output->height()));




}
void MainWindow::on_task42_clicked()
{
    Mat src = imread(inimg.toLocal8Bit().toStdString());
    Mat MedianBlurImg;
    medianBlur(src, MedianBlurImg, 5);
    Mat srcs = MedianBlurImg;
    Mat s = srcs.clone();
    Mat kernel;
    kernel = (Mat_<int>(3, 3) <<
                  0, -1, 0,
              -1, 4, -1,
              0, -1, 0
              );

    filter2D(s, s, s.depth(), kernel);
    result = srcs + s * 0.01 * 100;
    showImg(result);

}

void MainWindow::on_task43_clicked()
{
    Mat src = imread(inimg.toLocal8Bit().toStdString());

    Mat rect  = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    cv::Mat tmp;

    //可用开操作代替
    cv::erode(src, tmp, rect);
    cv::blur(tmp, tmp, cv::Size(3, 3));   //均值模糊使图片过渡自然
//    cv::imshow("1", tmp);//反向显示 图片更清晰
    showImg(tmp);

}


void MainWindow::on_task44_clicked()
{
    //实现的功能：边缘检测
    Mat src = imread(inimg.toLocal8Bit().toStdString());
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            if (src.at<Vec3b>(row, col) == Vec3b(255, 255, 255)) {
                src.at<Vec3b>(row, col)[0] = 0;
                src.at<Vec3b>(row, col)[1] = 0;
                src.at<Vec3b>(row, col)[2] = 0;
            }
        }
    }
    //锐化
    Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);//拉普拉斯算子
    Mat lapulasimg;
    Mat sharpimg = src;
    filter2D(src, lapulasimg, CV_32F, kernel);// 这里计算的颜色数据有可能是负值，所以深度传 CV_32F， 不要传 -1，原图的深度是 CV_8U，不能保存负值
    src.convertTo(sharpimg, CV_32F); // mat.type 由 CV_8UC3 转换为 CV_32FC3 ，为了下面的减法计算
    Mat resultimg = sharpimg - lapulasimg;
    lapulasimg.convertTo(lapulasimg, CV_8UC3);
    resultimg.convertTo(resultimg, CV_8UC3);
    //转为二值
    Mat binimg;
    cvtColor(resultimg, resultimg, CV_RGB2GRAY);
    threshold(resultimg, binimg, 40, 255, THRESH_BINARY);
    //距离变化
    Mat disimg;
    distanceTransform(binimg, disimg, DIST_L1, 3,5);// CV_32F表示输出图像的深度，通道数与输入图形一致

    imwrite("D://sobel.jpg", disimg);
    output.load("D://sobel.jpg");
    remove("D://sobel.jpg");
    ui->output->setPixmap(QPixmap::fromImage(output).scaled(ui->output->width(),ui->output->height()));


}



//滤波器代码
// 理想低通滤波器
cv::Mat ideal_low_pass_filter(cv::Mat &src, float sigma)
{
    cv::Mat padded = image_make_border(src);
    cv::Mat ideal_kernel = ideal_low_kernel(padded, sigma);
    cv::Mat result = frequency_filter(padded, ideal_kernel);
    return result;
}
// 理想低通滤波核函数
cv::Mat ideal_low_kernel(cv::Mat &scr, float sigma)
{
    cv::Mat ideal_low_pass(scr.size(), CV_32FC1); //，CV_32FC1
    float d0 = sigma;//半径D0越小，模糊越大；半径D0越大，模糊越小
    for (int i = 0; i < scr.rows; i++) {
        for (int j = 0; j < scr.cols; j++) {
            float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//分子,计算pow必须为float型
            if (d <= d0) {
                ideal_low_pass.at<float>(i, j) = 1;
            }
            else {
                ideal_low_pass.at<float>(i, j) = 0;
            }
        }
    }
    return ideal_low_pass;
}
// 理想高通滤波核函数
cv::Mat ideal_high_kernel(cv::Mat &scr, float sigma)
{
    cv::Mat ideal_high_pass(scr.size(), CV_32FC1); //，CV_32FC1
    float d0 = sigma;//半径D0越小，模糊越大；半径D0越大，模糊越小
    for (int i = 0; i < scr.rows; i++) {
        for (int j = 0; j < scr.cols; j++) {
            float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//分子,计算pow必须为float型
            if (d <= d0) {
                ideal_high_pass.at<float>(i, j) = 0;
            }
            else {
                ideal_high_pass.at<float>(i, j) = 1;
            }
        }
    }
    return ideal_high_pass;
}
// 理想高通滤波
cv::Mat ideal_high_pass_filter(cv::Mat &src, float sigma)
{
    cv::Mat padded = image_make_border(src);
    cv::Mat ideal_kernel = ideal_high_kernel(padded, sigma);
    cv::Mat result = frequency_filter(padded, ideal_kernel);
    return result;
}
// 高斯低通滤波核函数
cv::Mat gaussian_low_pass_kernel(cv::Mat scr, float sigma)
{
    cv::Mat gaussianBlur(scr.size(), CV_32FC1); //，CV_32FC1
    float d0 = sigma;//高斯函数参数，越小，频率高斯滤波器越窄，滤除高频成分越多，图像就越平滑
    for (int i = 0; i < scr.rows; i++) {
        for (int j = 0; j < scr.cols; j++) {
            float d = pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2);//分子,计算pow必须为float型
            gaussianBlur.at<float>(i, j) = expf(-d / (2 * d0*d0));//expf为以e为底求幂（必须为float型）
        }
    }
    return gaussianBlur;
}
// 高斯低通滤波
cv::Mat gaussian_low_pass_filter(cv::Mat &src, float d0)
{
    cv::Mat padded = image_make_border(src);
    cv::Mat gaussian_kernel = gaussian_low_pass_kernel(padded, d0);
    cv::Mat result = frequency_filter(padded, gaussian_kernel);
    return result;
}
// 高斯高通滤波核函数
cv::Mat gaussian_high_pass_kernel(cv::Mat scr, float sigma)
{
    cv::Mat gaussianBlur(scr.size(), CV_32FC1); //，CV_32FC1
    float d0 = sigma;
    for (int i = 0; i < scr.rows; i++) {
        for (int j = 0; j < scr.cols; j++) {
            float d = pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2);//分子,计算pow必须为float型
            gaussianBlur.at<float>(i, j) = 1 - expf(-d / (2 * d0*d0));;
        }
    }
    return gaussianBlur;
}
// 高斯高通滤波
cv::Mat gaussian_high_pass_filter(cv::Mat &src, float d0)
{
    cv::Mat padded = image_make_border(src);
    cv::Mat gaussian_kernel = gaussian_high_pass_kernel(padded, d0);//理想低通滤波器
    cv::Mat result = frequency_filter(padded, gaussian_kernel);
    return result;
}
// 巴特沃斯低通滤波核函数
cv::Mat butterworth_low_kernel(cv::Mat &scr, float sigma, int n)
{
    cv::Mat butterworth_low_pass(scr.size(), CV_32FC1); //，CV_32FC1
    float D0 = sigma;//半径D0越小，模糊越大；半径D0越大，模糊越小
    for (int i = 0; i < scr.rows; i++) {
        for (int j = 0; j < scr.cols; j++) {
            float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//分子,计算pow必须为float型
            butterworth_low_pass.at<float>(i, j) = 1.0f / (1.0f + pow(d / D0, 2 * n));
        }
    }
    return butterworth_low_pass;
}
// 巴特沃斯低通滤波
cv::Mat butterworth_low_pass_filter(cv::Mat &src, float d0, int n)
{
    // H = 1 / (1+(D/D0)^2n)   n表示巴特沃斯滤波器的次数
    // 阶数n=1 无振铃和负值    阶数n=2 轻微振铃和负值  阶数n=5 明显振铃和负值   阶数n=20 与ILPF相似
    cv::Mat padded = image_make_border(src);
    cv::Mat butterworth_kernel = butterworth_low_kernel(padded, d0, n);
    cv::Mat result = frequency_filter(padded, butterworth_kernel);
    return result;
}
// 巴特沃斯高通滤波核函数
cv::Mat butterworth_high_kernel(cv::Mat &scr, float sigma, int n)
{
    cv::Mat butterworth_high_pass(scr.size(), CV_32FC1); //，CV_32FC1
    float D0 = (float)sigma;  // 半径D0越小，模糊越大；半径D0越大，模糊越小
    for (int i = 0; i < scr.rows; i++) {
        for (int j = 0; j < scr.cols; j++) {
            float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//分子,计算pow必须为float型
            butterworth_high_pass.at<float>(i, j) =1.0f-1.0f / (1.0f + pow(d / D0, 2 * n));
        }
    }
    return butterworth_high_pass;
}
// 巴特沃斯高通滤波
cv::Mat butterworth_high_pass_filter(cv::Mat &src, float d0, int n)
{
    cv::Mat padded = image_make_border(src);
    cv::Mat butterworth_kernel = butterworth_high_kernel(padded, d0, n);
    cv::Mat result = frequency_filter(padded, butterworth_kernel);
    return result;
}
// 频率域滤波
cv::Mat frequency_filter(cv::Mat &scr, cv::Mat &blur)
{
    cv::Mat mask = scr == scr;
    scr.setTo(0.0f, ~mask);

    //创建通道，存储dft后的实部与虚部（CV_32F，必须为单通道数）
    cv::Mat plane[] = { scr.clone(), cv::Mat::zeros(scr.size() , CV_32FC1) };

    cv::Mat complexIm;
    cv::merge(plane, 2, complexIm); // 合并通道 （把两个矩阵合并为一个2通道的Mat类容器）
    cv::dft(complexIm, complexIm); // 进行傅立叶变换，结果保存在自身

    // 分离通道（数组分离）
    cv::split(complexIm, plane);

    // 以下的操作是频域迁移
    fftshift(plane[0], plane[1]);

    // *****************滤波器函数与DFT结果的乘积****************
    cv::Mat blur_r, blur_i, BLUR;
    cv::multiply(plane[0], blur, blur_r);  // 滤波（实部与滤波器模板对应元素相乘）
    cv::multiply(plane[1], blur, blur_i);  // 滤波（虚部与滤波器模板对应元素相乘）
    cv::Mat plane1[] = { blur_r, blur_i };

    // 再次搬移回来进行逆变换
    fftshift(plane1[0], plane1[1]);
    cv::merge(plane1, 2, BLUR); // 实部与虚部合并

    cv::idft(BLUR, BLUR);       // idft结果也为复数
    BLUR = BLUR / BLUR.rows / BLUR.cols;

    cv::split(BLUR, plane);//分离通道，主要获取通道

    return plane[0];
}
// 图像边界处理
cv::Mat image_make_border(cv::Mat &src)
{
    int w = cv::getOptimalDFTSize(src.cols); // 获取DFT变换的最佳宽度
    int h = cv::getOptimalDFTSize(src.rows); // 获取DFT变换的最佳高度

    cv::Mat padded;
    // 常量法扩充图像边界，常量 = 0
    cv::copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    padded.convertTo(padded, CV_32FC1);

    return padded;
}
// 实现频域滤波器的网格函数
void getcart(int rows, int cols, cv::Mat &x, cv::Mat &y) {
    x.create(rows, cols, CV_32FC1);
    y.create(rows, cols, CV_32FC1);
    //设置边界

    //计算其他位置的值
    for (int i = 0; i < rows; ++i) {
        if (i <= rows / 2) {
            x.row(i) = i;
        }
        else {
            x.row(i) = i - rows;
        }
    }
    for (int i = 0; i < cols; ++i) {
        if (i <= cols / 2) {
            y.col(i) = i;
        }
        else {
            y.col(i) = i - cols;
        }
    }
}
// fft变换后进行频谱搬移
void fftshift(cv::Mat &plane0, cv::Mat &plane1)
{
    // 以下的操作是移动图像  (零频移到中心)
    int cx = plane0.cols / 2;
    int cy = plane0.rows / 2;
    cv::Mat part1_r(plane0, cv::Rect(0, 0, cx, cy));  // 元素坐标表示为(cx, cy)
    cv::Mat part2_r(plane0, cv::Rect(cx, 0, cx, cy));
    cv::Mat part3_r(plane0, cv::Rect(0, cy, cx, cy));
    cv::Mat part4_r(plane0, cv::Rect(cx, cy, cx, cy));

    cv::Mat temp;
    part1_r.copyTo(temp);  //左上与右下交换位置(实部)
    part4_r.copyTo(part1_r);
    temp.copyTo(part4_r);

    part2_r.copyTo(temp);  //右上与左下交换位置(实部)
    part3_r.copyTo(part2_r);
    temp.copyTo(part3_r);

    cv::Mat part1_i(plane1, cv::Rect(0, 0, cx, cy));  //元素坐标(cx,cy)
    cv::Mat part2_i(plane1, cv::Rect(cx, 0, cx, cy));
    cv::Mat part3_i(plane1, cv::Rect(0, cy, cx, cy));
    cv::Mat part4_i(plane1, cv::Rect(cx, cy, cx, cy));

    part1_i.copyTo(temp);  //左上与右下交换位置(虚部)
    part4_i.copyTo(part1_i);
    temp.copyTo(part4_i);

    part2_i.copyTo(temp);  //右上与左下交换位置(虚部)
    part3_i.copyTo(part2_i);
    temp.copyTo(part3_i);
}
Mat powZ(cv::InputArray src, double power) {
    cv::Mat dst;
    cv::pow(src, power, dst);
    return dst;
}
Mat sqrtZ(cv::InputArray src) {
    cv::Mat dst;
    cv::sqrt(src, dst);
    return dst;
}












