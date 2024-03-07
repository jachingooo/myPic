#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include <QFileDialog>

#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    //槽函数
    void on_open_clicked();

    void on_sobel_clicked();

    void on_process_clicked();

    void on_canny_clicked();



    void on_smooth_clicked();

    void on_sharpen_clicked();

    void on_save_clicked();

    void on_histogram_clicked();

    void on_junheng_clicked();

    void on_spin_clicked();

    void on_cut_clicked();

    void on_task102_clicked();

    void on_task101_clicked();

    void on_pinlvyu_clicked();

    void on_highpass_clicked();

    void on_lowpass_clicked();

    void on_ideal_clicked();

    void on_gaosi_clicked();

    void on_batewosi_clicked();

    void on_task41_clicked();

    void on_task42_clicked();

    void on_task43_clicked();

    void on_task44_clicked();

private:
    //功能
    void sobel();
    void canny();
    void smooth();
    void sharpen();

    void processImg(int function);

    //其他细节
    QImage mat2qim(Mat  mat);//图像格式转化
    Mat qim2mat(QImage  qim);
    void showImg(Mat result);//展示图像
    //其他参数
    Ui::MainWindow *ui;
    QString inimg;
    QImage output;
    char opt = 'n';
    Mat result;
    char type ;
};
#endif // MAINWINDOW_H
