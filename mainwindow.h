#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <math.h>

namespace Ui {
class MainWindow;
}
typedef unsigned char   u8;
typedef unsigned short  u16;
typedef unsigned int    u32;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    double activation_func(double val);
    double derivative_of_activation_func(double val);
public slots:
    void _2_2_1_ann_train(void);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
