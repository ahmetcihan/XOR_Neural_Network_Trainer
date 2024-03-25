#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->pushButton_train,SIGNAL(clicked(bool)),this,SLOT(_2_2_1_ann_train()));

}
double MainWindow::activation_func(double val){
    return (1 / (1 + exp(-val)));     //sigmoid   - good
    //return tanh(val);                 //tanh      - works but slow and uses small learning rate
    //return val;                       //identity  - not properly worked
    //return atan(val);                   //atan      - not bad but slower
    //return (log(1+exp(val)));         //softplus  - good but slower
    /*********Leaky RELU**********/     //ReLU      -
    //if(val <= 0) return (0.01*val);
    //else return val;
    /*****************************/
    //return (val / (1 + exp(-val)));   //swish     - not bad but not good
    //return exp(-1*val*val);           //gaussien  - better than sigmoid
}
double MainWindow::derivative_of_activation_func(double val){
    return (activation_func(val) * (1 - activation_func(val)));
    //return (1 - tanh(val)*tanh(val));
    //return 1;
    //return (1 / (1 + val*val));
    //return (1 / (1 + exp(-val)));
    /***************RELU**********/
    //if(val < 0) return 0.01;
    //else return 1;
    /*****************************/
    //return (1 + exp(-val) + val*exp(-val))/((1 + exp(-val))*(1 + exp(-val)));
    //return -2*val*sigmoid_func(val);
}
void MainWindow::_2_2_1_ann_train(void){
    double input1[4] = {1,1,0,0};
    double input2[4] = {1,0,1,0};
    double desired_output[4] = {0,1,1,0};
    double calculated_output[4] = {0,0,0,0};
    double Y_in,Y_out;
    double w_input_to_hidden[2][2];
    double w_hidden_to_output[2];

    double A_in,B_in;
    double A_out,B_out;
    double output_error;

    double biasA = 0.17;
    double biasB = 0.18;
    double bias_output = 0.19;

    double errorA,errorB;

    u32 epoch = ui->spinBox->value();
    double learning_rate = ui->doubleSpinBox->value();

    double global_error;
    double err_sum;

    w_input_to_hidden[0][0] = 0.11;
    w_input_to_hidden[0][1] = 0.12;

    w_input_to_hidden[1][0] = 0.13;
    w_input_to_hidden[1][1] = 0.14;

    w_hidden_to_output[0] = 0.15;
    w_hidden_to_output[1] = 0.16;

    for(u32 era = 0; era < epoch; era++){
        for(u8 k = 0; k < 4; k++){
            A_in = input1[k]*w_input_to_hidden[0][0] + input2[k]*w_input_to_hidden[1][0] + biasA;
            B_in = input1[k]*w_input_to_hidden[0][1] + input2[k]*w_input_to_hidden[1][1] + biasB;

            A_out = activation_func(A_in);
            B_out = activation_func(B_in);

            Y_in = A_out*w_hidden_to_output[0] + B_out*w_hidden_to_output[1] + bias_output;
            Y_out = activation_func(Y_in);
            output_error = desired_output[k] - Y_out;
            calculated_output[k] = Y_out;

            err_sum = output_error;

            global_error = derivative_of_activation_func(Y_in) * output_error;

            w_hidden_to_output[0] += global_error * A_out * learning_rate;
            w_hidden_to_output[1] += global_error * B_out * learning_rate;

            bias_output += global_error * learning_rate;

            errorA = derivative_of_activation_func(A_in) * global_error * w_hidden_to_output[0];
            errorB = derivative_of_activation_func(B_in) * global_error * w_hidden_to_output[1];

            w_input_to_hidden[0][0] += errorA * input1[k] * learning_rate;
            w_input_to_hidden[0][1] += errorB * input1[k] * learning_rate;

            w_input_to_hidden[1][0] += errorA * input2[k] * learning_rate;
            w_input_to_hidden[1][1] += errorB * input2[k] * learning_rate;

            biasA += errorA * learning_rate;
            biasB += errorB * learning_rate;
        }
        ui->label_status->setText(QString("Era-%1 , Total error: %2").arg(era).arg(err_sum));
        //qDebug() << QString("era : %1").arg(era) << "error :" << err_sum;
        QApplication::processEvents();
    }

    for(u8 i = 0; i < 2; i++){
        for(u8 j = 0; j < 2; j++){
            qDebug() << QString("in_to_h_w[%1][%2] :").arg(i).arg(j) << w_input_to_hidden[i][j];
        }
    }
    for(u8 j = 0; j < 2; j++){
        qDebug() << QString("h_to_o_w[%1] :").arg(j) << w_hidden_to_output[j];
    }
    qDebug() << "biasA" << biasA;
    qDebug() << "biasB" << biasB;
    qDebug() << "bias_output" << bias_output;
    qDebug() << "desired 0 : " << desired_output[0] << "calculated 0 : " << calculated_output[0];
    qDebug() << "desired 1 : " << desired_output[1] << "calculated 1 : " << calculated_output[1];
    qDebug() << "desired 2 : " << desired_output[2] << "calculated 2 : " << calculated_output[2];
    qDebug() << "desired 3 : " << desired_output[3] << "calculated 3 : " << calculated_output[3];
}

MainWindow::~MainWindow()
{
    delete ui;
}
