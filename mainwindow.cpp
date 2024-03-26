#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->pushButton_train,SIGNAL(clicked(bool)),this,SLOT(_2_2_1_ann_train()));

    QString imagePath = QFileInfo(__FILE__).absolutePath() + "/net.png";
    QPixmap my_pixmap;
    my_pixmap.load(imagePath);
    ui->label_net->setPixmap(my_pixmap);
}
double MainWindow::activation_func(double val){
    return (1 / (1 + exp(-val)));     //sigmoid   - good
    //return tanh(val);                 //tanh      - works but slow and uses small learning rate
    //return val;                       //identity  - not properly worked
    //return atan(val);                 //atan      - not bad but slower
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
    double Y_in = 0,Y_out = 0;
    double w_input_to_hidden[2][2];
    double w_hidden_to_output[2];

    double A_in = 0,B_in = 0;
    double A_out = 0,B_out = 0;
    double output_error = 0;
    double global_error = 0;
    double err_sum = 0;
    double errorA = 0,errorB = 0;

    u32 total_epochs = ui->spinBox->value();
    double learning_rate = ui->doubleSpinBox->value();

    double biasA = ui->doubleSpinBox_biasA->value();
    double biasB = ui->doubleSpinBox_biasB->value();
    double bias_output = ui->doubleSpinBox_biasO->value();

    w_input_to_hidden[0][0] = ui->doubleSpinBox_w1->value();
    w_input_to_hidden[0][1] = ui->doubleSpinBox_w2->value();

    w_input_to_hidden[1][0] = ui->doubleSpinBox_w3->value();
    w_input_to_hidden[1][1] = ui->doubleSpinBox_w4->value();

    w_hidden_to_output[0] = ui->doubleSpinBox_w5->value();
    w_hidden_to_output[1] = ui->doubleSpinBox_w6->value();

    for(u32 epoch = 0; epoch < total_epochs; epoch++){
        err_sum = 0;
        for(u8 k = 0; k < 4; k++){
            A_in = input1[k]*w_input_to_hidden[0][0] + input2[k]*w_input_to_hidden[1][0] + biasA;
            B_in = input1[k]*w_input_to_hidden[0][1] + input2[k]*w_input_to_hidden[1][1] + biasB;

            A_out = activation_func(A_in);
            B_out = activation_func(B_in);

            Y_in = A_out*w_hidden_to_output[0] + B_out*w_hidden_to_output[1] + bias_output;
            Y_out = activation_func(Y_in);
            output_error = desired_output[k] - Y_out;
            calculated_output[k] = Y_out;

            err_sum += output_error;

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
        ui->label_status->setText(QString("Epoch : %1 , Total error: %2").arg(epoch).arg(err_sum));
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

    QString debugText = ""; // Initialize an empty string to build the output
    debugText += "w1: " + QString::number(w_input_to_hidden[0][0]) + "\n";
    debugText += "w2: " + QString::number(w_input_to_hidden[0][1]) + "\n";
    debugText += "w3: " + QString::number(w_input_to_hidden[1][0]) + "\n";
    debugText += "w4: " + QString::number(w_input_to_hidden[1][1]) + "\n";
    debugText += "w5: " + QString::number(w_hidden_to_output[0]) + "\n";
    debugText += "w6: " + QString::number(w_hidden_to_output[1]) + "\n";
    debugText += "biasA: " + QString::number(biasA) + "\n";
    debugText += "biasB: " + QString::number(biasB) + "\n";
    debugText += "bias_output: " + QString::number(bias_output) + "\n";

    // Loop through input/desired/calculated values and append to the string
    for (int i = 0; i < 4; ++i) {
        debugText += QString("input1: %1, input2: %2, output: %3, calculated output: %4\n")
                   .arg(input1[i])
                   .arg(input2[i])
                   .arg(desired_output[i])
                   .arg(calculated_output[i]);
    }
    ui->label_calculated->setText(debugText); // Set the label text with the accumulated string
}
MainWindow::~MainWindow()
{
    delete ui;
}
