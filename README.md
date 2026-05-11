# computational-fluid-dynamics
计算流体力学作业

## 作业二
代码见：homework2.py

相关图片为：

&emsp;&emsp;差分格式计算导数结果：

&emsp;&emsp;&emsp;&emsp;result_x=-1.png

&emsp;&emsp;&emsp;&emsp;result_x=0.png

&emsp;&emsp;差分格式误差：

&emsp;&emsp;&emsp;&emsp;error_x=-1.png

&emsp;&emsp;&emsp;&emsp;error_x=0.png

&emsp;&emsp;误差阶数展示图：

&emsp;&emsp;&emsp;&emsp;order_of_accuracy_x0 =  $x_0$  $h_{min}$  $h_{max}$.png

&emsp;&emsp;&emsp;&emsp;其中， $x_0$ 为计算时选取的点， $h_{min}$ ， $h_{max}$ 分布为选取的网格尺寸的最小值和最大值

&emsp;&emsp;对流方程差分格式解的稳定性、数值耗散误差与相位误差

&emsp;&emsp;&emsp;&emsp;u(t)_with_lambda= $\lambda$ .gif

&emsp;&emsp;&emsp;&emsp;其中， $\lambda$ 为网格比 $\lambda = \tau/h$ 的取值

### 运行环境说明

python 版本：Python 3.13.12

运行依赖的包：
&emsp;&emsp;name            version

&emsp;&emsp;matplotlib      3.10.9

&emsp;&emsp;numpy           2.4.4

&emsp;&emsp;pillow          12.2.0

&emsp;&emsp;scipy           1.17.1

### 编译运行说明
第 287-299 行代码实现运行功能。

说明：由于我是在 wsl 系统上完成代码运行，没有实时的图像窗口，生成的所有图像都将以 .png 或 .gif 的格式保存下来。

1、第 1 题

&emsp;&emsp;取消 288 - 293 行的注释，注释掉 299 行，运行文件 homework2.py，可得到第1题有关的所有图片。

&emsp;&emsp;若想得到特定的某一张图片，可通过运行第 296 行被注释掉的代码，$i$ 指示生成图片的类型，$x\_0$ 指示选取的坐标点。

&emsp;&emsp;&emsp;&emsp; $i$ = 0:

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;输出区间左侧单边的 5 节点模板差分格式的系数。

&emsp;&emsp;&emsp;&emsp; $i$ = 1:

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;绘制两幅图，一幅为两种差分格式及解析形式的计算结果随网格尺寸的变化图，一幅为两种差分格式的误差随网格尺寸的变化图

&emsp;&emsp;&emsp;&emsp; $i$ = 2:

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;生成比较不同点处不同差分格式误差的图片

&emsp;&emsp;&emsp;&emsp; $i$ = 3 - 6:

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;绘制差分格式误差和网格尺寸的双对数图，以计算差分格式的误差阶数，选取网格尺寸的范围及步长依次是： 0.05 到 1.05，步长为 0.01；0.01 到 0.51，步长为 0.005；0.001 到 0.101，步长为 0.01；0.0001 到 0.0101，步长为 0.01

2、第3题

&emsp;&emsp;相关运行代码为第 299 行， $l$ 代表网格比  $\lambda = \tau/h$ ，通过指定 $l$ = 1.1，1.0，0.5，0.2，可分别得到第3题的图片，图片保存为 .gif 格式。
    
        
