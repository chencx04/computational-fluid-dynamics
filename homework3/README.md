# computational-fluid-dynamics
计算流体力学作业

## 作业三
代码见：$calculate_equation.py$

相关图片为：

&emsp;&emsp;四种显式差分格式求解结果：

&emsp;&emsp;&emsp;&emsp;$u(x,t)_explicit with nt = 300.png$

&emsp;&emsp;&emsp;&emsp;$u(x,t)_explicit with nt = 3000.png$

&emsp;&emsp;四种显式差分格式的误差：

&emsp;&emsp;&emsp;&emsp;$error of explicit numerical solutions with nt = 300.png$

&emsp;&emsp;&emsp;&emsp;$error of explicit numerical solutions with nt = 3000.png$

&emsp;&emsp;迎风格式和Crank-Nicolson格式求解结果：

&emsp;&emsp;&emsp;&emsp;$u(x,t)_crank_nicolson with nt = 300.png$

&emsp;&emsp;迎风格式和Crank-Nicolson格式的误差对比：

&emsp;&emsp;&emsp;&emsp;$error of imexplicit numerical solutions with nt = 300.png$

&emsp;&emsp;迎风格式和Crank-Nicolson格式的稳定性对比：

&emsp;&emsp;&emsp;&emsp;$u(x,t)_with_lambda=1.10.gif$

### 运行环境说明

python 版本：Python 3.13.12

运行依赖的包：

&emsp;&emsp;matplotlib      3.10.9

&emsp;&emsp;numpy           2.4.4

&emsp;&emsp;pillow          12.2.0

&emsp;&emsp;scipy           1.17.1

### 编译运行说明
直接运行文件 $calculate_equation.py$ 即可

说明：代码不会弹出实时的图像窗口，生成的所有图像都将以 .png 或 .gif 的格式保存下来。

