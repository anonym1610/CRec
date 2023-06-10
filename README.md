# Getting Started with CRec

In software development, programmers reuse existing source code to improve their development workflow. However, source code is widely shared via images and videos, like scanned books or online learning video clips. It is time-consuming to manually extract source code and convert into editable code. This extraction can be automated via applying Optical Character Recognition (OCR) techniques. Due to the accuracy of recognizing source code via OCR, programmers have to manually recover the compilability for such OCRed source code, i.e., editable source code that is recognized via OCR.

CRec is an automatic tool for recovering the compilability of OCRed source code in Java programs. CRec aims to fix lexical and syntax errors in OCRed source code and make the output pass the compilation.

Before running CRec, you need to first configure its runtime environment. Navigate to this folder in the terminal and run this command:
```
pip install -r requirements.txt
```

Please put the OCR-based source code that need to be fixed into the ```./input``` folder, and you can begin to repair:
```
python ./main.py
```

The repair results will be saved in the ```./output``` folder. If exceptions are thrown during the repair process, the corresponding information will be saved in the ```./output/failures``` folder.

The training data for the n-gram language model used in CRec come from the source code of two popular Java projects, [Spring Framework](^[https://github.com/spring-projects/spring-framework]) and [Jenkins](https://github.com/jenkinsci/jenkins), from GitHub. We use [ANTLR](https://github.com/antlr/antlr4) to generate a lexer that tokenized the source code, resulting in 2,581,606 lines of tokens. The file containing these tokens is saved in the ```./tokens.zip```.

To obtain OCRed source code, we generate code images for OCR engines to recognize. We gather 563 compilable Java source code files from four classic Java textbooks, each with a maximum of 100 lines. Then we write a program to convert them into images using six different fonts. To increase realism, we choose one of four gray scale values as the background color and randomly adjust the rotation angle, brightness, contrast, and saturation of each image. Finally, we produce 563 code images. The collected code, generated code images, and recognition results from seven OCR engines are saved in the ```./dataset.zip```.