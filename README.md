# Getting Started with CRec

In software development, programmers reuse existing source code to improve their development workflow. However, source code is widely shared via images and videos, like scanned books or online learning video clips. It is time-consuming to manually extract source code and convert into editable code. This extraction can be automated via applying Optical Character Recognition (OCR) techniques. Due to the accuracy of recognizing source code via OCR, programmers have to manually fix lexical and syntax errors for such OCRed source code, i.e., recovering source code from the recognition results of OCR.

The state-of-the-art method for fixing errors in OCRed source code, [CodeT5-OCRfix](https://github.com/akmalkadi/ase23-main-771), relies on fine-tuning pre-trained code models, but a single fine-tuning
phase may not capture all lexical and syntax errors. Building on the success of CodeT5-OCRfix with lexical correction, CRec is designed to address fine-grained lexical and syntax errors that CodeT5-OCRfix may overlook. The goal of CRec is to make the output pass the compilation and save the time cost of manual modification by developers.

Before running CRec, you need to first configure its runtime environment. Navigate to this folder in the terminal and run this command:
```
pip install -r requirements.txt
```

After installing `nltk`, please download `punkt` with `nltk.download('punkt')`.

Please put the OCRed source code that needs to be fixed into the ```input``` folder, and you can begin to repair:
```
python main.py
```

The repair results will be saved in the ```output``` folder. If exceptions are thrown during the repair process, the corresponding information will be saved in the ```output/failures``` folder.


We use [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) to format code in the repair process. For Windows users, we have included the ClangFormat executable in the project. For Linux users, you need to install ClangFormat on your system and modify the relevant lines in `main.py` that call ClangFormat.

In the patch generation stage, We employ the [unixcoder-base](https://huggingface.co/microsoft/unixcoder-base) model from
[UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder) to predict and fill in the masked tokens. `unixcoder.py` is based on UniXcoder, with minor modifications to make it compatible with the CRec framework.

Please note that this project only implements CRec, not the CodeT5-OCRfix + CRec workflow. However, the latter can be easily implemented by integrating [our replicated CodeT5-OCRfix model](https://zenodo.org/records/11559865) with CRec.


To obtain OCRed source code, we generated code images for OCR engines to recognize. We gathered 563 compilable Java source code files from four classic Java textbooks, each with a maximum of 100 lines. Then we wrote a program to convert them into images using six different fonts. To increase realism, we chose one of four gray scale values as the background color and randomly adjusted the rotation angle, brightness, contrast, and saturation of each image. Finally, we produced 563 code images. The collected code, generated code images, and recognition results from seven OCR engines are saved in ```dataset.zip```.

This project is licensed under the MIT License - see the LICENSE file for details.

Additionally, this project includes components licensed under the BSD-3-Clause License. Please refer to the LICENSE-BSD-3-Clause file for details.