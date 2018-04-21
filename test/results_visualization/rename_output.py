import os

if __name__ == '__main__':
    folders = [
        'C:/Users/sssilvar/Documents/LaTeX/slides-seminar/results/curvelet/MCIc',
        'C:/Users/sssilvar/Documents/LaTeX/slides-seminar/results/curvelet/MCInc'
    ]
    ext = '.eps'

    for folder in folders:
        for root, dirs, files in os.walk(folder):
            for i, file in enumerate(files):
                if file.endswith(ext):
                    filename = os.path.join(root, file)
                    file_renamed = os.path.join(os.path.dirname(filename), '%d%s' % (i, ext))
                    print('File %i: %s --> %s' % (i, filename, file_renamed))
                    os.rename(filename, file_renamed)
