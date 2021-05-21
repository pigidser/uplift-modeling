from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    """
    Allows to suppress unwilling output

    print("Now you see it")
    with suppress_stdout():
        print("Now you don't")
    
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
def get_model_list():
    """Get all model names in work folder"""
    filenames = []
    for name in os.listdir(os.getcwd()):
        filename = name.lower()
        if os.path.splitext(filename)[1] == '.meta' and filename.find('_ext'):
            filenames.append(filename[:filename.find('_ext')])
    return filenames

def remove_model_data(filenames):
    """Remove all files related with models in filenames list"""
    for filename in filenames:
        meta_file1 = os.path.join(os.getcwd(), filename+'_ext.meta')
        meta_file2 = os.path.join(os.getcwd(), model.output_dir, filename+'.meta')
        model_file = os.path.join(os.getcwd(), model.output_dir, filename+'.pkl')
        metrics_file = os.path.join(os.getcwd(), model.output_dir, filename+'_all_metrics.xlsx')
        confidence_file = os.path.join(os.getcwd(), model.output_dir, filename+'_confidence_criteria.xlsx')
        if os.path.isfile(meta_file1):
            os.remove(meta_file1)
        if os.path.isfile(meta_file2):
            os.remove(meta_file2)
        if os.path.isfile(model_file):
            os.remove(model_file)
        if os.path.isfile(metrics_file):
            os.remove(metrics_file)
        if os.path.isfile(confidence_file):
            os.remove(confidence_file)
