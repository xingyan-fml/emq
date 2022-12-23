# emq

data/:
    The datasets. The two largest are not included. They can be downloaded from https://drive.google.com/drive/folders/1J9DwTdoLChWVZHAeYdbHx4G0HQHRmsZJ?usp=sharing

collect_results/:
    Collect results after finishing running all models. Run all .py in this folder.

emq/:
    The EMQ approach. Run "python exper.py" in it, then a folder "emq/results/" will be created.

emqw/:
    The EMQW approach. Run "python exper.py" in it, then a folder "emqw/results/" will be created.

deep_ensemble/:
    The HNN and Deep Ensemble methods. Run "python exper.py" ...

mc_dropout/:
    The MC Dropout method. Run "python exper.py" ...

concrete_dropout/:
    The Concrete Dropout method. Run "python exper.py" ...

vanilla/:
    The Vanilla QR method. Run "python exper.py" ...

vanilla.w/:
    The QRW method. Run "python exper.py" ...

sqr/:
    The SQR method. Run "python exper.py" ...

interval/:
    The Interval Score method. Run "python exper.py" ...

gbdt/:
    The GBDT method. Run "python exper.py" ...

lgbm/:
    The LightGBM method. Run "python exper.py" ...
