# MP2 2023 - G29

## Project structure

```
extra_files/               # folder with extra files used in the project
    models/                # contains the models trained and tested only with train.txt
        testing_logreg.py
        testing_nb.py
        testing_svc.py
    results/               # contains the .txt files for the models that weren't chosen
        results_logreg.txt
        results_svc.txt
    reviews/               # contains the .py files trained with train.txt and tested with
                           # test_just_reviews.txt (for the models that weren't chosen)
        testing_logreg.py
        testing_svc.py

given_files/               # folder with the files given by the teachers
    MP2_enunciado2023.pdf
    test_just_reviews.txt
    train.txt

results.txt                 # contains the results of the chosen model (output of reviews.py)
reviews.py                  # contains the python code with the chosen model (trained with train.txt and tested with test_just_reviews.txt)
```