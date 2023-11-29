# Judge-a-book-by-its-cover
Multi-label Classification task using image and text data

![contest](https://github.com/akpe12/Judge-a-book-by-its-cover/assets/77143331/490ace19-3971-46f2-be0a-153b63180f42)

Training, validation 시에 모델이 중복적으로 오답을 예측하는 training과 validation example을 filtering 한 후, filtered training, validation dataset을 만듦.

중복적으로 오답을 예측하는 training, validation example을 filtering 함으로써 outlier를 제거할 수 있을 것이라고 생각했음.

따라서, 약 6만 8천 여개의 full dataset에서 96개의 outlier example들을 제거하여 filtered training dataset 제작 후에 학습을 진행하였더니 성능의 개선을 확인할 수 있었음.
![중복_filtered](https://github.com/akpe12/Judge-a-book-by-its-cover/assets/77143331/48198792-58ae-46f7-917e-bec723ab748f)
