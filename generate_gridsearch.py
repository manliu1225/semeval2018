from itertools import product
import subprocess
classifier__num_leaves=[16,32,64, 128]
classifier__n_estimators=[100, 300, 700, 1000]
# classifier__num_leaves=[16, 32]
# classifier__n_estimators=[100]

for i, j in product(classifier__num_leaves, classifier__n_estimators):
	print(i, j)
	errf = open('{}{}error.txt'.format(i,j), 'w')
	outf = open('{}{}out.txt'.format(i,j), 'w')
	subprocess.Popen('python test.py --X_train data/data-split-20171027/1/train_1.text --y_train data/data-split-20171027/1/train_1.labels \
		--X_test data/data-split-20171027/1/test_1.text --y_test data/data-split-20171027/1/test_1.labels \
		--num_leaves {} --n_estimators {} --save_model'.format(i, j), shell=True, stderr=errf, stdout=outf)