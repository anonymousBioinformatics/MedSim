import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import numpy as np




#ATTENTION: put the model iters you want to plot into the list
model_iter = [i for i in xrange(10,100,20)] + [i for i in xrange(100,1000,100)] +[i for i in xrange(1000,50000,500)]
for one_iter in model_iter:
	plt.clf()
	y_true = np.load('./data/allans.npy')
	y_scores = np.load('./out/sample_allprob_iter_'+str(one_iter)+'.npy')


	cha = len(y_true) - len(y_scores)
	y_true = y_true[:-cha]

	precision,recall,threshold = precision_recall_curve(y_true,y_scores)
	average_precision = average_precision_score(y_true, y_scores)


	plt.plot(recall, precision, lw=2, color='navy',label='BGRU+2ATT')



	y_true = np.load('./data for parsing tree/allans.npy')
	y_scores = np.load('./out for parsing tree/sample_allprob_iter_'+str(one_iter)+'.npy')

	cha = len(y_true) - len(y_scores)
	y_true = y_true[:-cha]

	precision,recall,threshold = precision_recall_curve(y_true,y_scores)
	average_precision = average_precision_score(y_true, y_scores)

	plt.plot(recall, precision, lw=2, color='darkorange',label='BGRU+2ATT+parsing')



	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall Area={0:0.2f}'.format(average_precision))
	plt.legend(loc="upper right")
	plt.grid(True)
	plt.savefig('images/iter_'+str(one_iter))









