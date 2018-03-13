# -*- coding:utf-8 -*-

import numpy as np


class Parsing_tree():
	id = ''
	word = ''
	root = ''

	def __init__(self, line):
		self.id = line[0]
		self.word = line[1]
		self.root = line[2]
		self.e1 = line[3]
		self.e2 = line[4]
		self.dep = line[5]
		self.pos1 = line[6]
		self.pos2 = line[7]

	def get_info(self):
		try:
			res = [dic_root[self.root] + 1, dic_e1[self.e1] + 1, dic_e2[self.e2] + 1, dic_dep[self.dep] + 1
			, dic_pos1[self.pos1] + 1, dic_pos2[self.pos2] + 1]
		except:
			res = [0] * 6
		return res


dic_root = {"r_r": 0,"r_c": 1, "r_o": 2}
dic_e1 = {"e1_e1": 0,"e1_c": 1,"e1_p": 2,"e1_o": 3}
dic_e2 = {"e2_e2": 0,"e2_c": 1,"e2_p": 2,"e2_o": 3}

dic_dep = {"动补结构": 0, "定中关系": 1, "兼语": 2, "主谓关系": 3, "右附加关系": 4, "状中结构": 5, "并列关系": 6, "左附加关系": 7, "动宾关系": 8,
           "标点符号": 9, "前置宾语": 10, "介宾关系": 11, "核心关系": 12, "间宾关系" : 13}
dic_pos1 = {"b": 0, "ni": 1, "nz": 2, "ws": 3, "j": 4, "x": 5, "nd": 6, "nt": 7, "a": 8, "u": 9, "e": 10, "m": 11,
            "nh": 12, "n": 13, "d": 14, "k": 15, "p": 16, "q": 17, "c": 18, "r": 19, "wp": 20, "i": 21, "ns": 22,
            "v": 23, "o": 24, "nl": 25, }
dic_pos2 = {"vn": 0, "ry": 1, "nnd": 2, "g": 3, "rzt": 4, "gm": 5, "j": 6, "qt": 7, "ude1": 8, "gb": 9, "uzhi": 10,
            "pba": 11, "ulian": 12, "tg": 13, "t": 14, "ad": 15, "dg": 16, "d": 17, "cc": 18, "gc": 19, "f": 20,
            "p": 21, "ntu": 22, "mg": 23, "nnt": 24, "nit": 25, "nx": 26, "ng": 27, "an": 28, "usuo": 29, "nto": 30,
            "ag": 31, "ryv": 32, "nf": 33, "Rg": 34, "a": 35, "ntc": 36, "dl": 37, "vshi": 38, "ude2": 39, "Mg": 40,
            "k": 41, "vyou": 42, "rzv": 43, "rys": 44, "nhm": 45, "nmc": 46, "gg": 47, "vf": 48, "mq": 49, "al": 50,
            "nsf": 51, "rr": 52, "b": 53, "ule": 54, "ude3": 55, "bl": 56, "nt": 57, "s": 58, "uzhe": 59, "u": 60,
            "e": 61, "vx": 62, "l": 63, "w": 64, "n": 65, "gi": 66, "rzs": 67, "udeng": 68, "c": 69, "uls": 70,
            "nrf": 71, "o": 72, "nba": 73, "vd": 74, "vl": 75, "uguo": 76, "nz": 77, "x": 78, "uyy": 79, "y": 80,
            "nis": 81, "m": 82, "nhd": 83, "vi": 84, "pbei": 85, "nr": 86, "z": 87, "vg": 88, "udh": 89, "q": 90,
            "gp": 91, "ryt": 92, "r": 93, "i": 94, "qv": 95, "ns": 96, "v": 97, "rz": 98, }



dic_parsing_info = {}


def deal_with_sentences_with_sentences_with_PARSING_TREE():



		total = 0


		dic_dep = {"动补结构":0,"定中关系":1,"兼语":2,"主谓关系":3,"右附加关系":4,"状中结构":5,"并列关系":6,"左附加关系":7,"动宾关系":8,"标点符号":9,"前置宾语":10,"介宾关系":11,"核心关系":12,}
		dic_pos1 = {"b":0,"ni":1,"nz":2,"ws":3,"j":4,"x":5,"nd":6,"nt":7,"a":8,"u":9,"e":10,"m":11,"nh":12,"n":13,"d":14,"k":15,"p":16,"q":17,"c":18,"r":19,"wp":20,"i":21,"ns":22,"v":23,"o":24,"nl":25,}
		dic_pos2={"vn":0,"ry":1,"nnd":2,"g":3,"rzt":4,"gm":5,"j":6,"qt":7,"ude1":8,"gb":9,"uzhi":10,"pba":11,"ulian":12,"tg":13,"t":14,"ad":15,"dg":16,"d":17,"cc":18,"gc":19,"f":20,"p":21,"ntu":22,"mg":23,"nnt":24,"nit":25,"nx":26,"ng":27,"an":28,"usuo":29,"nto":30,"ag":31,"ryv":32,"nf":33,"Rg":34,"a":35,"ntc":36,"dl":37,"vshi":38,"ude2":39,"Mg":40,"k":41,"vyou":42,"rzv":43,"rys":44,"nhm":45,"nmc":46,"gg":47,"vf":48,"mq":49,"al":50,"nsf":51,"rr":52,"b":53,"ule":54,"ude3":55,"bl":56,"nt":57,"s":58,"uzhe":59,"u":60,"e":61,"vx":62,"l":63,"w":64,"n":65,"gi":66,"rzs":67,"udeng":68,"c":69,"uls":70,"nrf":71,"o":72,"nba":73,"vd":74,"vl":75,"uguo":76,"nz":77,"x":78,"uyy":79,"y":80,"nis":81,"m":82,"nhd":83,"vi":84,"pbei":85,"nr":86,"z":87,"vg":88,"udh":89,"q":90,"gp":91,"ryt":92,"r":93,"i":94,"qv":95,"ns":96,"v":97,"rz":98,}

		dic_parsing_info = {}

		for line in open("./origin_data/for parsing tree/for parsing tree/positive_sentences_with_parsing_tree.txt", mode='r'):
			# print(line)
			line = line.split(' ')

			en1 = line[2]
			en2 = line[3]
			ralation = line[4]

			line = ' '.join(line[5:])
			# print(line)
			if 'null' in line:
				# print('NULL~~~~~')
				continue

			line = line.split('|||')

			# print(line)
			parsing_list = []

			for i in line[:-1]:
				i = i.split(' ')
				if len(i) != 8:
					# print(i)
					continue
				parsing_tree_info = Parsing_tree(i)
				parsing_list += parsing_tree_info,

			dic_parsing_info[(en1,en2)] = dic_parsing_info.get((en1,en2),[]) + [parsing_tree_info]

			total += 1
			if total == -1000:
				break

		print len(dic_parsing_info)



		return dic_parsing_info



# embedding the position
def pos_embed(x):
	if x < -60:
		return 0
	if x >= -60 and x <= 60:
		return x + 61
	if x > 60:
		return 122


# reading data
def init(dic_parsing_info):
	print 'reading word embedding data...'


	word2id = np.load('./origin_data/for parsing tree/word2id.npy')
	word2id = word2id.item()

	print len(word2id.keys())


	print 'reading relation to id'
	relation2id = {}
	f = open('./origin_data/for parsing tree/relation2id.txt', 'r')
	while True:
		content = f.readline()
		if content == '':
			break
		content = content.strip().split()
		relation2id[content[0]] = int(content[1])
	f.close()


	fixlen = 100

	# max length of position embedding is 60 (-60~+60)
	maxlen = 60

	train_sen = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
	train_ans = {}  # {entity pair:label} the label is one-hot vector

	print 'reading train data...'
	for content in open("./origin_data/for parsing tree/train.txt", mode='r'):

		if content == '':
			break
		if 'null' in content:
			continue

		content = content.strip().split()
		try:
			# get entity name
			en1 = content[2]
			en2 = content[3]
		except:
			print content, 'this is except'
			continue

		# put the same entity pair sentences into a dict
		tup = (en1, en2)


		if tup not in train_sen:
			train_sen[tup] = []
			y_id = relation2id[content[4]]
			label = [0 for i in range(len(relation2id))]
			label[y_id] = 1
			train_ans[tup] = label

		content = ' '.join(content[5:])

		content = content.split('|||')


		parsing_list = []

		sentence = []

		for i in content[:-1]:
			i = i.split(' ')
			if len(i) != 8:
				continue
			sentence += i[1],
			parsing_tree_info = Parsing_tree(i)
			parsing_list += parsing_tree_info,


		en1pos = 0
		en2pos = 0

		for i in range(len(sentence)):
			if sentence[i] == en1:
				en1pos = i
			if sentence[i] == en2:
				en2pos = i
		output = []

		for i in range(fixlen):
			word = word2id['BLANK']
			rel_e1 = pos_embed(i - en1pos)
			rel_e2 = pos_embed(i - en2pos)

			if i < len(sentence):
				output.append([word, rel_e1, rel_e2] + parsing_list[i].get_info())
			else:
				output.append([word, rel_e1, rel_e2] + [0] * 6)

		for i in range(min(fixlen, len(sentence))):
			word = 0
			if sentence[i] not in word2id:
				word = word2id['UNK']
			else:
				word = word2id[sentence[i]]

			output[i][0] = word

		train_sen[tup].append(output)


	print('reading test data ...')

	test_sen = {}
	test_ans = {}

	f = open('./origin_data/for parsing tree/test.txt', 'r')

	while True:
		content = f.readline()
		if content == '':
			break
		if 'null' in content:
			continue

		content = content.strip().split()
		en1 = content[2]
		en2 = content[3]
		tup = (en1, en2)
		if tup not in test_sen:
			test_sen[tup] = []
			y_id = relation2id[content[4]]
			label = [0 for i in range(len(relation2id))]
			label[y_id] = 1
			test_ans[tup] = label


		content = ' '.join(content[5:])
		content = content.split('|||')

		parsing_list = []

		sentence = []

		for i in content[:-1]:
			i = i.split(' ')
			if len(i) != 8:
				# print(i)
				continue
			sentence += i[1],
			parsing_tree_info = Parsing_tree(i)
			parsing_list += parsing_tree_info,

		en1pos = 0
		en2pos = 0

		for i in range(len(sentence)):
			if sentence[i] == en1:
				en1pos = i
			if sentence[i] == en2:
				en2pos = i
		output = []

		for i in range(fixlen):
			word = word2id['BLANK']
			rel_e1 = pos_embed(i - en1pos)
			rel_e2 = pos_embed(i - en2pos)


			#ssc
			if i < len(sentence):
				output.append([word, rel_e1, rel_e2] + parsing_list[i].get_info())
			else:
				output.append([word, rel_e1, rel_e2] + [0] * 6)


		for i in range(min(fixlen, len(sentence))):
			word = 0
			if sentence[i] not in word2id:
				word = word2id['UNK']
			else:
				word = word2id[sentence[i]]

			output[i][0] = word
		test_sen[tup].append(output)





	train_x = []
	train_y = []
	test_x = []
	test_y = []

	print 'organizing train data'
	f = open('./data for parsing tree/train_q&a.txt','w')
	temp = 0
	for i in train_sen: #train_sen = {} #{entity pair:[[sentence 1],[sentence 2]...]}
		train_x.append(train_sen[i])
		train_y.append(train_ans[i])
		f.write(str(temp)+'\t'+i[0]+'\t'+i[1]+'\t'+str(np.argmax(train_ans[i]))+'\n')
		temp+=1
	f.close()

	print 'organizing test data'
	id2tup = []
	temp = 0
	f = open('./data for parsing tree/test_q&a.txt','w')
	for i in test_sen:
		test_x.append(test_sen[i])
		test_y.append(test_ans[i])
		id2tup.append(i)
		f.write(str(temp)+'\t'+i[0]+'\t'+i[1]+'\t'+str(np.argmax(test_ans[i]))+'\n')
		temp+=1
	f.close()




	train_x = np.array(train_x)
	train_y = np.array(train_y)
	test_x = np.array(test_x)
	test_y = np.array(test_y)

	np.save('./data for parsing tree/train_x.npy', train_x)
	np.save('./data for parsing tree/train_y.npy', train_y)
	np.save('./data for parsing tree/testall_x.npy', test_x)
	np.save('./data for parsing tree/testall_y.npy', test_y)



	#get test data for P@N evaluation, in which only entity pairs with more than 1 sentence exist
	print 'get test data for p@n test'

	pone_test_x = []
	pone_test_y = []

	ptwo_test_x = []
	ptwo_test_y = []

	pall_test_x = []
	pall_test_y = []

	for i in range(len(test_x)):
		if len(test_x[i]) > 1:

			pall_test_x.append(test_x[i])
			pall_test_y.append(test_y[i])

			onetest = []
			temp = np.random.randint(len(test_x[i]))
			onetest.append(test_x[i][temp])
			pone_test_x.append(onetest)
			pone_test_y.append(test_y[i])

			twotest = []
			temp1 = np.random.randint(len(test_x[i]))
			temp2 = np.random.randint(len(test_x[i]))
			while temp1 == temp2:
				temp2 = np.random.randint(len(test_x[i]))
			twotest.append(test_x[i][temp1])
			twotest.append(test_x[i][temp2])
			ptwo_test_x.append(twotest)
			ptwo_test_y.append(test_y[i])

	pone_test_x = np.array(pone_test_x)
	pone_test_y = np.array(pone_test_y)
	ptwo_test_x = np.array(ptwo_test_x)
	ptwo_test_y = np.array(ptwo_test_y)
	pall_test_x = np.array(pall_test_x)
	pall_test_y = np.array(pall_test_y)

	np.save('./data for parsing tree/pone_test_x.npy',pone_test_x)
	np.save('./data for parsing tree/pone_test_y.npy',pone_test_y)
	np.save('./data for parsing tree/ptwo_test_x.npy',ptwo_test_x)
	np.save('./data for parsing tree/ptwo_test_y.npy',ptwo_test_y)
	np.save('./data for parsing tree/pall_test_x.npy',pall_test_x)
	np.save('./data for parsing tree/pall_test_y.npy',pall_test_y)





def seperate():


	print 'reading training data'
	x_train = np.load('./data for parsing tree/train_x.npy')

	train_word = []
	train_pos1 = []
	train_pos2 = []

	# ssc parsing info
	train_root = []
	train_e1 = []
	train_e2 = []
	train_dep = []
	train_pospeech1 = []
	train_pospeech2 = []


	print 'seprating train data'
	for i in range(len(x_train)):
		word = []
		pos1 = []
		pos2 = []

		root = []
		e1 = []
		e2 = []
		dep = []
		pospeech1 = []
		pospeech2 = []

		for j in x_train[i]:
			temp_word = []
			temp_pos1 = []
			temp_pos2 = []

			temp_root = []
			temp_e1 = []
			temp_e2 = []
			temp_dep = []
			temp_pospeech1 = []
			temp_pospeech2 = []

			for k in j:

				temp_word.append(k[0])
				temp_pos1.append(k[1])
				temp_pos2.append(k[2])

				temp_root.append(k[3])
				temp_e1.append(k[4])
				temp_e2.append(k[5])
				temp_dep.append(k[6])
				temp_pospeech1.append(k[7])
				temp_pospeech2.append(k[8])

			word.append(temp_word)
			pos1.append(temp_pos1)
			pos2.append(temp_pos2)

			root.append(temp_root)
			e1.append(temp_e1)
			e2.append(temp_e2)
			dep.append(temp_dep)
			pospeech1.append(temp_pospeech1)
			pospeech2.append(temp_pospeech2)

		train_word.append(word)
		train_pos1.append(pos1)
		train_pos2.append(pos2)

		train_root.append(root)
		train_e1.append(e1)
		train_e2.append(e2)
		train_dep.append(dep)
		train_pospeech1.append(pospeech1)
		train_pospeech2.append(pospeech2)

	train_word = np.array(train_word)
	train_pos1 = np.array(train_pos1)
	train_pos2 = np.array(train_pos2)

	train_root = np.array(train_root)
	train_e1 = np.array(train_e1)
	train_e2 = np.array(train_e2)
	train_dep = np.array(train_dep)
	train_pospeech1 = np.array(train_pospeech1)
	train_pospeech2 = np.array(train_pospeech2)

	print len(train_word)
	print len(train_root)
	print len(train_pospeech1)

	np.save('./data for parsing tree/train_word.npy', train_word)
	np.save('./data for parsing tree/train_pos1.npy', train_pos1)
	np.save('./data for parsing tree/train_pos2.npy', train_pos2)

	np.save('./data for parsing tree/train_root.npy', train_root)
	np.save('./data for parsing tree/train_e1.npy', train_e1)
	np.save('./data for parsing tree/train_e2.npy', train_e2)
	np.save('./data for parsing tree/train_dep.npy', train_dep)
	np.save('./data for parsing tree/train_pospeech1.npy', train_pospeech1)
	np.save('./data for parsing tree/train_pospeech2.npy', train_pospeech2)




	print 'reading p-one test data'
	x_test = np.load('./data for parsing tree/pone_test_x.npy')
	print 'seperating p-one test data'
	test_word = []
	test_pos1 = []
	test_pos2 = []

	test_root = []
	test_e1 = []
	test_e2 = []
	test_dep = []
	test_pospeech1 = []
	test_pospeech2 = []

	for i in range(len(x_test)):
		word = []
		pos1 = []
		pos2 = []

		root = []
		e1 = []
		e2 = []
		dep = []
		pospeech1 = []
		pospeech2 = []

		for j in x_test[i]:
			temp_word = []
			temp_pos1 = []
			temp_pos2 = []

			temp_root = []
			temp_e1 = []
			temp_e2 = []
			temp_dep = []
			temp_pospeech1 = []
			temp_pospeech2 = []

			for k in j:
				temp_word.append(k[0])
				temp_pos1.append(k[1])
				temp_pos2.append(k[2])


				temp_root.append(k[3])
				temp_e1.append(k[4])
				temp_e2.append(k[5])
				temp_dep.append(k[6])
				temp_pospeech1.append(k[7])
				temp_pospeech2.append(k[8])

			word.append(temp_word)
			pos1.append(temp_pos1)
			pos2.append(temp_pos2)


			root.append(temp_root)
			e1.append(temp_e1)
			e2.append(temp_e2)
			dep.append(temp_dep)
			pospeech1.append(temp_pospeech1)
			pospeech2.append(temp_pospeech2)

		test_word.append(word)
		test_pos1.append(pos1)
		test_pos2.append(pos2)

		test_root.append(root)
		test_e1.append(e1)
		test_e2.append(e2)
		test_dep.append(dep)
		test_pospeech1.append(pospeech1)
		test_pospeech2.append(pospeech2)


	test_word = np.array(test_word)
	test_pos1 = np.array(test_pos1)
	test_pos2 = np.array(test_pos2)

	test_root = np.array(test_root)
	test_e1 = np.array(test_e1)
	test_e2 = np.array(test_e2)
	test_dep = np.array(test_dep)
	test_pospeech1 = np.array(test_pospeech1)
	test_pospeech2 = np.array(test_pospeech2)

	np.save('./data for parsing tree/pone_test_word.npy',test_word)
	np.save('./data for parsing tree/pone_test_pos1.npy',test_pos1)
	np.save('./data for parsing tree/pone_test_pos2.npy',test_pos2)

	np.save('./data for parsing tree/pone_test_root.npy', test_root)
	np.save('./data for parsing tree/pone_test_e1.npy', test_e1)
	np.save('./data for parsing tree/pone_test_e2.npy', test_e2)
	np.save('./data for parsing tree/pone_test_dep.npy', test_dep)
	np.save('./data for parsing tree/pone_test_pospeech1.npy', test_pospeech1)
	np.save('./data for parsing tree/pone_test_pospeech2.npy', test_pospeech2)



	print 'reading p-two test data'
	x_test = np.load('./data for parsing tree/ptwo_test_x.npy')
	print 'seperating p-two test data'
	test_word = []
	test_pos1 = []
	test_pos2 = []

	test_root = []
	test_e1 = []
	test_e2 = []
	test_dep = []
	test_pospeech1 = []
	test_pospeech2 = []


	for i in range(len(x_test)):
		word = []
		pos1 = []
		pos2 = []

		root = []
		e1 = []
		e2 = []
		dep = []
		pospeech1 = []
		pospeech2 = []

		for j in x_test[i]:
			temp_word = []
			temp_pos1 = []
			temp_pos2 = []


			temp_root = []
			temp_e1 = []
			temp_e2 = []
			temp_dep = []
			temp_pospeech1 = []
			temp_pospeech2 = []

			for k in j:
				temp_word.append(k[0])
				temp_pos1.append(k[1])
				temp_pos2.append(k[2])


				temp_root.append(k[3])
				temp_e1.append(k[4])
				temp_e2.append(k[5])
				temp_dep.append(k[6])
				temp_pospeech1.append(k[7])
				temp_pospeech2.append(k[8])

			word.append(temp_word)
			pos1.append(temp_pos1)
			pos2.append(temp_pos2)


			root.append(temp_root)
			e1.append(temp_e1)
			e2.append(temp_e2)
			dep.append(temp_dep)
			pospeech1.append(temp_pospeech1)
			pospeech2.append(temp_pospeech2)

		test_word.append(word)
		test_pos1.append(pos1)
		test_pos2.append(pos2)



		test_root.append(root)
		test_e1.append(e1)
		test_e2.append(e2)
		test_dep.append(dep)
		test_pospeech1.append(pospeech1)
		test_pospeech2.append(pospeech2)


	test_word = np.array(test_word)
	test_pos1 = np.array(test_pos1)
	test_pos2 = np.array(test_pos2)


	test_root = np.array(test_root)
	test_e1 = np.array(test_e1)
	test_e2 = np.array(test_e2)
	test_dep = np.array(test_dep)
	test_pospeech1 = np.array(test_pospeech1)
	test_pospeech2 = np.array(test_pospeech2)

	np.save('./data for parsing tree/ptwo_test_word.npy',test_word)
	np.save('./data for parsing tree/ptwo_test_pos1.npy',test_pos1)
	np.save('./data for parsing tree/ptwo_test_pos2.npy',test_pos2)



	np.save('./data for parsing tree/ptwo_test_root.npy', test_root)
	np.save('./data for parsing tree/ptwo_test_e1.npy', test_e1)
	np.save('./data for parsing tree/ptwo_test_e2.npy', test_e2)
	np.save('./data for parsing tree/ptwo_test_dep.npy', test_dep)
	np.save('./data for parsing tree/ptwo_test_pospeech1.npy', test_pospeech1)
	np.save('./data for parsing tree/ptwo_test_pospeech2.npy', test_pospeech2)


	print 'reading p-all test data'
	x_test = np.load('./data for parsing tree/pall_test_x.npy')
	print 'seperating p-all test data'
	test_word = []
	test_pos1 = []
	test_pos2 = []

	test_root = []
	test_e1 = []
	test_e2 = []
	test_dep = []
	test_pospeech1 = []
	test_pospeech2 = []


	for i in range(len(x_test)):
		word = []
		pos1 = []
		pos2 = []


		root = []
		e1 = []
		e2 = []
		dep = []
		pospeech1 = []
		pospeech2 = []

		for j in x_test[i]:
			temp_word = []
			temp_pos1 = []
			temp_pos2 = []


			temp_root = []
			temp_e1 = []
			temp_e2 = []
			temp_dep = []
			temp_pospeech1 = []
			temp_pospeech2 = []

			for k in j:
				temp_word.append(k[0])
				temp_pos1.append(k[1])
				temp_pos2.append(k[2])


				temp_root.append(k[3])
				temp_e1.append(k[4])
				temp_e2.append(k[5])
				temp_dep.append(k[6])
				temp_pospeech1.append(k[7])
				temp_pospeech2.append(k[8])

			word.append(temp_word)
			pos1.append(temp_pos1)
			pos2.append(temp_pos2)


			root.append(temp_root)
			e1.append(temp_e1)
			e2.append(temp_e2)
			dep.append(temp_dep)
			pospeech1.append(temp_pospeech1)
			pospeech2.append(temp_pospeech2)

		test_word.append(word)
		test_pos1.append(pos1)
		test_pos2.append(pos2)

		test_root.append(root)
		test_e1.append(e1)
		test_e2.append(e2)
		test_dep.append(dep)
		test_pospeech1.append(pospeech1)
		test_pospeech2.append(pospeech2)


	test_word = np.array(test_word)
	test_pos1 = np.array(test_pos1)
	test_pos2 = np.array(test_pos2)

	test_root = np.array(test_root)
	test_e1 = np.array(test_e1)
	test_e2 = np.array(test_e2)
	test_dep = np.array(test_dep)
	test_pospeech1 = np.array(test_pospeech1)
	test_pospeech2 = np.array(test_pospeech2)


	np.save('./data for parsing tree/pall_test_word.npy',test_word)
	np.save('./data for parsing tree/pall_test_pos1.npy',test_pos1)
	np.save('./data for parsing tree/pall_test_pos2.npy',test_pos2)

	np.save('./data for parsing tree/pall_test_root.npy', test_root)
	np.save('./data for parsing tree/pall_test_e1.npy', test_e1)
	np.save('./data for parsing tree/pall_test_e2.npy', test_e2)
	np.save('./data for parsing tree/pall_test_dep.npy', test_dep)
	np.save('./data for parsing tree/pall_test_pospeech1.npy', test_pospeech1)
	np.save('./data for parsing tree/pall_test_pospeech2.npy', test_pospeech2)




	print 'seperating test all data'
	x_test = np.load('./data for parsing tree/testall_x.npy')

	test_word = []
	test_pos1 = []
	test_pos2 = []


	# ssc parsing info
	test_root = []
	test_e1 = []
	test_e2 = []
	test_dep = []
	test_pospeech1 = []
	test_pospeech2 = []

	for i in range(len(x_test)):
		word = []
		pos1 = []
		pos2 = []

		root = []
		e1 = []
		e2 = []
		dep = []
		pospeech1 = []
		pospeech2 = []


		for j in x_test[i]:
			# print j
			temp_word = []
			temp_pos1 = []
			temp_pos2 = []


			temp_root = []
			temp_e1 = []
			temp_e2 = []
			temp_dep = []
			temp_pospeech1 = []
			temp_pospeech2 = []

			for k in j:
				temp_word.append(k[0])
				temp_pos1.append(k[1])
				temp_pos2.append(k[2])

				temp_root.append(k[3])
				temp_e1.append(k[4])
				temp_e2.append(k[5])
				temp_dep.append(k[6])
				temp_pospeech1.append(k[7])
				temp_pospeech2.append(k[8])

			word.append(temp_word)
			pos1.append(temp_pos1)
			pos2.append(temp_pos2)

			root.append(temp_root)
			e1.append(temp_e1)
			e2.append(temp_e2)
			dep.append(temp_dep)
			pospeech1.append(temp_pospeech1)
			pospeech2.append(temp_pospeech2)


		test_word.append(word)
		test_pos1.append(pos1)
		test_pos2.append(pos2)


		test_root.append(root)
		test_e1.append(e1)
		test_e2.append(e2)
		test_dep.append(dep)
		test_pospeech1.append(pospeech1)
		test_pospeech2.append(pospeech2)


	test_word = np.array(test_word)
	test_pos1 = np.array(test_pos1)
	test_pos2 = np.array(test_pos2)

	test_root = np.array(test_root)
	test_e1 = np.array(test_e1)
	test_e2 = np.array(test_e2)
	test_dep = np.array(test_dep)
	test_pospeech1 = np.array(test_pospeech1)
	test_pospeech2 = np.array(test_pospeech2)


	np.save('./data for parsing tree/testall_word.npy', test_word)
	np.save('./data for parsing tree/testall_pos1.npy', test_pos1)
	np.save('./data for parsing tree/testall_pos2.npy', test_pos2)

	np.save('./data for parsing tree/test_root.npy', test_root)
	np.save('./data for parsing tree/test_e1.npy', test_e1)
	np.save('./data for parsing tree/test_e2.npy', test_e2)
	np.save('./data for parsing tree/test_dep.npy', test_dep)
	np.save('./data for parsing tree/test_pospeech1.npy', test_pospeech1)
	np.save('./data for parsing tree/test_pospeech2.npy', test_pospeech2)

	print len(test_word)
	print len(test_root)
	print len(temp_pospeech2)




# get answer metric for PR curve evaluation
def getans():
	test_y = np.load('./data for parsing tree/testall_y.npy')
	eval_y = []
	for i in test_y:
		eval_y.append(i[1:])
	allans = np.reshape(eval_y, (-1))
	np.save('./data for parsing tree/allans.npy', allans)

init(dic_parsing_info)
seperate()
getans()
