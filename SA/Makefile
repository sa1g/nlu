DIR = dataset

TRAIN = https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/master/data/laptop14_train.txt
TEST = https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/master/data/laptop14_test.txt
EVAL = https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/master/evals.py

install:
	mkdir -p $(DIR)
	wget -P $(DIR) $(TRAIN)
	wget -P $(DIR) $(TEST)
	wget $(EVAL)

clean:
	rm -rf $(DIR)
