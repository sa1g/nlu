DIR = dataset

TRAIN = https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/master/data/laptop14_train.txt
TEST = https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/master/data/laptop14_test.txt

install:
	mkdir -p $(DIR)
	curl -L -o $(DIR)/laptop14_train.txt $(TRAIN)
	curl -L -o $(DIR)/laptop14_test.txt $(TEST)

clean:
	rm -rf $(DIR)
