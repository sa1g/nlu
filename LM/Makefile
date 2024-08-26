DIR = dataset

TRAIN = https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt
TEST = https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt
EVAL = https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt


install:
	mkdir -p $(DIR)
	curl -L -o $(DIR)/ptb.train.txt $(TRAIN)
	curl -L -o $(DIR)/ptb.test.txt $(TEST)
	curl -L -o $(DIR)/ptb.valid.txt $(EVAL)

clean:
	rm -rf $(DIR)
