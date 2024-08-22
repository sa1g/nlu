DIR = dataset

TRAIN = https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt
TEST = https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt
EVAL = https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt


install:
	mkdir -p $(DIR)
	wget -P $(DIR) $(TRAIN)
	wget -P $(DIR) $(TEST)
	wget $(EVAL)

clean:
	rm -rf $(DIR)
