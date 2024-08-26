DIR = dataset/ATIS

TRAIN = https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/test.json
TEST = https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/train.json

install:
	mkdir -p $(DIR)
	curl -L -o $(DIR)/train.json $(TRAIN)
	curl -L -o $(DIR)/test.json $(TEST)

clean:
	rm -rf $(DIR)

