DIR = dataset/ATIS

TRAIN = https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/test.json
TEST = https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/train.json

install:
	mkdir -p $(DIR)
	wget -P $(DIR) $(TRAIN)
	wget -P $(DIR) $(TEST)

clean:
	rm -rf $(DIR)
