#!/bin/bash
echo "Download the datasets"

if [ ! -d "dataset" ]; then
  mkdir dataset
fi  

cd dataset

wget -P dataset/PennTreeBank https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt  
wget -P dataset/PennTreeBank https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt  
wget -P dataset/PennTreeBank https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt

echo "Done"