#!/bin/bash

echo "Running all scripts in sequence..."

echo "1. Running preprocessor_01.py"
python packagename/preprocessor_01.py
if [ $? -ne 0 ]; then
    echo "Error running preprocessor_01.py"
    exit 1
fi

echo "2. Running test_and_train_split_02.py"
python packagename/test_and_train_split_02.py
if [ $? -ne 0 ]; then
    echo "Error running test_and_train_split_02.py"
    exit 1
fi

echo "3. Running encoding_03.py"
python packagename/encoding_03.py
if [ $? -ne 0 ]; then
    echo "Error running encoding_03.py"
    exit 1
fi

echo "4. Running modeling_04.py"
python packagename/modeling_04.py
if [ $? -ne 0 ]; then
    echo "Error running modeling_04.py"
    exit 1
fi

echo "5. Running training_05.py"
python packagename/training_05.py
if [ $? -ne 0 ]; then
    echo "Error running training_05.py"
    exit 1
fi

echo "6. Running evaluating_06.py"
python packagename/evaluating_06.py
if [ $? -ne 0 ]; then
    echo "Error running evaluating_06.py"
    exit 1
fi

echo "All scripts ran successfully!" 