
import os
import pandas as pd
import numpy as np

from EEG.dev03.preprocessor_01 import preprocess_data
from EEG.dev03.X_y_test_train_02 import test_train_split_save
from EEG.dev03.encoding_03 import encoder_LSTM
from EEG.dev03.modeling_04 import initialize_lstm_model, compile_lstm_model
from EEG.dev03.training_05 import train_lstm_model
from EEG.dev03.evaluating_06 import evaluate_lstm
#from predictor_07 import pred_lstm

input_path = "./raw_data"
file_name = "Epileptic Seizure Recognition.csv"
file_path = os.path.join(input_path, file_name)

output_path = "./processed_data"
data_file = "data.csv"
data_path = os.path.join(output_path, data_file)
data = pd.read_csv(file_path)

X_pred_file_name = "random_test_samples.csv"
X_pred_file_path = os.path.join(output_path, X_pred_file_name)

X_pred = pd.read_csv(X_pred_file_path)

#######preprocessor_01
data = preprocess_data(data, data_path)
X = data.drop(columns = 'y')
y = data.y
print(f"✅ Raw data cleaned and separated (X,y)")


####### test_and_train_split_02
X_train, X_test, y_train, y_test  = test_train_split_save(X,y,output_path)

#### encoding_03 for LSTM
X_train,y_train = encoder_LSTM(X_train,y_train)
X_test, y_test = encoder_LSTM(X_test, y_test)

### modeling_04 for LSTM
input_shape = (178,1)
model = initialize_lstm_model(input_shape)
modelLSTM = compile_lstm_model(model)


##### training_05
model,history = train_lstm_model(model,
        X_train,y_train,
        batch_size=128,
        patience=1,
        validation_data=None, # overrides validation_split
        validation_split=0.2
        )


# Save LSTM  trained model
model.save("models/LSTMmodel.h5")

# evaluating_06
results_df = evaluate_lstm(X_test, y_test)

#predicting_07
#pred_dictionary = pred_lstm(X_pred)
