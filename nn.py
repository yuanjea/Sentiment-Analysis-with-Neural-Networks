import argparse
import datasets
import pandas
import transformers
import tensorflow as tf
import numpy

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (SimpleRNN, Dense, Conv1D, Conv2D, MaxPooling2D,
                                      Flatten, Bidirectional, LSTM, GRU, Embedding, 
                                      Dropout, GlobalMaxPooling1D)

# GPU Check
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# use the tokenizer from DistilRoBERTa
tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")


def tokenize(examples):
    """Converts the text of each example to "input_ids", a sequence of integers
    representing 1-hot vectors for each token in the text"""
    return tokenizer(examples["text"], truncation=True, max_length=64,
                     padding="max_length")


def to_bow(example):
    """Converts the sequence of 1-hot vectors into a single many-hot vector"""
    vector = numpy.zeros(shape=(tokenizer.vocab_size,))
    vector[example["input_ids"]] = 1
    return {"input_bow": vector}


def train(model_path="model", train_path="../graduate-project-data/train.csv", dev_path="../graduate-project-data/dev.csv"):

    # load the CSVs into Huggingface datasets to allow use of the tokenizer
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    # the labels are the names of all columns except the first
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Converts the label columns into a list of 0s and 1s"""
        # the float here is because F1Score requires floats
        return {"labels": [float(example[l]) for l in labels]}

    # convert text and labels to format expected by model
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    #hf_dataset = hf_dataset.map(to_bow) # For Feed Forward NN

    # convert Huggingface datasets to Tensorflow datasets
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns="input_ids", # "input_bow" for FF
        label_cols="labels", 
        batch_size=16,
        shuffle=True)
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns="input_ids", # input_bow for FF
        label_cols="labels",
        batch_size=16)
    
    # define a model with a single fully connected layer
    # Default model
    #model = tf.keras.Sequential([tf.keras.layers.Dense(units=len(labels), input_dim=tokenizer.vocab_size, activation='sigmoid')])
    #model.add(SimpleRNN(32, activation='relu', input_shape=(None, tokenizer.vocab_size), return_sequences=True))

    model = Sequential()
    model.add(Embedding(input_dim=tokenizer.vocab_size, output_dim=128, input_length=64))
    model.add(Bidirectional(GRU(64, return_sequences=True), input_shape=(tokenizer.vocab_size,)))
    model.add(Dropout(0.5)) 
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Flatten())
    model.add(Dense(len(labels), activation='sigmoid'))

    # specify compilation hyperparameters
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)])
    
    # Model summmary
    model.summary()

    # fit the model to the training data, monitoring F1 on the dev data
    history = model.fit(
        train_dataset,
        epochs=10,
        batch_size=64,
        validation_data=dev_dataset,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_f1_score",
                mode="max",
                save_best_only=True),
                EarlyStopping(monitor='val_f1_score', mode='auto', patience=5, restore_best_weights=True)])

    # Print final validation F1 score
    final_val_f1 = history.history['val_f1_score'][-1]
    print(f"Final Validation F1 Score: {final_val_f1:.4f}")

def predict(model_path="model", input_path="../graduate-project-data/dev.csv"):

    # load the saved model
    model = tf.keras.models.load_model(model_path)

    # load the data for prediction
    # use Pandas here to make assigning labels easier later
    df = pandas.read_csv(input_path)

    # create input features in the same way as in train()
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    #hf_dataset = hf_dataset.map(to_bow) # For Feed Forward NN
    tf_dataset = hf_dataset.to_tf_dataset(
        columns="input_ids", # input_bow for FF
        batch_size=16)

    # generate predictions from model
    predictions = numpy.where(model.predict(tf_dataset) > 0.5, 1, 0)

    # assign predictions to label columns in Pandas data frame
    df.iloc[:, 1:] = predictions

    # write the Pandas dataframe to a zipped CSV file
    df.to_csv("submission.zip", index=False, compression=dict(
        method='zip', archive_name=f'submission.csv'))
    
    # Calculate F1 score using tf.keras.metrics.F1Score
    f1_metric = tf.keras.metrics.F1Score(average="micro", threshold=0.5)
    
    # Load the true labels for the dev set
    true_labels_df = pandas.read_csv("../graduate-project-data/dev.csv")
    
    # Calculate the F1 score
    true_labels = true_labels_df.iloc[:, 1:].values
    f1_metric.update_state(true_labels, predictions)
    f1_score = f1_metric.result().numpy()
    
    print(f"F1 Score (Micro): {f1_score}")


if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    # call either train() or predict()
    globals()[args.command]()
